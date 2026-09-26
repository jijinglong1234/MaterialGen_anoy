"""
Phase-2 runner tests (scripts/run_phase2.py).

Covers the parts of the runner that a fleet run would only discover after
thousands of GPU-hours: the configuration table against the paper's Table
(tab:param-configs), the three dataset cell axes, the per-cell hull
calibration, the task-reuse guard, and one end-to-end task + analysis on a
Lennard-Jones calculator (no NNP, no GPU) so the payload/analysis schema is
pinned by a test rather than by the first full run.

Cases:
  T1  configuration table: sigma/beta/K/M/alpha_lat/update_lattice/NFE per row
  T2  layer resolution: C8/C9 = Layers 1-3, C11 = Layer 1, C5 = no lattice
  T3  MP-20 axis = the frozen Phase-1 compositions, one cell each
  T4  Perov-5 axis: 20 dual-split compositions, class-stratified, cached
  T5  Carbon-24 axis: 20 unique density strata, ascending, lattice forced off
  T6  per-cell calibration: MP-20/Perov-5 anchor == registry anchor
  T7  Carbon-24: per-cell anchor scores exactly its own DFT E_form
  T8  task reuse guard: config change / missing generation marker -> stale
  T9  end-to-end task on LJ: payload schema, records aligned with candidates
  T10 analysis pass on that payload: S.U.N./dynamics/AMSD schema, no `unique`
      flag stored per task

The `unique` flag is deliberately absent from per-task records: it is defined
against a pool, so it can only be computed in the analysis pass (T10 asserts
this rather than trusting it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_phase2 as p2  # noqa: E402

from materialgen.data.registry import get_dataset  # noqa: E402
from materialgen.eval.diversity import TrainingIndex  # noqa: E402
from materialgen.eval.metrics import ReferenceIndex  # noqa: E402
from materialgen.eval.stability import HullEvaluator, HullTable, canonical_formula  # noqa: E402

# ---------------------------------------------------------------------------
# T1/T2 - configuration table
# ---------------------------------------------------------------------------

#: name -> (sigma_max, beta, K, M, alpha_lat, update_lattice, layers) as
#: published in the App. C table, restricted to the Phase-2 subset.
PAPER_TABLE = {
    "C1": (0.5, 38.68, 100, 2, 0.001, True, "bare"),
    "C3": (1.5, 11.60, 100, 2, 0.002, True, "bare"),
    "C5": (0.5, 38.68, 100, 2, 0.001, False, "bare"),
    "C8": (0.5, 38.68, 100, 2, 0.001, True, "l1l2l3"),
    "C9": (1.5, 11.60, 100, 2, 0.002, True, "l1l2l3"),
    "C10": (0.5, 38.68, 100, 2, 0.001, False, "l1l2l3"),
    "C11": (0.5, 38.68, 100, 2, 0.001, True, "l1"),
}


class TestConfigTable:
    def test_rows_match_the_paper(self):
        for name, row in PAPER_TABLE.items():
            cfg = p2.CONFIGS[name]
            got = (cfg.sigma_max, cfg.beta, cfg.K, cfg.M, cfg.alpha_lat,
                   cfg.update_lattice, cfg.layers)
            assert got == row, f"{name}: {got} != {row}"

    def test_nfe_is_200_for_the_subset(self):
        # the paper tabulates NFE = K*M = 200 for every configuration here
        for name in p2.PHASE2_SUBSET:
            assert p2.CONFIGS[name].nfe == 200

    def test_subset_is_the_streamlined_six(self):
        assert p2.PHASE2_SUBSET == ("C1", "C3", "C5", "C8", "C9", "C11")
        # C10 is tabulated but excluded from the benchmark subset
        assert "C10" in p2.CONFIGS and "C10" not in p2.PHASE2_SUBSET

    def test_pfode_arm_is_the_frozen_three(self):
        """The freeze: PF-ODE runs {C5, C8, C11}, ALD runs all six.

        The full cartesian grid costs 543 sequential GPU-h (2.5x the plan's
        stale ~220 estimate) and 71% of it is the ODE arm.  Dropping the
        sigma-dependence configs from the ODE arm -- whose sigma curves already
        exist from the Phase-1 10-point scan -- is what buys the 320 GPU-h
        budget.  Pinned here because a silent re-widening would be invisible
        until the bill arrived.
        """
        assert p2.PFODE_CONFIGS == ("C5", "C8", "C11")
        for name in p2.PFODE_CONFIGS:
            assert name in p2.PHASE2_SUBSET
        # the excluded ODE arms are exactly the two high-sigma configs
        assert set(p2.PHASE2_SUBSET) - set(p2.PFODE_CONFIGS) == {"C1", "C3", "C9"}
        assert p2.configs_for("pfode") == ("C5", "C8", "C11")
        assert p2.configs_for("ald") == p2.PHASE2_SUBSET
        # an explicit --configs still filters down on the ODE side
        assert p2.configs_for("pfode", ("C1", "C8")) == ("C8",)

    def test_grid_size_is_2700(self):
        """The frozen grid, counted off the real task builder (not a formula).

        Locks the freeze numerically: 3 datasets x (6 ALD + 3 PF-ODE) configs x
        20 cells x 5 seeds = 2700 tasks = 27000 trajectories at N_CAND=10.
        """
        n_cells, n_seeds = 4, 2                    # small stand-in axis
        tasks = p2.build_tasks("mace", datasets=("mp_20",), n_cells=n_cells,
                               seeds=tuple(p2.SEEDS[:n_seeds]))
        expected = 1 * (6 + 3) * n_cells * n_seeds
        assert len(tasks) == expected
        samplers = {t["sampler"] for t, _ in tasks}
        assert samplers == {"ald", "pfode"}
        ode_cfgs = {t["config"] for t, _ in tasks if t["sampler"] == "pfode"}
        assert ode_cfgs == set(p2.PFODE_CONFIGS), ode_cfgs
        ald_cfgs = {t["config"] for t, _ in tasks if t["sampler"] == "ald"}
        assert ald_cfgs == set(p2.PHASE2_SUBSET), ald_cfgs
        # no task is emitted twice (the grid is a set, not a multiset)
        assert len({(t["config"], t["sampler"], t["cell"], t["seed"])
                    for t, _ in tasks}) == len(tasks)

    def test_layer_subsets(self):
        # C8/C9 carry Layers 1-3 (never L4: the SafetyMonitor is an ablation),
        # C11 carries Layer 1 only
        assert p2.CONFIGS["C8"].layers == p2.CONFIGS["C9"].layers == "l1l2l3"
        assert set(p2.rp.level_subset("l1l2l3")) == {1, 2, 3}
        assert set(p2.rp.level_subset("l1")) == {1}
        assert set(p2.rp.level_subset("bare")) == set()
        # ... and the resolution the fleet actually runs
        assert p2.config_meta(p2.CONFIGS["C8"])["layers_subset"] == [1, 2, 3]
        assert p2.config_meta(p2.CONFIGS["C11"])["layers_subset"] == [1]

    def test_c8_is_c1_plus_layers(self):
        c1, c8 = p2.CONFIGS["C1"], p2.CONFIGS["C8"]
        assert c8.sigma_max == c1.sigma_max and c8.beta == c1.beta
        assert c8.alpha_lat == c1.alpha_lat
        assert (c8.K, c8.M, c8.update_lattice) == (c1.K, c1.M, c1.update_lattice)

    def test_config_sha1_detects_edits(self):
        c1 = p2.CONFIGS["C1"]
        assert p2.config_sha1(c1) == p2.config_sha1(p2.CONFIGS["C1"])
        edited = p2.Config(**{**p2.CONFIGS["C1"].__dict__, "beta": 11.60})
        assert p2.config_sha1(edited) != p2.config_sha1(c1)

    def test_cpu_threads_are_bounded(self):
        """set_cpu_threads must impose the bound, not just ask for it: torch's
        default (one thread per core) made a CPU run 6-25x slower than the same
        run at 4-16 threads (see the runner's CPU-budget note)."""
        import torch

        before = torch.get_num_threads()
        try:
            n = p2.set_cpu_threads(3)
            assert n == 3 and torch.get_num_threads() == 3
            assert p2.DEFAULT_THREADS <= 16
        finally:
            torch.set_num_threads(before)


# ---------------------------------------------------------------------------
# T3/T4/T5 - cell axes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refs_mace():
    return get_dataset("mp_20").load_reference("mace", "test")


class TestMP20Axis:
    def test_cells_are_the_frozen_phase1_compositions(self):
        spec = get_dataset("mp_20")
        cells = p2.mp20_cells(spec, spec.load_reference("mace", "test"))
        assert len(cells) == 20
        assert [c.id for c in cells] == [canonical_formula(c)
                                         for c in p2.rp.COMPOSITIONS_20]
        assert len({c.id for c in cells}) == 20

    def test_n_cells_truncates_the_axis(self):
        """--n-cells must bound MP-20 too: it silently ran 20 cells before, so
        a 2-cell smoke wrote a 20-cell grid whose files were indistinguishable
        from real grid output."""
        spec = get_dataset("mp_20")
        refs = spec.load_reference("mace", "test")
        full = [c.id for c in p2.mp20_cells(spec, refs)]
        for n in (1, 2, 5):
            assert [c.id for c in p2.mp20_cells(spec, refs, n)] == full[:n]
        assert len(p2.cells_for(spec, refs, 2)) == 2

    def test_each_cell_is_a_real_test_structure(self):
        spec = get_dataset("mp_20")
        refs = spec.load_reference("mace", "test")
        for cell in p2.mp20_cells(spec, refs):
            rec = refs[cell.ref_index]
            assert canonical_formula(spec.formula(rec["metadata"])) == cell.formula


class TestPerov5Axis:
    def test_axis_is_dual_split_and_cached(self):
        spec = get_dataset("perov_5")
        axis = p2.perov5_axis(spec)
        assert len(axis) == 20 and len(set(axis)) == 20

        import pandas as pd

        train = {canonical_formula(f)
                 for f in pd.read_csv(spec.csv_dir / "train.csv")["formula"]}
        test = {canonical_formula(f)
                for f in pd.read_csv(spec.csv_dir / "test.csv")["formula"]}
        # every cell must be present in BOTH splits, or Novelty is free and the
        # match-rate reference pool is empty
        assert set(axis) <= (train & test)

        cached = json.loads(p2.PEROV5_AXIS_CACHE.read_text())
        assert cached["cells"] == axis and cached["rule"] == p2._axis_rule()

    def test_cells_resolve_to_test_structures(self):
        spec = get_dataset("perov_5")
        refs = spec.load_reference("mace", "test")
        cells = p2.perov5_cells(spec, refs)
        assert len(cells) == 20
        assert all(len(refs[c.ref_index]["numbers"]) == 5 for c in cells)


class TestCarbon24Axis:
    def test_strata_are_unique_and_ordered(self):
        spec = get_dataset("carbon_24")
        cells = p2.carbon24_cells(spec, spec.load_reference("mace", "test"))
        assert len(cells) == 20
        assert len({c.id for c in cells}) == 20
        rho = [c.meta["vol_per_atom"] for c in cells]
        assert rho == sorted(rho)
        assert all(5.4 < r < 8.95 for r in rho)

    def test_lattice_updates_are_forced_off(self):
        spec = get_dataset("carbon_24")
        assert spec.lattice_mode == "fixed"
        for name in ("C1", "C3", "C8", "C9", "C11"):
            assert not p2.effective_update_lattice(p2.CONFIGS[name], spec)
        # ... while the variable-lattice datasets keep the table's setting
        mp = get_dataset("mp_20")
        assert p2.effective_update_lattice(p2.CONFIGS["C1"], mp)
        assert not p2.effective_update_lattice(p2.CONFIGS["C5"], mp)


# ---------------------------------------------------------------------------
# T6/T7 - per-cell calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_mp20_cell_anchors_equal_registry_anchors(self):
        ctx = p2.context("mp_20", "mace")
        anchors = {r["formula_canonical"]: r for r in ctx.spec.calib("mace", "test")}
        for cell in ctx.cells:
            cc = ctx.cell_calib[cell.id]
            want = anchors[cell.formula]
            assert cc["e_ref"] == pytest.approx(want["nnp_energy_ref"], abs=1e-6)
            assert cc["n_atoms_ref"] == want["n_atoms_ref"]
            assert cc["e_form_ref_per_atom"] == pytest.approx(
                want["e_form_ref_per_atom"], abs=1e-9)

    def test_perov5_cell_anchors_equal_registry_anchors(self):
        ctx = p2.context("perov_5", "mace")
        anchors = {r["formula_canonical"]: r for r in ctx.spec.calib("mace", "test")}
        for cell in ctx.cells:
            cc = ctx.cell_calib[cell.id]
            assert cc["e_ref"] == pytest.approx(
                anchors[cell.formula]["nnp_energy_ref"], abs=1e-6)

    def test_carbon_anchor_scores_its_own_dft_formation_energy(self):
        """Per-cell anchoring is what keeps the 20 strata on one E_hull scale."""
        ctx = p2.context("carbon_24", "mace")
        for cell in ctx.cells:
            cc = ctx.calibrate_cell(cell)
            e_form = cc["e_ref"] / cc["n_atoms_ref"] - (
                cc["e_ref"] / cc["n_atoms_ref"] - cc["e_form_ref_per_atom"])
            assert e_form == pytest.approx(cc["e_form_ref_per_atom"], abs=1e-12)
            # and the hull evaluator must return exactly that DFT E_form
            assert ctx.hull.e_hull(cc["formula"], cc["e_ref"],
                                   cc["n_atoms_ref"]) == pytest.approx(
                ctx.spec.hull().e_above_hull("C", cc["e_form_ref_per_atom"]), abs=1e-9)


# ---------------------------------------------------------------------------
# T8 - task reuse guard
# ---------------------------------------------------------------------------

class TestTaskGuard:
    def _payload(self, cfg_name="C1"):
        return {"init_cell_gen": p2.rp.INIT_CELL_GEN,
                "config_sha1": p2.config_sha1(p2.CONFIGS[cfg_name])}

    def test_fresh_payload_is_reused(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(self._payload()))
        assert p2._task_ok(path, p2.CONFIGS["C1"])

    def test_config_edit_invalidates(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps(self._payload("C1")))
        assert not p2._task_ok(path, p2.CONFIGS["C3"])

    def test_missing_generation_marker_invalidates(self, tmp_path):
        path = tmp_path / "t.json"
        blob = self._payload()
        del blob["init_cell_gen"]
        path.write_text(json.dumps(blob))
        assert not p2._task_ok(path, p2.CONFIGS["C1"])

    def test_corrupt_file_invalidates(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text("{not json")
        assert not p2._task_ok(path, p2.CONFIGS["C1"])


# ---------------------------------------------------------------------------
# T9/T10 - end-to-end task + analysis on Lennard-Jones (no NNP, no GPU)
# ---------------------------------------------------------------------------

#: Lennard-Jones stand-in for a chemical bond: minimum at 2^(1/6)*2.0 = 2.245 A,
#: inside the covalent-radius cutoff that run_phase1.structure_diagnostics uses
#: (Si: max(2.0, 2*1.11+0.3) = 2.52 A).  That keeps the frozen Phase-1 d_min
#: and the metric chain's full-matrix d_min in agreement, which is what the
#: n_validity_definition_mismatch counter exists to police -- see
#: TestDiluteCellBoundary for the regime where they legitimately differ.
STUB_LJ = {"sigma": 2.0, "epsilon": 0.1}

#: Anchor cell: 8-atom diamond at a = 4.8, i.e. nearest neighbour 2.078 A --
#: compressed relative to the LJ minimum (2.245) and comfortably inside the
#: 2.52 A cutoff, so sampled structures relax *upward* in energy (E_hull > 0,
#: like a real generated structure) without ever approaching either edge.
STUB_A = 4.8
STUB_FRAC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                      [.25, .25, .25], [.25, .75, .75], [.75, .25, .75],
                      [.75, .75, .25]])


def _stub_calc():
    from ase.calculators.lj import LennardJones

    return LennardJones(**STUB_LJ)


class _StubCtx:
    """Minimal DatasetContext stand-in: one Si diamond cell, anchored hull."""

    def __init__(self, lattice_mode: str = "variable"):
        spec = SimpleNamespace(key="stub", name="stub", lattice_mode=lattice_mode)
        self.spec = spec
        self.cell = p2.Cell(id="Si", formula="Si", label="Si", ref_index=0)
        self.cells = [self.cell]
        self.record = {
            "numbers": np.array([14] * len(STUB_FRAC)),
            "frac_coords": STUB_FRAC.copy(),
            "lattice": np.eye(3) * STUB_A,
            "metadata": {"pretty_formula": "Si", "material_id": "stub-1"},
        }
        table = HullTable(name="stub", systems={"Si": {
            "elements": ["Si"],
            "entries": [{"formula": "Si", "natoms": len(STUB_FRAC),
                         "e_form_per_atom": 0.0}]}})
        self.hull = HullEvaluator(table)
        # `index` is the name DatasetContext exposes (run_task reads ctx.index)
        self.index = self.training = TrainingIndex({})
        self._reference_index = {}
        self.e_ref = float(self._anchor_atoms().get_potential_energy())

    def _anchor_atoms(self):
        from materialgen.core.crystal import CrystalStructure

        atoms = CrystalStructure.from_frac_coords(
            self.record["numbers"], self.record["frac_coords"],
            self.record["lattice"]).ase_atoms
        atoms.calc = _stub_calc()
        return atoms

    def cell_ref(self, cell):
        return {**self.record, "metadata": dict(self.record["metadata"]),
                "energy": self.e_ref}

    def calibrate_cell(self, cell):
        cc = {"formula": "Si", "e_ref": self.e_ref,
              "n_atoms_ref": len(STUB_FRAC),
              "e_form_ref_per_atom": 0.0, "material_id": "stub-1"}
        self.hull.calibrate(cc["formula"], cc["e_ref"], cc["n_atoms_ref"],
                            cc["e_form_ref_per_atom"])
        return cc

    def reference_structures(self, cell):
        from materialgen.utils.crystal_io import to_pymatgen

        from materialgen.core.crystal import CrystalStructure

        return [to_pymatgen(CrystalStructure.from_frac_coords(
            self.record["numbers"], self.record["frac_coords"], self.record["lattice"]))]

    def reference_index(self, cell):
        """Same object the real ctx hands analyze_cell (see ReferenceIndex)."""
        if cell.id not in self._reference_index:
            self._reference_index[cell.id] = ReferenceIndex(
                self.reference_structures(cell))
        return self._reference_index[cell.id]


@pytest.fixture
def stub_env(monkeypatch):
    monkeypatch.setattr(p2, "N_CAND", 3)
    return _StubCtx(), _stub_calc()


class TestTaskEndToEnd:
    def _run(self, ctx, calc, config="C1", sampler="ald"):
        task = {"dataset": "stub", "nnp": "mace", "config": config,
                "sampler": sampler, "cell": ctx.cell.id, "seed": 42}
        return task, p2.run_task(task, ctx, calc)

    def test_payload_schema_ald(self, stub_env):
        ctx, calc = stub_env
        task, payload = self._run(ctx, calc)
        for key in ("dataset", "nnp", "config", "sampler", "cell", "seed",
                    "config_spec", "config_sha1", "init_cell_gen", "e_hull_units",
                    "effective_update_lattice", "lattice_forced_off",
                    "n_atoms", "n_candidates", "ref", "candidates", "records"):
            assert key in payload, key
        assert payload["e_hull_units"] == "eV/atom"
        assert payload["init_cell_gen"] == p2.rp.INIT_CELL_GEN
        assert payload["n_candidates"] == len(payload["candidates"]) == 3
        assert payload["config_sha1"] == p2.config_sha1(p2.CONFIGS["C1"])
        assert payload["ref"]["n_atoms_ref"] == len(STUB_FRAC)
        assert payload["effective_update_lattice"] is True
        assert payload["lattice_forced_off"] is False
        # every candidate keeps the Phase-1 fields the analysis reads
        for c in payload["candidates"]:
            for key in ("nfe", "valid", "ood_steps", "ood_bitmask", "type1_steps",
                        "type2_steps", "nan_steps", "final", "n_rejections"):
                assert key in c, key
            assert "structure" in c["final"]

    def test_records_align_with_candidates(self, stub_env):
        ctx, calc = stub_env
        _, payload = self._run(ctx, calc)
        assert len(payload["records"]) == len(payload["candidates"]) == 3
        for rec, cand in zip(payload["records"], payload["candidates"]):
            assert rec["formula"] == "Si"
            # `unique` is pool-defined and must not be frozen per task
            assert "unique" not in rec
            # `physical` is derived from e_hull, so it is stored and reproduced
            assert rec["physical"] == (rec["e_hull"] is None
                                       or rec["e_hull"] >= -0.05)
            assert rec["valid"] == (cand["final"]["d_min"] >= 0.5)
            if rec["energy"] is not None:
                # the payload rounds E_hull to 4 decimals (frozen Phase-1
                # payload format); the record keeps the full value
                assert rec["e_hull"] == pytest.approx(
                    cand["final"]["e_hull"], abs=2e-4)

    def test_energy_and_e_hull_are_on_the_dft_scale(self, stub_env):
        """The anchored structure must sit exactly on the hull it defines."""
        ctx, calc = stub_env
        _, payload = self._run(ctx, calc)
        assert ctx.hull.e_form_per_atom("Si", ctx.e_ref, len(STUB_FRAC)) == \
            pytest.approx(0.0, abs=1e-12)

    def test_pfode_arm_runs(self, stub_env):
        ctx, calc = stub_env
        _, payload = self._run(ctx, calc, config="C11", sampler="pfode")
        assert payload["sampler"] == "pfode"
        assert payload["config_spec"]["layers_subset"] == [1]
        assert all(c["nfe"] > 0 for c in payload["candidates"])

    def test_lattice_forced_off_for_fixed_lattice_dataset(self, stub_env, monkeypatch):
        ctx = _StubCtx(lattice_mode="fixed")
        _, payload = self._run(ctx, _stub_calc(), config="C1")
        assert payload["effective_update_lattice"] is False
        assert payload["lattice_forced_off"] is True


class TestDiluteCellBoundary:
    """Where the frozen Phase-1 d_min and the metric chain's d_min diverge.

    ``run_phase1.structure_diagnostics`` builds a neighbor list with cutoff
    max(2.0, 2*R_cov+0.3) and returns inf when no pair is inside it, while the
    metric chain's ``validity_reasons`` uses the full minimum-image matrix.
    For a dense cell the two agree (that is the point of STUB_LJ); for a cell
    expanded past the cutoff the frozen definition calls the structure invalid
    (non-finite d_min) and the chain calls it valid, which is exactly what the
    per-task ``n_validity_definition_mismatch`` counter counts.  Phase 2
    reports validity from the chain (paper definition: composition + d_min
    >= 0.5 A) and flags the disagreement instead of hiding it.
    """

    def test_expanded_cell_is_counted_as_a_mismatch(self):
        ctx = _StubCtx()
        ctx.record["lattice"] = np.eye(3) * (STUB_A * 2.0)   # nn 4.7 A > 2.52
        ctx.e_ref = float(ctx._anchor_atoms().get_potential_energy())
        task = {"dataset": "stub", "nnp": "mace", "config": "C5",
                "sampler": "ald", "cell": ctx.cell.id, "seed": 42}
        payload = p2.run_task(task, ctx, _stub_calc())
        assert payload["n_validity_definition_mismatch"] > 0
        cand = payload["candidates"][0]
        assert not np.isfinite(cand["final"]["d_min"])       # frozen: inf
        assert payload["records"][0]["valid"] is True        # chain: fine


class TestTwoSidedStability:
    """The stability band: a blow-up must not read as a stability success.

    Numbers taken from the Carbon-24 PF-ODE pilot probe (max|F| = 611 eV/A,
    d_min = 0.834 A, NNP E/N = -67.583 against a reference at -9.057, i.e.
    E_hull = -58.5154 eV/atom).
    """

    def _rec(self, e_hull):
        from materialgen.eval.metrics import StructureRecord

        return StructureRecord(formula="C", structure=None, e_hull=e_hull)

    def test_floor_and_threshold_band(self):
        for e_hull, physical, stable in [
                (-58.5154, False, False),      # the pilot's blow-up
                (-0.051, False, False),        # just outside the floor
                (-0.05, True, True),           # the floor itself is inclusive
                (-0.001, True, True),          # NNP error scale, legitimately below
                (0.0, True, True),             # on the hull
                (0.0999, True, True),
                (0.1, True, False),            # threshold is strict
                (0.5, True, False),
        ]:
            rec = self._rec(e_hull)
            assert rec.physical is physical, (e_hull, "physical")
            assert rec.stable is stable, (e_hull, "stable")

    def test_blowup_is_not_sun(self):
        from materialgen.eval.metrics import StructureRecord, compute_sun

        recs = [StructureRecord(formula="C", structure=None, e_hull=-58.5154,
                                valid=True, novel=True, unique=True),
                StructureRecord(formula="C", structure=None, e_hull=0.02,
                                valid=True, novel=True, unique=True)]
        sun = compute_sun(recs, mark=False)
        assert sun["n_sun"] == 1                    # only the real one
        assert sun["stable_rate"] == pytest.approx(0.5)
        assert sun["n_unphysical"] == 1
        assert sun["unphysical_rate"] == pytest.approx(0.5)
        assert sun["unphysical_e_hull_min"] == pytest.approx(-58.5154)
        assert sun["e_hull_floor"] == 0.05

    def test_unphysical_is_never_the_cluster_representative(self):
        """A blow-up must not steal uniqueness from a real structure.

        The two are within e_hull_tol (|−0.06 − (−0.045)| = 15 meV) of each
        other and share space group and volume, so the duplicate rule pairs
        them; ordering by E_hull alone would keep the blow-up (it is lower)
        and mark the real one a duplicate.
        """
        from materialgen.eval.metrics import StructureRecord, mark_unique

        bad = StructureRecord(formula="C", structure=None, e_hull=-0.06,
                              spacegroup=227, volume=20.0)
        good = StructureRecord(formula="C", structure=None, e_hull=-0.045,
                               spacegroup=227, volume=20.0)
        mark_unique([bad, good])
        assert good.unique is True and bad.unique is False

    def test_hull_summary_keeps_both_medians(self):
        from materialgen.eval.stability import hull_summary

        s = hull_summary([-58.5154, 0.0232, 0.0603])
        assert s["n"] == 3 and s["n_unphysical"] == 1
        assert s["median"] == pytest.approx(0.0232)          # raw, dragged
        assert s["median_physical"] == pytest.approx((0.0232 + 0.0603) / 2)
        assert s["min"] == pytest.approx(-58.5154)
        assert s["frac_unphysical"] == pytest.approx(1 / 3)

    def test_evaluator_derives_the_same_band(self, stub_env):
        """The record builder and the record dataclass must agree."""
        ctx, calc = stub_env
        from materialgen.core.crystal import CrystalStructure
        from materialgen.eval.metrics import StructureEvaluator

        ctx.calibrate_cell(ctx.cell)
        struct = CrystalStructure.from_frac_coords(
            ctx.record["numbers"], ctx.record["frac_coords"], ctx.record["lattice"])
        ev = StructureEvaluator(training=None, hull=ctx.hull, formula="Si")
        # a deliberately absurd energy: 1 keV below the anchor
        rec = ev.build_records([struct], [ctx.e_ref - 1000.0], ["Si"])[0]
        assert rec.physical is False and rec.stable is False
        assert rec.e_hull < -100.0


class TestAnalysis:
    def test_analyze_cell_schema(self, stub_env):
        ctx, calc = stub_env
        task = {"dataset": "stub", "nnp": "mace", "config": "C1", "sampler": "ald",
                "cell": ctx.cell.id, "seed": 42}
        payload = p2.run_task(task, ctx, calc)
        out = p2.analyze_cell([payload], ctx, ctx.cell)

        assert out["n"] == payload["n_candidates"]
        assert out["n_seeds"] == 1 and out["seed_list"] == [42]
        sun = out["sun"]
        assert set(sun) >= {"n", "n_sun", "sun_rate", "validity_rate",
                            "stable_rate", "unique_rate", "novel_rate",
                            "e_hull_threshold", "e_hull_units",
                            "n_unphysical", "unphysical_rate", "e_hull_floor"}
        assert sun["n_sun"] <= sun["n"]
        assert out["dynamics"]["n_traj"] == payload["n_candidates"]
        assert out["e_hull_valid"]["units"] == "eV/atom"
        # the physical-only E_hull panel is always present next to the raw one
        assert out["e_hull_valid_physical"]["units"] == "eV/atom"
        assert set(out["unphysical"]) == {"n", "rate", "e_hull_min",
                                          "e_hull_median", "e_hull_floor"}
        # the reference pool is non-empty, so the reference-side metrics exist
        assert "match_coverage" in out and "r_angle_kl" in out
        assert set(out["per_seed"]["42"]) >= {"validity", "ood_rate",
                                             "sun_rate_within_seed"}
        assert out["n_validity_definition_mismatch"] == 0

    def test_analysis_pools_seeds_for_uniqueness(self, stub_env):
        ctx, calc = stub_env
        payloads = []
        for seed in (42, 123):
            task = {"dataset": "stub", "nnp": "mace", "config": "C5",
                    "sampler": "ald", "cell": ctx.cell.id, "seed": seed}
            payloads.append(p2.run_task(task, ctx, calc))
        out = p2.analyze_cell(payloads, ctx, ctx.cell)
        assert out["n"] == 2 * p2.N_CAND
        assert out["n_seeds"] == 2
        # uniqueness is measured in the pool, so the pooled count can only be
        # <= the sum of the per-seed counts
        per_seed_sum = sum(v["n_sun_within_seed"] for v in out["per_seed"].values())
        assert out["sun"]["n_sun"] <= per_seed_sum


class TestAnalysisAggregateScope:
    """A task file can outlive its cell's membership in the axis.

    A truncated ``--n-cells`` smoke leaves a full grid of files behind a short
    axis, and an axis edit does the same to a finished run.  Those cells are
    dropped from the per-cell rows, so their trajectories must not re-enter the
    *pooled* rates either -- otherwise `n_cells` and the OOD/E_hull aggregates
    describe different sets and the row is internally inconsistent.
    """

    def test_off_axis_tasks_are_counted_but_not_pooled(
            self, stub_env, tmp_path, monkeypatch):
        ctx, calc = stub_env
        task = {"dataset": "stub", "nnp": "mace", "config": "C1", "sampler": "ald",
                "cell": ctx.cell.id, "seed": 42}
        payload = p2.run_task(task, ctx, calc)
        off_axis = json.loads(json.dumps(payload))
        off_axis["cell"]["id"] = "OffAxis"

        monkeypatch.setattr(p2, "OUT", tmp_path)
        monkeypatch.setattr(p2, "SUMMARY", tmp_path / "summary")
        monkeypatch.setattr(p2, "context", lambda *a, **k: ctx)
        d = tmp_path / "stub" / "C1_ald"
        d.mkdir(parents=True)
        (d / f"{ctx.cell.id}_seed42.json").write_text(json.dumps(payload))
        (d / "OffAxis_seed42.json").write_text(json.dumps(off_axis))

        out = p2.analyze("mace", ["stub"], ["C1"], ["ald"], n_cells=1)
        row = out["rows"][0]
        assert row["n_cells"] == 1
        assert row["n_traj"] == payload["n_candidates"]      # not 2x
        written = json.loads((tmp_path / "summary" / "phase2_stub_mace.json").read_text())
        agg = written["configs"]["C1_ald"]["aggregate"]
        assert agg["n_cells"] == 1
        assert agg["n_tasks"] == 1 and agg["n_tasks_on_disk"] == 2
        assert agg["n_traj"] == payload["n_candidates"]
        assert agg["dynamics"]["n_traj"] == payload["n_candidates"]

    def test_row_is_skipped_when_no_task_is_on_the_axis(
            self, stub_env, tmp_path, monkeypatch):
        ctx, calc = stub_env
        payload = p2.run_task(
            {"dataset": "stub", "nnp": "mace", "config": "C1", "sampler": "ald",
             "cell": ctx.cell.id, "seed": 42}, ctx, calc)
        payload["cell"]["id"] = "OffAxis"
        monkeypatch.setattr(p2, "OUT", tmp_path)
        monkeypatch.setattr(p2, "SUMMARY", tmp_path / "summary")
        monkeypatch.setattr(p2, "context", lambda *a, **k: ctx)
        d = tmp_path / "stub" / "C1_ald"
        d.mkdir(parents=True)
        (d / "OffAxis_seed42.json").write_text(json.dumps(payload))
        # an empty mean would write a NaN row that looks like a real measurement
        assert p2.analyze("mace", ["stub"], ["C1"], ["ald"], n_cells=1)["rows"] == []
