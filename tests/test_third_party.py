"""
Third-party baseline adapters (materialgen/eval/third_party.py,
scripts/score_third_party.py).

The adapters exist because the learned baselines are trained and sampled in
their own repositories: what we get is an official artifact, and every metric
we report for them depends on us reading that artifact's layout correctly.
The layouts are therefore pinned by tests with synthetic artifacts written in
the exact shape the vendored official code produces (DiffCSP
``generation.py`` / ``evaluate.py``, CDVAE ``eval_gen.pt``) rather than by
the first real file, which would only be available after a long generation
run.

Cases:
  P1  DiffCSP generation layout: logits -> Z = argmax+1, per-sample slicing
  P2  CDVAE layout: padded per batch, integer Z, left-aligned slicing
  P3  DiffCSP CSP layout: (eval, offset) slicing, conditioned formulas
  P4  dispatch: rank-3 layouts are told apart by `input_data_batch`
  P5  malformed artifacts fail loudly (vocabulary, padding Z, atom-count sum)
  P6  --limit truncates the generation arm only
  P7  grouped_match_coverage == compute_match_and_coverage (prefilter exact)
  P8  grouped_amsd: per-composition mean, singleton compositions excluded
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import score_third_party as s3  # noqa: E402

from materialgen.core.crystal import CrystalStructure  # noqa: E402
from materialgen.eval.metrics import compute_match_and_coverage  # noqa: E402
from materialgen.eval.third_party import (DIFFCSP_VOCAB, detect_format,  # noqa: E402
                                         load_generated)
from materialgen.utils.crystal_io import to_pymatgen  # noqa: E402

torch = pytest.importorskip("torch")

#: (a, b, c, alpha, beta, gamma), Z list, frac list -- two compositions
CELLS = [
    ((4.0, 4.0, 4.0, 90.0, 90.0, 90.0), [11, 17], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ((5.0, 5.0, 5.0, 90.0, 90.0, 90.0), [14, 8, 8], [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]]),
]


def _lattice(spec):
    from materialgen.eval.third_party import _lattice_matrix

    return _lattice_matrix(spec[:3], spec[3:])


def _stub(spec, zs, fracs) -> CrystalStructure:
    return CrystalStructure.from_frac_coords(np.array(zs), np.array(fracs), _lattice(spec))


def _diffcsp_blob():
    """DiffCSP generation.py: concatenated rows, logits, no batch axis."""
    frac, z, lengths, angles = [], [], [], []
    for spec, zs, fracs in CELLS:
        frac.append(np.array(fracs, dtype=np.float32))
        z.append(np.array(zs))
        lengths.append(spec[:3])
        angles.append(spec[3:])
    frac = torch.tensor(np.concatenate(frac))
    z = np.concatenate(z)
    logits = np.full((len(z), DIFFCSP_VOCAB), -5.0, dtype=np.float32)
    logits[np.arange(len(z)), z - 1] = 5.0
    return {"frac_coords": frac, "num_atoms": torch.tensor([2, 3]),
            "atom_types": torch.tensor(logits), "lengths": torch.tensor(lengths),
            "angles": torch.tensor(angles)}


def _cdvae_blob():
    """CDVAE eval_gen.pt: dummy batch axis, flat atom axis, Z as ints."""
    frac = np.zeros((1, 5, 3), dtype=np.float32)
    z = np.zeros((1, 5), dtype=np.int64)
    lengths = np.zeros((1, 2, 3), dtype=np.float32)
    angles = np.zeros((1, 2, 3), dtype=np.float32)
    offset = 0
    for j, (spec, zs, fracs) in enumerate(CELLS):
        n = len(zs)
        frac[0, offset:offset + n] = fracs
        z[0, offset:offset + n] = zs
        lengths[0, j] = spec[:3]
        angles[0, j] = spec[3:]
        offset += n
    return {"frac_coords": torch.tensor(frac), "num_atoms": torch.tensor([[2, 3]]),
            "atom_types": torch.tensor(z), "lengths": torch.tensor(lengths),
            "angles": torch.tensor(angles)}


def _csp_blob(n_evals=2):
    """evaluate.py: dim 0 = eval, dim 1 = concatenated structures."""
    frac = np.zeros((n_evals, 5, 3), dtype=np.float32)
    z = np.zeros((n_evals, 5), dtype=np.int64)
    lengths = np.zeros((n_evals, 2, 3), dtype=np.float32)
    angles = np.zeros((n_evals, 2, 3), dtype=np.float32)
    offset = 0
    for j, (spec, zs, fracs) in enumerate(CELLS):
        n = len(zs)
        frac[:, offset:offset + n] = np.array(fracs, dtype=np.float32)
        z[:, offset:offset + n] = zs
        lengths[:, j] = spec[:3]
        angles[:, j] = spec[3:]
        offset += n
    return {"frac_coords": torch.tensor(frac), "num_atoms": torch.tensor([[2, 3]] * n_evals),
            "atom_types": torch.tensor(z), "lengths": torch.tensor(lengths),
            "angles": torch.tensor(angles), "input_data_batch": object()}


def _save(tmp_path, blob, name):
    path = tmp_path / name
    torch.save(blob, path)
    return path


class TestDiffCSPGenerationLayout:
    def test_logits_map_to_z_by_argmax_plus_one(self, tmp_path):
        structures, prov = load_generated(_save(tmp_path, _diffcsp_blob(), "g.pt"))
        assert prov["format"] == "diffcsp"
        assert [list(s.atomic_numbers) for s in structures] == [[11, 17], [14, 8, 8]]
        assert (prov["z_min"], prov["z_max"]) == (8, 17)
        assert prov["n_structures"] == 2 and prov["n_atoms_total"] == 5

    def test_lattice_and_coordinates_survive(self, tmp_path):
        structures, _ = load_generated(_save(tmp_path, _diffcsp_blob(), "g.pt"))
        for s, (spec, zs, fracs) in zip(structures, CELLS):
            assert np.allclose(s.lattice, _lattice(spec), atol=1e-5)
            assert np.allclose(s.frac_coords, fracs, atol=1e-5)

    def test_limit_truncates_by_sample(self, tmp_path):
        structures, prov = load_generated(_save(tmp_path, _diffcsp_blob(), "g.pt"), limit=1)
        assert len(structures) == 1
        assert list(structures[0].atomic_numbers) == [11, 17]
        assert prov["truncated_to"] == 1

    def test_limit_is_refused_where_not_implemented(self, tmp_path):
        path = _save(tmp_path, _cdvae_blob(), "c.pt")
        with pytest.raises(ValueError, match="generation arm only"):
            load_generated(path, limit=1)

    def test_vocabulary_mismatch_raises(self, tmp_path):
        blob = _diffcsp_blob()
        blob["atom_types"] = blob["atom_types"][:, :50]
        with pytest.raises(ValueError, match="vocabulary"):
            load_generated(_save(tmp_path, blob, "g.pt"), fmt="diffcsp")


class TestCDVAELayout:
    def test_padded_batches_slice_by_count(self, tmp_path):
        structures, prov = load_generated(_save(tmp_path, _cdvae_blob(), "c.pt"))
        assert prov["format"] == "cdvae"
        assert [list(s.atomic_numbers) for s in structures] == [[11, 17], [14, 8, 8]]

    def test_zero_z_is_refused(self, tmp_path):
        blob = _cdvae_blob()
        z = blob["atom_types"].clone()
        z[0, 0] = 0                      # no element 0: unmapped vocabulary
        blob["atom_types"] = z
        with pytest.raises(ValueError, match="Z=0"):
            load_generated(_save(tmp_path, blob, "c.pt"), fmt="cdvae")

    def test_atom_count_that_does_not_fill_the_axis_is_refused(self, tmp_path):
        blob = _cdvae_blob()
        blob["num_atoms"] = torch.tensor([[2, 2]])
        with pytest.raises(ValueError, match="sums to 4"):
            load_generated(_save(tmp_path, blob, "c.pt"), fmt="cdvae")


class TestCSPLayout:
    def test_eval_major_slicing_and_conditioned_formulas(self, tmp_path):
        structures, prov = load_generated(_save(tmp_path, _csp_blob(), "d.pt"))
        assert prov["format"] == "diffcsp_csp"
        assert len(structures) == 4          # 2 evals x 2 structures
        assert prov["conditioned_formulas"] == ["NaCl", "SiO2"] * 2
        assert [list(s.atomic_numbers) for s in structures] == [
            [11, 17], [14, 8, 8], [11, 17], [14, 8, 8]]

    def test_atom_count_that_does_not_fill_the_axis_is_refused(self, tmp_path):
        blob = _csp_blob()
        blob["frac_coords"] = blob["frac_coords"][:, :4]
        with pytest.raises(ValueError, match="frac_coords holds 4"):
            load_generated(_save(tmp_path, blob, "d.pt"), fmt="diffcsp_csp")


class TestDispatch:
    def test_rank3_layouts_are_told_apart_by_input_batch(self):
        assert detect_format(_cdvae_blob()) == "cdvae"
        assert detect_format(_csp_blob()) == "diffcsp_csp"
        assert detect_format(_diffcsp_blob()) == "diffcsp"

    def test_missing_keys_raise(self):
        with pytest.raises(KeyError):
            detect_format({"frac_coords": torch.zeros(1, 3)})

    def test_unrecognised_ranks_raise(self):
        blob = _diffcsp_blob()
        blob["num_atoms"] = torch.zeros(1, 2, dtype=torch.int64)
        with pytest.raises(ValueError, match="unrecognised eval layout"):
            detect_format(blob)


class TestGroupedMatching:
    """The composition prefilter must be exact, not merely fast."""

    def _ensemble(self):
        gen = [_stub(*c) for c in CELLS]
        # a second sample of the first composition, slightly displaced
        spec, zs, fracs = CELLS[0]
        moved = [[0.02, 0.0, 0.0], [0.52, 0.5, 0.5]]
        ref = [_stub(spec, zs, fracs), _stub(*CELLS[1]), _stub(spec, zs, moved)]
        return gen, ref

    def test_grouped_equals_ungrouped(self):
        gen, ref = self._ensemble()
        grouped = s3.grouped_match_coverage(gen, ref)
        plain = compute_match_and_coverage(gen, ref)
        assert grouped["matched"] == plain["matched"]
        assert grouped["covered"] == plain["covered"]
        assert grouped["n_generated"] == plain["n_generated"]
        assert grouped["n_reference"] == plain["n_reference"]

    def test_identical_sets_are_fully_matched(self):
        gen, _ = self._ensemble()
        out = s3.grouped_match_coverage(gen, gen)
        assert out["match_rate"] == 1.0 and out["coverage"] == 1.0

    def test_coverage_is_capped_by_the_compositions_generated(self):
        """A reference in a composition no sample reproduced cannot be covered."""
        gen, ref = self._ensemble()
        only_si = [s for s in gen if to_pymatgen(s).composition.reduced_formula == "SiO2"]
        out = s3.grouped_match_coverage(only_si, ref)
        assert out["match_rate"] == 1.0            # identical to its reference
        assert out["coverage"] == 1 / 3            # the two NaCl refs are unreachable
        assert out["generated_in_reference_composition"] == 1
        assert out["n_formulas_shared"] == 1


class TestGroupedAMSD:
    def test_mean_over_compositions_not_over_structures(self):
        spec, zs, fracs = CELLS[0]
        moved = [[0.2, 0.0, 0.0], [0.7, 0.5, 0.5]]
        ensemble = [_stub(spec, zs, fracs), _stub(spec, zs, moved),
                    _stub(*CELLS[1]), _stub(*CELLS[1])]
        out = s3.grouped_amsd(ensemble)
        assert out["n_groups"] == 2 and out["n_groups_used"] == 2
        assert out["mean"] is not None and np.isfinite(out["mean"])

    def test_singletons_are_excluded(self):
        ensemble = [_stub(*c) for c in CELLS]
        out = s3.grouped_amsd(ensemble)
        assert out["n_groups"] == 2 and out["n_groups_used"] == 0
        assert out["mean"] is None
