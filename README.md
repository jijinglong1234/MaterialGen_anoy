# MaterialGen

**Zero-training crystal structure generation: a pre-trained neural network potential *is* a score function.**

A pre-trained neural network interatomic potential (NNP) is an energy function $U_{\text{NNP}}(x)$,
so its negative gradient is exactly the score of a Boltzmann distribution:

$$s_{\text{NNP}}(x) = -\beta \nabla_x U_{\text{NNP}}(x), \qquad \beta = 1/(k_B T)$$

Crystal generation therefore becomes sampling from $p_B(x) \propto e^{-\beta U_{\text{NNP}}(x)}$,
with **no generative model trained at all** — the score comes from physics, not from data.

---

## What this code implements

The identity above is exact. Its *usefulness* is not.

Modern NNPs are trained on DFT-relaxed structures and their mild distortions, so their training
data contain no deep-overlap configurations. The score is therefore **chemically complete** —
faithful for bond stretching, angle bending, and electrostatics inside the descriptor cutoff —
but **sterically incomplete** where Pauli repulsion dominates. Using the bare score in a sampler
drives trajectories across that boundary into an out-of-distribution (OOD) regime, where the
extrapolated energy is meaningless and structures collapse through atom overlap.

This codebase implements the resulting method: the **NNP score completeness boundary**
$\partial\mathcal{C}$, and a repair that decomposes "physics" for generative purposes.

- **Long-range chemistry** is system-specific and learned → it stays in the NNP.
- **Short-range sterics** is universal for a given element pair and analytically expressible
  → it is supplied as a ZBL-matched Pauli repulsion term active only below the boundary.

The augmented score is zero-training, exactly conservative, and vanishes at equilibrium.

### The defense stack (L1–L4)

| Layer | Module | Mechanism | Role |
|---|---|---|---|
| **L1** Pauli repulsion | [core/pauli.py](materialgen/core/pauli.py) | Analytic ZBL-matched pair potential active below $0.75\,(R_i^{\text{cov}}+R_j^{\text{cov}})$, plus a bounded-repulsion cap | ~half the OOD reduction on its own, at no energy cost |
| **L2** Density-adaptive noise | [samplers/ald.py](materialgen/samplers/ald.py) | Per-atom noise scaled by $1/(1+\gamma\rho_i)$, $\rho_i$ the local density | Small marginal |
| **L3** Reflection operator | [core/reflection.py](materialgen/core/reflection.py) | Mirrors any pair violating the covalent boundary back across it; also caps the outer cell volume | **Dominant layer** |
| **L4** SafetyMonitor | [core/safety_monitor.py](materialgen/core/safety_monitor.py) | Step rejection on energy/force/overlap/cell thresholds + phase-aware $\sigma$ decay | **Net-harmful — excluded from the paper's arm** |

The paper's protective arm is `l1l2l3` (L1+L2+L3). L4 loses validity, inflates trajectory dwell
time, and worsens $E_\text{hull}$ while buying nothing; the ablation is retained as evidence.

### Samplers

| Sampler | Module | Role |
|---|---|---|
| **ALD** — annealed Langevin dynamics | [samplers/ald.py](materialgen/samplers/ald.py) | Main line; stochastic, carries the full defense stack |
| **PF-ODE** — probability flow ODE | [samplers/pf_ode.py](materialgen/samplers/pf_ode.py) | Main line; deterministic, no step size |
| Underdamped / Cyclical / U-Adaptive | `samplers/*.py` | Appendix material |
| FP-Optimal | `samplers/fp_optimal.py` | Dropped |

### NNP backends

**MACE-MP-0** (`medium`) via `mace-torch`, and **eSEN-30M-MPTrj** via `fairchem-core`. Any ASE
calculator can also be driven directly through `NNPScore` in
[core/score_function.py](materialgen/core/score_function.py). The CHGNet / M3GNet / ensemble
modules are design-stage stubs, not wired into the drivers.

---

## Repository layout

```
materialgen/            # the package
├── core/               #   CrystalStructure, ScoreFunction, Scheduler, Sampler
│                       #   + the protection layers: pauli.py (L1), reflection.py (L3),
│                       #     safety_monitor.py (L4)
├── nnp/                #   eSEN wrapper (live); mace/chgnet/m3gnet are design-stage
├── samplers/           #   ald.py, pf_ode.py (L2 lives inside ald.py)
├── eval/               #   validity, E_hull / phase-diagram stability, diversity, relaxation
├── analysis/           #   score-field comparison, curl violation, lambda-mixing, figures
├── data/               #   mp_20 / perov_5 / carbon_24 loaders, registry, hull lookup
├── baselines/          #   MD annealing, basin hopping, DiffCSP / MatterGen / CDVAE interfaces
├── experiments/        #   config, runner, logging
└── utils/              #   CIF/POSCAR I/O, periodic boundary helpers, constants

scripts/                # drivers and analysis entry points (below)
tests/                  # pytest suite: samplers, score, protection layers, metrics, drivers
configs/                # per-sampler YAML configs (see the note in "Running")
```

### Scripts

| Script | Purpose |
|---|---|
| [minimal_demo.py](scripts/minimal_demo.py) | Self-contained introduction to the core idea — Lennard-Jones argon, no NNP and no dataset required |
| [preprocess_datasets.py](scripts/preprocess_datasets.py) | Parse the benchmark CIFs and write `data/processed/` |
| [build_hulls.py](scripts/build_hulls.py) | Build the per-system convex-hull entries in `data/hulls/` |
| [run_phase1.py](scripts/run_phase1.py) | Noise scan and layer ablation (σ sweep × protection arm × sampler) |
| [run_phase2.py](scripts/run_phase2.py) | The frozen benchmark grid across MP-20 / Perov-5 / Carbon-24 |
| [fit_pauli_params.py](scripts/fit_pauli_params.py) | Regenerate the per-element-pair `(A, B)` wall parameters from the ZBL function |
| `analyze_*.py`, `verify_*.py` | Metric aggregation, gate checks, and ablation diagnostics |

---

## Installation

No package build is required. Run everything from the repository root: the scripts and the
test suite put the repo on `sys.path` themselves.

```bash
pip install numpy scipy ase pyyaml tqdm                      # core
pip install pymatgen spglib scikit-learn pandas matplotlib   # eval / analysis
pip install mace-torch                                       # MACE backend
pip install fairchem-core                                    # eSEN backend
pip install pytest                                           # tests
```

Model weights are not bundled. MACE-MP-0 is fetched automatically by `mace-torch`. The eSEN
loader expects a local checkpoint at `checkpoints/OMAT24/esen_30m_mptrj.pt` (or an explicit
`checkpoint_path=`).

## Running

```bash
# 1. Core idea, no NNP and no dataset needed
python scripts/minimal_demo.py

# 2. Unit tests
pytest

# 3. Prepare the benchmarks (needs the CDVAE dataset CSVs; creates data/)
python scripts/preprocess_datasets.py
python scripts/build_hulls.py

# 4. Sampling drivers
python scripts/run_phase2.py --list                                # print the task grid and exit
python scripts/run_phase2.py --pilot --nnp mace                    # a few tasks end-to-end
python scripts/run_phase2.py --nnp mace --shard 0/10               # sharded workers
python scripts/run_phase1.py --subexp 1 --nnp mace --shard 0/10
```

Both drivers shard by `--shard k/N`, skip tasks whose output already exists, and write one JSON
payload per task. **They are configured in code, not by YAML** — Phase-2 configs are the `CONFIGS`
table in [run_phase2.py](scripts/run_phase2.py). The `configs/*.yaml` files belong to the older
generic [run_experiment.py](scripts/run_experiment.py) path and are not read by the phase drivers.

### The benchmark grid

Datasets `mp_20`, `perov_5`, `carbon_24` (CDVAE splits) × configs `C1, C3, C5, C8, C9, C11`
× 5 seeds × 20 cells × 10 candidates. ALD runs all six configs; PF-ODE runs `C5, C8, C11` only.
`C1`/`C5` are the bare-score references at high and low noise, `C8`–`C10` the protected arms, and
`C11` isolates the Pauli term (L1) alone.

### Data

`data/` is generated by the two preparation scripts above; it is not part of this repository.
`data/processed/` holds primitive-cell-standardized, supercell-expanded structures per split, and
`data/hulls/` the per-chemical-system formation-energy entries used to compute $E_\text{above hull}$.

---

## Notes on the code

- **`l1l4` is a legacy token meaning the full four-layer stack** (L1+L2+L3+L4), not "L1+L4".
  See `LEVEL_SUBSETS` in [run_phase1.py](scripts/run_phase1.py) — that dict is the single place
  that maps an arm name to active layers. On the PF-ODE path it happens to reduce to L1+L3, since
  L2 (noise modulation) and L4 (step rejection) are undefined for a deterministic ODE.
- **Fixed cell is the Phase-1 practice** (`update_lattice=False`). Stress-driven lattice updates
  are an ablation (`C5`/`C10` vs `C1`/`C8`).
- **Per-task payloads carry an `init_cell_gen` stamp** recording the initial-cell convention that
  produced them; payloads with different stamps must not be pooled into one statistic.
- All samplers operate on Cartesian coordinates with wrapped fractional positions, and derive the
  per-candidate RNG from `(seed, candidate_index)` so that every shard reproduces the same
  trajectories.
