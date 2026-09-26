"""
ESENNNP — wrapper for the eSEN (equivariant Scalar-Energy Network) potential.

Functionality:
    Loads a local eSEN checkpoint (e.g. OMAT24/esen_30m_mptrj.pt, trained on
    OMat24) through fairchem-core and exposes it as an ASE Calculator for
    use with NNPScore (energy, forces, stress).

    Two fairchem-core APIs are supported:
    - fairchem-core >= 2.0: FAIRChemCalculator + load_predict_unit
    - fairchem-core 1.x:    OCPCalculator(checkpoint_path=...)

    The package is an optional dependency: a clear ImportError is raised
    with install instructions if fairchem is not available.

Usage:
    from materialgen.nnp.esen import load_esen_calculator
    calc = load_esen_calculator(device="cuda")
    score = NNPScore(calc, beta=BETA_300K, label="eSEN-30M-MPTrj")

Dependencies:
    fairchem-core (pip install fairchem-core), torch, ase
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.crystal import CrystalStructure

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "OMAT24" / "esen_30m_mptrj.pt"
)


def load_esen_calculator(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
):
    """
    Build an ASE Calculator from a local eSEN checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint. Defaults to the repo's
            OMAT24/esen_30m_mptrj.pt.
        device: "cuda" or "cpu", or "cuda:<i>" on fairchem-core >= 2.0, which
            forwards the string to load_predict_unit.  The 1.x OCPCalculator
            path cannot select a GPU index (see the note in its branch) and
            accepts "cuda:0"/"cuda" only -- use CUDA_VISIBLE_DEVICES for any
            other device.

    Returns:
        An ASE Calculator providing energy/forces/stress.
    """
    path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(
            f"eSEN checkpoint not found: {path}. "
            "Download esen_30m_mptrj.pt into checkpoints/OMAT24/ or pass checkpoint_path."
        )

    # --- fairchem-core >= 2.0 API ---
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        predict_unit = load_predict_unit(str(path), device=device, **kwargs)
        return FAIRChemCalculator(predict_unit, task_name="omat")
    except ImportError:
        pass

    # --- fairchem-core 1.x API ---
    try:
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        # OCPCalculator cannot select a GPU, so refuse to pretend it did.  Its
        # ``cpu`` argument is a bare bool and the trainer it builds resolves its
        # device as torch.device(f"cuda:{local_rank}") with local_rank taken
        # from the checkpoint config -- always 0 (base_trainer.py:108).  Any
        # index in ``device`` is therefore discarded: device="cuda:3" runs on
        # the visible set's cuda:0, silently.  torch.cuda.set_device() does not
        # help either, because the index is written into the device object
        # before set_device is consulted.  The visible device set is the only
        # lever, so a nonzero index is an error naming the workaround rather
        # than a run on the wrong GPU (measured: an eSEN preflight
        # launched with --device cuda:3 executed on physical GPU 0).
        idx = None
        if device.startswith("cuda:"):
            idx = int(device.split(":", 1)[1])
        if idx not in (None, 0):
            raise ValueError(
                f"fairchem-core 1.x cannot place eSEN on {device!r}: OCPCalculator "
                "always uses the visible set's cuda:0.  Launch under "
                f"CUDA_VISIBLE_DEVICES={idx} and pass device='cuda' instead."
            )
        return OCPCalculator(checkpoint_path=str(path), cpu=(device == "cpu"), **kwargs)
    except ImportError as exc:
        raise ImportError(
            "fairchem-core is required for eSEN. Install with:\n"
            "    pip install fairchem-core\n"
            "(see https://github.com/facebookresearch/fairchem for CUDA wheels)"
        ) from exc


class ESENNNP:
    """
    Thin object wrapper around the eSEN ASE calculator.

    Provides predict() -> dict for direct use, and .calculator for plugging
    into NNPScore / baselines.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        label: str = "eSEN-30M-MPTrj",
        **kwargs,
    ):
        self.label = label
        self._calc = load_esen_calculator(checkpoint_path, device=device, **kwargs)

    @property
    def calculator(self):
        return self._calc

    def predict(self, crystal: CrystalStructure) -> dict:
        """Compute energy/forces/stress for a CrystalStructure."""
        atoms = crystal.ase_atoms
        atoms.calc = self._calc
        try:
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            try:
                stress = atoms.get_stress()
            except Exception:
                stress = None
        finally:
            atoms.calc = None
        return {
            "energy": energy,
            "forces": np.asarray(forces),
            "stress": None if stress is None else np.asarray(stress),
        }

    def __repr__(self) -> str:
        return f"ESENNNP(label={self.label!r})"
