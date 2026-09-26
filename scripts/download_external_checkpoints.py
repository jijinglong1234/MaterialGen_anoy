"""Download DiffCSP/CDVAE MP-20 checkpoints from public HF mirrors."""
import os

# If huggingface.co is slow or unreachable from your network, point HF_ENDPOINT
# at a mirror of your choice before running this script.

from huggingface_hub import hf_hub_download

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(_ROOT, "third_party", "checkpoints")

ITEMS = [
    ("anitay/cdvae-mp20-cowboys", "generator/epoch=745-step=79076.ckpt",
     f"{DEST}/cdvae_mp20_generator.ckpt"),
    ("anitay/cdvae-mp20-cowboys", "generator/lattice_scaler.pt",
     f"{DEST}/cdvae_mp20_lattice_scaler.pt"),
    ("anitay/cdvae-mp20-cowboys", "generator/hparams.yaml",
     f"{DEST}/cdvae_mp20_hparams.yaml"),
]

if __name__ == "__main__":
    import shutil

    os.makedirs(DEST, exist_ok=True)
    for repo, fname, dest in ITEMS:
        p = hf_hub_download(repo, fname)
        shutil.copy(p, dest)
        print(f"{repo}/{fname} -> {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
