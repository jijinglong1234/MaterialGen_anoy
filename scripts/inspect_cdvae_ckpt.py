"""Locate the stale absolute path baked into the CDVAE checkpoint."""
import os
import re

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_ROOT, "third_party", "checkpoints", "cdvae_mp20",
                    "epoch=745-step=79076.ckpt")
NEW_PREFIX = os.path.join(_ROOT, "third_party", "cdvae").replace(os.sep, "/")

# Same structural match as patch_cdvae_ckpt.py: the original trainer's checkout
# path is baked in, and it is not ours to hardcode.
STALE_RE = re.compile(r"/home/[^/]+/(?:.*/)?cdvae(?=/|$)")

ckpt = torch.load(CKPT, map_location="cpu")
hp = ckpt.get("hyper_parameters", {})
print("hyper_parameters type:", type(hp), "| keys:", list(hp.keys())[:25])

found = []


def walk(obj, path=""):
    if isinstance(obj, str) and STALE_RE.search(obj):
        found.append((path, obj))
        return 1
    if hasattr(obj, "items"):
        return sum(walk(v, f"{path}.{k}") for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sum(walk(v, f"{path}[{i}]") for i, v in enumerate(obj))
    return 0


print("occurrences:", walk(hp))
for p, v in found:
    print("  ", p, "=", v[:110])
