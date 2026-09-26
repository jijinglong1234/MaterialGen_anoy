"""Patch the mirrored CDVAE checkpoint: rewrite baked-in absolute paths
(from the original trainer's machine) to our local repo."""

import os
import re

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_ROOT, "third_party", "checkpoints", "cdvae_mp20",
                    "epoch=745-step=79076.ckpt")
NEW_ROOT = os.path.join(_ROOT, "third_party", "cdvae").replace(os.sep, "/")

# The upstream checkpoint was produced on a machine whose checkout lived under
# a home directory that does not exist here, and that prefix is baked into
# hyper_parameters.  Match it structurally -- an absolute home path ending in
# the package directory -- rather than hardcoding a path that is not ours.
OLD_ROOT_RE = re.compile(r"/home/[^/]+/(?:.*/)?cdvae(?=/|$)")

ckpt = torch.load(CKPT, map_location="cpu")
changed = []


def fix(obj, path=""):
    if isinstance(obj, str):
        if OLD_ROOT_RE.search(obj):
            changed.append(path)
            return OLD_ROOT_RE.sub(NEW_ROOT, obj)
        return obj
    if hasattr(obj, "items"):  # dict, DictConfig, AttributeDict, ...
        new = {k: fix(v, f"{path}.{k}") for k, v in obj.items()}
        try:
            obj.update(new)
            return obj
        except Exception:
            return new
    if isinstance(obj, (list, tuple)):
        return type(obj)(fix(v, f"{path}[{i}]") for i, v in enumerate(obj))
    return obj


ckpt["hyper_parameters"] = fix(ckpt.get("hyper_parameters", {}))
torch.save(ckpt, CKPT)
print("patched:", *changed, sep="\n  ")
