"""EqR Sudoku weights: scan the pickle before loading, merge EMA over raw, export a pickle-free npz.

The upstream checkpoint is a torch pickle (it embeds an OmegaConf config).  ``scan_checkpoint`` walks the
pickle with a restricted unpickler that only knows a short allow-list of globals, so nothing can execute;
``export_npz`` then writes just the arrays, which is what the cluster loads (numpy only, no pickle).
"""
from __future__ import annotations

import os
import pickle
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "data/raw/loopscape_cache/eqr_sudoku.pth"
DEFAULT_NPZ = ROOT / "data/raw/eqr_sudoku_weights.npz"
_PREFIX = "_orig_mod.model."


class UnsafeCheckpoint(Exception):
    """The checkpoint pickle references a global outside the allow-list."""


_ALLOWED_BUILTINS = {"dict", "list", "long", "int", "float", "set", "tuple", "bool", "str", "bytes", "object"}
_ALLOWED = {
    "collections": {"OrderedDict", "defaultdict"},
    "typing": {"Any"},
    "omegaconf.base": {"ContainerMetadata", "Metadata"},
    "omegaconf.dictconfig": {"DictConfig"},
    "omegaconf.listconfig": {"ListConfig"},
}


def _allowed(module: str, name: str) -> bool:
    if module in ("builtins", "__builtin__"):
        return name in _ALLOWED_BUILTINS
    if module == "torch":
        return name.endswith("Storage") or name in {"Size"}
    if module == "torch._utils":
        return name.startswith("_rebuild_")
    if module == "omegaconf.nodes":
        return name.endswith("Node")
    return name in _ALLOWED.get(module, ())


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()

    def __setstate__(self, state):
        pass

    # the unpickler fills dict-/list-/set-like objects through these; the scan only needs them not to fail
    def __setitem__(self, key, value):
        pass

    def append(self, item):
        pass

    def extend(self, items):
        pass

    def update(self, *a, **k):
        pass

    def add(self, item):
        pass


def scan_checkpoint(path: str | Path) -> list[str]:
    """Return the sorted globals a torch checkpoint's pickle uses; raise if any is not allow-listed."""
    seen: set[str] = set()

    class _Scan(pickle.Unpickler):
        def find_class(self, module, name):  # noqa: D401
            if not _allowed(module, name):
                raise UnsafeCheckpoint(f"checkpoint references disallowed global {module}.{name}")
            seen.add(f"{module}.{name}")
            return _Stub

        def persistent_load(self, pid):
            return _Stub()

    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise UnsafeCheckpoint("not a torch zip checkpoint (no data.pkl)")
        with z.open(pkl[0]) as f:
            _Scan(f).load()
    return sorted(seen)


def merge_state_dict(ck: dict) -> dict:
    """EMA weights override the raw model's; prefix ``_orig_mod.model.`` stripped (as loopscape does)."""
    model_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    merged = dict(model_sd)
    ema = ck.get("ema") if isinstance(ck, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        merged.update(ema["shadow"])
    return {(k[len(_PREFIX):] if k.startswith(_PREFIX) else k): v for k, v in merged.items()}


def load_checkpoint(path: str | Path = DEFAULT_CKPT) -> dict[str, np.ndarray]:
    import torch

    scan_checkpoint(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().float().numpy() for k, v in merge_state_dict(ck).items()}


def export_npz(dest: str | Path = DEFAULT_NPZ, ckpt: str | Path = DEFAULT_CKPT) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **load_checkpoint(ckpt))
    return dest


def load(path: str | Path | None = None, ckpt: str | Path = DEFAULT_CKPT) -> dict[str, np.ndarray]:
    """Weights as float32 numpy: explicit npz, else $RF_WEIGHTS_NPZ, else the default npz, else the checkpoint."""
    candidates = [Path(path)] if path else []
    if os.environ.get("RF_WEIGHTS_NPZ"):
        candidates.append(Path(os.environ["RF_WEIGHTS_NPZ"]))
    candidates.append(DEFAULT_NPZ)
    for c in candidates:
        if c.exists():
            with np.load(c) as z:
                return {k: z[k] for k in z.files}
    return load_checkpoint(ckpt)


def puzzle_tokens(puzzle: str) -> np.ndarray:
    """EqR Sudoku input tokens: cell value + 1 (blank '.' -> 1, digit d -> d + 1), int64 (81,)."""
    s = "".join(puzzle.split())
    if len(s) != 81:
        raise ValueError(f"puzzle must have 81 cells, got {len(s)}")
    return np.array([0 if ch in "._0" else int(ch) for ch in s], dtype=np.int64) + 1
