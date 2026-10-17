"""Weights handling: safe checkpoint scan, clean npz export, loopscape-equivalent merge."""
import io
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from render import weights as W

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/raw/loopscape_cache/eqr_sudoku.pth"
PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="EqR checkpoint not downloaded")


def _fake_pth(tmp_path, payload) -> Path:
    """A torch-style zip whose data.pkl is an arbitrary pickle."""
    p = tmp_path / "evil.pth"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("m/data.pkl", pickle.dumps(payload))
    return p


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_scan_rejects_pickles_that_import_dangerous_callables(tmp_path):
    with pytest.raises(W.UnsafeCheckpoint, match="os|posix|system"):
        W.scan_checkpoint(_fake_pth(tmp_path, _Evil()))


@needs_ckpt
def test_scan_accepts_the_real_checkpoint_and_reports_its_globals():
    globs = W.scan_checkpoint(CKPT)
    mods = {g.split(".")[0] for g in globs}
    assert mods <= {"torch", "collections", "omegaconf", "builtins", "__builtin__", "typing"}
    assert any("_rebuild_tensor_v2" in g for g in globs)


@needs_ckpt
def test_export_npz_round_trips_and_equals_loopscape_merge(tmp_path):
    out = W.export_npz(tmp_path / "w.npz", ckpt=CKPT)
    w = W.load(out)
    import torch
    from loopscape.eqr.solver import _extract_state_dict

    ref = {k: v.float().numpy() for k, v in _extract_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)).items()}
    assert set(w) == set(ref)
    for k in ref:
        np.testing.assert_array_equal(w[k], ref[k], err_msg=k)
    assert w["inner.L_level.layers.0.mlp.gate_up_proj.weight"].shape == (3072, 512)
    assert "inner.puzzle_emb.weights" in w  # comes from the raw model dict, not the EMA shadow


@needs_ckpt
def test_load_falls_back_to_checkpoint_when_no_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_WEIGHTS_NPZ", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(W, "DEFAULT_NPZ", tmp_path / "also_missing.npz")
    w = W.load(ckpt=CKPT)
    assert w["inner.lm_head.weight"].shape == (11, 512)


def test_puzzle_tokens_are_cell_plus_one():
    t = W.puzzle_tokens(PUZZLE)
    assert t.shape == (81,) and t.dtype == np.int64
    assert t[0] == 1 and t[5] == 7 and t.max() == 10  # blank -> 1, digit 6 -> 7


@needs_ckpt
def test_puzzle_tokens_match_loopscape_codec():
    from loopscape.tasks import get_task

    np.testing.assert_array_equal(W.puzzle_tokens(PUZZLE), get_task("sudoku").encode(PUZZLE).numpy()[0])
