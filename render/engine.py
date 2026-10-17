"""Deterministic EqR rollouts on one Sudoku, from arbitrary initial latents.

This is the map loopscape calls with ``noise_scale=0``: ``steps`` loops of EqR's ``deep_recursion``
(H_cycles=3 x [L_cycles=6 L-updates of z_L, then one z_H update]), each loop decoded with ``lm_head``.
Two interchangeable backends compute exactly this map:

* ``torch`` -- the upstream modules from the pinned locuslab/EqR source (reference, CPU or MPS).
* ``mlx``   -- the same arithmetic as a compiled MLX graph, laid out token-major so every matmul is
               one large 2-D GEMM.  About 1.5-2x faster than torch-MPS on an M4.

Decoded cells follow loopscape: ``argmax(logits) - 1`` (so -1 = pad token, 0 = blank, 1..9 digits).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOOPSCAPE_CACHE_DIR", str(ROOT / "data/raw/loopscape_cache"))
os.environ.setdefault("LOOPSCAPE_EQR_SRC", str(ROOT / "vendor/EqR"))

H_CYCLES, L_CYCLES, LAMBDA, EPS = 3, 6, 0.95, 1e-5
N_TOK, HIDDEN, REG = 97, 512, 16


def _load_weights() -> dict[str, np.ndarray]:
    """Checkpoint -> numpy weights, merging EMA over raw exactly as loopscape does."""
    import torch
    from loopscape.eqr.download import ensure_eqr_checkpoint
    from loopscape.eqr.solver import _extract_state_dict

    ck = torch.load(ensure_eqr_checkpoint(task="sudoku"), map_location="cpu", weights_only=False)
    return {k: v.detach().float().numpy() for k, v in _extract_state_dict(ck).items()}


def _tokens(puzzle: str) -> np.ndarray:
    from loopscape.tasks import get_task

    return get_task("sudoku").encode(puzzle).numpy()[0].astype(np.int64)  # (81,) cell + 1


class _MlxBackend:
    name = "mlx"

    def __init__(self, puzzle: str, w: dict[str, np.ndarray]):
        import mlx.core as mx
        import mlx.nn as nn

        self.mx = mx
        a = lambda x: mx.array(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
        p = "inner.L_level.layers."
        self.layers = []
        for i in range(2):
            self.layers.append(
                dict(
                    tok_gu=a(w[f"{p}{i}.mlp_t.gate_up_proj.weight"]),          # (1024, 97)
                    tok_d=a(w[f"{p}{i}.mlp_t.down_proj.weight"]),              # (97, 512)
                    ch_gu=a(w[f"{p}{i}.mlp.gate_up_proj.weight"].T),           # (512, 3072)
                    ch_d=a(w[f"{p}{i}.mlp.down_proj.weight"].T),               # (1536, 512)
                )
            )
        self.lm_head = a(w["inner.lm_head.weight"].T)                          # (512, 11)
        emb = w["inner.embed_tokens.embedding_weight"][_tokens(puzzle)]        # (81, 512)
        puz = np.zeros((REG, HIDDEN), dtype=np.float32)
        puz.reshape(-1)[:HIDDEN] = w["inner.puzzle_emb.weights"][0]
        self.x = a(np.sqrt(HIDDEN) * np.concatenate([puz, emb], axis=0))[:, None, :]  # (97, 1, 512)
        self.ones = mx.ones((HIDDEN,), dtype=mx.float32)
        silu = nn.silu
        layers, ones = self.layers, self.ones

        def l_level(h, inj):  # h, inj: (97, B, 512)
            t_, b_, c_ = h.shape
            u = h + inj
            for p_ in layers:
                t = u.reshape(t_, b_ * c_)
                gu = p_["tok_gu"] @ t
                t = t + p_["tok_d"] @ (silu(gu[:512]) * gu[512:])
                t = t * mx.rsqrt(mx.mean(t * t, axis=0, keepdims=True) + EPS)
                h2 = t.reshape(t_ * b_, c_)
                gu = h2 @ p_["ch_gu"]
                h2 = h2 + (silu(gu[:, :1536]) * gu[:, 1536:]) @ p_["ch_d"]
                h2 = mx.fast.rms_norm(h2, ones, EPS)
                u = h2.reshape(t_, b_, c_)
            return (1.0 - LAMBDA) * h + LAMBDA * u

        self._l_level = mx.compile(l_level)

    def _step(self, zh, zl):
        x = self.x
        for _ in range(H_CYCLES):
            for _ in range(L_CYCLES):
                zl = self._l_level(zl, zh + x)
            zh = self._l_level(zh, zl)
        return zh, zl

    def _decode(self, zh):
        mx = self.mx
        logits = zh[REG:] @ self.lm_head  # (81, B, 11)
        return (mx.argmax(logits, axis=-1) - 1).astype(mx.int8)  # (81, B)

    def step(self, z_h: np.ndarray, z_l: np.ndarray):
        mx = self.mx
        zh = mx.array(np.ascontiguousarray(z_h.transpose(1, 0, 2)))
        zl = mx.array(np.ascontiguousarray(z_l.transpose(1, 0, 2)))
        zh, zl = self._step(zh, zl)
        mx.eval(zh, zl)
        return np.asarray(zh).transpose(1, 0, 2).copy(), np.asarray(zl).transpose(1, 0, 2).copy()

    def rollout(self, z_h: np.ndarray, z_l: np.ndarray, steps: int) -> np.ndarray:
        mx = self.mx
        zh = mx.array(np.ascontiguousarray(z_h.transpose(1, 0, 2)))
        zl = mx.array(np.ascontiguousarray(z_l.transpose(1, 0, 2)))
        out = []
        for _ in range(steps):
            zh, zl = self._step(zh, zl)
            cells = self._decode(zh)
            mx.eval(zh, zl, cells)
            out.append(np.asarray(cells).T.copy())  # (B, 81)
        return np.stack(out, axis=1)


class _TorchBackend:
    def __init__(self, puzzle: str, device: str = "cpu"):
        import torch
        from loopscape import get_solver

        self.torch = torch
        solver = get_solver("eqr", device=device)
        self.inner = solver.model.inner
        self.inner.L_level.noise_scale = 0.0
        self.device = device
        tok = torch.tensor(_tokens(puzzle), dtype=torch.int32, device=device)[None, :]
        self.tokens = tok
        self.name = f"torch-{device}"

    def _x(self, b):
        torch = self.torch
        ids = torch.zeros((b,), dtype=torch.int32, device=self.device)
        return self.inner._input_embeddings(self.tokens.repeat(b, 1), ids)

    def step(self, z_h, z_l):
        torch = self.torch
        with torch.inference_mode():
            zh = torch.as_tensor(z_h, device=self.device)
            zl = torch.as_tensor(z_l, device=self.device)
            zh, zl = self.inner.deep_recursion(zh, zl, self._x(len(zh)), {"cos_sin": None})
            return zh.float().cpu().numpy(), zl.float().cpu().numpy()

    def rollout(self, z_h, z_l, steps):
        torch = self.torch
        out = []
        with torch.inference_mode():
            zh = torch.as_tensor(z_h, device=self.device)
            zl = torch.as_tensor(z_l, device=self.device)
            x = self._x(len(zh))
            for _ in range(steps):
                zh, zl = self.inner.deep_recursion(zh, zl, x, {"cos_sin": None})
                logits = self.inner.lm_head(zh[:, REG:])
                out.append((logits.argmax(-1) - 1).to(torch.int8).cpu().numpy())
        return np.stack(out, axis=1)


class Engine:
    """Run the deterministic EqR map for one puzzle from arbitrary initial latents."""

    def __init__(self, puzzle: str, backend: str = "mlx", steps: int = 24, device: str = "cpu"):
        self.puzzle, self.steps = puzzle, steps
        if backend == "mlx":
            self._b = _MlxBackend(puzzle, _load_weights())
        elif backend == "torch":
            self._b = _TorchBackend(puzzle, device=device)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

    def step(self, z_h: np.ndarray, z_l: np.ndarray):
        """One EqR loop. Inputs/outputs are float32 (B, 97, 512)."""
        return self._b.step(z_h, z_l)

    def rollout(self, z_h: np.ndarray, z_l: np.ndarray | None = None) -> np.ndarray:
        """Decoded grids after each loop: int8 (B, steps, 81). ``z_l`` defaults to zeros (notebook setting)."""
        z_h = np.ascontiguousarray(z_h, dtype=np.float32)
        z_l = np.zeros_like(z_h) if z_l is None else np.ascontiguousarray(z_l, dtype=np.float32)
        return self._b.rollout(z_h, z_l, self.steps)
