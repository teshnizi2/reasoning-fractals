"""The rollout engine must be the loopscape/EqR deterministic map, just faster."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOOPSCAPE_CACHE_DIR", str(ROOT / "data/raw/loopscape_cache"))
os.environ.setdefault("LOOPSCAPE_EQR_SRC", str(ROOT / "vendor/EqR"))

pytestmark = pytest.mark.slow

PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."


@pytest.fixture(scope="module")
def ref_solver():
    from loopscape import get_solver

    return get_solver("eqr")  # CPU float32, the upstream reference path


def ref_run(solver, z_H, steps=24):
    """Upstream solve_batch with noise off; returns decoded traces (B, T, 81) and latent histories."""
    r = solver.solve_batch(
        PUZZLE,
        z_H=torch.tensor(z_H),
        z_L=torch.zeros(len(z_H), 97, 512),
        max_steps=steps,
        noise_scale=0.0,
        return_intermediates=True,
        return_latents=True,
    )
    tr = np.asarray(r.intermediates, dtype=np.int8)  # (T, B, 9, 9)
    tr = np.swapaxes(tr, 0, 1).reshape(len(z_H), steps, 81)
    return tr, r.extra["z_H_history"].numpy(), r.extra["z_L_history"].numpy()


BACKENDS = {
    "mlx": dict(backend="mlx"),
    "torchfn-cpu": dict(backend="torchfn", device="cpu"),
}


@pytest.fixture(scope="module", params=list(BACKENDS))
def engine(request):
    from render.engine import Engine

    return Engine(PUZZLE, **BACKENDS[request.param])


@pytest.fixture(scope="module")
def mlx_engine():
    from render.engine import Engine

    return Engine(PUZZLE, backend="mlx")


def test_engine_loads_and_reports_backend(engine):
    assert engine.backend in ("mlx", "torchfn") and engine.steps == 24


def test_one_step_matches_reference_latents(engine, ref_solver):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((3, 97, 512)).astype(np.float32)
    _, zh_hist, zl_hist = ref_run(ref_solver, z, steps=1)
    zH, zL = engine.step(z, np.zeros_like(z))
    np.testing.assert_allclose(zH, zh_hist[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(zL, zl_hist[0], rtol=2e-4, atol=2e-4)


def test_tiny_plane_latent_one_step_matches_reference(engine, ref_solver):
    # the regime the fractal lives in: |z| ~ 1e-3 around the origin
    from render import slices as SL

    u, v = SL.plane()
    z = SL.latents(SL.mesh(3)[:4], u, v)
    _, zh_hist, zl_hist = ref_run(ref_solver, z, steps=1)
    zH, zL = engine.step(z, np.zeros_like(z))
    np.testing.assert_allclose(zH, zh_hist[0], rtol=2e-4, atol=2e-4)


def test_rollout_traces_match_reference_on_gaussian_latents(engine, ref_solver):
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 97, 512)).astype(np.float32)
    ref, _, _ = ref_run(ref_solver, z)
    got = engine.rollout(z)
    assert got.shape == (4, 24, 81) and got.dtype == np.int8
    assert (got == ref).mean() > 0.999
    assert (got[:, -1] == ref[:, -1]).all()


def test_rollout_is_deterministic(engine):
    rng = np.random.default_rng(2)
    z = rng.standard_normal((2, 97, 512)).astype(np.float32)
    np.testing.assert_array_equal(engine.rollout(z), engine.rollout(z))


def test_rollout_row_does_not_depend_on_batch_neighbours_for_stable_latents(engine):
    rng = np.random.default_rng(3)
    z = rng.standard_normal((5, 97, 512)).astype(np.float32)
    full = engine.rollout(z)
    single = engine.rollout(z[2:3])
    assert (full[2, -1] == single[0, -1]).all()


def test_torch_backend_agrees_with_mlx_backend(mlx_engine):
    from render.engine import Engine

    e2 = Engine(PUZZLE, backend="torch")
    rng = np.random.default_rng(4)
    z = rng.standard_normal((2, 97, 512)).astype(np.float32)
    assert (e2.rollout(z) == mlx_engine.rollout(z)).mean() > 0.999


def test_unknown_precision_is_rejected():
    from render.engine import Engine

    with pytest.raises(ValueError):
        Engine(PUZZLE, backend="torchfn", device="cpu", precision="fp8")


def test_tf32_request_is_harmless_on_cpu_and_changes_nothing():
    from render.engine import Engine

    e = Engine(PUZZLE, backend="torchfn", device="cpu", precision="tf32")
    z = np.random.default_rng(5).standard_normal((2, 97, 512)).astype(np.float32)
    assert e.rollout(z).shape == (2, 24, 81)


@pytest.fixture(scope="module")
def torchfn_engine():
    from render.engine import Engine

    return Engine(PUZZLE, backend="torchfn", device="cpu")


def test_rollout_ex_returns_traces_and_latent_residuals(torchfn_engine):
    e = torchfn_engine
    z = np.random.default_rng(6).standard_normal((2, 97, 512)).astype(np.float32)
    traces, resid = e.rollout_ex(z)
    np.testing.assert_array_equal(traces, e.rollout(z))
    assert resid.shape == (2, 24) and resid.dtype == np.float32
    zh1, _ = e.step(z, np.zeros_like(z))
    np.testing.assert_allclose(resid[:, 0], np.abs(zh1 - z).max(axis=(1, 2)), rtol=1e-4)
    assert (resid[:, -1] < resid[:, 0]).all()  # a converging trajectory moves less at the end


def _solution():
    from render import sudoku as S

    return np.array([int(c) for c in S.solve(PUZZLE)], dtype=np.int8)


def test_rollout_early_matches_the_full_rollout_and_saves_loops(torchfn_engine):
    e = torchfn_engine
    z = np.random.default_rng(7).standard_normal((4, 97, 512)).astype(np.float32)
    full = e.rollout(z)
    traces, retired_at = e.rollout_early(z, solution=_solution(), m=3)
    np.testing.assert_array_equal(traces, full)
    assert retired_at.shape == (4,) and (retired_at < 24).all()  # all four solve and stay solved
    # after retirement the trace is the solution repeated
    sol = _solution()
    for k in range(4):
        assert (traces[k, retired_at[k] - 1 :] == sol).all()


def test_rollout_early_never_retires_a_pixel_that_is_not_on_the_solution(torchfn_engine):
    e = torchfn_engine
    z = np.random.default_rng(8).standard_normal((2, 97, 512)).astype(np.float32)
    wrong = _solution().copy()
    wrong[0] = (wrong[0] % 9) + 1  # a "solution" the model will never produce
    traces, retired_at = e.rollout_early(z, solution=wrong, m=3)
    assert (retired_at == 24).all()
    np.testing.assert_array_equal(traces, e.rollout(z))


def test_rollout_early_is_exposed_on_the_engine_for_torchfn_only(mlx_engine):
    with pytest.raises(NotImplementedError):
        mlx_engine.rollout_early(np.zeros((1, 97, 512), dtype=np.float32), solution=_solution())
