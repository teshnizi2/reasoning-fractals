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


@pytest.fixture(scope="module")
def engine():
    from render.engine import Engine

    return Engine(PUZZLE, backend="mlx")


def test_engine_loads_and_reports_backend(engine):
    assert engine.backend == "mlx" and engine.steps == 24


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


def test_torch_backend_agrees_with_mlx_backend(engine):
    from render.engine import Engine

    e2 = Engine(PUZZLE, backend="torch")
    rng = np.random.default_rng(4)
    z = rng.standard_normal((2, 97, 512)).astype(np.float32)
    assert (e2.rollout(z) == engine.rollout(z)).mean() > 0.999
