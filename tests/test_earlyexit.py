"""Early-exit policy: retire a trajectory once its decoded grid and its latent have both stopped moving."""
import numpy as np

from render import earlyexit as EE


def tr(*grids):
    return np.stack([np.full(81, g, dtype=np.int8) for g in grids])


def test_retire_requires_m_identical_grids_and_a_small_residual():
    stable = np.stack([tr(5, 5, 5), tr(5, 5, 6), tr(5, 5, 5)])  # (B, m, 81)
    resid = np.array([1e-5, 1e-5, 1.0])
    out = EE.retire_mask(stable, resid, tol=1e-3)
    assert out.tolist() == [True, False, False]


def test_simulate_retires_at_the_first_checkpoint_where_the_policy_holds():
    traces = np.stack([tr(1, 2, 3, 3, 3, 3, 3, 3), tr(1, 2, 3, 4, 5, 6, 7, 8)])[:, :, :]  # (2, 8, 81)
    resid = np.array([[9, 5, 1, 1e-6, 1e-7, 1e-8, 1e-9, 1e-9], [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.float32)
    r = EE.simulate(traces, resid, m=2, tol=1e-3, checkpoints=(4, 6))
    assert r["retired_at"].tolist() == [4, 8]  # pixel 1 never retires: runs the full 8 loops
    assert (r["predicted"] == traces).all()    # exact here, since pixel 0 really did settle
    assert r["mean_loops"] == 6.0 and abs(r["saving"] - (1 - 6.0 / 8)) < 1e-12


def test_simulate_reports_the_pixels_whose_prediction_would_be_wrong():
    # pixel leaves its plateau right after the policy fired: a false retirement
    traces = np.stack([tr(1, 1, 3, 3, 3, 9, 9, 9)])
    resid = np.array([[9, 9, 1e-6, 1e-6, 1e-6, 9, 9, 9]], dtype=np.float32)
    r = EE.simulate(traces, resid, m=3, tol=1e-3, checkpoints=(5,))
    assert r["retired_at"].tolist() == [5] and r["n_wrong"] == 1
    assert (r["predicted"][0, -1] == 3).all() and not (traces[0, -1] == 3).all()


def test_solution_only_policy_never_retires_on_a_wrong_plateau():
    sol = np.full(81, 5, dtype=np.int8)
    wrong_plateau = tr(4, 4, 4, 4)  # long plateau on a wrong grid
    right_plateau = tr(4, 5, 5, 5)
    traces = np.stack([wrong_plateau, right_plateau])
    resid = np.ones((2, 4), dtype=np.float32)
    r = EE.simulate(traces, resid, m=3, tol=np.inf, checkpoints=(3, 4), solution=sol)
    assert r["retired_at"].tolist() == [4, 4]  # nobody retires before the cap...
    r = EE.simulate(traces, resid, m=2, tol=np.inf, checkpoints=(3,), solution=sol)
    assert r["retired_at"].tolist() == [4, 3]  # ...except the pixel plateaued on the solution
