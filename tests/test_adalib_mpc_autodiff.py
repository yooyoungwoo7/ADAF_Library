"""
Tests for gradient-based (autodiff) and CEM surrogate MPC.

Covers three claims:
 1. run_mpc(gradient="autodiff") runs a horizon-H closed loop end-to-end.
 2. The autodiff gradient dJ/dz through the operator (net + LPA basis,
    chained over H segments) matches central finite differences.
 3. run_mpc(optimizer="CEM") runs a batched sampling closed loop.
"""
import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import adalib
from adalib.mpc.options import MPCOptions

X0      = [80.0, 60.0, 70.0]
TARGET  = {"h3": 150.0}
HIDDEN  = 32
LAYERS  = 2
HORIZON = 3


def _base_opts(work_dir, **overrides):
    opts = MPCOptions(
        mode="tracking",
        basis="lpa",
        target=dict(TARGET),
        n_steps=2,
        horizon=HORIZON,
        n_train=10,
        n_val=4,
        epochs=2,
        batch_size=64,
        hidden=HIDDEN,
        n_layers=LAYERS,
        work_dir=str(work_dir),
        verbose=False,
        seed=7,
    )
    for k, v in overrides.items():
        setattr(opts, k, v)
    return opts


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("tt_autodiff_mpc")


@pytest.fixture(scope="module")
def autodiff_result(work_dir):
    opts = _base_opts(work_dir, gradient="autodiff")
    return adalib.run_mpc(system="triple_tank", x0=X0, options=opts)


def test_autodiff_closed_loop(autodiff_result):
    r = autodiff_result
    assert r.x.shape == (3, 3)          # n_steps+1, n_state
    assert r.u.shape == (2, 2)          # n_steps, n_control
    assert np.all(np.isfinite(r.x))
    assert np.all(np.isfinite(r.u))
    assert np.all(r.u >= 0.0) and np.all(r.u <= 100.0)
    assert r.metadata["mpc_optimizer"] == "SLSQP-autodiff"
    assert r.metadata["mpc_horizon"] == HORIZON
    assert r.metadata["opt_njev_mean"] >= 1.0


def test_autodiff_gradient_matches_fd(autodiff_result, work_dir):
    """dJ/dz from tf.GradientTape must match central finite differences."""
    import tensorflow as tf
    from adalib.workflows.mpc_workflow import _ensure_legacy_on_path
    from adalib.utils.legacy_context import reload_legacy_chain
    from adalib.mpc._surrogate_mpc import _build_tracking_spec, _make_rollout_fns

    _ensure_legacy_on_path()
    reload_legacy_chain("triple_tank_mpc", "lpa")
    import problems.registry as _preg
    import models.learner as _ml
    import data.dataset_builder as _db

    problem = _preg.get_problem("triple_tank_mpc")
    seg_path = os.path.join(str(work_dir), "operator", "data",
                            "triple_tank_mpc_train_segments.npz")
    train = _db.load_segments(seg_path)
    learner = _ml.OperatorLearner(
        problem=problem, hidden=HIDDEN, n_layers=LAYERS,
        x_mean=train.get("X_mean"), x_std=train.get("X_std"),
    )
    ckpt = autodiff_result.operator_result["best_checkpoint"]
    assert ckpt and os.path.exists(ckpt)
    learner.load_weights(ckpt)

    opts = _base_opts(work_dir, gradient="autodiff")
    spec = _build_tracking_spec("triple_tank_mpc", problem, opts)
    batch_cost, cost_and_grad = _make_rollout_fns(learner, HORIZON, spec)

    DTYPE = learner.basis.dtype
    xk = tf.constant(np.asarray(X0, dtype=np.float32), dtype=DTYPE)
    rng = np.random.default_rng(0)
    z = rng.uniform(0.2, 0.8, size=HORIZON * 2)

    J0, g_ad = cost_and_grad(xk, tf.constant(z, dtype=DTYPE))
    g_ad = np.asarray(g_ad.numpy(), dtype=np.float64)
    assert np.all(np.isfinite(g_ad))

    def _J(z_np):
        Z = tf.constant(z_np.reshape(1, HORIZON, 2), dtype=DTYPE)
        return float(batch_cost(xk, Z).numpy()[0])

    h = 1e-2
    g_fd = np.zeros_like(g_ad)
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        g_fd[i] = (_J(zp) - _J(zm)) / (2.0 * h)

    # cosine similarity + relative error on the dominant components
    denom = np.linalg.norm(g_ad) * np.linalg.norm(g_fd)
    assert denom > 0.0
    cos = float(np.dot(g_ad, g_fd) / denom)
    assert cos > 0.99, f"cosine(g_ad, g_fd)={cos:.4f}"

    scale = np.max(np.abs(g_fd))
    big = np.abs(g_fd) > 0.05 * scale
    rel = np.abs(g_ad[big] - g_fd[big]) / np.abs(g_fd[big])
    assert np.max(rel) < 0.05, f"max relative gradient error {np.max(rel):.3e}"


def test_cstr_autodiff_gradient_matches_fd(tmp_path):
    """Gradient consistency on the stiff CSTR (Arrhenius kinetics)."""
    import tensorflow as tf
    from adalib.workflows.mpc_workflow import _ensure_legacy_on_path
    from adalib.utils.legacy_context import reload_legacy_chain
    from adalib.mpc._surrogate_mpc import _build_tracking_spec, _make_rollout_fns

    wd = str(tmp_path / "cstr_ad")
    opts = MPCOptions(
        mode="tracking", basis="lpa", target={"T_R": 136.0},
        n_steps=1, horizon=3, n_train=40, n_val=8, epochs=3,
        batch_size=64, hidden=32, n_layers=2,
        work_dir=wd, verbose=False, seed=5, gradient="autodiff",
    )
    res = adalib.run_mpc(system="cstr", x0=[0.8, 0.5, 141.0, 141.0], options=opts)

    _ensure_legacy_on_path()
    reload_legacy_chain("cstr_mpc", "lpa")
    import problems.registry as _preg
    import models.learner as _ml
    import data.dataset_builder as _db
    problem = _preg.get_problem("cstr_mpc")
    train = _db.load_segments(os.path.join(
        wd, "operator", "data", "cstr_mpc_train_segments.npz"))
    learner = _ml.OperatorLearner(
        problem=problem, hidden=32, n_layers=2,
        x_mean=train.get("X_mean"), x_std=train.get("X_std"))
    learner.load_weights(res.operator_result["best_checkpoint"])

    spec = _build_tracking_spec("cstr_mpc", problem, opts)
    H = 3
    batch_cost, cost_and_grad = _make_rollout_fns(learner, H, spec)
    DTYPE = learner.basis.dtype
    xk = tf.constant([0.8, 0.5, 141.0, 141.0], dtype=DTYPE)
    rng = np.random.default_rng(1)
    z = rng.uniform(0.2, 0.8, size=H * 1)   # 1 control (Q)

    _, g_ad = cost_and_grad(xk, tf.constant(z, dtype=DTYPE))
    g_ad = np.asarray(g_ad.numpy(), dtype=np.float64)
    assert np.all(np.isfinite(g_ad))

    def _J(z_np):
        Z = tf.constant(z_np.reshape(1, H, 1), dtype=DTYPE)
        return float(batch_cost(xk, Z).numpy()[0])

    h = 1e-2
    g_fd = np.array([(_J(z + h * np.eye(len(z))[i])
                      - _J(z - h * np.eye(len(z))[i])) / (2 * h)
                     for i in range(len(z))])
    denom = np.linalg.norm(g_ad) * np.linalg.norm(g_fd)
    assert denom > 0.0
    cos = float(np.dot(g_ad, g_fd) / denom)
    assert cos > 0.99, f"cosine(g_ad, g_fd)={cos:.4f}"


def test_mppi_closed_loop(autodiff_result, work_dir):
    opts = _base_opts(
        work_dir, optimizer="MPPI", mppi_samples=64, mppi_iters=2,
        generate_data=False, reuse_existing_data=True,
        train_operator=False, reuse_existing_operator=True,
    )
    r = adalib.run_mpc(system="triple_tank", x0=X0, options=opts)
    assert r.x.shape == (3, 3)
    assert np.all(np.isfinite(r.x))
    assert np.all(r.u >= 0.0) and np.all(r.u <= 100.0)
    assert r.metadata["mpc_optimizer"] == "MPPI"
    assert r.metadata["rollouts_per_step"] == 64 * 2


def test_cem_closed_loop(autodiff_result, work_dir):
    opts = _base_opts(
        work_dir,
        optimizer="CEM",
        cem_samples=64,
        cem_elites=8,
        cem_iters=2,
        generate_data=False,
        reuse_existing_data=True,
        train_operator=False,
        reuse_existing_operator=True,
    )
    r = adalib.run_mpc(system="triple_tank", x0=X0, options=opts)
    assert r.x.shape == (3, 3)
    assert r.u.shape == (2, 2)
    assert np.all(np.isfinite(r.x))
    assert np.all(r.u >= 0.0) and np.all(r.u <= 100.0)
    assert r.metadata["mpc_optimizer"] == "CEM"
    assert r.metadata["rollouts_per_step"] == 64 * 2
