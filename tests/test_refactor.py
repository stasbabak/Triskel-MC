import numpy as np
import jax.numpy as jnp

from triskel_mc import (
    PTState,
    PSState,
    RunTrace,
    TraceConfig,
    compute_bd_hazards_all,
    make_batched_loglik_masked,
    redblue_mask_np,
)
from triskel_mc import bd_mh_step_numpy  # compatibility import


def test_runtrace_selection_and_snapshot():
    betas = np.array([1.0, 0.5])
    cfg = TraceConfig(chain_inds=[1])
    tr = RunTrace.init(cfg, betas=betas, W=1, D=2, Kmax=2, d=1)
    ps = PSState(
        phi=np.zeros((2, 1, 2, 1)),
        m=np.zeros((2, 1, 2), dtype=bool),
        rest=None,
        logpi=np.zeros((2, 1)),
    )
    tr.add_bd_event(
        type("ev", (), {"t_abs": 0.0, "dt": 0.0, "kind": 0, "c": 0, "w": 0, "slot": 0, "k_before": 0, "k_after": 1}),
        ps,
    )
    assert tr.bd_events[0].phi_sel.shape[0] == 1


def test_compute_bd_hazards_all_shapes():
    rng = np.random.default_rng(0)
    C, W, Kmax, d = 1, 2, 3, 1
    phi = rng.standard_normal((C, W, Kmax, d))
    m = rng.random((C, W, Kmax)) > 0.5
    ps = PSState(phi=phi, m=m, rest=None, logpi=np.zeros((C, W)))

    def qb_density_np(phi_j, ctx, phi_all, rest):
        return np.ones_like(ctx[..., 0], dtype=float)

    def log_prior_phi_np(phi_vec):
        return float(np.sum(phi_vec**2) * -0.5)

    def log_pseudo_phi_np(phi_vec):
        return 0.0

    def log_p_k_np(k):
        return np.zeros_like(k, dtype=float)

    def log_lik_masked_jax(phi_jax, m_jax, rest_jax):
        return jnp.sum(phi_jax * m_jax[..., None])

    batched = make_batched_loglik_masked(log_lik_masked_jax)
    log_on, log_off, lam_total, _ = compute_bd_hazards_all(
        ps,
        betas=np.ones(C),
        qb_density_np=qb_density_np,
        qb_eval_variant="child",
        log_prior_phi_np=log_prior_phi_np,
        log_pseudo_phi_np=log_pseudo_phi_np,
        log_p_k_np=log_p_k_np,
        batched_loglik_masked=batched,
        bd_rate_scale=1.0,
    )

    assert log_on.shape == (C, W, Kmax)
    assert log_off.shape == (C, W, Kmax)
    assert lam_total.shape == (C, W)


def test_redblue_mask_np_partition():
    rng = np.random.default_rng(0)
    red, blue = redblue_mask_np(rng, C=2, W=4)
    assert red.shape == blue.shape == (2, 4)
    assert np.all(~(red & blue))
    assert np.all((red | blue))


# ensure compatibility module exposes runner symbols
assert hasattr(bd_mh_step_numpy, "run_ct_mcmc")
assert hasattr(bd_mh_step_numpy, "run_epoch_ct_numpy")
