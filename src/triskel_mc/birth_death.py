"""Birth–death process utilities extracted from :mod:`bd_mh_step_numpy`."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .states import PSState


def _log_uniform_masks_given_k(Kmax: int, k: np.ndarray) -> np.ndarray:
    # log p(m | k) = -log binom(Kmax, k)
    k = k.astype(np.int32)
    return -(gammaln(Kmax + 1.0) - gammaln(k + 1.0) - gammaln(Kmax - k + 1.0))


def _log_symmetrization(k: np.ndarray) -> np.ndarray:
    # log (1/k!) = -log(k!)
    return -gammaln(k + 1.0)


def make_batched_loglik_masked(
    log_lik_masked_jax: Callable[
        [jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray
    ],
):
    """Return two JAX-compiled batched evaluators.
        1) batched_cur(phi: (B,Kmax,d), m: (B,Kmax), rest: (B,Drest or 0)) -> (B,)
        2) batched_off(phi: (B*Kmax,Kmax,d), m: (B*Kmax,Kmax), rest: (B*Kmax,Drest or 0)) -> (B*Kmax,)
    """

    f = jax.jit(
        jax.vmap(log_lik_masked_jax, in_axes=(0, 0, 0))
    )  # batch over first axis

    def batched(
        phi_b: np.ndarray, m_b: np.ndarray, rest_b: Optional[np.ndarray]
    ) -> np.ndarray:
        return np.array(
            f(
                jnp.asarray(phi_b),
                jnp.asarray(m_b),
                None if rest_b is None else jnp.asarray(rest_b),
            )
        )

    return batched



# def make_batched_loglik_masked(log_lik_masked_jax):
#     """Return a batched evaluator that supports rest_b being None."""
#
#     # case A: rest provided (batched)
#     f_with_rest = jax.jit(jax.vmap(log_lik_masked_jax, in_axes=(0, 0, 0)))
#
#     # case B: rest is None (do NOT vmap over None)
#     def _ll_no_rest(phi, m):
#         return log_lik_masked_jax(phi, m, None)
#     f_no_rest = jax.jit(jax.vmap(_ll_no_rest, in_axes=(0, 0)))
#
#     def batched(phi_b, m_b, rest_b):
#         phi_b = jnp.asarray(phi_b)
#         m_b = jnp.asarray(m_b)
#         if rest_b is None:
#             out = f_no_rest(phi_b, m_b)
#         else:
#             out = f_with_rest(phi_b, m_b, jnp.asarray(rest_b))
#         return np.asarray(out)
#
#     return batched



def masked_ll_for_phi_batch(
    phi: np.ndarray,  # (C,W,Kmax,d)
    m: np.ndarray,  # (C,W,Kmax) bool
    rest: np.ndarray | None,
    batched_loglik_masked,  # fn(B,Kmax,d),(B,Kmax),(B,Drest|) -> (B,)
) -> np.ndarray:
    C, W, Kmax, d = phi.shape
    B = C * W
    ll = batched_loglik_masked(
        phi.reshape(B, Kmax, d),
        m.reshape(B, Kmax),
        None if rest is None else rest.reshape(B, -1),
    ).reshape(C, W)
    return np.asarray(ll, dtype=np.float64)


def _logsigmoid(x):
    # -softplus(-x)
    return -np.log1p(np.exp(-np.clip(x, -700, 700)))


# def compute_bd_hazards_all(
#     ps: PSState,
#     betas: np.ndarray,  # (C,) 1/temps
#     *,
#     qb_density_np: Callable[
#         [np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
#     ],
#     qb_eval_variant: Literal["child", "parent"],
#     log_prior_phi_np: Callable[[np.ndarray], float],
#     log_pseudo_phi_np: Callable[[np.ndarray], float],
#     log_p_k_np: Callable[[np.ndarray], np.ndarray],  # vectorized over k (B,)
#     batched_loglik_masked: Callable[
#         [np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
#     ],
#     bd_rate_scale=1.0,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

# log_p_mask_np(m, slot_type, must_be_on, can_toggle) -> (C,W)
# lot_type can be int/str codes; we'll keep it as np.ndarray

def compute_bd_hazards_all(
    ps: PSState,
    betas: np.ndarray,
    *,
    qb_density_np: Callable[..., np.ndarray],
    qb_eval_variant: Literal["child", "parent"],
    log_prior_phi_np: Callable[[np.ndarray, int], float],
    log_pseudo_phi_np: Callable[[np.ndarray, int], float],
    log_p_mask_np: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], np.ndarray], ##(C, W)
    batched_loglik_masked: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    bd_rate_scale=1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Compute hazards (lam_on, lam_off, lam_total) for ALL (c,w) in one shot."""

    phi, m, rest, logpi_cur = ps.phi, ps.m, ps.rest, ps.logpi
    slot_dim  = ps.slot_dim
    slot_type = ps.slot_type
    must_be_on = ps.must_be_on
    can_toggle = ps.can_toggle

    C, W, Kmax, d = phi.shape
    B = C * W
    beta_cw = betas[:, None]  # (C,1)
    log_beta = np.log(np.clip(beta_cw, 1e-300, None))

    def _log_pos(x):
        return np.where(x > 0.0, np.log(x), -np.inf)

    log_bd_scale = np.log(max(bd_rate_scale, 1e-300))

    ### I want some model to be always on (like noise model) and others can toggle (like signal)
    # [CHANGED] defaults for constraints if None
    if must_be_on is None:
        must_be_on = np.zeros((Kmax,), dtype=bool)
    if can_toggle is None:
        can_toggle = np.ones((Kmax,), dtype=bool)

    # [ADDED] eligibility masks (C,W,Kmax) enforcing constraints
    eligible_on = (~m) & can_toggle[None, None, :] & (~must_be_on[None, None, :])
    eligible_off = (m) & can_toggle[None, None, :] & (~must_be_on[None, None, :])

    # ----- Build current-mask batch to compute current LL -----
    phi_cur = phi.reshape(B, Kmax, d)  # (B,Kmax,d)
    m_cur = m.reshape(B, Kmax)  # (B,Kmax)
    rest_cur = None if rest is None else rest.reshape(B, -1)

    ll_cur = batched_loglik_masked(phi_cur, m_cur, rest_cur)  # (B,)
    ll_cur = ll_cur.reshape(C, W)

    # ----- Build "turn OFF i" batches -----
    m_off = np.repeat(m_cur[:, None, :], Kmax, axis=1)  # (B,Kmax,Kmax)
    arK = np.arange(Kmax)
    m_off[np.arange(B)[:, None], arK[None, :], arK[None, :]] = False
    phi_off = np.repeat(phi_cur[:, None, :, :], Kmax, axis=1)  # (B,Kmax,Kmax,d)
    rest_off = (
        None if rest_cur is None else np.repeat(rest_cur[:, None, :], Kmax, axis=1)
    )

    phi_off_flat = phi_off.reshape(B * Kmax, Kmax, d)
    m_off_flat = m_off.reshape(B * Kmax, Kmax)
    rest_off_flat = None if rest_off is None else rest_off.reshape(B * Kmax, -1)

    ll_off_flat = batched_loglik_masked(
        phi_off_flat, m_off_flat, rest_off_flat
    )  # (B*Kmax,)
    ll_off = ll_off_flat.reshape(C, W, Kmax)  # (C,W,Kmax)

    # --- build "turn ON j" batches (mirror of OFF) ---
    m_on = np.repeat(m_cur[:, None, :], Kmax, axis=1)  # (B,Kmax,Kmax)
    arK = np.arange(Kmax)
    m_on[np.arange(B)[:, None], arK[None, :], arK[None, :]] = True
    phi_on = np.repeat(phi_cur[:, None, :, :], Kmax, axis=1)  # (B,Kmax,Kmax,d)
    rest_on = (
        None if rest_cur is None else np.repeat(rest_cur[:, None, :], Kmax, axis=1)
    )

    phi_on_flat = phi_on.reshape(B * Kmax, Kmax, d)
    m_on_flat = m_on.reshape(B * Kmax, Kmax)
    rest_on_flat = None if rest_on is None else rest_on.reshape(B * Kmax, -1)
    ll_on_flat = batched_loglik_masked(
        phi_on_flat, m_on_flat, rest_on_flat
    )  # (B*Kmax,)
    ll_on = ll_on_flat.reshape(C, W, Kmax)  # (C,W,Kmax)

    # ----- per-slot prior/pseudo with true slot dimension and slot type
    comp_cur = np.zeros((C, W), dtype=np.float64)
    logp   = np.zeros((C, W, Kmax), dtype=np.float64)
    logpsi = np.zeros((C, W, Kmax), dtype=np.float64)

    for j in range(Kmax):
        dj = int(slot_dim[j])
        tj = int(slot_type[j])
        phi_j = phi[:, :, j, :dj]  # <-- [CHANGED] use slot_dim

        logp[:, :, j] = np.vectorize(lambda v: log_prior_phi_np(v, tj), signature="(d)->()")(phi_j)
        logpsi[:, :, j] = np.vectorize(lambda v: log_pseudo_phi_np(v, tj), signature="(d)->()")(phi_j)
        comp_cur += np.where(m[:, :, j], logp[:, :, j], logpsi[:, :, j])

    # [ADDED] generic mask prior p(m) (replaces p(k)*binom*1/k!)
    logpm_cur = log_p_mask_np(m, slot_type, must_be_on, can_toggle)  # (C,W)
    logpi_unscaled_cur = logpm_cur + comp_cur  # (C,W)

    logpm_off = np.full((C, W, Kmax), -np.inf, dtype=np.float64)
    logpm_on = np.full((C, W, Kmax), -np.inf, dtype=np.float64)
    for j in range(Kmax):
        m_off_j = m.copy()
        m_off_j[:, :, j] = False
        logpm_off[:, :, j] = log_p_mask_np(m_off_j, slot_type, must_be_on, can_toggle)

        m_on_j = m.copy()
        m_on_j[:, :, j] = True
        logpm_on[:, :, j] = log_p_mask_np(m_on_j, slot_type, must_be_on, can_toggle)

    # [CHANGED] unscaled deltas now use log_p_mask differences (no k_cur/k_on/k_off etc.)
    delta_unscaled_off = (logpm_off - logpm_cur[:, :, None]) + (logpsi - logp) * m
    delta_unscaled_on = (logpm_on - logpm_cur[:, :, None]) + (logp - logpsi) * (~m)


    beta_cw = betas[:, None]  # (C,1)

    Delta_off = delta_unscaled_off + beta_cw[:, :, None] * (ll_off - ll_cur[:, :, None])
    Delta_on = delta_unscaled_on + beta_cw[:, :, None] * (ll_on - ll_cur[:, :, None])

    log_lam_on = np.full((C, W, Kmax), -np.inf, dtype=np.float64)
    log_lam_off = np.full((C, W, Kmax), -np.inf, dtype=np.float64)


    for j in range(Kmax):
        dj = int(slot_dim[j])
        tj = int(slot_type[j])
        phi_j = phi[:, :, j, :dj]  # <-- [CHANGED] use slot_dim

        # q_fwd for ON (inactive -> active)
        if qb_eval_variant == "child":
            ctx_fwd = m.copy()
            ctx_fwd[:, :, j] = True
        else:
            ctx_fwd = m

        #  pass slot metadata; and pass phi_j not padded
        q_fwd = qb_density_np(phi_j, ctx_fwd, phi, ps.rest, slot=j, slot_type=tj)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        # q_rev for reverse OFF at destination
        if qb_eval_variant == "child":
            ctx_rev = m.copy()
            ctx_rev[:, :, j] = False
        else:
            ctx_rev = m.copy()
            ctx_rev[:, :, j] = True

        # [CHANGED] pass slot metadata
        q_rev = qb_density_np(phi_j, ctx_rev, phi, ps.rest, slot=j, slot_type=tj)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        Delta_on_tilde = Delta_on[:, :, j] + (log_q_rev - log_q_fwd)

        # [CHANGED] enforce eligibility_on (constraints)
        log_lam_on[:, :, j] = np.where(
            eligible_on[:, :, j],
            log_beta + log_q_fwd + _logsigmoid(Delta_on_tilde),
            -np.inf,
        )

    for i in range(Kmax):
        di = int(slot_dim[i])
        ti = int(slot_type[i])
        phi_i = phi[:, :, i, :di]  # <-- [CHANGED] use slot_dim

        # q_fwd for OFF (active -> inactive)
        if qb_eval_variant == "child":
            ctx_fwd = m.copy()
            ctx_fwd[:, :, i] = False
        else:
            ctx_fwd = m

        #  pass slot metadata
        q_fwd = qb_density_np(phi_i, ctx_fwd, phi, ps.rest, slot=i, slot_type=ti)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        # q_rev for reverse ON at destination
        if qb_eval_variant == "child":
            ctx_rev = m.copy()
            ctx_rev[:, :, i] = True
        else:
            ctx_rev = m.copy()
            ctx_rev[:, :, i] = False

        #  pass slot metadata
        q_rev = qb_density_np(phi_i, ctx_rev, phi, ps.rest, slot=i, slot_type=ti)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        Delta_off_tilde = Delta_off[:, :, i] + (log_q_rev - log_q_fwd)

        #  enforce eligibility_off (constraints)
        log_lam_off[:, :, i] = np.where(
            eligible_off[:, :, i],
            log_beta + log_q_fwd + _logsigmoid(Delta_off_tilde),
            -np.inf,
        )

    assert not np.any(np.isfinite(log_lam_on[m])), "ON hazard finite where active"
    assert not np.any(np.isfinite(log_lam_off[~m])), "OFF hazard finite where inactive"
    log_lam_all = np.concatenate([log_lam_on, log_lam_off], axis=2)  # (C,W,2K)
    maxv = np.max(log_lam_all, axis=2)  # (C,W)
    with np.errstate(over="ignore", invalid="ignore"):
        sumexp = np.sum(np.exp(log_lam_all - maxv[..., None]), axis=2)  # (C,W)
    log_lam_total = np.where(np.isfinite(maxv), maxv + np.log(sumexp), -np.inf)

    lam_total = np.where(np.isfinite(log_lam_total), np.exp(log_lam_total), 0.0)

    return log_lam_on, log_lam_off, lam_total, log_lam_total


__all__ = [name for name in globals() if not name.startswith("_")]
