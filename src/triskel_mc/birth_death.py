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


def compute_bd_hazards_all(
    ps: PSState,
    betas: np.ndarray,  # (C,) 1/temps
    *,
    qb_density_np: Callable[
        [np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
    ],
    qb_eval_variant: Literal["child", "parent"],
    log_prior_phi_np: Callable[[np.ndarray], float],
    log_pseudo_phi_np: Callable[[np.ndarray], float],
    log_p_k_np: Callable[[np.ndarray], np.ndarray],  # vectorized over k (B,)
    batched_loglik_masked: Callable[
        [np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
    ],
    bd_rate_scale=1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute hazards (lam_on, lam_off, lam_total) for ALL (c,w) in one shot."""

    phi, m, rest, logpi_cur = ps.phi, ps.m, ps.rest, ps.logpi
    C, W, Kmax, d = phi.shape
    B = C * W
    beta_cw = betas[:, None]  # (C,1)
    log_beta = np.log(np.clip(beta_cw, 1e-300, None))

    def _log_pos(x):
        return np.where(x > 0.0, np.log(x), -np.inf)

    log_bd_scale = np.log(max(bd_rate_scale, 1e-300))

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

    # ----- Static prior/pseudoprior + k-terms for current and off -----
    comp_cur = np.zeros((C, W), dtype=np.float64)
    for i in range(Kmax):
        act = m[:, :, i]
        comp_cur += np.where(
            act,
            np.vectorize(log_prior_phi_np, signature="(d)->()")(
                phi[:, :, i, :]
            ),
            np.vectorize(log_pseudo_phi_np, signature="(d)->()")(
                phi[:, :, i, :]
            ),
        )

    k_cur = m.astype(np.int32).sum(axis=-1)  # (C,W)
    comb_cur = (
        log_p_k_np(k_cur)
        + _log_uniform_masks_given_k(Kmax, k_cur)
        + _log_symmetrization(k_cur)
    )  # (C,W)

    logpi_unscaled_cur = comb_cur + comp_cur  # (C,W)

    logpsi = np.zeros((C, W, Kmax), dtype=np.float64)
    logp = np.zeros((C, W, Kmax), dtype=np.float64)
    for i in range(Kmax):
        logpsi[:, :, i] = np.vectorize(log_pseudo_phi_np, signature="(d)->()")(
            phi[:, :, i, :]
        )
        logp[:, :, i] = np.vectorize(log_prior_phi_np, signature="(d)->()")(
            phi[:, :, i, :]
        )
    act = m  # (C,W,Kmax) bool

    delta_comp_off = (logpsi - logp) * act  # (C,W,Kmax)
    k_off = np.clip(k_cur - 1, 0, Kmax)  # (C,W)
    comb_off = (
        log_p_k_np(k_off)
        + _log_uniform_masks_given_k(Kmax, k_off)
        + _log_symmetrization(k_off)
    )
    comb_off_b = np.repeat(comb_off[:, :, None], Kmax, axis=2)
    comb_cur_b = np.repeat(comb_cur[:, :, None], Kmax, axis=2)
    delta_comb_off = (comb_off_b - comb_cur_b) * act  # (C,W,Kmax)

    delta_unscaled_off = delta_comp_off + delta_comb_off
    delta_comp_on = (logp - logpsi) * (~m)  # sign flip vs OFF
    k_on = np.clip(k_cur + 1, 0, Kmax)
    comb_on = (
        log_p_k_np(k_on)
        + _log_uniform_masks_given_k(Kmax, k_on)
        + _log_symmetrization(k_on)
    )
    delta_comb_on = (
        np.repeat(comb_on[:, :, None], Kmax, axis=2)
        - np.repeat(comb_cur[:, :, None], Kmax, axis=2)
    ) * (~m)
    delta_unscaled_on = delta_comp_on + delta_comb_on

    beta_cw = betas[:, None]  # (C,1)

    Delta_off = delta_unscaled_off + beta_cw[:, :, None] * (ll_off - ll_cur[:, :, None])
    Delta_on = delta_unscaled_on + beta_cw[:, :, None] * (ll_on - ll_cur[:, :, None])

    log_lam_on = np.full((C, W, Kmax), -np.inf, dtype=np.float64)
    log_lam_off = np.full((C, W, Kmax), -np.inf, dtype=np.float64)

    for j in range(Kmax):
        if qb_eval_variant == "child":
            ctx_fwd = m.copy()
            ctx_fwd[:, :, j] = True
        else:
            ctx_fwd = m
        q_fwd = qb_density_np(phi[:, :, j, :], ctx_fwd, phi, ps.rest)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        if qb_eval_variant == "child":
            ctx_rev = m.copy()
            ctx_rev[:, :, j] = False
        else:
            ctx_rev = m.copy()
            ctx_rev[:, :, j] = True
        q_rev = qb_density_np(phi[:, :, j, :], ctx_rev, phi, ps.rest)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        Delta_on_tilde = Delta_on[:, :, j] + (log_q_rev - log_q_fwd)

        log_lam_on[:, :, j] = np.where(
            ~m[:, :, j], log_beta + log_q_fwd + _logsigmoid(Delta_on_tilde), -np.inf
        )

    for i in range(Kmax):
        if qb_eval_variant == "child":
            ctx_fwd = m.copy()
            ctx_fwd[:, :, i] = False
        else:
            ctx_fwd = m
        q_fwd = qb_density_np(phi[:, :, i, :], ctx_fwd, phi, ps.rest)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        if qb_eval_variant == "child":
            ctx_rev = m.copy()
            ctx_rev[:, :, i] = True
        else:
            ctx_rev = m.copy()
            ctx_rev[:, :, i] = False
        q_rev = qb_density_np(phi[:, :, i, :], ctx_rev, phi, ps.rest)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        Delta_off_tilde = Delta_off[:, :, i] + (log_q_rev - log_q_fwd)

        log_lam_off[:, :, i] = np.where(
            m[:, :, i], log_beta + log_q_fwd + _logsigmoid(Delta_off_tilde), -np.inf
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
