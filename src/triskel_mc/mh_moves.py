"""Metropolis–Hastings move kernels and helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .states import MHEvent, PTState, PSState
from .birth_death import compute_bd_hazards_all


def _resolve_slot_indices(D: int, slot_sel) -> np.ndarray:
    """Return resolved slot indices as a 1D ndarray."""

    if isinstance(slot_sel, slice):
        return np.arange(D)[slot_sel]
    return np.where(np.asarray(slot_sel, bool))[0]


def redblue_mask_np(rng: np.random.Generator, C: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    red = np.zeros((C, W), dtype=bool)
    for c in range(C):
        perm = rng.permutation(W)
        red_idx = perm[: W // 2]
        red[c, red_idx] = True
    return red, ~red


def propose_stretch_redblue_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,  # (C,W,D)
    subset_mask: np.ndarray,  # (C,W) True: moved in this half
    slot_sel,  # slice or boolean mask for the slot dims
    a: float = 2.0,
    partner_pool_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slot-aware Goodman–Weare stretch in *slot_sel* only."""

    C, W, D = X.shape
    idx = _resolve_slot_indices(D, slot_sel)
    d_slot = idx.size
    if d_slot == 0:
        return X.copy(), np.zeros((C, W), dtype=X.dtype), np.zeros((C, W), dtype=bool)

    comp = (~subset_mask) if partner_pool_mask is None else partner_pool_mask

    X_prop = X.copy()
    logJ = np.zeros((C, W), dtype=X.dtype)
    moved = np.zeros((C, W), dtype=bool)

    sa = np.sqrt(a)
    for c in range(C):
        comp_idx = np.where(comp[c])[0]
        if comp_idx.size == 0:
            continue
        u = rng.random(W)
        z = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2
        for w in range(W):
            if not subset_mask[c, w]:
                continue
            choices = comp_idx[comp_idx != w]
            if choices.size == 0:
                continue
            y = choices[rng.integers(0, choices.size)]
            x_cw = X[c, w, idx]
            y_cw = X[c, y, idx]
            zcw = z[w]
            X_prop[c, w, idx] = y_cw + zcw * (x_cw - y_cw)
            logJ[c, w] = (d_slot - 1.0) * np.log(zcw)
            moved[c, w] = True

    return X_prop, logJ, moved


def propose_rw_fullcov_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,  # (C,W,D)
    Ls: np.ndarray,  # (C,D,D) Cholesky for full θ
    slot_sel,
) -> np.ndarray:
    C, W, D = X.shape
    out = X.copy()
    idx = _resolve_slot_indices(D, slot_sel)
    d_slot = idx.size
    if d_slot == 0:
        return out

    Ls_slot = Ls[:, idx[:, None], idx[None, :]]  # (C,d_slot,d_slot)

    Z = rng.standard_normal(size=(C, W, d_slot))
    eps = np.einsum("cij,cwj->cwi", Ls_slot, Z)  # (C,W,d_slot)

    step = (2.38 / np.sqrt(d_slot) * 0.5) * eps
    out[:, :, idx] = X[:, :, idx] + step
    return out


def propose_rw_eigenline_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,  # (C,W,D)
    U: np.ndarray,  # (C,D,D) eigenvectors for full θ
    S: np.ndarray,  # (C,D)   eigenvalues
    slot_sel,
) -> np.ndarray:
    C, W, D = X.shape
    out = X.copy()
    idx = _resolve_slot_indices(D, slot_sel)
    d_slot = idx.size
    if d_slot == 0:
        return out

    axes_local = rng.integers(0, d_slot, size=(C, W))
    axes = idx[axes_local]

    r = rng.standard_normal(size=(C, W))
    for c in range(C):
        U_slot = np.zeros((W, d_slot), dtype=X.dtype)
        u_cols = U[c][idx, axes[c]]
        U_slot[:, :] = u_cols.T

        S_ax = S[c, axes[c]]
        step_slot = (r[c] * np.sqrt(S_ax))[:, None] * U_slot
        out[c, :, idx] = X[c, :, idx] + step_slot

    return out


def propose_rw_student_t_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,  # (C,W,D)
    Ls: np.ndarray,  # (C,D,D)
    slot_sel,
    nu: float = 5.0,
) -> np.ndarray:
    C, W, D = X.shape
    out = X.copy()
    idx = _resolve_slot_indices(D, slot_sel)
    d_slot = idx.size
    if d_slot == 0:
        return out

    Ls_slot = Ls[:, idx[:, None], idx[None, :]]  # (C,d_slot,d_slot)

    Z = rng.standard_normal(size=(C, W, d_slot))
    g = rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=(C, W))
    T = Z / np.sqrt(g[..., None])  # Student-t on slot
    t_tr = np.einsum("cij,cwj->cwi", Ls_slot, T)
    step = (2.38 / np.sqrt(d_slot) * 0.5) * t_tr
    out[:, :, idx] = X[:, :, idx] + step
    return out


def propose_de_two_point_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,  # (C,W,D)
    eligible_mask: np.ndarray,  # (C,W) who can be parent/partner (usually move_mask)
    slot_sel,  # slice or boolean mask for slot dims
    crossover_rate: float = 0.8,
    gamma_scale: float = 2.38,
) -> tuple[np.ndarray, np.ndarray]:
    """Slot-only differential evolution step."""

    C, W, D = X.shape
    out = X.copy()

    idx = _resolve_slot_indices(D, slot_sel)
    d_slot = idx.size
    if d_slot == 0:
        return out, np.zeros((C, W), dtype=bool)

    sigma = gamma_scale / np.sqrt(2.0 * d_slot)
    has_pair = np.zeros((C, W), dtype=bool)

    for c in range(C):
        pool = np.where(eligible_mask[c])[0]
        for w in range(W):
            choices = pool[pool != w]
            if choices.size < 2:
                continue
            y, z = rng.choice(choices, size=2, replace=False)
            diff = X[c, y, idx] - X[c, z, idx]
            gam = rng.normal(0.0, sigma)

            if crossover_rate >= 1.0:
                mask_slot = np.ones(d_slot, dtype=bool)
            else:
                mask_slot = rng.random(d_slot) < crossover_rate
                if not mask_slot.any():
                    mask_slot[rng.integers(0, d_slot)] = True

            step_slot = np.zeros(d_slot, dtype=X.dtype)
            step_slot[mask_slot] = gam * diff[mask_slot]
            out[c, w, idx] = X[c, w, idx] + step_slot
            has_pair[c, w] = True

    return out, has_pair


def restrict_slot_np(prop_all: np.ndarray, cur_all: np.ndarray, slot_slice_or_mask) -> np.ndarray:
    out = cur_all.copy()
    if isinstance(slot_slice_or_mask, slice):
        sl = slot_slice_or_mask
        out[..., sl] = prop_all[..., sl]
    else:
        maskD = np.asarray(slot_slice_or_mask, dtype=bool)[None, None, :]
        out = np.where(maskD, prop_all, cur_all)
    return out


def mh_accept_tempered_np(
    rng: np.random.Generator,
    ll_cur: np.ndarray,  # (C,W)
    ll_prop: np.ndarray,  # (C,W)
    lprior_cur_j: np.ndarray,  # (C,W)
    lprior_prop_j: np.ndarray,  # (C,W)
    betas: np.ndarray,  # (C,)
    log_qcorr: np.ndarray,  # (C,W)
    move_mask: np.ndarray,  # (C,W)
) -> np.ndarray:
    delta = (
        (lprior_prop_j - lprior_cur_j) + betas[:, None] * (ll_prop - ll_cur) + log_qcorr
    )
    log_u = np.log(rng.random(ll_cur.shape))
    return move_mask & (log_u < delta)


def apply_mh_and_record_np(
    rng: np.random.Generator,
    event_log,
    t_abs: float,
    dt: float,
    slot_j: int,
    move_type: str,
    pt: PTState,
    thetas_prop: np.ndarray,
    ll_cur: np.ndarray,
    ll_prop: np.ndarray,
    lprior_cur_j: np.ndarray,
    lprior_prop_j: np.ndarray,
    betas: np.ndarray,
    log_qcorr: np.ndarray,
    move_mask: np.ndarray,
):
    accept = mh_accept_tempered_np(
        rng,
        ll_cur,
        ll_prop,
        lprior_cur_j,
        lprior_prop_j,
        betas,
        log_qcorr,
        move_mask,
    )
    thetas_new = np.where(accept[:, :, None], thetas_prop, pt.thetas)
    ll_new = np.where(accept, ll_prop, ll_cur)
    pt_new = PTState(thetas=thetas_new, log_probs=ll_new)

    C, W = pt.thetas.shape[:2]
    for c in range(C):
        for w in range(W):
            if move_mask[c, w]:
                event_log.mh_events.append(
                    MHEvent(
                        t_abs=t_abs,
                        dt=dt,
                        c=c,
                        w=w,
                        slot=slot_j,
                        move_type=move_type,
                        accepted=bool(accept[c, w]),
                    )
                )
    return pt_new, accept


def _indices(mask2d: np.ndarray):
    assert mask2d.ndim == 2
    idx = np.where(mask2d)
    return idx[0], idx[1]


def _scatter_into(base: np.ndarray, data: np.ndarray, idx: tuple[np.ndarray, np.ndarray]):
    base = base.copy()
    base[idx] = data
    return base


def gibbs_mh_sweep_active_np(
    # rng: np.random.Generator,
    # event_log,
    # run_trace,
    # t_abs,
    # dt,
    # # thetas,
    # # lps,
    # pt_state,
    # ps_state,
    # betas,
    # # Kmax,
    # slot_slices,
    # Ls = None,
    # U = None,
    # S = None,
    # do_stretch=True,
    # do_rw_fullcov=True,
    # do_rw_eigenline=True,
    # do_rw_student_t=True,
    # do_de=True,
    # do_PTswap=True,
    # qb_density_np=None,
    # qb_eval_variant="child",
    # log_prior_phi_np=None,
    # # log_pseudo_phi_np=None,
    # # log_p_k_np=None,
    # batched_loglik_masked=None,
    # bd_rate_scale=1.0,
    # cross_rate=0.8,
    # gamma_de=2.38,
    # stretch_a=2.0,
    rng: np.random.Generator,
    *,
    t_abs,
    dt,
    pt_state: PTState,
    ps_state,
    slot_slices,
    betas,
    batched_loglik_masked=None,
    log_prior_phi_np=None,
    Ls=None,
    U=None,
    S=None,
    do_stretch=True,
    do_rw_fullcov=True,
    do_rw_eigenline=True,
    do_rw_student_t=True,
    do_de=True,
    do_PTswap=True,
    stretch_a=2.0,
    cross_rate=0.8,
    gamma_de=2.38,
    event_log=None,
    run_trace=None,
):

    thetas, lps = pt_state.thetas, pt_state.log_probs
    Kmax = ps_state.m.shape[-1]

    def _slot_prior_np(phi_cwd: np.ndarray) -> np.ndarray:
        return np.vectorize(log_prior_phi_np, signature="(d)->()")(phi_cwd)

    def _masked_ll(phi: np.ndarray) -> np.ndarray:
        C, W, Kmax, d = phi.shape
        ll = batched_loglik_masked(
            phi.reshape(C * W, Kmax, d),
            ps_state.m.reshape(C * W, Kmax),
            None if ps_state.rest is None else ps_state.rest.reshape(C * W, -1),
        ).reshape(C, W)
        return np.asarray(ll, dtype=np.float64)

    def _masked_ll_subset(phi_full, m_full, rest_full, idx_subset):
        c_idx, w_idx = idx_subset
        phi_subset = phi_full[c_idx, w_idx]
        m_subset = m_full[c_idx, w_idx]
        rest_subset = None if rest_full is None else rest_full[c_idx, w_idx]
        ll_subset = batched_loglik_masked(phi_subset, m_subset, rest_subset)
        ll_full = _scatter_into(
            np.zeros((phi_full.shape[0], phi_full.shape[1])),
            ll_subset,
            idx_subset,
        )
        return ll_subset, ll_full

    def _slot_prior_subset(phi_j_full, idx_subset):
        c_idx, w_idx = idx_subset
        phi_subset = phi_j_full[c_idx, w_idx]
        prior_subset = np.vectorize(log_prior_phi_np, signature="(d)->()")(
            phi_subset
        )
        prior_full = _scatter_into(
            np.zeros((phi_j_full.shape[0], phi_j_full.shape[1])),
            prior_subset,
            idx_subset,
        )
        return prior_full

    C, W, D = thetas.shape
    # run_trace.begin_mh_tick(t_abs, dt)
    lps = np.asarray(lps, dtype=np.float64)
    ps_state.logpi = np.asarray(ps_state.logpi, dtype=np.float64)

    phi = ps_state.phi
    m = ps_state.m
    rest = ps_state.rest

    if do_rw_eigenline:
        eig_mask = np.array([np.allclose(u @ u.T, np.eye(D)) for u in U])
        if not np.all(eig_mask):
            print("[warn] rw_eigenline disabled: non-orthogonal eigenvectors")
            do_rw_eigenline = False

    for j in range(Kmax):
        move_mask = ps_state.m[:, :, j].astype(bool)
        if not move_mask.any():
            continue
        slot_sel = slot_slices[j]

        ll_cur = _masked_ll(ps_state.phi)  # (C,W)
        lprior_cur_j = _slot_prior_np(ps_state.phi[:, :, j, :])  # (C,W)

        if do_stretch:
            red, blue = redblue_mask_np(rng, C, W)
            red &= move_mask
            blue &= move_mask

            prop1, logJ1, moved1 = propose_stretch_redblue_slot_np(
                rng,
                thetas,
                subset_mask=red,
                slot_sel=slot_sel,
                a=stretch_a,
                partner_pool_mask=~red,
            )
            attempt_mask = red & moved1
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop1[c_idx, w_idx, slot_sel]

            ll_prop = lps.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "stretch",
                pt_state,
                thetas_prop=prop1,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=logJ1,
                move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop1[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )
            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="stretch",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )

            prop2, logJ2, moved2 = propose_stretch_redblue_slot_np(
                rng,
                thetas,
                subset_mask=blue,
                slot_sel=slot_sel,
                a=stretch_a,
                partner_pool_mask=~blue,
            )
            attempt_mask = blue & moved2
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop2[c_idx, w_idx, slot_sel]

            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "stretch",
                pt_state,
                thetas_prop=prop2,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=logJ2,
                move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop2[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )
            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="stretch",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )

        if do_rw_fullcov:
            prop = propose_rw_fullcov_slot_np(rng, thetas, Ls, slot_sel)
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "rw_fullcov",
                pt_state,
                thetas_prop=prop,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=zeros,
                move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )
            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_fullcov",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )

        if do_rw_eigenline:
            prop = propose_rw_eigenline_slot_np(rng, thetas, U, S, slot_sel)
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "rw_eigenline",
                pt_state,
                thetas_prop=prop,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=zeros,
                move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )
            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_eigenline",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )

        if do_rw_student_t:
            prop = propose_rw_student_t_slot_np(rng, thetas, Ls, slot_sel)
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "rw_student_t",
                pt_state,
                thetas_prop=prop,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=zeros,
                move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )

            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_student_t",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )

        if do_de:
            prop, has_pair = propose_de_two_point_slot_np(
                rng,
                thetas,
                eligible_mask=move_mask,
                slot_sel=slot_sel,
                crossover_rate=cross_rate,
                gamma_scale=gamma_de,
            )

            attempt_mask = move_mask & has_pair
            c_idx, w_idx = _indices(attempt_mask)

            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng,
                event_log,
                t_abs,
                dt,
                j,
                "de",
                pt_state,
                thetas_prop=prop,
                ll_cur=ll_cur,
                ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j,
                lprior_prop_j=lprior_prop_j_full,
                betas=betas,
                log_qcorr=zeros,
                move_mask=attempt_mask,
            )

            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(
                accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :]
            )
            ll_cur = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="de",
                    slot_j=j,
                    accepted_mask=accept,
                    pt=pt_state,
                    ps=ps_state,
                )
            # maxdiff, diffs = validate_ll_snapshot(
            #     batched_loglik_masked,
            #     run_trace.mh_ticks[-1].submoves[-1].phi_sel,
            #     run_trace.mh_ticks[-1].submoves[-1].m_sel,
            #     None,
            #     run_trace.mh_ticks[-1].submoves[-1].ll_sel,
            # )
            # if maxdiff > 1e-6:
            #     print(f"[warn] LL mismatch after DE slot {j}: max |Δ|={maxdiff:.3e}")

    if do_PTswap:
        # even pass
        pt_state, acc, att = pt_swap_pass_numpy(rng, pt_state, betas, even_pass=True)
        ps_swap_pass_inplace(ps_state, acc, even_pass=True)

        # odd pass
        pt_state, acc, att = pt_swap_pass_numpy(rng, pt_state, betas, even_pass=False)
        ps_swap_pass_inplace(ps_state, acc, even_pass=False)

    return pt_state, ps_state


def pt_swap_pass_numpy(
    rng: np.random.Generator,
    state: PTState,
    betas: np.ndarray,
    even_pass: bool,
) -> Tuple[PTState, np.ndarray, np.ndarray]:
    thetas, log_probs = state.thetas, state.log_probs
    C, W, D = thetas.shape

    acc = np.zeros((C, W), dtype=bool)
    att = np.zeros((C, W), dtype=bool)

    pair_indices = list(range(0 if even_pass else 1, C - 1, 2))

    for c in pair_indices:
        c_next = c + 1
        delta = (betas[c_next] - betas[c]) * (log_probs[c_next] - log_probs[c])
        log_u = np.log(rng.random(W))
        accept = log_u < delta
        acc[c, :] = accept
        acc[c_next, :] = accept
        att[c, :] = True
        att[c_next, :] = True

        thetas_c = thetas[c].copy()
        thetas[c][accept] = thetas[c_next][accept]
        thetas[c_next][accept] = thetas_c[accept]

        lp_c = log_probs[c].copy()
        log_probs[c][accept] = log_probs[c_next][accept]
        log_probs[c_next][accept] = lp_c[accept]

    return PTState(thetas, log_probs), acc, att


def ps_swap_pass_inplace(ps: "PSState", acc: np.ndarray, even_pass: bool):
    C, W = ps.m.shape[:2]
    pair_indices = list(range(0 if even_pass else 1, C - 1, 2))
    for c in pair_indices:
        c_next = c + 1
        mask = acc[c]
        if not mask.any():
            continue
        ps.m[c][mask], ps.m[c_next][mask] = ps.m[c_next][mask].copy(), ps.m[c][mask].copy()
        ps.phi[c][mask], ps.phi[c_next][mask] = (
            ps.phi[c_next][mask].copy(),
            ps.phi[c][mask].copy(),
        )
        if ps.rest is not None:
            ps.rest[c][mask], ps.rest[c_next][mask] = (
                ps.rest[c_next][mask].copy(),
                ps.rest[c][mask].copy(),
            )
        ps.logpi[c][mask], ps.logpi[c_next][mask] = (
            ps.logpi[c_next][mask].copy(),
            ps.logpi[c][mask].copy(),
        )


def validate_ll_snapshot(batched_ll_masked, phi_sel, m_sel, rest_sel, ll_sel):
    Csel = phi_sel.shape[0]
    diffs = np.zeros((Csel, ll_sel.shape[1]))
    for ci in range(Csel):
        phi_c = phi_sel[ci]
        m_c = m_sel[ci]
        rest_c = None if rest_sel is None else rest_sel[ci]
        ll_re = batched_ll_masked(phi_c, m_c, rest_c)
        diffs[ci] = np.abs(ll_re - ll_sel[ci])
    return float(diffs.max()), diffs


__all__ = [name for name in globals() if not name.startswith("_")]
