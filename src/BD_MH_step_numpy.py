# src/ptdamh/BD_MH_step_numpy.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Literal, List

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

# ==== import your JAX proposal kernels exactly as in your project ====
from .proposals import (
    redblue_mask,
    _propose_stretch_redblue,
    _propose_fullcov,
    _propose_eigenline,
    _propose_student_t,
    _propose_de_two_point,
)

# =============================================================================
#                                State containers
# =============================================================================

MOVE_IDS = {"stretch": 0, "rw_fullcov": 1, "rw_eigenline": 2,
            "rw_student_t": 3, "de": 4, "ptswap": 5}

@dataclass
class PTState: # for parallel tempering state
    # C- temperature W - walker D- parameter (dimension)
    thetas: np.ndarray      # (C, W, D)
    log_probs: np.ndarray   # (C, W)

@dataclass
class PSState: # product space state,
    # Kmax - max number of components (sources)
    phi: np.ndarray         # (C, W, Kmax, d)
    m: np.ndarray           # (C, W, Kmax) bool
    rest: Optional[np.ndarray]   # (C, W, Drest) or None
    logpi: np.ndarray       # (C, W)

# BD event kinds: 0=birth, 1=death
@dataclass
class BDEvent:  ## birth-death Process event storage
    t_abs: float
    dt: float
    kind: int
    c: int
    w: int
    slot: int
    k_before: int
    k_after: int

# MH move types: "stretch","rw_fullcov","rw_eigenline","rw_student_t","de","ptswap"
@dataclass
class MHEvent:  ## Metropolis-Hastings Process event storage
    t_abs: float
    dt: float
    c: int
    w: int
    slot: int                 # active slot index for Gibbs move; -1 for PT swap
    move_type: str
    accepted: bool

@dataclass
class EventLog:
    bd_events: List[BDEvent]
    mh_events: List[MHEvent]


@dataclass
class TraceConfig:
    # Which temperature indices to snapshot (None => all C)
    chain_inds: Optional[List[int]] = None

# ---- Snapshots captured after each submove ----
@dataclass
class SubmoveSnapshot:
    move_id: int                  # as in MOVE_IDS
    slot_j: int                   # -1 for PT swap
    accepted: np.ndarray          # (C,W) bool
    # selected chains (Csel) snapshots AFTER the submove
    thetas_sel: np.ndarray        # (Csel, W, D)
    ll_sel: np.ndarray            # (Csel, W)   masked log-lik
    phi_sel: np.ndarray           # (Csel, W, Kmax, d)
    m_sel: np.ndarray             # (Csel, W, Kmax) bool
    logpi_sel: np.ndarray         # (Csel, W)

# ---- One MH tick groups all its submoves ----
@dataclass
class MHTick:
    t_abs: float
    dt: float
    submoves: List[SubmoveSnapshot] = field(default_factory=list)


# ---- BD event + optional PS snapshot (selected chains) ----
@dataclass
class BDEventWithState:
    t_abs: float
    dt: float
    kind: int         # 0=birth, 1=death
    c: int
    w: int
    slot: int
    k_before: int
    k_after: int
    # optional selected-chain PS snapshot AFTER the BD event
    phi_sel: Optional[np.ndarray] = None   # (Csel, W, Kmax, d)
    m_sel: Optional[np.ndarray]   = None   # (Csel, W, Kmax)
    logpi_sel: Optional[np.ndarray]= None  # (Csel, W)


# ---- Top-level run trace ----
@dataclass
class RunTrace:
    cfg: TraceConfig
    betas: np.ndarray             # (C,)
    chain_inds: np.ndarray        # (Csel,)
    C: int; W: int; D: int; Kmax: int; d: int

    # event streams
    bd_events: List[BDEventWithState] = field(default_factory=list)
    mh_ticks:  List[MHTick]           = field(default_factory=list)

    # ---- helpers ----
    @staticmethod
    def init(cfg: TraceConfig, betas: np.ndarray, W: int, D: int, Kmax: int, d: int) -> "RunTrace":
        C = int(betas.shape[0])
        if cfg.chain_inds is None:
            chain_inds = np.arange(C, dtype=np.int32)
        else:
            chain_inds = np.array(cfg.chain_inds, dtype=np.int32)
        return RunTrace(cfg=cfg, betas=np.asarray(betas), chain_inds=chain_inds,
                        C=C, W=W, D=D, Kmax=Kmax, d=d)

    def _sel(self, arr: np.ndarray) -> np.ndarray:
        """Select configured temperature indices."""
        return arr[self.chain_inds]

    # record a BD event + (optionally) PS snapshot of selected chains
    def add_bd_event(self, ev, ps: "PSState", with_snapshot: bool = True):
        rec = BDEventWithState(
            t_abs=ev.t_abs, dt=ev.dt, kind=ev.kind,
            c=ev.c, w=ev.w, slot=ev.slot,
            k_before=ev.k_before, k_after=ev.k_after,
        )
        if with_snapshot:
            rec.phi_sel   = self._sel(ps.phi).copy()
            rec.m_sel     = self._sel(ps.m).copy()
            rec.logpi_sel = self._sel(ps.logpi).copy()
        self.bd_events.append(rec)

    # start a new MH tick (call once per tick)
    def begin_mh_tick(self, t_abs: float, dt: float):
        self.mh_ticks.append(MHTick(t_abs=t_abs, dt=dt))

    # add a submove snapshot to the current (most recent) MH tick
    def add_submove_snapshot(self, move_type: str, slot_j: int,
                             accepted_mask: np.ndarray,
                             pt: "PTState", ps: "PSState"):
        assert len(self.mh_ticks) > 0, "begin_mh_tick() before adding submoves"
        snap = SubmoveSnapshot(
            move_id=MOVE_IDS[move_type],
            slot_j=int(slot_j),
            accepted=accepted_mask.copy(),
            thetas_sel=self._sel(pt.thetas).copy(),
            ll_sel=self._sel(pt.log_probs).copy(),
            phi_sel=self._sel(ps.phi).copy(),
            m_sel=self._sel(ps.m).copy(),
            logpi_sel=self._sel(ps.logpi).copy(),
        )
        self.mh_ticks[-1].submoves.append(snap)


# =============================================================================
#                      Utils: model combinatoric terms (NumPy)
# =============================================================================

def _log_uniform_masks_given_k(Kmax: int, k: np.ndarray) -> np.ndarray:
    # log p(m | k) = -log binom(Kmax, k)
    k = k.astype(np.int32)
    return - (gammaln(Kmax + 1.) - gammaln(k + 1.) - gammaln(Kmax - k + 1.))

def _log_symmetrization(k: np.ndarray) -> np.ndarray:
    # log (1/k!) = -log(k!)
    return -gammaln(k + 1.0)



# =============================================================================
#                         NumPy proposals (CPU-friendly)
# =============================================================================

def redblue_mask_np(rng: np.random.Generator, C: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    red = np.zeros((C, W), dtype=bool)
    for c in range(C):
        perm = rng.permutation(W)
        red_idx = perm[: W // 2]
        red[c, red_idx] = True
    return red, ~red

# def propose_stretch_redblue_np(
#     rng: np.random.Generator,
#     X: np.ndarray,                 # (C,W,D)
#     subset_mask: np.ndarray,       # (C,W) True: moved in this half
#     a: float = 2.0,
#     partner_pool_mask: Optional[np.ndarray] = None,  # if provided, partners restricted to this pool
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     X' = Y + z * (X - Y), z ~ g(z;a) with E[log z] = 0, classic Goodman-Weare.
#     Only walkers with subset_mask=True attempt a stretch move.
#     Partners chosen from complement (or partner_pool_mask if provided).
#     Returns:
#       X_prop (C,W,D), logJ (C,W), moved_mask (C,W)
#     """
#     C, W, D = X.shape
#     comp = (~subset_mask) if partner_pool_mask is None else partner_pool_mask
#     # choose partner per (c,w) within comp
#     X_prop = X.copy()
#     logJ = np.zeros((C, W), dtype=X.dtype)
#     moved = np.zeros((C, W), dtype=bool)

#     sa = np.sqrt(a)
#     # sample z via u in [0,1]: z = (u*(sqrt(a)-1/sqrt(a)) + 1/sqrt(a))^2
#     for c in range(C):
#         comp_idx = np.where(comp[c])[0]
#         # pre-sample z for all w
#         u = rng.random(W)
#         z = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2
#         for w in range(W):
#             if not subset_mask[c, w]:
#                 continue
#             # partner from comp excluding self; if empty, skip
#             choices = comp_idx[comp_idx != w]
#             if choices.size == 0:
#                 continue
#             y = choices[rng.integers(0, choices.size)]
#             Y = X[c, y]
#             zcw = z[w]
#             X_prop[c, w] = Y + zcw * (X[c, w] - Y)
#             logJ[c, w] = (D - 1.0) * np.log(zcw)
#             moved[c, w] = True

#     return X_prop, logJ, moved


def propose_stretch_redblue_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,                 # (C,W,D)
    subset_mask: np.ndarray,       # (C,W) True: moved in this half
    slot_sel,                      # slice or boolean mask for the slot dims
    a: float = 2.0,
    partner_pool_mask: Optional[np.ndarray] = None,  # if provided, partners restricted to this pool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slot-aware Goodman–Weare stretch in *slot_sel* only:
      x' = y + z * (x - y),  z ~ g(z;a)
    Returns:
      X_prop (C,W,D), logJ (C,W), moved_mask (C,W)
    """
    C, W, D = X.shape
    # resolve slot indices and its dimensionality
    if isinstance(slot_sel, slice):
        idx = np.arange(D)[slot_sel]
    else:
        idx = np.where(np.asarray(slot_sel, bool))[0]
    d_slot = idx.size
    if d_slot == 0:
        return X.copy(), np.zeros((C, W), dtype=X.dtype), np.zeros((C, W), dtype=bool)

    comp = (~subset_mask) if partner_pool_mask is None else partner_pool_mask

    X_prop = X.copy()
    logJ = np.zeros((C, W), dtype=X.dtype)
    moved = np.zeros((C, W), dtype=bool)

    sa = np.sqrt(a)
    # sample z via u in [0,1]: z = (u*(sqrt(a)-1/sqrt(a)) + 1/sqrt(a))^2
    for c in range(C):
        comp_idx = np.where(comp[c])[0]
        if comp_idx.size == 0:
            continue
        u = rng.random(W)
        z = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2
        for w in range(W):
            if not subset_mask[c, w]:
                continue
            # choose partner from available pool (exclude self)
            choices = comp_idx[comp_idx != w]
            if choices.size == 0:
                continue
            y = choices[rng.integers(0, choices.size)]
            # stretch ONLY slot coordinates
            x_cw = X[c, w, idx]
            y_cw = X[c, y, idx]
            zcw = z[w]
            X_prop[c, w, idx] = y_cw + zcw * (x_cw - y_cw)
            # Jacobian power uses d_slot - 1 (Goodman–Weare)
            logJ[c, w] = (d_slot - 1.0) * np.log(zcw)
            moved[c, w] = True

    return X_prop, logJ, moved

# def propose_rw_fullcov_np(rng: np.random.Generator, X: np.ndarray, Ls: np.ndarray) -> np.ndarray:
#     C, W, D = X.shape
#     Z = rng.standard_normal(size=(C, W, D))
#     # eps_cw = L_c @ z_cw  for each (c,w)
#     eps = np.einsum("cij,cwj->cwi", Ls, Z)  # (C,W,D)
#     step = (2.38 / np.sqrt(D) * 0.5) * eps
#     return X + step

def propose_rw_fullcov_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,            # (C,W,D)
    Ls: np.ndarray,           # (C,D,D) Cholesky for full θ
    slot_sel,                 # slice or boolean mask for the slot dims
) -> np.ndarray:
    C, W, D = X.shape
    out = X.copy()
    # resolve slot indices and dimension
    if isinstance(slot_sel, slice):
        idx = np.arange(D)[slot_sel]
    else:
        idx = np.where(np.asarray(slot_sel, bool))[0]
    d_slot = idx.size
    if d_slot == 0:
        return out

    # slot-sub Cholesky
    Ls_slot = Ls[:, idx[:, None], idx[None, :]]      # (C,d_slot,d_slot)

    # standard normal on slot only
    Z = rng.standard_normal(size=(C, W, d_slot))
    # eps_cw = L_c(slot,slot) @ z_cw
    eps = np.einsum("cij,cwj->cwi", Ls_slot, Z)      # (C,W,d_slot)

    step = (2.38 / np.sqrt(d_slot) * 0.5) * eps      # <<< use d_slot
    out[:, :, idx] = X[:, :, idx] + step
    return out

# def propose_rw_eigenline_np(rng: np.random.Generator, X: np.ndarray, U: np.ndarray, S: np.ndarray) -> np.ndarray:
#     C, W, D = X.shape
#     axes = rng.integers(0, D, size=(C, W))
#     U_ax = U[np.arange(C)[:, None], :, axes]     # (C,W,D)
#     S_ax = S[np.arange(C)[:, None], axes]        # (C,W)
#     r = rng.standard_normal(size=(C, W))
#     step = (r * np.sqrt(S_ax))[..., None] * U_ax
#     return X + step

def propose_rw_eigenline_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,            # (C,W,D)
    U: np.ndarray,            # (C,D,D) eigenvectors for full θ
    S: np.ndarray,            # (C,D)   eigenvalues
    slot_sel,
) -> np.ndarray:
    
    C, W, D = X.shape
    out = X.copy()
    if isinstance(slot_sel, slice):
        idx = np.arange(D)[slot_sel]
    else:
        idx = np.where(np.asarray(slot_sel, bool))[0]
    d_slot = idx.size
    if d_slot == 0:
        return out

    # choose a principal axis **within the slot** for each (c,w)
    axes_local = rng.integers(0, d_slot, size=(C, W))
    axes = idx[axes_local]  # map to global D indices (C,W)

    # gather eigenvector column at that axis, but zero outside the slot
    # U_ax = np.zeros((C, W, D), dtype=X.dtype)
    r = rng.standard_normal(size=(C, W))
    for c in range(C):
        U_slot = np.zeros((W, d_slot), dtype=X.dtype)
        # gather per-(w) axis components for this c
        u_cols = U[c][idx, axes[c]]     # shape (d_slot, W)
        U_slot[:, :] = u_cols.T         # to (W, d_slot)
        # U_ax[c, :, idx] = U[c][:, axes[c]]  # copy the chosen eigenvector’s components
        # zero outside slot automatically since we only wrote idx slice

        S_ax = S[c, axes[c]]            # (W,)
        step_slot = (r[c] * np.sqrt(S_ax))[:, None] * U_slot  # (W, d_slot)
        out[c, :, idx] = X[c, :, idx] + step_slot

    return out


# def propose_rw_student_t_np(rng: np.random.Generator, X: np.ndarray, Ls: np.ndarray, nu: float = 5.0) -> np.ndarray:
#     C, W, D = X.shape
#     # z ~ N(0,I), g ~ Gamma(ν/2, 1/2) => t = z / sqrt(g * 2/ν)
#     Z = rng.standard_normal(size=(C, W, D))
#     g = rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=(C, W))  # so that E[g]=1
#     T = Z / np.sqrt(g[..., None])
#     t_tr = np.einsum("cij,cwj->cwi", Ls, T)
#     step = (2.38 / np.sqrt(D) * 0.5) * t_tr
#     return X + step

def propose_rw_student_t_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,            # (C,W,D)
    Ls: np.ndarray,           # (C,D,D)
    slot_sel,
    nu: float = 5.0,
) -> np.ndarray:
    C, W, D = X.shape
    out = X.copy()
    if isinstance(slot_sel, slice):
        idx = np.arange(D)[slot_sel]
    else:
        idx = np.where(np.asarray(slot_sel, bool))[0]
    d_slot = idx.size
    if d_slot == 0:
        return out

    Ls_slot = Ls[:, idx[:, None], idx[None, :]]      # (C,d_slot,d_slot)

    Z = rng.standard_normal(size=(C, W, d_slot))
    g = rng.gamma(shape=nu/2.0, scale=2.0/nu, size=(C, W))
    T = Z / np.sqrt(g[..., None])                    # Student-t on slot
    t_tr = np.einsum("cij,cwj->cwi", Ls_slot, T)
    step = (2.38 / np.sqrt(d_slot) * 0.5) * t_tr     # <<< use d_slot
    out[:, :, idx] = X[:, :, idx] + step
    return out

def propose_de_two_point_slot_np(
    rng: np.random.Generator,
    X: np.ndarray,                        # (C,W,D)
    eligible_mask: np.ndarray,            # (C,W) who can be parent/partner (usually move_mask)
    slot_sel,                             # slice or boolean mask for slot dims
    crossover_rate: float = 0.8,
    gamma_scale: float = 2.38,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slot-only DE: x' = x + γ * (y - z) on slot dims only, γ ~ N(0, σ^2),
    σ = gamma_scale / sqrt(2 * d_slot). Returns (X_prop, has_pair).
    """
    C, W, D = X.shape
    out = X.copy()

    # resolve slot indices
    if isinstance(slot_sel, slice):
        idx = np.arange(D)[slot_sel]
    else:
        idx = np.where(np.asarray(slot_sel, bool))[0]
    d_slot = idx.size
    if d_slot == 0:
        return out, np.zeros((C, W), dtype=bool)

    sigma = gamma_scale / np.sqrt(2.0 * d_slot)
    has_pair = np.zeros((C, W), dtype=bool)

    for c in range(C):
        pool = np.where(eligible_mask[c])[0]
        for w in range(W):
            # need two distinct partners (≠ w)
            choices = pool[pool != w]
            if choices.size < 2:
                continue
            y, z = rng.choice(choices, size=2, replace=False)
            diff = X[c, y, idx] - X[c, z, idx]  # only slot dims
            gam = rng.normal(0.0, sigma)

            # crossover mask within the slot
            if crossover_rate >= 1.0:
                mask_slot = np.ones(d_slot, dtype=bool)
            else:
                mask_slot = rng.random(d_slot) < crossover_rate
                if not mask_slot.any():  # ensure at least one dim in slot changes
                    mask_slot[rng.integers(0, d_slot)] = True

            # apply only on slot dims
            step_slot = np.zeros(d_slot, dtype=X.dtype)
            step_slot[mask_slot] = gam * diff[mask_slot]
            out[c, w, idx] = X[c, w, idx] + step_slot
            has_pair[c, w] = True

    return out, has_pair

# =============================================================================
#                                   PT swap (NumPy)
# =============================================================================

def pt_swap_pass_numpy(
    rng: np.random.Generator,
    state: PTState,
    betas: np.ndarray,
    even_pass: bool,
) -> Tuple[PTState, np.ndarray, np.ndarray]:
    thetas, lps = state.thetas, state.log_probs
    C, W, D = thetas.shape
    start = 0 if even_pass else 1
    idx_low  = np.arange(start, C-1, 2, dtype=np.int32)
    idx_high = idx_low + 1

    bi, bj = betas[idx_low], betas[idx_high]
    li = lps[idx_low, :]
    lj = lps[idx_high, :]
    delta = (bi - bj)[:, None] * (lj - li)     # (P,W)
    log_u = np.log(rng.random(delta.shape))
    accept_pairs = log_u < np.minimum(0.0, delta)

    th = thetas.copy()
    lp = lps.copy()
    for p in range(idx_low.shape[0]):
        i = idx_low[p]; j = idx_high[p]
        mask = accept_pairs[p]                  # (W,)
        mask_wd = mask[:, None]
        thi, thj = th[i].copy(), th[j].copy()
        lpi, lpj = lp[i].copy(), lp[j].copy()
        th[i] = np.where(mask_wd, thj, thi)
        th[j] = np.where(mask_wd, thi, thj)
        lp[i] = np.where(mask, lpj, lpi)
        lp[j] = np.where(mask, lpi, lpj)

    att_mask = np.zeros((C, W), dtype=bool)
    for p in range(idx_low.shape[0]):
        i = idx_low[p]; j = idx_high[p]
        att_mask[i, :] = True
        att_mask[j, :] = True

    acc_mask = np.zeros((C, W), dtype=bool)
    for p in range(idx_low.shape[0]):
        i = idx_low[p]; j = idx_high[p]
        acc_mask[i] |= accept_pairs[p]
        acc_mask[j] |= accept_pairs[p]

    return PTState(thetas=th, log_probs=lp), acc_mask, att_mask

### need to swap the whole state
def ps_swap_pass_inplace(
    ps_state: PSState,
    accept_pairs: np.ndarray,   # (C,W) from pt_swap_pass_numpy
    even_pass: bool,
) -> None:
    """Mirror the accepted PT swaps onto PSState (phi, m, rest, logpi)."""
    C, W, Kmax, d = ps_state.phi.shape
    start = 0 if even_pass else 1
    idx_low  = np.arange(start, C-1, 2, dtype=np.int32)
    idx_high = idx_low + 1

    for p in range(idx_low.shape[0]):
        i = idx_low[p]; j = idx_high[p]
        mask_w = accept_pairs[i] | accept_pairs[j]            # (W,) booleans for walkers at this pair
        if not mask_w.any():
            continue
        # swap (φ, m, rest, logπ) at (i,<mask_w>) <-> (j,<mask_w>)
        mw = mask_w
        ps_state.phi[i, mw],  ps_state.phi[j, mw]  = ps_state.phi[j, mw].copy(),  ps_state.phi[i, mw].copy()
        ps_state.m[i, mw],    ps_state.m[j, mw]    = ps_state.m[j, mw].copy(),    ps_state.m[i, mw].copy()
        if ps_state.rest is not None:
            ps_state.rest[i, mw], ps_state.rest[j, mw] = ps_state.rest[j, mw].copy(), ps_state.rest[i, mw].copy()
        ps_state.logpi[i, mw], ps_state.logpi[j, mw] = ps_state.logpi[j, mw].copy(), ps_state.logpi[i, mw].copy()

# =============================================================================
#     JAX masked-likelihood: single-config -> scalar  (YOU MUST PROVIDE)
# =============================================================================
# Signature you should supply:
#   log_lik_masked_jax(phi_kwd: (Kmax,d), m_k: (Kmax,), rest_d: (Drest,) or None) -> ()
#
# We'll vmap+jit over a batch dimension to evaluate the LL for
#   - every (c,w) "current" mask, and
#   - every (c,w,i) with slot i turned off (for all i)
# all at once (or in chunks if needed).


def make_batched_loglik_masked(log_lik_masked_jax: Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]):
    """Return two JAX-compiled batched evaluators:

    1) batched_cur(phi: (B,Kmax,d), m: (B,Kmax), rest: (B,Drest or 0)) -> (B,)
    2) batched_off(phi: (B*Kmax,Kmax,d), m: (B*Kmax,Kmax), rest: (B*Kmax,Drest or 0)) -> (B*Kmax,)
    """
    f = jax.jit(jax.vmap(log_lik_masked_jax, in_axes=(0, 0, 0)))  # batch over first axis

    def batched(phi_b: np.ndarray, m_b: np.ndarray, rest_b: Optional[np.ndarray]) -> np.ndarray:
        return np.array(f(jnp.asarray(phi_b), jnp.asarray(m_b),
                          None if rest_b is None else jnp.asarray(rest_b)))

    return batched



def masked_ll_for_phi_batch(
    phi: np.ndarray,   # (C,W,Kmax,d)
    m: np.ndarray,     # (C,W,Kmax) bool
    rest: np.ndarray | None,
    batched_loglik_masked,   # fn(B,Kmax,d),(B,Kmax),(B,Drest|) -> (B,)
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

# =============================================================================
#     Vectorized hazards for ALL chains via one batched LL call (NumPy+JAX)
# =============================================================================

def compute_bd_hazards_all(
    ps: PSState,
    betas: np.ndarray,                        # (C,) 1/temps
    *,
    qb_density_np: Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    qb_eval_variant: Literal["child","parent"],
    log_prior_phi_np: Callable[[np.ndarray], float],
    log_pseudo_phi_np: Callable[[np.ndarray], float],
    log_p_k_np: Callable[[np.ndarray], np.ndarray],  # vectorized over k (B,)
    batched_loglik_masked: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    bd_rate_scale = 1.0, 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hazards (lam_on, lam_off, lam_total) for ALL (c,w) in one shot.

    Returns:
      lam_on:  (C,W,Kmax)
      lam_off: (C,W,Kmax)
      lam_total: (C,W)
      log_lam_total: (C,W)
    """
    phi, m, rest, logpi_cur = ps.phi, ps.m, ps.rest, ps.logpi
    C, W, Kmax, d = phi.shape
    B = C * W
    beta_cw = betas[:, None]                   # (C,1)
    log_beta = np.log(np.clip(beta_cw, 1e-300, None))               # <<< NEW

    def _log_pos(x):                                                # <<< NEW
        return np.where(x > 0.0, np.log(x), -np.inf)

    log_bd_scale = np.log(max(bd_rate_scale, 1e-300))               # <<< NEW

    # ----- Build current-mask batch to compute current LL -----
    phi_cur = phi.reshape(B, Kmax, d)          # (B,Kmax,d)
    m_cur   = m.reshape(B, Kmax)               # (B,Kmax)
    rest_cur = None if rest is None else rest.reshape(B, -1)

    ll_cur = batched_loglik_masked(phi_cur, m_cur, rest_cur)     # (B,)
    ll_cur = ll_cur.reshape(C, W)

    # ----- Build "turn OFF i" batches -----
    # For each (c,w), we create Kmax masks with exactly index i turned off.
    # Note: this is vectorized across B*Kmax in one call.
    m_off = np.repeat(m_cur[:, None, :], Kmax, axis=1)           # (B,Kmax,Kmax)
    arK = np.arange(Kmax)
    m_off[np.arange(B)[:, None], arK[None, :], arK[None, :]] = False
    phi_off = np.repeat(phi_cur[:, None, :, :], Kmax, axis=1)    # (B,Kmax,Kmax,d)
    rest_off = None if rest_cur is None else np.repeat(rest_cur[:, None, :], Kmax, axis=1)  # (B,Kmax,Drest)

    # collapse (B,Kmax,...) -> (B*Kmax,...)
    phi_off_flat = phi_off.reshape(B * Kmax, Kmax, d)
    m_off_flat   = m_off.reshape(B * Kmax, Kmax)
    rest_off_flat = None if rest_off is None else rest_off.reshape(B * Kmax, -1)

    ll_off_flat = batched_loglik_masked(phi_off_flat, m_off_flat, rest_off_flat)  # (B*Kmax,)
    ll_off = ll_off_flat.reshape(C, W, Kmax)                                      # (C,W,Kmax)

    # --- build "turn ON j" batches (mirror of OFF) ---
    m_on = np.repeat(m_cur[:, None, :], Kmax, axis=1)                   # (B,Kmax,Kmax)
    arK  = np.arange(Kmax)
    m_on[np.arange(B)[:, None], arK[None, :], arK[None, :]] = True
    phi_on  = np.repeat(phi_cur[:, None, :, :], Kmax, axis=1)           # (B,Kmax,Kmax,d)
    rest_on = None if rest_cur is None else np.repeat(rest_cur[:, None, :], Kmax, axis=1)

    phi_on_flat  = phi_on.reshape(B*Kmax, Kmax, d)
    m_on_flat    = m_on.reshape(B*Kmax, Kmax)
    rest_on_flat = None if rest_on is None else rest_on.reshape(B*Kmax, -1)
    ll_on_flat   = batched_loglik_masked(phi_on_flat, m_on_flat, rest_on_flat)  # (B*Kmax,)
    ll_on        = ll_on_flat.reshape(C, W, Kmax)                                # (C,W,Kmax)

    # ----- Static prior/pseudoprior + k-terms for current and off -----
    # component terms for current mask
    comp_cur = np.zeros((C, W), dtype=np.float64)
    for i in range(Kmax):
        act = m[:, :, i]
        # (optional clarity: add signature to avoid accidental broadcasting)
        comp_cur += np.where(
            act,
            np.vectorize(log_prior_phi_np,     signature="(d)->()")(phi[:, :, i, :]),   # <<< CHANGED (clarity)
            np.vectorize(log_pseudo_phi_np,    signature="(d)->()")(phi[:, :, i, :])    # <<< CHANGED (clarity)
        )

    k_cur = m.astype(np.int32).sum(axis=-1)                                        # (C,W)
    comb_cur = log_p_k_np(k_cur) + _log_uniform_masks_given_k(Kmax, k_cur) + _log_symmetrization(k_cur)  # (C,W)

    logpi_unscaled_cur = comb_cur + comp_cur                                       # (C,W)

    # logpi_tempered_cur = logpi_unscaled_cur + beta_cw * ll_cur                     # (C,W)

    # For off masks: only i-th site flips active->inactive; others unchanged.
    # We can construct the *difference* of the unscaled part quickly:
    # Δcomp_i = log ψ(φ_i) - log p(φ_i)  if currently active, else 0 (but off hazard masked later).
    # k -> k-1 => update combinatorial terms (vectorized via k_off = k_cur-1 but masked by act).
    logpsi = np.zeros((C, W, Kmax), dtype=np.float64)
    logp   = np.zeros((C, W, Kmax), dtype=np.float64)
    for i in range(Kmax):
        logpsi[:, :, i] = np.vectorize(log_pseudo_phi_np, signature="(d)->()")(phi[:, :, i, :])  # <<< CHANGED (clarity)
        logp[:, :, i]   = np.vectorize(log_prior_phi_np,  signature="(d)->()")(phi[:, :, i, :])  # <<< CHANGED (clarity)
    act = m  # (C,W,Kmax) bool

    delta_comp_off = (logpsi - logp) * act                                        # (C,W,Kmax)
    k_off = np.clip(k_cur - 1, 0, Kmax)                                           # (C,W)
    comb_off = log_p_k_np(k_off) + _log_uniform_masks_given_k(Kmax, k_off) + _log_symmetrization(k_off)  # (C,W)
    # broadcast to (C,W,Kmax)
    comb_off_b = np.repeat(comb_off[:, :, None], Kmax, axis=2)
    comb_cur_b = np.repeat(comb_cur[:, :, None], Kmax, axis=2)
    delta_comb_off = (comb_off_b - comb_cur_b) * act                               # (C,W,Kmax)


    delta_unscaled_off = delta_comp_off + delta_comb_off
    delta_comp_on  = (logp - logpsi) * (~m)                                      # sign flip vs OFF
    k_on           = np.clip(k_cur + 1, 0, Kmax)
    comb_on        = log_p_k_np(k_on) + _log_uniform_masks_given_k(Kmax, k_on) + _log_symmetrization(k_on)
    delta_comb_on  = (np.repeat(comb_on[:, :, None], Kmax, axis=2)
                    - np.repeat(comb_cur[:, :, None], Kmax, axis=2)) * (~m)
    delta_unscaled_on = delta_comp_on + delta_comb_on

    beta_cw = betas[:, None]                     # (C,1)

    Delta_off = delta_unscaled_off + beta_cw[:, :, None] * (ll_off - ll_cur[:, :, None])
    Delta_on  = delta_unscaled_on  + beta_cw[:, :, None] * (ll_on  - ll_cur[:, :, None])

    # --- Barker/logistic hazards in log-space ---
    if bd_rate_scale > 0.0:
        log_base = (np.log(bd_rate_scale)
                    + np.log(np.maximum(betas[:, None], 1e-300)))      # (C,1)
    else:
        log_base = -np.inf * np.ones((C, 1))


    

      # Δ untempered part when turning i off:
    # delta_unscaled_off = delta_comp_off + delta_comb_off                          # (C,W,Kmax)
    # # log ratio: log π_off - log π_cur = Δunscaled_off + β * (ll_off - ll_cur)
    # log_ratio_off = delta_unscaled_off + beta_cw[:, :, None] * (ll_off - ll_cur[:, :, None])  # (C,W,Kmax)

    # # full logπ_off (tempered) for each i:
    # logpi_off = (logpi_unscaled_cur[:, :, None] + delta_comp_off + delta_comb_off) \
    #             + beta_cw[:, :, None] * (ll_off - ll_cur[:, :, None])             # (C,W,Kmax)

    # ----- Hazards -----
    # On: for j with m=0, λ_on = β * q_b(φ_j | ctx), where ctx is parent or child-evaluated.
    # Off: for i with m=1, λ_off = β * q_b(φ_i | ctx) * exp(logπ_off(i) - logπ_cur)
    # lam_on = np.zeros((C, W, Kmax), dtype=np.float64)
    # lam_off = np.zeros((C, W, Kmax), dtype=np.float64)

     # ---------------------------- log hazards ----------------------------
    log_lam_on  = np.full((C, W, Kmax), -np.inf, dtype=np.float64)  # <<< CHANGED
    log_lam_off = np.full((C, W, Kmax), -np.inf, dtype=np.float64)  # <<< CHANGED

    # evaluate qb on needed contexts
    # for j in range(Kmax):
    #     # build ctx for child/parent
    #     if qb_eval_variant == "child":
    #         ctx = m.copy()
    #         ctx[:, :, j] = True
    #     else:
    #         ctx = m
    #     # density per (c,w)
    #     # qb_density_np takes (phi_j:(C,W,d), m_ctx:(C,W,Kmax), phi_all:(C,W,Kmax,d), rest:(C,W,Drest) or None)
    #     val = bd_rate_scale*qb_density_np(phi[:, :, j, :], ctx, phi, ps.rest)  # (C,W)
    #     lam_on[:, :, j] = np.where(~m[:, :, j], betas[:, None] * np.maximum(0.0, val), 0.0)

    # On-hazards, per slot j in log
    # for j in range(Kmax):
    #     if qb_eval_variant == "child":
    #         ctx = m.copy(); ctx[:, :, j] = True
    #     else:
    #         ctx = m
    #     val = qb_density_np(phi[:, :, j, :], ctx, phi, ps.rest)     # (C,W)
    #     log_q = _log_pos(val) + log_bd_scale                        # <<< NEW
    #     # only where currently inactive
    #     log_lam_on[:, :, j] = np.where(~m[:, :, j], (log_beta + log_q), -np.inf)

    # ---------- ON hazards (flip j: inactive -> active) ----------
    for j in range(Kmax):
        # q_fwd for ON: x (m_j=False) -> x' (m'_j=True)
        if qb_eval_variant == "child":
            # child = destination of forward (evaluate in proposed ON context)
            ctx_fwd = m.copy(); ctx_fwd[:, :, j] = True
        else:  # "parent"
            # parent = origin of forward (evaluate in current context)
            ctx_fwd = m
        q_fwd = qb_density_np(phi[:, :, j, :], ctx_fwd, phi, ps.rest)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        # q_rev for reverse OFF at x' (x' -> x)
        if qb_eval_variant == "child":
            # child = destination of reverse (evaluate in OFF context)
            ctx_rev = m.copy(); ctx_rev[:, :, j] = False
        else:  # "parent"
            # parent = origin of reverse (at x', j is active)
            ctx_rev = m.copy(); ctx_rev[:, :, j] = True
        q_rev = qb_density_np(phi[:, :, j, :], ctx_rev, phi, ps.rest)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        # Barker/logistic with full MH log-ratio
        Delta_on_tilde = Delta_on[:, :, j] + (log_q_rev - log_q_fwd)

        log_lam_on[:, :, j] = np.where(
            ~m[:, :, j],
            log_beta + log_q_fwd + _logsigmoid(Delta_on_tilde),
            -np.inf
        )

    # ---------- OFF hazards (flip i: active -> inactive) ----------
    for i in range(Kmax):
        # q_fwd for OFF: x (m_i=True) -> x' (m'_i=False)
        if qb_eval_variant == "child":
            # child = destination of forward (evaluate in proposed OFF context)
            ctx_fwd = m.copy(); ctx_fwd[:, :, i] = False
        else:  # "parent"
            # parent = origin of forward (evaluate in current context)
            ctx_fwd = m
        q_fwd = qb_density_np(phi[:, :, i, :], ctx_fwd, phi, ps.rest)
        log_q_fwd = _log_pos(q_fwd) + log_bd_scale

        # q_rev for reverse ON at x' (x' -> x)
        if qb_eval_variant == "child":
            # child = destination of reverse (evaluate in ON context)
            ctx_rev = m.copy(); ctx_rev[:, :, i] = True
        else:  # "parent"
            # parent = origin of reverse (at x', i is inactive)
            ctx_rev = m.copy(); ctx_rev[:, :, i] = False
        q_rev = qb_density_np(phi[:, :, i, :], ctx_rev, phi, ps.rest)
        log_q_rev = _log_pos(q_rev) + log_bd_scale

        Delta_off_tilde = Delta_off[:, :, i] + (log_q_rev - log_q_fwd)

        log_lam_off[:, :, i] = np.where(
            m[:, :, i],
            log_beta + log_q_fwd + _logsigmoid(Delta_off_tilde),
            -np.inf
        )

    # for i in range(Kmax):
    #     if qb_eval_variant == "child":
    #         ctx = m
    #     else:
    #         ctx = m.copy()
    #         ctx[:, :, i] = False
    #     val = bd_rate_scale*qb_density_np(phi[:, :, i, :], ctx, phi, ps.rest)  # (C,W)
    #     lam_off[:, :, i] = np.where(
    #         m[:, :, i],
    #         betas[:, None] * np.maximum(0.0, val) * np.exp(log_ratio_off[:, :, i]),
    #         0.0
    #     )

    # slots that are currently active must have ON = -inf
    assert not np.any(np.isfinite(log_lam_on[m])),  "ON hazard finite where active"
    # slots that are currently inactive must have OFF = -inf
    assert not np.any(np.isfinite(log_lam_off[~m])), "OFF hazard finite where inactive"
    # Total rate for waiting-time (linear), but sum via logsumexp
    log_lam_all = np.concatenate([log_lam_on, log_lam_off], axis=2) # (C,W,2K)
    # stable log-sum-exp along last axis
    maxv = np.max(log_lam_all, axis=2)                                  # (C,W)
    with np.errstate(over='ignore', invalid='ignore'):
        sumexp = np.sum(np.exp(log_lam_all - maxv[..., None]), axis=2)  # (C,W)
    # if all entries were -inf, maxv=-inf → set logsumexp to -inf
    log_lam_total = np.where(np.isfinite(maxv), maxv + np.log(sumexp), -np.inf)  # (C,W)

    # linear total rate for clocks (OK for zeros)
    lam_total = np.where(np.isfinite(log_lam_total), np.exp(log_lam_total), 0.0)  # (C,W)

    # lam_total = lam_on.sum(axis=-1) + lam_off.sum(axis=-1)  # (C,W)
    return log_lam_on, log_lam_off, lam_total, log_lam_total


# =============================================================================
#                Restrict proposals to a slot & MH accept/update
# =============================================================================

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
    # <<< CHANGED: now pass split terms, not full log-posteriors >>>
    ll_cur: np.ndarray,         # (C,W) masked log-likelihood (current)
    ll_prop: np.ndarray,        # (C,W) masked log-likelihood (proposed)
    lprior_cur_j: np.ndarray,   # (C,W) prior for UPDATED slot j (current), untempered
    lprior_prop_j: np.ndarray,  # (C,W) prior for UPDATED slot j (proposed), untempered
    betas: np.ndarray,          # (C,)
    log_qcorr: np.ndarray,      # (C,W)
    move_mask: np.ndarray,      # (C,W)
) -> np.ndarray:
    
    delta = (lprior_prop_j - lprior_cur_j) + betas[:, None] * (ll_prop - ll_cur) + log_qcorr
    log_u = np.log(rng.random(ll_cur.shape))
    accept = (log_u < np.minimum(0.0, delta)) & move_mask
    return accept

def apply_mh_and_record_np(
    rng: np.random.Generator,
    event_log: EventLog,
    t_abs: float, dt: float, slot_j: int, move_type: str,
    pt_state: PTState,
    thetas_prop: np.ndarray,     # (C,W,D)

    # <<< CHANGED: pass masked LL and slot-j priors separately >>>
    ll_cur: np.ndarray,          # (C,W)
    ll_prop: np.ndarray,         # (C,W)
    lprior_cur_j: np.ndarray,    # (C,W)
    lprior_prop_j: np.ndarray,   # (C,W)

    betas: np.ndarray,           # (C,)
    log_qcorr: np.ndarray,       # (C,W)
    move_mask: np.ndarray,       # (C,W)
) -> Tuple[PTState, np.ndarray]:  # <<< CHANGED: also return accept mask

    # <<< CHANGED: call the split-version acceptance >>>
    accept = mh_accept_tempered_np(
        rng, ll_cur, ll_prop, lprior_cur_j, lprior_prop_j, betas, log_qcorr, move_mask
    )

    # record every attempt (accepted/rejected)
    C, W = accept.shape
    for c in range(C):
        for w in range(W):
            if move_mask[c, w]:  # log only attempts
                event_log.mh_events.append(
                    MHEvent(t_abs=t_abs, dt=dt, c=c, w=w, slot=slot_j,
                            move_type=move_type, accepted=bool(accept[c, w]))
                )

    # <<< CHANGED: pt_state.log_probs is masked LL; update with ll_prop where accepted >>>
    th_new = np.where(accept[:, :, None], thetas_prop, pt_state.thetas)
    lp_new = np.where(accept, ll_prop, pt_state.log_probs)
    return PTState(thetas=th_new, log_probs=lp_new), accept     # <<< CHANGED


def _indices(mask2d: np.ndarray):
    """Return (c_idx, w_idx) 1D arrays for True entries."""
    return np.where(mask2d)

def _scatter_into(full: np.ndarray, idx: tuple[np.ndarray,np.ndarray], vals: np.ndarray) -> np.ndarray:
    """Write vals (shape = idx length) into full (C,W) at idx; return new array."""
    out = full.copy()
    out[idx] = vals
    return out




# =============================================================================
#                       One Gibbs MH sweep over active slots
# =============================================================================

def gibbs_mh_sweep_active_np(
    rng: np.random.Generator,
    *,
    t_abs: float,
    dt: float,
    pt_state: PTState,                 # (C,W,D), (C,W)
    ps_state: PSState,                 # provides active mask per slot
    slot_slices: Tuple,                # len Kmax; slice or bool mask over D per φ_j
    betas: np.ndarray,                 # (C,)
    # batched likelihood for φ with mask m  
    batched_loglik_masked: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    # component prior/pseudoprior and p(k)   # <<< NEW
    log_prior_phi_np: Callable[[np.ndarray], float],        # <<< NEW

    # RW resources for MH
    Ls: Optional[np.ndarray] = None,   # (C,D,D)
    U: Optional[np.ndarray] = None,    # (C,D,D)
    S: Optional[np.ndarray] = None,    # (C,D)
    # knobs
    do_stretch: bool = True,
    do_rw_fullcov: bool = True,
    do_rw_eigenline: bool = False,
    do_rw_student_t: bool = False,
    do_de: bool = True,
    do_PTswap: bool = True,
    # proposal params
    stretch_a: float = 1.3,
    cross_rate: float = 0.7,
    gamma_de: float = 2.38,
    # logging
    event_log: Optional[EventLog] = None,
    run_trace: Optional["RunTrace"] = None,
) -> PTState:
    
    C, W, D = pt_state.thetas.shape
    thetas = pt_state.thetas
    lps    = pt_state.log_probs
    Kmax = ps_state.m.shape[-1]

    m = ps_state.m.astype(bool)                                  
    rest = ps_state.rest

    # --- helpers --------------------------------------------------------------  # <<< NEW
    def _slot_prior_np(phi_cwd: np.ndarray) -> np.ndarray:
        """phi_cwd: (C,W,d) -> (C,W) via log_prior_phi_np"""
        out = np.empty((C, W), dtype=np.float64)
        for c in range(C):
            for w in range(W):
                out[c, w] = float(log_prior_phi_np(phi_cwd[c, w]))
        return out

    def _masked_ll(phi: np.ndarray) -> np.ndarray:
        """phi: (C,W,Kmax,d) -> (C,W) masked log-likelihood"""
        B = C * W
        ll = batched_loglik_masked(
            phi.reshape(B, Kmax, -1),
            m.reshape(B, Kmax),
            None if rest is None else rest.reshape(B, -1)
        ).reshape(C, W)
        return np.asarray(ll, dtype=np.float64)

    def _masked_ll_subset(phi_full: np.ndarray,
                      m_full: np.ndarray,
                      rest_full: Optional[np.ndarray],
                      idx: tuple[np.ndarray,np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate masked LL only on subset 'idx', return (vals, full_with_vals).
        """
        if idx[0].size == 0:
            return np.empty((0,), dtype=np.float64), None
        C, W, Kmax, d = phi_full.shape
        # pack subset into a batch
        phi_sub  = phi_full[idx][:, :, :]         # (B_sub, Kmax, d)
        m_sub    = m_full[idx][:, :]              # (B_sub, Kmax)
        rest_sub = None if rest_full is None else rest_full[idx][:, :]
        ll_sub = batched_loglik_masked(phi_sub, m_sub, rest_sub)  # (B_sub,)
        return ll_sub, _scatter_into(np.zeros((C, W), dtype=np.float64), idx, ll_sub)

    def _slot_prior_subset(phi_slot_full: np.ndarray, idx: tuple[np.ndarray,np.ndarray]) -> np.ndarray:
        """Compute prior for φ_j only on subset; returns 1D (B_sub,)"""
        if idx[0].size == 0:
            return np.empty((0,), dtype=np.float64)
        vals = np.empty(idx[0].shape[0], dtype=np.float64)
        for n, (c, w) in enumerate(zip(idx[0], idx[1])):
            vals[n] = float(log_prior_phi_np(phi_slot_full[c, w]))
        return vals
    
    def validate_ll_snapshot(batched_ll_masked, phi_sel, m_sel, rest_sel, ll_sel):
        """
        Compare stored ll_sel (Csel,W) to a recomputation from phi_sel/m_sel/rest_sel.
        Returns (max_abs_diff, per-temp per-walker diffs).
        """
        Csel, W, Kmax, d = phi_sel.shape
        diffs = np.zeros((Csel, W), dtype=float)
        for ci in range(Csel):
            phi_c  = phi_sel[ci]          # (W,Kmax,d)
            m_c    = m_sel[ci]            # (W,Kmax)
            rest_c = None if rest_sel is None else rest_sel[ci]  # (W,Drest) or None
            ll_re  = batched_ll_masked(phi_c, m_c, rest_c)       # (W,)
            diffs[ci] = np.abs(ll_re - ll_sel[ci])
        return float(diffs.max()), diffs

    # --- end helpers ----------------------------------------------------------


    for j in range(Kmax):  ### loop over all sources
        move_mask = ps_state.m[:, :, j].astype(bool)
        if not move_mask.any():
            continue
        slot_sel = slot_slices[j]

        # ---------- Common CURRENT pieces for this slot j ----------
        # ll_cur       = lps.copy() ### TODO not sure I I can use it.... is it true loglikelihood for the current state or log_pi?
        ll_cur        = _masked_ll(ps_state.phi)             # (C,W)
        lprior_cur_j  = _slot_prior_np(ps_state.phi[:, :, j, :])# (C,W)

        # --- Stretch (two half-steps via red/blue) ---
        if do_stretch:
            red, blue = redblue_mask_np(rng, C, W)
            red  &= move_mask
            blue &= move_mask

            # ---------- A half: red moves using blue pool ----------
            prop1, logJ1, moved1 = propose_stretch_redblue_slot_np(
                rng, thetas, subset_mask=red, slot_sel=slot_sel, a=stretch_a, partner_pool_mask=~red
            )
            attempt_mask = red & moved1
            c_idx, w_idx = _indices(attempt_mask)

            # Build φ_prop only for attempted
            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop1[c_idx, w_idx, slot_sel]

            # masked-LL only for attempted subset; scatter back
            ll_prop = lps.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            # slot prior only for attempted subset; scatter back
            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            pt_state, accept = apply_mh_and_record_np(
                rng, event_log, t_abs, dt, j, "stretch",
                PTState(thetas, lps),
                thetas_prop=prop1,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=logJ1, move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop1[:, :, slot_sel], ps_state.phi[:, :, j, :])
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)

            # ---------- B half: blue moves using UPDATED red pool ----------
            prop2, logJ2, moved2 = propose_stretch_redblue_slot_np(
                rng, thetas, subset_mask=blue, slot_sel=slot_sel, a=stretch_a, partner_pool_mask=~blue
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
                rng, event_log, t_abs, dt, j, "stretch",
                PTState(thetas, lps),
                thetas_prop=prop2,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=logJ2, move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop2[:, :, slot_sel], ps_state.phi[:, :, j, :])
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="stretch", slot_j=j, accepted_mask=accept,
                    pt=pt_state, ps=ps_state
                )
            # --- LL consistency self-check (debug) ---
            maxdiff, diffs = validate_ll_snapshot(
                batched_loglik_masked,
                run_trace.mh_ticks[-1].submoves[-1].phi_sel,
                run_trace.mh_ticks[-1].submoves[-1].m_sel,
                None,
                run_trace.mh_ticks[-1].submoves[-1].ll_sel,
            )
            if maxdiff > 1e-6:  # choose your tolerance
                print(f"[warn] LL mismatch after STRETCH slot {j}: max |Δ|={maxdiff:.3e}")

        # --- RW fullcov ---
        if do_rw_fullcov and (Ls is not None):
            # propose only slot dims (no need to restrict later)
            prop = propose_rw_fullcov_slot_np(rng, thetas, Ls, slot_sel)

            # attempt only where slot j is active
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)
            # build φ_prop only for attempted chains
            phi_prop = ps_state.phi.copy()
            phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            # compute masked-LL only on attempted subset
            ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
            # scatter into a full (C,W) array for the accept function
            ll_prop = ll_cur.copy()          # start from current masked LL
            if c_idx.size:
                ll_prop[c_idx, w_idx] = ll_prop_sub

            # slot prior only on attempted subset
            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng, event_log, t_abs, dt, j, "rw_fullcov",
                PTState(thetas, lps),
                thetas_prop=prop,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=zeros, move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :])

            # carry-forward only where accepted
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_fullcov", slot_j=j, accepted_mask=accept,
                    pt=pt_state, ps=ps_state
                )
            # --- LL consistency self-check (debug) ---
            maxdiff, diffs = validate_ll_snapshot(
                batched_loglik_masked,
                run_trace.mh_ticks[-1].submoves[-1].phi_sel,
                run_trace.mh_ticks[-1].submoves[-1].m_sel,
                None,
                run_trace.mh_ticks[-1].submoves[-1].ll_sel,
            )
            if maxdiff > 1e-6:  # choose your tolerance
                print(f"[warn] LL mismatch after RW slot {j}: max |Δ|={maxdiff:.3e}")

        # --- RW eigenline ---
        if do_rw_eigenline and (U is not None) and (S is not None):
            # propose only slot dims (no need to restrict later)
            prop = propose_rw_eigenline_slot_np(rng, thetas, U, S, slot_sel)

            # attempt only where slot j is active
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)
            # build φ_prop only for attempted chains
            phi_prop = ps_state.phi.copy()
            phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            # compute masked-LL only on attempted subset
            ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
            # scatter into a full (C,W) array for the accept function
            ll_prop = ll_cur.copy()          # start from current masked LL
            if c_idx.size:
                ll_prop[c_idx, w_idx] = ll_prop_sub

            # slot prior only on attempted subset
            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

 
            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng, event_log, t_abs, dt, j, "rw_eigenline",
                PTState(thetas, lps),
                thetas_prop=prop,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=zeros, move_mask=attempt_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :])

            # carry-forward                                                      # <<< NEW
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_eigenline", slot_j=j, accepted_mask=accept,
                    pt=pt_state, ps=ps_state
                )

        # --- RW student-t ---
        if do_rw_student_t and (Ls is not None):
           # propose only slot dims (no need to restrict later)
            prop =  propose_rw_student_t_slot_np(rng, thetas, Ls, slot_sel)

            # attempt only where slot j is active
            attempt_mask = move_mask
            c_idx, w_idx = _indices(attempt_mask)
            # build φ_prop only for attempted chains
            phi_prop = ps_state.phi.copy()
            phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            # compute masked-LL only on attempted subset
            ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
            # scatter into a full (C,W) array for the accept function
            ll_prop = ll_cur.copy()          # start from current masked LL
            if c_idx.size:
                ll_prop[c_idx, w_idx] = ll_prop_sub

            # slot prior only on attempted subset
            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub


            zeros = np.zeros((C, W), dtype=np.float64)
            pt_state, accept = apply_mh_and_record_np(
                rng, event_log, t_abs, dt, j, "rw_student_t",
                PTState(thetas, lps),
                thetas_prop=prop,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=zeros, move_mask=move_mask,
            )
            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :])

            # carry-forward                                                      # <<< NEW
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="rw_student_t", slot_j=j, accepted_mask=accept,
                    pt=pt_state, ps=ps_state
                )

        # --- DE two-point ---
        if do_de:
            prop, has_pair = propose_de_two_point_slot_np(
                rng, thetas, eligible_mask=move_mask, slot_sel=slot_sel,
                crossover_rate=cross_rate, gamma_scale=gamma_de
            )

            attempt_mask = move_mask & has_pair
            c_idx, w_idx = _indices(attempt_mask)

            # build φ_prop only for attempted
            phi_prop = ps_state.phi.copy()
            if c_idx.size:
                phi_prop[c_idx, w_idx, j, :] = prop[c_idx, w_idx, slot_sel]

            # masked-LL only on attempted subset; scatter back
            ll_prop = ll_cur.copy()
            if c_idx.size:
                ll_prop_sub, _ = _masked_ll_subset(phi_prop, m, rest, (c_idx, w_idx))
                ll_prop[c_idx, w_idx] = ll_prop_sub

            # slot prior only on attempted subset; scatter back
            lprior_prop_j_full = lprior_cur_j.copy()
            if c_idx.size:
                pj_sub = _slot_prior_subset(phi_prop[:, :, j, :], (c_idx, w_idx))
                lprior_prop_j_full[c_idx, w_idx] = pj_sub

            zeros = np.zeros((C, W), dtype=np.float64)  # symmetric -> log_qcorr = 0
            pt_state, accept = apply_mh_and_record_np(
                rng, event_log, t_abs, dt, j, "de",
                PTState(thetas, lps),
                thetas_prop=prop,
                ll_cur=ll_cur, ll_prop=ll_prop,
                lprior_cur_j=lprior_cur_j, lprior_prop_j=lprior_prop_j_full,
                betas=betas, log_qcorr=zeros, move_mask=attempt_mask,
            )

            thetas, lps = pt_state.thetas, pt_state.log_probs
            ps_state.phi[:, :, j, :] = np.where(accept[:, :, None], prop[:, :, slot_sel], ps_state.phi[:, :, j, :])
            ll_cur       = np.where(accept, ll_prop, ll_cur)
            lprior_cur_j = np.where(accept, lprior_prop_j_full, lprior_cur_j)
            if run_trace is not None:
                run_trace.add_submove_snapshot(
                    move_type="de", slot_j=j, accepted_mask=accept,
                    pt=pt_state, ps=ps_state
                )
            # --- LL consistency self-check (debug) ---
            maxdiff, diffs = validate_ll_snapshot(
                batched_loglik_masked,
                run_trace.mh_ticks[-1].submoves[-1].phi_sel,
                run_trace.mh_ticks[-1].submoves[-1].m_sel,
                None,
                run_trace.mh_ticks[-1].submoves[-1].ll_sel,
            )
            if maxdiff > 1e-6:  # choose your tolerance
                print(f"[warn] LL mismatch after DE slot {j}: max |Δ|={maxdiff:.3e}")

    if do_PTswap:
        # PT swaps (even + odd), record attempts
        pt_state = PTState(thetas=thetas, log_probs=lps)
        # even pass
        pt_state, acc_e, att_e = pt_swap_pass_numpy(rng, pt_state, betas, even_pass=True)
        ps_swap_pass_inplace(ps_state, acc_e, even_pass=True)
        # odd pass  
        pt_state, acc_o, att_o = pt_swap_pass_numpy(rng, pt_state, betas, even_pass=False)
        ps_swap_pass_inplace(ps_state, acc_o, even_pass=False)

        C, W = pt_state.thetas.shape[:2]
        acc_pt = acc_e | acc_o
        att_pt = att_e | att_o
        for c in range(C):
            for w in range(W):
                if att_pt[c, w]:
                    event_log.mh_events.append(
                        MHEvent(t_abs=t_abs, dt=dt, c=c, w=w, slot=-1, move_type="ptswap", accepted=bool(acc_pt[c, w]))
                    )
        if run_trace is not None:
            run_trace.add_submove_snapshot(
                move_type="ptswap", slot_j=-1, accepted_mask=acc_pt,
                pt=pt_state, ps=ps_state
            )
    return pt_state


# =============================================================================
#                              Main CTMC + MH runner
# =============================================================================

from tqdm import tqdm

# --- DEBUG knobs ---------------------------------------------------------  # ### ADDED
_DO_BD_DB_CHECK = True                                                      # ### ADDED
_DB_TOL = 1e-2              # |lhs - rhs| tolerance for detailed balance    # ### ADDED
_EPS = 1e-300               # log-safe epsilon for hazards                  # ### ADDED
_LL_HARD_MIN = -1e12        # flag very bad LL                               # ### ADDED

DO_PSEUDO_REFRESH = True  # set True to enable

def run_epoch_ct_numpy(
    *,
    # RNG seed
    seed: int,
    # initial states
    pt_init: PTState,
    ps_init: PSState,
    # time horizon
    T_end: float,
    # MH Poisson rate
    rho_mh: float,
    # product-space / likelihood bits
    betas: np.ndarray,                     # (C,) 1/T_c
    qb_density_np: Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    qb_eval_variant: Literal["child","parent"],
    log_prior_phi_np: Callable[[np.ndarray], float],
    log_pseudo_phi_np: Callable[[np.ndarray], float],
    log_p_k_np: Callable[[np.ndarray], np.ndarray],   # vectorized over k (C,W) -> (C,W)
    # masked-likelihood (JAX, single-config) -> scalar
    log_lik_masked_jax: Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray],
    # θ slot mapping (length Kmax): each element is a slice or a bool mask over D
    slot_slices: Tuple,
    ### scale for BD rate
    bd_rate_scale: float = 1.0,    # <<< NEW
    # RW resources for MH
    Ls: Optional[np.ndarray] = None,       # (C,D,D)
    U: Optional[np.ndarray]  = None,       # (C,D,D)
    S: Optional[np.ndarray]  = None,       # (C,D)
    # MH knobs
    do_stretch: bool = True,
    do_rw_fullcov: bool = True,
    do_rw_eigenline: bool = False,
    do_rw_student_t: bool = False,
    do_de: bool = True,
    do_PTswap: bool = True,
    stretch_a: float = 1.3,
    cross_rate: float = 0.7,
    gamma_de: float = 2.38,
    # pseudo-prior sampler
    sample_pseudo_phi: Callable[[], np.ndarray],
    trace_cfg: Optional["TraceConfig"] = None,
) -> Tuple[PTState, PSState, EventLog, RunTrace]:
    """
    Hybrid CTMC:
     - Each (c,w) has its own BD clock λ_bd(c,w) derived from hazards.
     - Global MH Poisson clock with rate ρ.
     - At each MH tick: Gibbs over active slots (Stretch→RW→DE) and PT swaps.
     - Logs every BD event and every MH submove (accepted or not).
    """

    rng = np.random.default_rng(seed)
    batched_ll_masked = make_batched_loglik_masked(log_lik_masked_jax)
    DO_BD = (float(bd_rate_scale) > 0.0) 

    C, W, D = pt_init.thetas.shape
    Kmax = ps_init.m.shape[-1]

    # clone inputs
    pt = PTState(np.array(pt_init.thetas, copy=True), np.array(pt_init.log_probs, copy=True))
    ps = PSState(np.array(ps_init.phi, copy=True), np.array(ps_init.m, copy=True),
                 None if ps_init.rest is None else np.array(ps_init.rest, copy=True),
                 np.array(ps_init.logpi, copy=True))
    
    # Ensure pt.log_probs stores the *masked log-likelihood* (not full log posterior)
    # Recompute if pt_init.log_probs is missing/stale.
    if (pt.log_probs is None) or (pt.log_probs.shape != (ps.m.shape[0], ps.m.shape[1])):
        C, W, Kmax, D = ps.phi.shape
        B = C * W
        pt.log_probs = batched_ll_masked(
            ps.phi.reshape(B, Kmax, D),
            ps.m.reshape(B, Kmax),
            None if ps.rest is None else ps.rest.reshape(B, -1)
        ).reshape(C, W).astype(np.float64)


    events = EventLog(bd_events=[], mh_events=[])
    tr = None ## saving all events
    if trace_cfg is not None:
        C, W, D = pt.thetas.shape
        Kmax, d = ps.phi.shape[-2], ps.phi.shape[-1]
        tr = RunTrace.init(trace_cfg, betas=betas, W=W, D=D, Kmax=Kmax, d=d)

    # initial BD clocks: sample from Exp(Λ(c,w))
    # compute hazards once to seed clocks
    if DO_PSEUDO_REFRESH:
        idxs = np.argwhere(~ps.m)  # (n_inactive, 3) over (c,w,slot)
        for c_i, w_i, j_i in idxs:
            ps.phi[c_i, w_i, j_i, :] = sample_pseudo_phi()
    T_bd = np.full((C, W), np.inf, dtype=np.float64)
    if DO_BD:
        _, _, lam_total, _ = compute_bd_hazards_all(
            ps, betas,
            qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
            log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
            log_p_k_np=log_p_k_np,
            batched_loglik_masked=batched_ll_masked,  bd_rate_scale=bd_rate_scale
        )
        with np.errstate(divide='ignore'):
            mask_pos = lam_total > 0.0
            T_bd[mask_pos] = rng.exponential(1.0 / lam_total[mask_pos])
    # absolute next times
    t = 0.0
    T_bd = T_bd + t

    # N_ticks_est = int(rho_mh * T_end)
    # with tqdm(total=N_ticks_est, desc="MH ticks") as pbar:

    # main loop
    with tqdm(total=T_end, desc="CT-MCMC run", unit="time") as pbar:   
        while t < T_end:
            t_in = t
            # next MH tick
            tau_mh = rng.exponential(1.0 / rho_mh)
            t_next = min(T_end, t + tau_mh)

            # process BD events up to t_next in chronological order
            # print ('debug: entering BD event processing loop')
            cnt = 0
            if DO_BD:
                while True:
                    # pick the nearest BD event in (absolute) time across all walkers (C, W)
                    idx_min = np.argmin(T_bd)  # flat index
                    c_min = int(idx_min // W)
                    w_min = int(idx_min %  W)
                    t_bd = T_bd[c_min, w_min]
                    if not np.isfinite(t_bd) or t_bd >= t_next:
                        break # no more BD events before next MH tick

                    # determine per-(c_min,w_min) hazards to sample a concrete slot event
                    # (We already have lam_on/off global, but they are stale if ps changed before; recompute for this chain.)
                    
                    # lam_on_cw, lam_off_cw, lam_total_cw = compute_bd_hazards_all(
                    #     PSState(ps.phi[c_min:c_min+1, w_min:w_min+1],
                    #             ps.m[c_min:c_min+1, w_min:w_min+1],
                    #             None if ps.rest is None else ps.rest[c_min:c_min+1, w_min:w_min+1],
                    #             ps.logpi[c_min:c_min+1, w_min:w_min+1]),
                    #     betas[c_min:c_min+1],
                    #     qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
                    #     log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
                    #     log_p_k_np=lambda k: log_p_k_np(k).reshape(1,1),  # adapter
                    #     batched_loglik_masked=batched_ll_masked
                    # )
                    

                    log_lam_on_cw, log_lam_off_cw, _, _ = compute_bd_hazards_all(           # <<< CHANGED
                        PSState(ps.phi[c_min:c_min+1, w_min:w_min+1],
                                ps.m[c_min:c_min+1,   w_min:w_min+1],
                                None if ps.rest is None else ps.rest[c_min:c_min+1, w_min:w_min+1],
                                ps.logpi[c_min:c_min+1, w_min:w_min+1]),
                        betas[c_min:c_min+1],
                        qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
                        log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
                        log_p_k_np=lambda k: log_p_k_np(k).reshape(1,1),
                        batched_loglik_masked=batched_ll_masked, bd_rate_scale=bd_rate_scale
                    )
                    log_lam_on_cw  = log_lam_on_cw[0, 0]    # (Kmax,)                           # <<< CHANGED
                    log_lam_off_cw = log_lam_off_cw[0, 0]   # (Kmax,)                           # <<< CHANGED

                    # lam_on_cw = lam_on_cw[0, 0]     # (Kmax,)
                    # lam_off_cw = lam_off_cw[0, 0]   # (Kmax,)
        


                    # ===== DEBUG BD CHECK (pre-event) =====
                    # Recompute the *tempered* logπ for current (c_min,w_min) state
                    m_cw_now    = ps.m[c_min, w_min].astype(bool)                # (Kmax,)   <-- ADDED
                    phi_cw_now  = ps.phi[c_min, w_min].copy()                    # (Kmax,d)  <-- ADDED
                    def tempered_logpi_single(phi_cw, m_cw, rest_cw, beta_c):
                        # masked LL
                        ll_here = batched_ll_masked(phi_cw[None, ...], m_cw[None, ...], None if rest_cw is None else rest_cw[None, ...])[0]
                        # component + combinatorics
                        comp = 0.0
                        for i in range(phi_cw.shape[0]):  # Kmax
                            comp += (log_prior_phi_np(phi_cw[i]) if m_cw[i] else log_pseudo_phi_np(phi_cw[i]))
                        k_here = int(m_cw.astype(np.int32).sum())
                        comb = float(log_p_k_np(np.array([[k_here]])).reshape(())) \
                            + float(_log_uniform_masks_given_k(phi_cw.shape[0], np.array(k_here))) \
                            + float(_log_symmetrization(np.array(k_here)))
                        logpi_unscaled = comb + comp
                        return float(logpi_unscaled + beta_c * ll_here), float(ll_here), float(logpi_unscaled)

                    beta_c = float(betas[c_min])
                    phi_cw_now  = ps.phi[c_min, w_min].copy()      # (Kmax,d)
                    m_cw_now    = ps.m[c_min, w_min].astype(bool)  # (Kmax,)
                    rest_cw_now = None if ps.rest is None else ps.rest[c_min, w_min].copy()

                    logpi_cur_tempered, ll_cur, logpi_cur_unscaled = tempered_logpi_single(phi_cw_now, m_cw_now, rest_cw_now, beta_c)

                    bad_on  = np.any(np.isfinite(log_lam_on_cw[m_cw_now]), axis=None)   # should be False
                    bad_off = np.any(np.isfinite(log_lam_off_cw[~m_cw_now]), axis=None) # should be False
                    if bad_on:  print("[BD DEBUG] λ_on active>0 ...")
                    if bad_off: print("[BD DEBUG] λ_off inactive>0 ...")

                    
                    # Invariants: hazards must be finite and non-negative on proper supports
                    # if not np.all(np.isfinite(lam_on_cw)) or not np.all(np.isfinite(lam_off_cw)):
                    #     print(f"[BD DEBUG] non-finite hazard at c={c_min}, w={w_min}")
                    # if (lam_on_cw[m_cw_now] > 0).any():   # on-rate must be zero where active
                    #     print(f"[BD DEBUG] λ_on active>0 at c={c_min}, w={w_min}")
                    # if (lam_off_cw[~m_cw_now] > 0).any(): # off-rate must be zero where inactive
                    #     print(f"[BD DEBUG] λ_off inactive>0 at c={c_min}, w={w_min}")
                    ######################################################

                    # hazards = np.concatenate([lam_on_cw, lam_off_cw])
                    # total = hazards.sum()
                    # if total <= 0.0 or not np.isfinite(total):
                    #     # no event; disable this clock
                    #     T_bd[c_min, w_min] = np.inf
                    #     continue

                    # probs = hazards / total
                    # idx = rng.choice(2 * Kmax, p=probs)
                    # chosen_is_on = (idx < Kmax)
                    # slot = idx if chosen_is_on else (idx - Kmax)

                    # lam_used = float(lam_on_cw[slot] if chosen_is_on else lam_off_cw[slot])  # ### ADDED

                    # --- robustness checks (now look for NaNs; -inf is OK) ---
                    if np.isnan(log_lam_on_cw).any() or np.isnan(log_lam_off_cw).any():      # <<< CHANGED
                        print(f"[BD DEBUG] NaN in log-hazard at c={c_min}, w={w_min}")

                    # Build one log-vector [on..., off...]
                    log_hazard_vec = np.concatenate([log_lam_on_cw, log_lam_off_cw])                # (2K,)  # <<< CHANGED
                    if not np.isfinite(np.max(log_hazard_vec)):                                      # <<< CHANGED
                        # all -inf → no BD event possible
                        T_bd[c_min, w_min] = np.inf
                        continue

                    # ---- Gumbel–max selection in log-space ----                                   # <<< NEW
                    g = -np.log(-np.log(np.clip(rng.random(log_hazard_vec.shape[0]), 1e-16, 1-1e-16)))
                    idx = int(np.argmax(log_hazard_vec + g))                                         # <<< NEW
                    chosen_is_on = (idx < Kmax)                                               # <<< CHANGED
                    slot = idx if chosen_is_on else (idx - Kmax)                              # <<< CHANGED

                    log_lam_used = float(log_hazard_vec[idx])                                        # <<< NEW


                    # apply BD toggle
                    phi_new = ps.phi.copy()
                    m_new   = ps.m.copy()
                    k_before = int(m_new[c_min, w_min].astype(np.int32).sum())
                    if chosen_is_on:   # birth
                        m_new[c_min, w_min, slot] = True
                    else:              # death
                        m_new[c_min, w_min, slot] = False
                        # phi_new[c_min, w_min, slot, :] = sample_pseudo_phi()

                    # recompute current LL and logπ at (c_min,w_min)
                    # Get ll_cur for this modified state
                    phi_cw = phi_new[c_min, w_min][None, ...]   # (1,Kmax,d)
                    m_cw   = m_new[c_min, w_min][None, ...]     # (1,Kmax)
                    rest_cw = None if ps.rest is None else ps.rest[c_min, w_min][None, ...]
                    
                    ll_new = batched_ll_masked(phi_cw, m_cw, rest_cw)[0]


                    # component & combinatorial terms
                    comp = 0.0
                    for i in range(Kmax):
                        if m_new[c_min, w_min, i]:
                            comp += float(log_prior_phi_np(phi_new[c_min, w_min, i]))
                        else:
                            comp += float(log_pseudo_phi_np(phi_new[c_min, w_min, i]))
                    k_cur = int(m_new[c_min, w_min].astype(np.int32).sum())
                    comb = float(log_p_k_np(np.array([[k_cur]])).reshape(())) \
                        + float(_log_uniform_masks_given_k(Kmax, np.array(k_cur))) \
                        + float(_log_symmetrization(np.array(k_cur)))
                    logpi_cw = comb + comp + float(betas[c_min] * ll_new)

                    ps = PSState(phi=phi_new, m=m_new, rest=ps.rest, logpi=ps.logpi.copy())
                    ps.logpi[c_min, w_min] = logpi_cw

                    # Keep PT and PS consistent right after BD
                    pt.log_probs[c_min, w_min] = ll_new

                    # Also sync PT θ for the affected slot so proposals start from PS state
                    sl = slot_slices[slot]                   # slice/mask in θ-space for this φ_slot
                    pt.thetas[c_min, w_min, sl] = phi_new[c_min, w_min, slot, :]

                                    # ------------------- BD DEBUG (post-event) --------------------  # ### ADDED
                    if _DO_BD_DB_CHECK:                                             # ### ADDED
                        # recompute single-chain hazards at NEW state
                        log_lam_on_new, log_lam_off_new, _, _ = compute_bd_hazards_all(          # <<< CHANGED
                            PSState(ps.phi[c_min:c_min+1, w_min:w_min+1],
                                    ps.m[c_min:c_min+1,   w_min:w_min+1],
                                    None if ps.rest is None else ps.rest[c_min:c_min+1, w_min:w_min+1],
                                    ps.logpi[c_min:c_min+1, w_min:w_min+1]),
                            betas[c_min:c_min+1],
                            qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
                            log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
                            log_p_k_np=lambda k: log_p_k_np(k).reshape(1,1),
                            batched_loglik_masked=batched_ll_masked, bd_rate_scale=bd_rate_scale
                        )
                        log_lam_on_new  = log_lam_on_new[0, 0]                                    # <<< CHANGED
                        log_lam_off_new = log_lam_off_new[0, 0]                                   # <<< CHANGED

                        log_lam_rev = float(log_lam_off_new[slot] if chosen_is_on else log_lam_on_new[slot])
                        if log_lam_rev < -800:
                            print(f"[BD REV-FLOOR] c={c_min} w={w_min} slot={slot} logλ_rev={log_lam_rev:.1f} "
                                    f"Δlogπ≈{(logpi_cw - logpi_cur_tempered):.1f}")

                        lhs = logpi_cur_tempered + log_lam_used                                   # <<< CHANGED
                        rhs = logpi_cw            + log_lam_rev                                   # <<< CHANGED
                        db_resid = lhs - rhs

                        if (not np.isfinite(db_resid)) or (abs(db_resid) > _DB_TOL):  # ### ADDED
                            print(                                                    # ### ADDED
                                f"[BD DB-RESID] c={c_min} w={w_min} "                 # ### ADDED
                                f"{'birth' if chosen_is_on else 'death'} slot={slot} " # ### ADDED
                                f"resid={db_resid:.3e}  ll_cur={ll_cur:.3e} ll_new={ll_new:.3e}"  # ### ADDED
                            )                                                         # ### ADDED

                        # LL anomaly check                                            # ### ADDED
                        if (not np.isfinite(ll_new)) or (ll_new < _LL_HARD_MIN):      # ### ADDED
                            print(                                                    # ### ADDED
                                f"[BD LL-ANOM] c={c_min} w={w_min} "                  # ### ADDED
                                f"{'birth' if chosen_is_on else 'death'} slot={slot} " # ### ADDED
                                f"k {k_before}->{k_cur}  ll_new={ll_new:.3e}"         # ### ADDED
                            )                                                         # ### ADDED

                                             # DEBUG this is the birth slot we just turned on; check if it was pseudo right before
                        was_pseudo = not m_cw_now[slot]   # True by definition for a birth
                        # if was_pseudo:
                        #     delta_ll = ll_new - ll_cur
                        #     if delta_ll < -200:     # tune threshold to your scale
                        #         print(f"[BD BIRTH→PSEUDO] c={c_min} w={w_min} slot={slot} Δll={delta_ll:.1f}")
                    # ----------------- end BD DEBUG (post-event) -------------------  # ### ADDED

                    # log event
                    ev = BDEvent(
                        t_abs=t_bd,
                        dt=t_bd - t,
                        kind=(0 if chosen_is_on else 1),
                        c=c_min, w=w_min,
                        slot=int(slot),
                        k_before=k_before,
                        k_after=int(m_new[c_min, w_min].astype(np.int32).sum())
                    )
                    events.bd_events.append(ev)
                    if tr is not None:
                        tr.add_bd_event(ev, ps, with_snapshot=True)

                    # advance "now" to this BD time for subsequent events
                    t = t_bd
                    if DO_PSEUDO_REFRESH and (not chosen_is_on):           # we just did a death
                        old_phi = ps.phi[c_min, w_min, slot, :].copy()
                        new_phi = sample_pseudo_phi()
                        ps.phi[c_min, w_min, slot, :] = new_phi
                        # keep cached tempered logπ consistent (optional)
                        ps.logpi[c_min, w_min] += (
                            log_pseudo_phi_np(new_phi) - log_pseudo_phi_np(old_phi)
                        )

                    # redraw this chain's next absolute BD time from its new Λ(c,w)
                    _, _, lam_total_single, _ = compute_bd_hazards_all(
                    PSState(ps.phi[c_min:c_min+1, w_min:w_min+1],
                            ps.m[c_min:c_min+1, w_min:w_min+1],
                            None if ps.rest is None else ps.rest[c_min:c_min+1, w_min:w_min+1],
                            ps.logpi[c_min:c_min+1, w_min:w_min+1]),
                            betas[c_min:c_min+1],
                            qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
                            log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
                            log_p_k_np=lambda k: log_p_k_np(k).reshape(1,1),
                            batched_loglik_masked=batched_ll_masked,  bd_rate_scale=bd_rate_scale
                    )
                    lam_next = lam_total_single[0, 0]
                    if lam_next > 0.0 and np.isfinite(lam_next):
                        T_bd[c_min, w_min] = t + rng.exponential(1.0 / lam_next)
                    else:
                        T_bd[c_min, w_min] = np.inf
                    cnt += 1
            # print ('debug cnt =', cnt)

            if tr is not None:
                tr.begin_mh_tick(t_abs=t_next, dt=(t_next - t))

            # Perform MH sweep at t_next
            pt = gibbs_mh_sweep_active_np(
                rng,
                t_abs=t_next, dt=(t_next - t),
                pt_state=pt, ps_state=ps, slot_slices=slot_slices,
                betas=betas,
                batched_loglik_masked=batched_ll_masked,
                log_prior_phi_np=log_prior_phi_np,
                Ls=Ls, U=U, S=S,
                do_stretch=do_stretch,
                do_rw_fullcov=do_rw_fullcov,
                do_rw_eigenline=do_rw_eigenline,
                do_rw_student_t=do_rw_student_t,
                do_de=do_de,
                do_PTswap=do_PTswap,
                stretch_a=stretch_a,
                cross_rate=cross_rate,
                gamma_de=gamma_de,
                event_log=events,
                run_trace=tr,
            )

            # reset all BD clocks after MH (memoryless, and hazards may change via θ)
            # lam_on, lam_off, lam_total = compute_bd_hazards_all(
            #     ps, betas,
            #     qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
            #     log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
            #     log_p_k_np=log_p_k_np,
            #     batched_loglik_masked=batched_ll_masked
            # )
            if DO_BD:

                if DO_PSEUDO_REFRESH:
                    idxs = np.argwhere(~ps.m)  # (n_inactive, 3) over (c,w,slot)
                    for c_i, w_i, j_i in idxs:
                        ps.phi[c_i, w_i, j_i, :] = sample_pseudo_phi()
                
                _, _, lam_total, _ = compute_bd_hazards_all(
                    ps, betas,
                    qb_density_np=qb_density_np, qb_eval_variant=qb_eval_variant,
                    log_prior_phi_np=log_prior_phi_np, log_pseudo_phi_np=log_pseudo_phi_np,
                    log_p_k_np=log_p_k_np,
                    batched_loglik_masked=batched_ll_masked,  bd_rate_scale=bd_rate_scale
                )
                T_bd[:] = np.inf
                with np.errstate(divide='ignore'):
                    mask_pos = lam_total > 0.0
                    T_bd[mask_pos] = rng.exponential(1.0 / lam_total[mask_pos])
                T_bd += t_next

            # advance to MH time
            t = t_next
            dt_advance = t - t_in
            pbar.update(dt_advance)
            # pbar.n = t                                   # set absolute progress
            # pbar.refresh()

    return pt, ps, events, tr




# =============================================================================
#                               End of file
# =============================================================================

# For validation of log-likelihood of the state:
# def validate_ll_snapshot(batched_ll_masked, phi_sel, m_sel, rest_sel, ll_sel, temp_index):
#     phi_c = phi_sel[temp_index]   # (W,Kmax,d)
#     m_c   = m_sel[temp_index]     # (W,Kmax)
#     rest_c = None if rest_sel is None else rest_sel[temp_index]
#     ll_re = batched_ll_masked(phi_c, m_c, rest_c)      # (W,)
#     ll_st = ll_sel[temp_index]                          # (W,)
#     diff = np.abs(ll_re - ll_st)
#     return diff.max(), diff
