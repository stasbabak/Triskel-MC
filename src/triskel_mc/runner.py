from __future__ import annotations

from typing import Callable, Optional, Tuple, Literal

import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm


from .birth_death import (
    _log_symmetrization,
    _log_uniform_masks_given_k,
    compute_bd_hazards_all,
    make_batched_loglik_masked,
)
from .mh_moves import gibbs_mh_sweep_active_np
from .states import EventLog, PTState, PSState, RunTrace, TraceConfig, BDEvent, MHEvent



class _NullTqdm:
    def __init__(self, total, desc="", unit=""):
        self.total = total
        self.desc = desc
        self.unit = unit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, _):
        pass


# def tqdm(total, desc="", unit=""):
#     return _NullTqdm(total, desc, unit)

# --- DEBUG knobs ---------------------------------------------------------  # ### ADDED
_DO_BD_DB_CHECK = True  # ### ADDED
_DB_TOL = 1e-2  # |lhs - rhs| tolerance for detailed balance    # ### ADDED
_EPS = 1e-300  # log-safe epsilon for hazards                  # ### ADDED
_LL_HARD_MIN = -1e12  # flag very bad LL                               # ### ADDED

DO_PSEUDO_REFRESH = True  # set True to enable


def recompute_logpi(ps: PSState, pt_ll: np.ndarray, betas: np.ndarray,
                    log_prior_phi_np, log_pseudo_phi_np, log_p_mask_np) -> np.ndarray:
    phi, m = ps.phi, ps.m
    slot_dim  = ps.slot_dim
    slot_type = ps.slot_type
    must_be_on = ps.must_be_on
    can_change = ps.can_change
    can_toggle = ps.can_toggle

    C, W, Kmax, dmax = phi.shape ### assume that each slot is padded to maximim dimension
    comp = np.zeros((C, W), dtype=np.float64)

    for j in range(Kmax):
        dj = int(slot_dim[j])
        tj = int(slot_type[j])
        phi_j = phi[:, :, j, :dj]
        logp   = np.vectorize(lambda v: log_prior_phi_np(v, tj), signature="(d)->()")(phi_j)
        logpsi = np.vectorize(lambda v: log_pseudo_phi_np(v, tj), signature="(d)->()")(phi_j)
        comp += np.where(m[:, :, j], logp, logpsi)

    # logpm = log_p_mask_np(m, slot_type, must_be_on, can_change)  # (C,W)
    logpm = log_p_mask_np(m, slot_type, must_be_on, can_toggle)  # (C,W)
    return (logpm + comp + betas[:, None] * pt_ll).astype(np.float64)

def run_ct_mcmc(
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
    betas: np.ndarray,  # (C,) 1/T_c
    qb_density_np: Callable[
        [np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
    ],
    qb_eval_variant: Literal["child", "parent"],

    log_prior_phi_np: Callable[[np.ndarray, int], float],          # (phi_j, type_j) -> float
    log_pseudo_phi_np: Callable[[np.ndarray, int], float],         # (phi_j, type_j) -> float

    # >>>  generic mask prior
    log_p_mask_np: Callable[
        [np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        np.ndarray
    ],  # (m, slot_type, must_be_on, can_toggle) -> (C,W)


    log_lik_masked: Callable[
    [np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray],
    float]
    # (phi_one, m_one, rest_one, slot_dim, slot_type) -> scalar float

    # θ slot mapping (length Kmax): each element is a slice or a bool mask over D
    slot_slices: Tuple,
    ### scale for BD rate
    bd_rate_scale: float = 1.0,  # <<< NEW
    # RW resources for MH
    Ls: Optional[np.ndarray] = None,  # (C,D,D)
    U: Optional[np.ndarray] = None,  # (C,D,D)
    S: Optional[np.ndarray] = None,  # (C,D)
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
    # batched_ll_masked = make_batched_loglik_masked_jax(log_lik_masked_jax)
    batched_ll_masked = make_batched_loglik_masked_np(log_lik_masked_np)

    DO_BD = float(bd_rate_scale) > 0.0

    C, W, D = pt_init.thetas.shape
    Kmax = ps_init.m.shape[-1]

    # clone inputs
    pt = PTState(
        np.array(pt_init.thetas, copy=True), np.array(pt_init.log_probs, copy=True)
    )
    ps = PSState(
        phi=np.array(ps_init.phi, copy=True),
        m=np.array(ps_init.m, copy=True),
        rest=None if ps_init.rest is None else np.array(ps_init.rest, copy=True),
        logpi=np.array(ps_init.logpi, copy=True),

        # >>> NEW: preserve slot metadata
        slot_dim=np.array(ps_init.slot_dim, copy=True),
        slot_type=np.array(ps_init.slot_type, copy=True),
        must_be_on=None if ps_init.must_be_on is None else np.array(ps_init.must_be_on, copy=True),
        can_change=None if ps_init.can_change is None else np.array(ps_init.can_change, copy=True),
        can_toggle=None if ps_init.can_toggle is None else np.array(ps_init.can_toggle, copy=True),
    )

    def _ps_cw_view(ps: PSState, c: int, w: int) -> PSState:
        return PSState(
            phi=ps.phi[c:c+1, w:w+1],
            m=ps.m[c:c+1, w:w+1],
            rest=None if ps.rest is None else ps.rest[c:c+1, w:w+1],
            logpi=ps.logpi[c:c+1, w:w+1],
            slot_dim=ps.slot_dim,
            slot_type=ps.slot_type,
            must_be_on=ps.must_be_on,
            can_toggle= ps.can_toggle,
            can_change=ps.can_change,
        )

    # Ensure pt.log_probs stores the *masked log-likelihood* (not full log posterior)
    # Recompute if pt_init.log_probs is missing/stale.
    if (pt.log_probs is None) or (pt.log_probs.shape != (ps.m.shape[0], ps.m.shape[1])):
        C, W, Kmax, D = ps.phi.shape
        B = C * W
        pt.log_probs = (
            batched_ll_masked(
                ps.phi.reshape(B, Kmax, D),
                ps.m.reshape(B, Kmax),
                None if ps.rest is None else ps.rest.reshape(B, -1),
                ps.slot_dim,
                ps.slot_type,
            )
            .reshape(C, W)
            .astype(np.float64)
        )

    events = EventLog(bd_events=[], mh_events=[])
    tr = None  ## saving all events
    if trace_cfg is not None:
        C, W, D = pt.thetas.shape
        Kmax, d = ps.phi.shape[-2], ps.phi.shape[-1]
        tr = RunTrace.init(trace_cfg, betas=betas, W=W, D=D, Kmax=Kmax, d=d)

    # initial BD clocks: sample from Exp(Λ(c,w))
    # compute hazards once to seed clocks
    if DO_PSEUDO_REFRESH:
        idxs = np.argwhere(~ps.m)  # (n_inactive, 3) over (c,w,slot)
        for c_i, w_i, j_i in idxs:
            dj = int(ps.slot_dim[j_i])
            draw = sample_pseudo_phi()          # assumed length >= dj (or exactly dj)
            ps.phi[c_i, w_i, j_i, :dj] = draw[:dj]
            ps.phi[c_i, w_i, j_i, dj:] = 0.0
        ps.logpi[...] = recompute_logpi(ps, pt.log_probs, betas,
                                    log_prior_phi_np, log_pseudo_phi_np, log_p_mask_np)


    T_bd = np.full((C, W), np.inf, dtype=np.float64)
    if DO_BD:
        _, _, lam_total, _ = compute_bd_hazards_all(
            ps,
            betas,
            qb_density_np=qb_density_np,
            qb_eval_variant=qb_eval_variant,
            log_prior_phi_np=log_prior_phi_np,
            log_pseudo_phi_np=log_pseudo_phi_np,
            log_p_mask_np=log_p_mask_np,
            batched_loglik_masked=batched_ll_masked,
            bd_rate_scale=bd_rate_scale,
        )
        with np.errstate(divide="ignore"):
            mask_pos = lam_total > 0.0
            T_bd[mask_pos] = rng.exponential(1.0 / lam_total[mask_pos])
    # absolute next times
    t = 0.0
    T_bd = T_bd + t

    # N_ticks_est = int(rho_mh * T_end)
    # with tqdm(total=N_ticks_est, desc="MH ticks") as pbar:

    # main loop
    with tqdm(total=float(T_end), desc="CT-MCMC run", unit="time", disable=False) as pbar:
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
                    w_min = int(idx_min % W)
                    t_bd = T_bd[c_min, w_min]
                    if not np.isfinite(t_bd) or t_bd >= t_next:
                        break  # no more BD events before next MH tick

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

                    log_lam_on_cw, log_lam_off_cw, _, _ = (
                        compute_bd_hazards_all(  # <<< CHANGED
                            _ps_cw_view(ps, c_min, w_min),
                            betas[c_min : c_min + 1],
                            qb_density_np=qb_density_np,
                            qb_eval_variant=qb_eval_variant,
                            log_prior_phi_np=log_prior_phi_np,
                            log_pseudo_phi_np=log_pseudo_phi_np,
                            log_p_mask_np=log_p_mask_np,
                            batched_loglik_masked=batched_ll_masked,
                            bd_rate_scale=bd_rate_scale,
                        )
                    )
                    log_lam_on_cw = log_lam_on_cw[
                        0, 0
                    ]  # (Kmax,)                           # <<< CHANGED
                    log_lam_off_cw = log_lam_off_cw[
                        0, 0
                    ]  # (Kmax,)                           # <<< CHANGED

                    # lam_on_cw = lam_on_cw[0, 0]     # (Kmax,)
                    # lam_off_cw = lam_off_cw[0, 0]   # (Kmax,)

                    # ===== DEBUG BD CHECK (pre-event) =====
                    # Recompute the *tempered* logπ for current (c_min,w_min) state
                    m_cw_now = ps.m[c_min, w_min].astype(bool)  # (Kmax,)   <-- ADDED
                    phi_cw_now = ps.phi[c_min, w_min].copy()  # (Kmax,d)  <-- ADDED

                    def tempered_logpi_single(phi_cw: np.ndarray, m_cw: np.ndarray, beta_c: float) -> tuple[float, float, float]:
                        """
                        phi_cw: (Kmax, dmax) padded
                        m_cw:   (Kmax,) bool
                        returns: (logpi_tempered, ll, logpi_unscaled)
                        """
                        # masked LL (no rest in current PSState design)
                        ll_here = float(
                            batched_ll_masked(
                                phi_cw[None, ...],
                                m_cw[None, ...],
                                None if ps.rest is None else ps.rest[c_min, w_min][None, ...],  # <<< CHANGE
                                ps.slot_dim,
                                ps.slot_type,
                            )[0]
                        )

                        # component terms: sum_j log p(phi_j|type_j) if active else log pseudo
                        comp = 0.0
                        Kmax = phi_cw.shape[0]
                        for j in range(Kmax):
                            dj = int(ps.slot_dim[j])
                            tj = int(ps.slot_type[j])
                            vj = phi_cw[j, :dj]
                            comp += float(log_prior_phi_np(vj, tj) if m_cw[j] else log_pseudo_phi_np(vj, tj))

                        # mask prior
                        logpm = float(
                            log_p_mask_np(
                                m_cw[None, None, :],   # (1,1,Kmax)
                                ps.slot_type,          # (Kmax,)
                                ps.must_be_on,         # (Kmax,) or None
                                ps.can_toggle,         # (Kmax,) or None
                            ).reshape(())
                        )

                        logpi_unscaled = logpm + comp
                        logpi_tempered = logpi_unscaled + beta_c * ll_here
                        return float(logpi_tempered), float(ll_here), float(logpi_unscaled)


                    beta_c = float(betas[c_min])
                    phi_cw_now = ps.phi[c_min, w_min].copy()  # (Kmax,d)
                    m_cw_now = ps.m[c_min, w_min].astype(bool)  # (Kmax,)
                    rest_cw_now = (
                        None if ps.rest is None else ps.rest[c_min, w_min].copy()
                    )

                    logpi_cur_tempered, ll_cur, logpi_cur_unscaled = tempered_logpi_single(phi_cw_now, m_cw_now, beta_c)

                    bad_on = np.any(
                        np.isfinite(log_lam_on_cw[m_cw_now]), axis=None
                    )  # should be False
                    bad_off = np.any(
                        np.isfinite(log_lam_off_cw[~m_cw_now]), axis=None
                    )  # should be False
                    if bad_on:
                        print("[BD DEBUG] λ_on active>0 ...")
                    if bad_off:
                        print("[BD DEBUG] λ_off inactive>0 ...")

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
                    if (
                        np.isnan(log_lam_on_cw).any() or np.isnan(log_lam_off_cw).any()
                    ):  # <<< CHANGED
                        print(f"[BD DEBUG] NaN in log-hazard at c={c_min}, w={w_min}")

                    # Build one log-vector [on..., off...]
                    log_hazard_vec = np.concatenate(
                        [log_lam_on_cw, log_lam_off_cw]
                    )  # (2K,)  # <<< CHANGED
                    if not np.isfinite(np.max(log_hazard_vec)):  # <<< CHANGED
                        # all -inf → no BD event possible
                        T_bd[c_min, w_min] = np.inf
                        continue

                    # ---- Gumbel–max selection in log-space ----                                   # <<< NEW
                    g = -np.log(
                        -np.log(
                            np.clip(
                                rng.random(log_hazard_vec.shape[0]), 1e-16, 1 - 1e-16
                            )
                        )
                    )
                    idx = int(np.argmax(log_hazard_vec + g))  # <<< NEW
                    chosen_is_on = idx < Kmax  # <<< CHANGED
                    slot = idx if chosen_is_on else (idx - Kmax)  # <<< CHANGED

                    log_lam_used = float(log_hazard_vec[idx])  # <<< NEW

                    # if (ps.can_change is not None) and (not bool(ps.can_change[slot])):
                    #     # slot cannot toggle at all
                    #     T_bd[c_min, w_min] = np.inf
                    #     continue

                    if (not chosen_is_on) and (ps.must_be_on is not None) and bool(ps.must_be_on[slot]):
                        raise RuntimeError(
                            f"Impossible BD event: attempted death of must_be_on slot {slot} "
                            f"at chain={c_min}, walker={w_min}"
                        )

                    # apply BD toggle
                    phi_new = ps.phi.copy()
                    m_new = ps.m.copy()
                    k_before = int(m_new[c_min, w_min].astype(np.int32).sum())
                    if chosen_is_on:  # birth
                        m_new[c_min, w_min, slot] = True
                    else:  # death
                        m_new[c_min, w_min, slot] = False
                        # phi_new[c_min, w_min, slot, :] = sample_pseudo_phi()

                    # recompute current LL and logπ at (c_min,w_min)
                    # Get ll_cur for this modified state
                    phi_cw = phi_new[c_min, w_min][None, ...]  # (1,Kmax,d)
                    m_cw = m_new[c_min, w_min][None, ...]  # (1,Kmax)
                    rest_cw = (
                        None if ps.rest is None else ps.rest[c_min, w_min][None, ...]
                    )

                    ll_new = batched_ll_masked(phi_cw, m_cw, rest_cw, ps.slot_dim, ps.slot_type,)[0]

                    # component & combinatorial terms
                    comp = 0.0
                    for j in range(Kmax):
                        dj = int(ps.slot_dim[j])
                        tj = int(ps.slot_type[j])
                        vj = phi_new[c_min, w_min, j, :dj]
                        comp += float(log_prior_phi_np(vj, tj) if m_new[c_min, w_min, j] else log_pseudo_phi_np(vj, tj))

                    # mask prior (generic)
                    logpm = float(
                        log_p_mask_np(
                            m_new[c_min, w_min][None, None, :],   # (1,1,Kmax)  <<< FIX
                            ps.slot_type,
                            ps.must_be_on,
                            ps.can_toggle,
                        ).reshape(())
                    )

                    logpi_cw = logpm + comp + float(betas[c_min] * ll_new)

                    # rebuild ps (KEEP metadata!)
                    ps = PSState(
                        phi=phi_new,
                        m=m_new,
                        rest=ps.rest,
                        logpi=ps.logpi.copy(),
                        slot_dim=ps.slot_dim,
                        slot_type=ps.slot_type,
                        must_be_on=ps.must_be_on,
                        can_change=ps.can_change,
                        can_toggle=ps.can_toggle,
                    )
                    ps.logpi[c_min, w_min] = logpi_cw

                    # Keep PT and PS consistent right after BD
                    pt.log_probs[c_min, w_min] = ll_new

                    # Also sync PT θ for the affected slot so proposals start from PS state
                    sl = slot_slices[slot]  # slice/mask in θ-space for this φ_slot
                    dj = int(ps.slot_dim[slot])
                    pt.thetas[c_min, w_min, sl] = phi_new[c_min, w_min, slot, :dj]

                    # ------------------- BD DEBUG (post-event) --------------------  # ### ADDED
                    if _DO_BD_DB_CHECK:  # ### ADDED
                        # recompute single-chain hazards at NEW state
                        log_lam_on_new, log_lam_off_new, _, _ = (
                            compute_bd_hazards_all(  # <<< CHANGED
                                _ps_cw_view(ps, c_min, w_min),
                                betas[c_min : c_min + 1],
                                qb_density_np=qb_density_np,
                                qb_eval_variant=qb_eval_variant,
                                log_prior_phi_np=log_prior_phi_np,
                                log_pseudo_phi_np=log_pseudo_phi_np,
                                log_p_mask_np=log_p_mask_np,
                                batched_loglik_masked=batched_ll_masked,
                                bd_rate_scale=bd_rate_scale,
                            )
                        )
                        log_lam_on_new = log_lam_on_new[0, 0]  # <<< CHANGED
                        log_lam_off_new = log_lam_off_new[0, 0]  # <<< CHANGED

                        log_lam_rev = float(
                            log_lam_off_new[slot]
                            if chosen_is_on
                            else log_lam_on_new[slot]
                        )
                        if log_lam_rev < -800:
                            print(
                                f"[BD REV-FLOOR] c={c_min} w={w_min} slot={slot} logλ_rev={log_lam_rev:.1f} "
                                f"Δlogπ≈{(logpi_cw - logpi_cur_tempered):.1f}"
                            )

                        lhs = logpi_cur_tempered + log_lam_used  # <<< CHANGED
                        rhs = logpi_cw + log_lam_rev  # <<< CHANGED
                        db_resid = lhs - rhs

                        if (not np.isfinite(db_resid)) or (
                            abs(db_resid) > _DB_TOL
                        ):  # ### ADDED
                            print(  # ### ADDED
                                f"[BD DB-RESID] c={c_min} w={w_min} "  # ### ADDED
                                f"{'birth' if chosen_is_on else 'death'} slot={slot} "  # ### ADDED
                                f"resid={db_resid:.3e}  ll_cur={ll_cur:.3e} ll_new={ll_new:.3e}"  # ### ADDED
                            )  # ### ADDED

                        # LL anomaly check                                            # ### ADDED
                        if (not np.isfinite(ll_new)) or (
                            ll_new < _LL_HARD_MIN
                        ):  # ### ADDED
                            print(  # ### ADDED
                                f"[BD LL-ANOM] c={c_min} w={w_min} "  # ### ADDED
                                f"{'birth' if chosen_is_on else 'death'} slot={slot} "  # ### ADDED
                                f"k {k_before}->{k_cur}  ll_new={ll_new:.3e}"  # ### ADDED
                            )  # ### ADDED

                            # DEBUG this is the birth slot we just turned on; check if it was pseudo right before
                        was_pseudo = not m_cw_now[
                            slot
                        ]  # True by definition for a birth
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
                        c=c_min,
                        w=w_min,
                        slot=int(slot),
                        k_before=k_before,
                        k_after=int(m_new[c_min, w_min].astype(np.int32).sum()),
                    )
                    events.bd_events.append(ev)
                    if tr is not None:
                        tr.add_bd_event(ev, ps, with_snapshot=True)

                    # advance "now" to this BD time for subsequent events
                    t = t_bd
                    if DO_PSEUDO_REFRESH and (not chosen_is_on):  # we just did a death

                        dj = int(ps.slot_dim[slot])
                        tj = int(ps.slot_type[slot])

                        old_v = ps.phi[c_min, w_min, slot, :dj].copy()
                        draw = sample_pseudo_phi()
                        new_v = np.asarray(draw[:dj], dtype=ps.phi.dtype)

                        ps.phi[c_min, w_min, slot, :dj] = new_v
                        ps.phi[c_min, w_min, slot, dj:] = 0.0
                        ps.logpi[c_min, w_min] = float(
                            recompute_logpi(
                                _ps_cw_view(ps, c_min, w_min),
                                pt.log_probs[c_min:c_min+1, w_min:w_min+1],
                                betas[c_min:c_min+1],
                                log_prior_phi_np,
                                log_pseudo_phi_np,
                                log_p_mask_np,
                            )[0, 0]
                        )

                    # redraw this chain's next absolute BD time from its new Λ(c,w)
                    _, _, lam_total_single, _ = compute_bd_hazards_all(
                        _ps_cw_view(ps, c_min, w_min),
                        betas[c_min : c_min + 1],
                        qb_density_np=qb_density_np,
                        qb_eval_variant=qb_eval_variant,
                        log_prior_phi_np=log_prior_phi_np,
                        log_pseudo_phi_np=log_pseudo_phi_np,
                        log_p_mask_np=log_p_mask_np,
                        batched_loglik_masked=batched_ll_masked,
                        bd_rate_scale=bd_rate_scale,
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
            pt, ps = gibbs_mh_sweep_active_np(
                rng,
                t_abs=t_next,
                dt=(t_next - t),
                pt_state=pt,
                ps_state=ps,
                slot_slices=slot_slices,
                betas=betas,
                batched_loglik_masked=batched_ll_masked,
                log_prior_phi_np=log_prior_phi_np,
                Ls=Ls,
                U=U,
                S=S,
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
            # keep cached logpi consistent for traces / swaps
            # ps.logpi[...] = pt.log_probs
            ps.logpi[...] = recompute_logpi(ps, pt.log_probs, betas, log_prior_phi_np, log_pseudo_phi_np, log_p_mask_np)

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
                        dj = int(ps.slot_dim[j_i])
                        draw = sample_pseudo_phi()
                        ps.phi[c_i, w_i, j_i, :dj] = np.asarray(draw[:dj], dtype=ps.phi.dtype)
                        ps.phi[c_i, w_i, j_i, dj:] = 0.0
                    ps.logpi[...] = recompute_logpi(ps, pt.log_probs, betas, log_prior_phi_np, log_pseudo_phi_np, log_p_mask_np)

                _, _, lam_total, _ = compute_bd_hazards_all(
                    ps,
                    betas,
                    qb_density_np=qb_density_np,
                    qb_eval_variant=qb_eval_variant,
                    log_prior_phi_np=log_prior_phi_np,
                    log_pseudo_phi_np=log_pseudo_phi_np,
                    log_p_mask_np=log_p_mask_np,
                    batched_loglik_masked=batched_ll_masked,
                    bd_rate_scale=bd_rate_scale,
                )
                T_bd[:] = np.inf
                with np.errstate(divide="ignore"):
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


# Backwards compatibility
run_epoch_ct_numpy = run_ct_mcmc
