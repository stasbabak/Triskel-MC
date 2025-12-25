"""Shared dataclasses and constants for Triskel-MC states and traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Mapping of move names to compact integer identifiers used in traces.
MOVE_IDS = {
    "stretch": 0,
    "rw_fullcov": 1,
    "rw_eigenline": 2,
    "rw_student_t": 3,
    "de": 4,
    "ptswap": 5,
}


@dataclass
class PTState:
    """Parallel tempering state container."""

    thetas: np.ndarray  # (C, W, D)
    log_probs: np.ndarray  # (C, W)


@dataclass
class PSState:
    """Product space state container."""

    phi: np.ndarray  # (C, W, Kmax, d)
    m: np.ndarray  # (C, W, Kmax) bool
    rest: Optional[np.ndarray]  # (C, W, Drest) or None
    logpi: np.ndarray  # (C, W)


@dataclass
class BDEvent:
    """Birth–death process event metadata."""

    t_abs: float
    dt: float
    kind: int
    c: int
    w: int
    slot: int
    k_before: int
    k_after: int


@dataclass
class MHEvent:
    """Metropolis–Hastings move record."""

    t_abs: float
    dt: float
    c: int
    w: int
    slot: int  # active slot index for Gibbs move; -1 for PT swap
    move_type: str
    accepted: bool


@dataclass
class EventLog:
    """Container for BD/MH event streams."""

    bd_events: List[BDEvent]
    mh_events: List[MHEvent]


@dataclass
class TraceConfig:
    """Configuration for optional tracing."""

    # Which temperature indices to snapshot (None => all C)
    chain_inds: Optional[List[int]] = None


@dataclass
class SubmoveSnapshot:
    """Snapshot captured after a submove."""

    move_id: int  # as in MOVE_IDS
    slot_j: int  # -1 for PT swap
    accepted: np.ndarray  # (C,W) bool
    # selected chains (Csel) snapshots AFTER the submove
    thetas_sel: np.ndarray  # (Csel, W, D)
    ll_sel: np.ndarray  # (Csel, W)   masked log-lik
    phi_sel: np.ndarray  # (Csel, W, Kmax, d)
    m_sel: np.ndarray  # (Csel, W, Kmax) bool
    logpi_sel: np.ndarray  # (Csel, W)


@dataclass
class MHTick:
    """Group of submoves executed within a single MH tick."""

    t_abs: float
    dt: float
    submoves: List[SubmoveSnapshot] = field(default_factory=list)


@dataclass
class BDEventWithState:
    """BD event along with optional state snapshot."""

    t_abs: float
    dt: float
    kind: int  # 0=birth, 1=death
    c: int
    w: int
    slot: int
    k_before: int
    k_after: int
    # optional selected-chain PS snapshot AFTER the BD event
    phi_sel: Optional[np.ndarray] = None  # (Csel, W, Kmax, d)
    m_sel: Optional[np.ndarray] = None  # (Csel, W, Kmax)
    logpi_sel: Optional[np.ndarray] = None  # (Csel, W)


@dataclass
class RunTrace:
    """Top-level trace for a continuous-time run."""

    cfg: TraceConfig
    betas: np.ndarray  # (C,)
    chain_inds: np.ndarray  # (Csel,)
    C: int
    W: int
    D: int
    Kmax: int
    d: int

    # event streams
    bd_events: List[BDEventWithState] = field(default_factory=list)
    mh_ticks: List[MHTick] = field(default_factory=list)

    # ---- helpers ----
    @staticmethod
    def init(
        cfg: TraceConfig, betas: np.ndarray, W: int, D: int, Kmax: int, d: int
    ) -> "RunTrace":
        C = int(betas.shape[0])
        if cfg.chain_inds is None:
            chain_inds = np.arange(C, dtype=np.int32)
        else:
            chain_inds = np.array(cfg.chain_inds, dtype=np.int32)
        return RunTrace(
            cfg=cfg,
            betas=np.asarray(betas),
            chain_inds=chain_inds,
            C=C,
            W=W,
            D=D,
            Kmax=Kmax,
            d=d,
        )

    def _sel(self, arr: np.ndarray) -> np.ndarray:
        """Select configured temperature indices."""

        return arr[self.chain_inds]

    # record a BD event + (optionally) PS snapshot of selected chains
    def add_bd_event(self, ev, ps: "PSState", with_snapshot: bool = True):
        rec = BDEventWithState(
            t_abs=ev.t_abs,
            dt=ev.dt,
            kind=ev.kind,
            c=ev.c,
            w=ev.w,
            slot=ev.slot,
            k_before=ev.k_before,
            k_after=ev.k_after,
        )
        if with_snapshot:
            rec.phi_sel = self._sel(ps.phi).copy()
            rec.m_sel = self._sel(ps.m).copy()
            rec.logpi_sel = self._sel(ps.logpi).copy()
        self.bd_events.append(rec)

    # start a new MH tick (call once per tick)
    def begin_mh_tick(self, t_abs: float, dt: float):
        self.mh_ticks.append(MHTick(t_abs=t_abs, dt=dt))

    # add a submove snapshot to the current (most recent) MH tick
    def add_submove_snapshot(
        self,
        move_type: str,
        slot_j: int,
        accepted_mask: np.ndarray,
        pt: "PTState",
        ps: "PSState",
    ):
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


__all__ = [name for name in globals() if not name.startswith("_")]
