import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List

from shared.types import TemporalEvidence


class TemporalFilter:
    """Computes adaptive Tukey-fence bounds [L, U] over the rolling behavioral gap
    history and classifies each incoming gap g_i as PASS, REJECT_HIGH_FREQ,
    or REJECT_STRAGGLER.  Maintains the rolling gap history internally —
    only PASS gaps are appended.
    """

    def __init__(self, config: dict):
        self.K_base: int = config.get("K_base", 50)
        self.lam: float = config.get("lam", 0.3)
        self.kappa: float = config.get("kappa", 1.5)
        self.N_burn: int = config.get("burn_in_count", config.get("N_burn", 80))
        self.fixed_K: bool = config.get("fixed_K", False)
        self.use_tukey: bool = config.get("use_tukey", True)
        self.temporal_min_samples: int = config.get("temporal_min_samples", min(20, self.N_burn))
        self.warm_start_mode: str = config.get("warm_start_mode", "state_maturity")

        self.gap_history: list[float] = []
        self.client_gap_history: Dict[int, list[float]] = {}
        self._total_seen: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def is_temporal_mature(self) -> bool:
        """Returns True when temporal statistics are mature and ready for active filtering."""
        if len(self.gap_history) < self.temporal_min_samples:
            return False
        gaps = list(self.gap_history)
        q25, q75 = float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))
        iqr = q75 - q25
        return iqr > 1e-6 or (q25 > 0.0 and len(gaps) >= self.temporal_min_samples)

    def is_burn_in(self) -> bool:
        """Returns True while still in temporal warmup."""
        if self.warm_start_mode == "state_maturity":
            return not self.is_temporal_mature()
        return self._total_seen <= self.N_burn

    def extract_evidence(self, g_i: float, client_id: Optional[int] = None) -> TemporalEvidence:
        """Extracts continuous temporal features without altering state."""
        temp_mature = self.is_temporal_mature()
        is_burn = not temp_mature if self.warm_start_mode == "state_maturity" else self._total_seen <= self.N_burn

        if not temp_mature:
            L, U = None, None
            fence_margin = 0.0
        else:
            K_t = self._compute_Kt()
            L, U = self._compute_tukey_fences(K_t)
            # Compute normalized fence margin
            if L is not None and U is not None and (U - L) > 1e-9:
                if g_i < L:
                    fence_margin = (g_i - L) / (U - L)
                elif g_i > U:
                    fence_margin = (g_i - U) / (U - L)
                else:
                    fence_margin = 0.0
            else:
                fence_margin = 0.0

        # Compute client-specific z-score
        if client_id is not None and client_id in self.client_gap_history:
            client_gaps = self.client_gap_history[client_id]
            if len(client_gaps) >= 2:
                mean_g = float(np.mean(client_gaps))
                std_g = float(np.std(client_gaps))
                client_z_score = (g_i - mean_g) / (std_g + 1e-6)
            else:
                client_z_score = 0.0
        else:
            client_z_score = 0.0

        return TemporalEvidence(
            g_i=g_i,
            lower_fence=L,
            upper_fence=U,
            fence_margin=fence_margin,
            client_z_score=client_z_score,
            is_burn_in=is_burn,
            temporal_mature=temp_mature,
        )

    def step_seen(self) -> None:
        """Increments the total submissions counter for burn-in tracking."""
        self._total_seen += 1

    def record_gap(self, g_i: float, client_id: Optional[int] = None) -> None:
        """Records an accepted gap into global and per-client history."""
        self.gap_history.append(g_i)
        if client_id is not None:
            if client_id not in self.client_gap_history:
                self.client_gap_history[client_id] = []
            self.client_gap_history[client_id].append(g_i)

    def evaluate(self, g_i: float, client_id: Optional[int] = None) -> str:
        """Primary public method.  Called by AggregatorServer on every
        incoming update.

        Returns one of: ``"PASS"``, ``"REJECT_HIGH_FREQ"``,
        ``"REJECT_STRAGGLER"``.
        """
        self._total_seen += 1

        # During burn-in, accept unconditionally and record the gap.
        if self.is_burn_in():
            self.record_gap(g_i, client_id)
            return "PASS"

        K_t = self._compute_Kt()
        L, U = self._compute_tukey_fences(K_t)

        # Not enough data to compute fences — accept and record.
        if L is None:
            self.record_gap(g_i, client_id)
            return "PASS"

        if g_i < L:
            return "REJECT_HIGH_FREQ"

        if g_i > U:
            return "REJECT_STRAGGLER"

        # Within [L, U] — accept and record the gap.
        self.record_gap(g_i, client_id)
        return "PASS"

    def is_burn_in(self) -> bool:
        """Returns ``True`` while still in the burn-in phase."""
        return self._total_seen <= self.N_burn

    def get_stats(self) -> dict:
        """Returns a dict with diagnostic information for logging."""
        K_t = self._compute_Kt()
        L, U = self._compute_tukey_fences(K_t)
        return {
            "K_t": K_t,
            "history_len": len(self.gap_history),
            "is_burn_in": self.is_burn_in(),
            "L": L,
            "U": U,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _compute_Kt(self) -> int:
        """Computes adaptive window size K_t."""
        if self.fixed_K:
            return self.K_base

        recent = self.gap_history[-(max(2, self.K_base // 2)):]
        if len(recent) < 2:
            return self.K_base

        sigma = float(np.std(recent))
        mean_gap = float(np.mean(recent))
        if mean_gap <= 0:
            mean_gap = 1e-6

        cv = sigma / mean_gap
        K_t = int(self.K_base * (1 + self.lam * cv))
        return max(K_t, 4)

    def _compute_tukey_fences(self, K_t: int) -> Tuple[Optional[float], Optional[float]]:
        """Computes ``(L, U)`` from the rolling window."""
        window = self.gap_history[-K_t:]
        if len(window) < 4:
            return (None, None)

        Q1 = float(np.percentile(window, 25))
        Q3 = float(np.percentile(window, 75))
        IQR = Q3 - Q1

        # Prevent scale collapse by enforcing a lower bound on the IQR.
        # Since history contains only accepted gaps (conditional distribution),
        # the variance mathematically contracts over time. We bound the IQR to
        # be at least 5% of the median of the current window.
        min_iqr = 0.05 * float(np.median(window))
        IQR = max(IQR, min_iqr)

        if self.use_tukey:
            L = Q1 - self.kappa * IQR
            U = Q3 + self.kappa * IQR
        else:
            # Ablation — raw Q1/Q3
            L = Q1
            U = Q3

        # Gaps cannot be negative
        L = max(0.0, L)
        return (L, U)
