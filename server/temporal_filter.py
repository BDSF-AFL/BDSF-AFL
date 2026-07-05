import numpy as np
from collections import deque
from typing import Optional, Tuple


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
        self.N_burn: int = config["burn_in_count"]
        self.fixed_K: bool = config.get("fixed_K", False)
        self.use_tukey: bool = config.get("use_tukey", True)

        self.gap_history: list[float] = []
        self._total_seen: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self, g_i: float) -> str:
        """Primary public method.  Called by AggregatorServer on every
        incoming update.

        Returns one of: ``"PASS"``, ``"REJECT_HIGH_FREQ"``,
        ``"REJECT_STRAGGLER"``.
        """
        self._total_seen += 1

        # During burn-in, accept unconditionally and record the gap.
        if self.is_burn_in():
            self.gap_history.append(g_i)
            return "PASS"

        K_t = self._compute_Kt()
        L, U = self._compute_tukey_fences(K_t)

        # Not enough data to compute fences — accept and record.
        if L is None:
            self.gap_history.append(g_i)
            return "PASS"

        if g_i < L:
            return "REJECT_HIGH_FREQ"

        if g_i > U:
            return "REJECT_STRAGGLER"

        # Within [L, U] — accept and record the gap.
        self.gap_history.append(g_i)
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
