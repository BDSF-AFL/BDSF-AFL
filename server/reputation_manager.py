from typing import Tuple, Optional


class ReputationManager:
    """Maintains per-client Integrity Score I_i (spatial trust) and Pace Score P_i
    (temporal trust) with asymmetric slash rates and recovery rates.

    The asymmetry (beta_I < beta_P, alpha_I >= alpha_P) is a core research
    contribution of BDSF-AFL: integrity violations are punished harder and
    recover slower than pace violations.
    """

    def __init__(self, client_ids: list[int], config: dict) -> None:
        # Slash rates (multiplicative)
        self.alpha_I: float = config.get("alpha_I", 0.30)  # integrity slash (moderated)
        self.alpha_P: float = config.get("alpha_P", 0.20)  # pace slash

        # Recovery rates (additive)
        self.beta_I: float = config.get("beta_I", 0.05)    # integrity recovery (accelerated)
        self.beta_P: float = config.get("beta_P", 0.08)    # pace recovery (fast)

        # Core invariants — asymmetry constraint
        assert self.beta_I < self.beta_P, (
            f"Asymmetry violated: beta_I ({self.beta_I}) must be < beta_P ({self.beta_P})"
        )
        assert self.alpha_I >= self.alpha_P, (
            f"Asymmetry violated: alpha_I ({self.alpha_I}) must be >= alpha_P ({self.alpha_P})"
        )

        # Per-client scores — both start at 1.0 (full trust)
        self.scores: dict[int, dict] = {
            cid: {"I": 1.0, "P": 1.0} for cid in client_ids
        }

        # History for Fig 7 reputation trajectory plots
        self._history: dict[int, list[tuple]] = {cid: [] for cid in client_ids}
        self._round: int = 0

        # Spatial Grace Counter parameters (default: 2)
        self.spatial_grace_k: int = config.get("spatial_grace_k", 2)
        # Borderline Suspicion Counter parameters (default: margin=0.10, limit=5)
        self.borderline_margin: float = config.get("borderline_margin", 0.10)
        self.borderline_limit: int = config.get("borderline_limit", 5)
        self.theta_cos: float = config.get("theta_cos", 0.1)

        # Initialize streaks
        self.spatial_reject_streak: dict[int, int] = {cid: 0 for cid in client_ids}
        self.borderline_streak: dict[int, int] = {cid: 0 for cid in client_ids}
        self.min_integrity_floor: float = float(config.get("min_integrity_floor", 0.10))

    # ------------------------------------------------------------------
    # Slash methods
    # ------------------------------------------------------------------

    def slash_integrity(self, cid: int) -> None:
        """Multiplicative integrity slash with bounded recovery floor.
        Called on: spatial cosine failure, HIGH_FREQ temporal rejection."""
        self.scores[cid]["I"] *= (1.0 - self.alpha_I)
        self.scores[cid]["I"] = max(self.min_integrity_floor, self.scores[cid]["I"])

    def reduce_pace(self, cid: int) -> None:
        """Multiplicative pace slash with bounded recovery floor.
        Called on: STRAGGLER temporal rejection. Does NOT touch I_i."""
        self.scores[cid]["P"] *= (1.0 - self.alpha_P)
        self.scores[cid]["P"] = max(self.min_integrity_floor, self.scores[cid]["P"])

    def record_spatial_rejection(self, cid: int) -> None:
        """Increments spatial reject streak. Applies integrity slash only if the streak
        reaches or exceeds spatial_grace_k.
        """
        self.spatial_reject_streak[cid] += 1
        if self.spatial_reject_streak[cid] >= self.spatial_grace_k:
            self.slash_integrity(cid)

    def record_accepted_update(self, cid: int) -> None:
        """Resets the spatial reject streak to 0 upon a successfully accepted update.
        """
        self.spatial_reject_streak[cid] = 0

    def record_borderline_check(self, cid: int, sim: Optional[float]) -> None:
        """Tracks borderline streak for diagnostic telemetry.
        No integrity penalty is applied for accepted borderline updates.
        """
        if sim is not None and self.theta_cos <= sim <= (self.theta_cos + self.borderline_margin):
            self.borderline_streak[cid] += 1
        else:
            self.borderline_streak[cid] = 0

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover(self, cid: int) -> None:
        """Additive recovery on both scores after a successful accepted update.
        Scores are capped at 1.0."""
        s = self.scores[cid]
        s["I"] = min(1.0, s["I"] + self.beta_I)
        s["P"] = min(1.0, s["P"] + self.beta_P)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, cid: int) -> Tuple[float, float]:
        """Returns (I_i, P_i) for the given client."""
        return self.scores[cid]["I"], self.scores[cid]["P"]

    def effective_weight(self, cid: int) -> float:
        """Returns I_i * P_i. Used by SpatialValidator for Top-K sorting."""
        I, P = self.get(cid)
        return I * P

    # ------------------------------------------------------------------
    # Logging / history
    # ------------------------------------------------------------------

    def log_round(self, round_number: int) -> None:
        """Called once per accepted round by AggregatorServer (after recovery).
        Stores a snapshot of all scores for Fig 7 trajectory plots."""
        self._round = round_number
        for cid, s in self.scores.items():
            self._history[cid].append((round_number, s["I"], s["P"]))

    def get_trajectory(self, cid: int) -> list[tuple]:
        """Returns [(round, I, P), ...] history for a single client.
        Called by fig7_reputation_traj.py."""
        return self._history[cid]

    def get_sorted_by_reputation(self) -> list[tuple[int, float]]:
        """Returns all clients sorted by I_i * P_i descending,
        as [(cid, score), ...]."""
        scored = [
            (cid, self.scores[cid]["I"] * self.scores[cid]["P"])
            for cid in self.scores
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)
