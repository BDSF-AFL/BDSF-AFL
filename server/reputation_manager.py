from typing import Tuple


class ReputationManager:
    """Maintains per-client Integrity Score I_i (spatial trust) and Pace Score P_i
    (temporal trust) with asymmetric slash rates and recovery rates.

    The asymmetry (beta_I < beta_P, alpha_I >= alpha_P) is a core research
    contribution of BDSF-AFL: integrity violations are punished harder and
    recover slower than pace violations.
    """

    def __init__(self, client_ids: list[int], config: dict) -> None:
        # Slash rates (multiplicative)
        self.alpha_I: float = config.get("alpha_I", 0.4)   # integrity slash
        self.alpha_P: float = config.get("alpha_P", 0.2)   # pace slash

        # Recovery rates (additive)
        self.beta_I: float = config.get("beta_I", 0.02)    # integrity recovery (slow)
        self.beta_P: float = config.get("beta_P", 0.05)    # pace recovery (fast)

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

    # ------------------------------------------------------------------
    # Slash methods
    # ------------------------------------------------------------------

    def slash_integrity(self, cid: int) -> None:
        """Multiplicative integrity slash.
        Called on: spatial cosine failure, HIGH_FREQ temporal rejection."""
        self.scores[cid]["I"] *= (1.0 - self.alpha_I)
        self.scores[cid]["I"] = max(0.0, self.scores[cid]["I"])

    def reduce_pace(self, cid: int) -> None:
        """Multiplicative pace slash. Called on: STRAGGLER temporal rejection.
        Does NOT touch I_i."""
        self.scores[cid]["P"] *= (1.0 - self.alpha_P)
        self.scores[cid]["P"] = max(0.0, self.scores[cid]["P"])

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
