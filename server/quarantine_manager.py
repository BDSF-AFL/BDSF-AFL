import uuid
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
import torch

from shared.types import QuarantinedEntry


class QuarantineManager:
    """Bounded, CPU-isolated quarantine state machine for BDSF-AFL.
    
    Temporarily holds ambiguous or borderline candidate updates without
    allowing them to contaminate global model weights, accepted buffers,
    or reference centroids. Automatically re-evaluates held entries against
    refreshed Top-K reference vectors upon subsequent server rounds.
    """

    def __init__(self, config: dict):
        self.capacity: int = config.get("quarantine_capacity", 20)
        self.horizon: int = config.get("quarantine_horizon", 5)
        self.theta_cos: float = config.get("theta_cos", 0.10)
        self.buffer: deque[QuarantinedEntry] = deque(maxlen=self.capacity)

    def enqueue(
        self,
        client_id: int,
        delta_W_clipped: torch.Tensor,
        current_round: int,
        virtual_time: float,
        reputation: Tuple[float, float],
        reason: str
    ) -> QuarantinedEntry:
        """Enqueues a candidate update in quarantine buffer on host CPU."""
        # Ensure detached 1D float32 CPU tensor
        vec_cpu = delta_W_clipped.detach().cpu().flatten().float()
        
        # If client already has a pending entry in quarantine, evict the stale one
        self.evict_for_client(client_id)

        entry = QuarantinedEntry(
            entry_id=str(uuid.uuid4())[:8],
            client_id=client_id,
            delta_W_clipped=vec_cpu,
            entry_round=current_round,
            entry_virtual_time=virtual_time,
            reputation_at_entry=reputation,
            primary_reason=reason,
        )
        self.buffer.append(entry)
        return entry

    def evict_for_client(self, client_id: int) -> Optional[QuarantinedEntry]:
        """Evicts any prior quarantined entry for the given client."""
        matching = [e for e in self.buffer if e.client_id == client_id]
        if matching:
            for e in matching:
                self.buffer.remove(e)
            return matching[-1]
        return None

    def re_evaluate_pending(
        self,
        current_round: int,
        reference_vector: Optional[torch.Tensor],
        theta_cos: Optional[float] = None
    ) -> List[Tuple[QuarantinedEntry, str, float]]:
        """Re-evaluates all pending quarantined entries against refreshed Top-K reference.
        
        Returns:
            List of (entry, action, age_delay_multiplier) for all resolved entries.
            Action is either 'ACCEPT' or 'REJECT'.
        """
        if theta_cos is None:
            theta_cos = self.theta_cos

        resolved = []
        remaining = deque(maxlen=self.capacity)

        # Compute normalized reference vector on CPU if available
        ref_normed = None
        if reference_vector is not None:
            r_flat = reference_vector.detach().cpu().flatten().float()
            r_norm = torch.norm(r_flat).item()
            if r_norm > 1e-9:
                ref_normed = r_flat / r_norm

        while self.buffer:
            entry = self.buffer.popleft()
            rounds_held = current_round - entry.entry_round

            # Condition 1: Check Expiry
            if rounds_held > self.horizon:
                resolved.append((entry, "REJECT", 0.0))
                continue

            # Condition 2: Re-evaluate against fresh reference
            if ref_normed is not None:
                dW = entry.delta_W_clipped
                dW_norm = torch.norm(dW).item()
                if dW_norm > 1e-9:
                    sim = torch.dot(dW / dW_norm, ref_normed).item()
                else:
                    sim = 1.0

                if sim >= theta_cos:
                    # Successfully resolved to ACCEPT with age-attenuated multiplier
                    age_delay = max(0, rounds_held)
                    age_multiplier = 1.0 / (1.0 + 0.1 * age_delay)
                    resolved.append((entry, "ACCEPT", age_multiplier))
                    continue

            # Still unresolved and within horizon -> retain in quarantine
            remaining.append(entry)

        self.buffer = remaining
        return resolved

    @property
    def depth(self) -> int:
        """Current number of quarantined updates in memory."""
        return len(self.buffer)
