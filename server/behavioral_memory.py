from collections import deque
from typing import Optional, Dict, List
import torch
import numpy as np

from shared.types import BehavioralEvidence


class ClientBehavioralProfile:
    """Maintains bounded rolling history of accepted gradients, telemetry,
    and the long-term Genesis Anchor for a single client on the CPU.
    """

    def __init__(self, maxlen: int = 10):
        # Flattened 1D unit float32 tensors on CPU
        self.gradient_memory: deque[torch.Tensor] = deque(maxlen=maxlen)
        # Accepted update norms
        self.norm_history: deque[float] = deque(maxlen=maxlen)
        self.total_accepted: int = 0
        
        # Dual-Horizon Genesis Anchor
        self.genesis_anchor: Optional[torch.Tensor] = None
        self.early_vectors: List[torch.Tensor] = []
        self.consecutive_downweights: int = 0

    def append(self, delta_W: torch.Tensor, norm_val: float, is_downweight: bool = False) -> None:
        """Stores unit-normalized float32 1D vector on CPU and its norm.
        Guarantees no GPU tensors or computation graphs are retained.
        """
        vec = delta_W.detach().cpu().flatten().float()
        norm = torch.norm(vec).item()
        if norm > 1e-9:
            unit_vec = vec / norm
        else:
            unit_vec = vec

        self.gradient_memory.append(unit_vec)
        self.norm_history.append(float(norm_val))
        self.total_accepted += 1

        if is_downweight:
            self.consecutive_downweights += 1
        else:
            self.consecutive_downweights = 0
            self._update_anchor(unit_vec)

    def _update_anchor(self, unit_vec: torch.Tensor, lambda_anchor: float = 0.05) -> None:
        """Initializes or slowly updates the long-term Genesis Anchor on full ACCEPT."""
        if self.genesis_anchor is None:
            self.early_vectors.append(unit_vec.clone())
            if len(self.early_vectors) >= 3:
                centroid = torch.stack(self.early_vectors).mean(dim=0)
                norm_c = torch.norm(centroid).item()
                if norm_c > 1e-9:
                    self.genesis_anchor = centroid / norm_c
                else:
                    self.genesis_anchor = unit_vec.clone()
        else:
            updated = (1.0 - lambda_anchor) * self.genesis_anchor + lambda_anchor * unit_vec
            norm_u = torch.norm(updated).item()
            if norm_u > 1e-9:
                self.genesis_anchor = updated / norm_u

    def compute_anchor_similarity(self, delta_W: torch.Tensor) -> Optional[float]:
        """Computes cosine similarity between candidate update and Genesis Anchor (available when depth >= 1)."""
        vec = delta_W.detach().cpu().flatten().float()
        norm = torch.norm(vec).item()
        if norm < 1e-9:
            return 1.0
        unit_vec = vec / norm

        if self.genesis_anchor is not None:
            return float(torch.dot(unit_vec, self.genesis_anchor).item())
        elif len(self.early_vectors) >= 1:
            early_c = torch.stack(self.early_vectors).mean(dim=0)
            c_norm = torch.norm(early_c).item()
            if c_norm > 1e-9:
                return float(torch.dot(unit_vec, early_c / c_norm).item())
        return None

    @property
    def depth(self) -> int:
        """Number of valid historical updates currently in memory."""
        return len(self.gradient_memory)


class BehavioralMemoryManager:
    """Central manager for per-client behavioral profiling and deterministic
    continuous evidence extraction in BDSF-AFL.
    """

    def __init__(self, config: dict):
        self.history_size: int = config.get("behavioral_history_size", 10)
        self.min_history: int = config.get("behavioral_min_depth", config.get("behavioral_min_history", 3))
        self.profiles: Dict[int, ClientBehavioralProfile] = {}

    def get_or_create_profile(self, client_id: int) -> ClientBehavioralProfile:
        """Retrieves or creates a ClientBehavioralProfile for a client."""
        if client_id not in self.profiles:
            self.profiles[client_id] = ClientBehavioralProfile(maxlen=self.history_size)
        return self.profiles[client_id]

    def extract_evidence(
        self,
        client_id: int,
        delta_W: torch.Tensor,
        g_i: Optional[float] = None,
        client_gap_history: Optional[List[float]] = None
    ) -> BehavioralEvidence:
        """Extracts continuous behavioral evidence strictly from prior history.
        Side-effect free: does NOT mutate history or internal state.
        """
        profile = self.get_or_create_profile(client_id)
        depth = profile.depth
        behavioral_mature = (depth >= self.min_history)

        # 1. Compute cadence consistency and genesis anchor similarity
        cadence_consistency = self._compute_cadence_consistency(g_i, client_gap_history)
        sim_anchor = profile.compute_anchor_similarity(delta_W)
        consecutive_dw = profile.consecutive_downweights

        # If not enough gradient trajectory history exists yet, return None for self-trajectory metrics
        if not behavioral_mature:
            return BehavioralEvidence(
                sim_self_mean=None,
                sim_self_max=None,
                norm_deviation_self=None,
                cadence_consistency=cadence_consistency,
                history_depth=depth,
                sim_anchor=sim_anchor,
                consecutive_dw=consecutive_dw,
                behavioral_mature=False,
            )

        # 2. Self-Similarity: Cosine similarity against prior stored unit vectors
        dW_flat = delta_W.detach().cpu().flatten().float()
        norm_raw = torch.norm(dW_flat).item()

        if norm_raw < 1e-9:
            sim_self_mean = 1.0
            sim_self_max = 1.0
        else:
            unit_candidate = dW_flat / norm_raw
            # Stack stored unit vectors -> shape (|H_i|, D)
            M = torch.stack(list(profile.gradient_memory))
            # Matrix-vector multiply for single-pass dot products
            sims = torch.mv(M, unit_candidate).numpy()
            sim_self_mean = float(np.mean(sims))
            sim_self_max = float(np.max(sims))

        # 3. Self Norm Deviation: Robust MAD-based normalized deviation
        norms = np.array(profile.norm_history)
        med_norm = float(np.median(norms))
        mad_norm = float(np.median(np.abs(norms - med_norm)))
        norm_deviation_self = float(np.abs(norm_raw - med_norm) / (1.4826 * mad_norm + 1e-6))

        return BehavioralEvidence(
            sim_self_mean=sim_self_mean,
            sim_self_max=sim_self_max,
            norm_deviation_self=norm_deviation_self,
            cadence_consistency=cadence_consistency,
            history_depth=depth,
            sim_anchor=sim_anchor,
            consecutive_dw=consecutive_dw,
            behavioral_mature=True,
        )

    def _compute_cadence_consistency(
        self,
        g_i: Optional[float],
        client_gap_history: Optional[List[float]]
    ) -> Optional[float]:
        """Calculates robust normalized MAD deviation of incoming gap g_i
        from that client's prior gap distribution.
        """
        if g_i is None or client_gap_history is None or len(client_gap_history) < self.min_history:
            return None

        gaps = np.array(client_gap_history)
        med_gap = float(np.median(gaps))
        mad_gap = float(np.median(np.abs(gaps - med_gap)))
        cadence_dev = float(np.abs(g_i - med_gap) / (1.4826 * mad_gap + 1e-6))
        return cadence_dev

    def on_accept(
        self,
        client_id: int,
        delta_W: torch.Tensor,
        norm_val: Optional[float] = None,
        is_downweight: bool = False
    ) -> None:
        """Updates per-client behavioral memory strictly after an update is accepted.
        Guarantees only legitimate, accepted updates enter historical memory.
        """
        profile = self.get_or_create_profile(client_id)
        if norm_val is None:
            norm_val = torch.norm(delta_W.detach().cpu().flatten().float()).item()
        profile.append(delta_W, norm_val, is_downweight=is_downweight)
