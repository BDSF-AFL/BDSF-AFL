from collections import deque
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np

from shared.types import BehavioralEvidence


class ClientBehavioralProfile:
    """Maintains bounded rolling history of accepted gradients, telemetry,
    and the dual-horizon Genesis Anchor (frozen ground-truth + bounded adaptive) for a single client on the CPU.
    """

    def __init__(self, maxlen: int = 10):
        # Flattened 1D unit float32 tensors on CPU
        self.gradient_memory: deque[torch.Tensor] = deque(maxlen=maxlen)
        # Accepted update norms
        self.norm_history: deque[float] = deque(maxlen=maxlen)
        self.total_accepted: int = 0
        
        # Dual-Horizon Genesis Anchor
        self.genesis_anchor: Optional[torch.Tensor] = None
        self.frozen_genesis_anchor: Optional[torch.Tensor] = None
        self.early_vectors: List[torch.Tensor] = []
        self.consecutive_downweights: int = 0

    def append(self, delta_W: torch.Tensor, norm_val: float, is_downweight: bool = False) -> None:
        """Stores unit-normalized float16 1D vector on CPU and its norm.
        Guarantees no GPU tensors or computation graphs are retained.
        """
        vec = delta_W.detach().cpu().flatten().float()
        norm = torch.norm(vec).item()
        if norm > 1e-9:
            unit_vec = (vec / norm).half()
        else:
            unit_vec = vec.half()

        # Compute sim_self against prior gradient memory before appending
        if len(self.gradient_memory) >= 1:
            prior_vecs = [v.float() for v in self.gradient_memory]
            M = torch.stack(prior_vecs)
            sims = torch.mv(M, unit_vec.float()).numpy()
            sim_self = float(np.mean(sims))
        else:
            sim_self = 0.0

        self.gradient_memory.append(unit_vec)
        self.norm_history.append(float(norm_val))
        self.total_accepted += 1

        if is_downweight:
            self.consecutive_downweights += 1
            # Damped micro-adaptation on verified self-consistent non-IID downweight to prevent anchor starvation
            if sim_self >= 0.35:
                self._update_anchor(unit_vec, lambda_anchor=0.02)
        else:
            self.consecutive_downweights = 0
            self._update_anchor(unit_vec, lambda_anchor=0.15)

    def _update_anchor(self, unit_vec: torch.Tensor, lambda_anchor: float = 0.15) -> None:
        """Initializes or slowly updates the long-term Genesis Anchor while guarding against adversarial drift."""
        if self.genesis_anchor is None:
            self.early_vectors.append(unit_vec.clone())
            if len(self.early_vectors) >= 3:
                centroid = torch.stack([v.float() for v in self.early_vectors]).mean(dim=0)
                norm_c = torch.norm(centroid).item()
                if norm_c > 1e-9:
                    self.genesis_anchor = (centroid / norm_c).half()
                else:
                    self.genesis_anchor = unit_vec.clone().half()
                # Initialize immutable frozen ground-truth anchor
                self.frozen_genesis_anchor = self.genesis_anchor.clone()
        else:
            if self.frozen_genesis_anchor is None:
                self.frozen_genesis_anchor = self.genesis_anchor.clone()
            # Dual-horizon bounded adaptation: update adaptive anchor while bounding divergence from frozen anchor
            updated = (1.0 - lambda_anchor) * self.genesis_anchor.float() + lambda_anchor * unit_vec.float()
            norm_u = torch.norm(updated).item()
            if norm_u > 1e-9:
                cand_anchor = updated / norm_u
                sim_to_frozen = float(torch.dot(cand_anchor, self.frozen_genesis_anchor.float()).item())
                # Guard against uncoordinated boiling-frog drift
                if sim_to_frozen >= 0.10:
                    self.genesis_anchor = cand_anchor.half()
            else:
                self.genesis_anchor = (updated / norm_u).half()

    def compute_anchor_similarity(self, delta_W: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
        """Computes cosine similarity between candidate update and Genesis Anchor (available when depth >= 1).
        Returns (sim_adaptive, sim_frozen).
        """
        vec = delta_W.detach().cpu().flatten().float()
        norm = torch.norm(vec).item()
        if norm < 1e-9:
            return 1.0, 1.0
        unit_vec = vec / norm

        if self.genesis_anchor is not None:
            sim_adaptive = float(torch.dot(unit_vec, self.genesis_anchor.float()).item())
            sim_frozen = (
                float(torch.dot(unit_vec, self.frozen_genesis_anchor.float()).item())
                if self.frozen_genesis_anchor is not None
                else sim_adaptive
            )
            return sim_adaptive, sim_frozen
        elif len(self.early_vectors) >= 1:
            early_c = torch.stack([v.float() for v in self.early_vectors]).mean(dim=0)
            c_norm = torch.norm(early_c).item()
            if c_norm > 1e-9:
                sim = float(torch.dot(unit_vec, early_c / c_norm).item())
                return sim, sim
        return None, None

    def compute_frozen_anchor_similarity(self, delta_W: torch.Tensor) -> Optional[float]:
        """Computes cosine similarity between candidate update and immutable frozen Genesis Anchor."""
        if self.frozen_genesis_anchor is None:
            return None
        vec = delta_W.detach().cpu().flatten().float()
        norm = torch.norm(vec).item()
        if norm < 1e-9:
            return 1.0
        unit_vec = vec / norm
        return float(torch.dot(unit_vec, self.frozen_genesis_anchor.float()).item())

    def compute_anchor_drift(self) -> float:
        """Computes angular/cosine divergence between adaptive anchor and immutable frozen anchor."""
        if self.genesis_anchor is not None and self.frozen_genesis_anchor is not None:
            g = self.genesis_anchor.float()
            f = self.frozen_genesis_anchor.float()
            norm_g = torch.norm(g).item()
            norm_f = torch.norm(f).item()
            if norm_g > 1e-9 and norm_f > 1e-9:
                dot_val = torch.dot(g / norm_g, f / norm_f).item()
            else:
                dot_val = torch.dot(g, f).item()
            return float(max(0.0, 1.0 - dot_val))
        return 0.0

    @property
    def depth(self) -> int:
        """Number of valid historical updates currently in memory."""
        return len(self.gradient_memory)

    def get_state(self) -> dict:
        """Serializes single profile state for checkpointing."""
        return {
            "gradient_memory": [v.clone().cpu() for v in self.gradient_memory],
            "norm_history": list(self.norm_history),
            "total_accepted": self.total_accepted,
            "genesis_anchor": self.genesis_anchor.clone().cpu() if self.genesis_anchor is not None else None,
            "frozen_genesis_anchor": self.frozen_genesis_anchor.clone().cpu() if self.frozen_genesis_anchor is not None else None,
            "early_vectors": [v.clone().cpu() for v in self.early_vectors],
            "consecutive_downweights": self.consecutive_downweights,
        }

    def load_state(self, state: dict) -> None:
        """Restores single profile state from checkpoint."""
        if "gradient_memory" in state:
            self.gradient_memory = deque(
                [v.clone().cpu() for v in state.get("gradient_memory", [])],
                maxlen=self.gradient_memory.maxlen,
            )
        if "norm_history" in state:
            self.norm_history = deque(
                list(state.get("norm_history", [])),
                maxlen=self.norm_history.maxlen,
            )
        self.total_accepted = int(state.get("total_accepted", 0))
        if "genesis_anchor" in state and state["genesis_anchor"] is not None:
            self.genesis_anchor = state["genesis_anchor"].clone().cpu()
        elif "anchor" in state and state["anchor"] is not None:
            self.genesis_anchor = state["anchor"].clone().cpu()
        else:
            self.genesis_anchor = None

        if "frozen_genesis_anchor" in state and state["frozen_genesis_anchor"] is not None:
            self.frozen_genesis_anchor = state["frozen_genesis_anchor"].clone().cpu()
        elif self.genesis_anchor is not None:
            self.frozen_genesis_anchor = self.genesis_anchor.clone()
        else:
            self.frozen_genesis_anchor = None

        if "early_vectors" in state:
            self.early_vectors = [v.clone().cpu() for v in state.get("early_vectors", [])]
        self.consecutive_downweights = int(state.get("consecutive_downweights", 0))


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

        # 1. Compute cadence consistency, genesis anchor similarities, and drift
        cadence_consistency = self._compute_cadence_consistency(g_i, client_gap_history)
        sim_anchor, sim_frozen_anchor = profile.compute_anchor_similarity(delta_W)
        anchor_drift = profile.compute_anchor_drift()
        consecutive_dw = profile.consecutive_downweights

        # If not enough gradient trajectory history exists yet, return None for self-trajectory metrics
        if not behavioral_mature:
            return BehavioralEvidence(
                sim_self_mean=None,
                sim_self_max=None,
                sim_self_mad=None,
                norm_deviation_self=None,
                cadence_consistency=cadence_consistency,
                history_depth=depth,
                sim_anchor=sim_anchor,
                sim_frozen_anchor=sim_frozen_anchor,
                anchor_drift=anchor_drift,
                consecutive_dw=consecutive_dw,
                behavioral_mature=False,
            )

        # 2. Self-Similarity: Cosine similarity against prior stored unit vectors
        dW_flat = delta_W.detach().cpu().flatten().float()
        norm_raw = torch.norm(dW_flat).item()

        if norm_raw < 1e-9:
            sim_self_mean = 1.0
            sim_self_max = 1.0
            sim_self_mad = 0.0
        else:
            unit_candidate = dW_flat / norm_raw
            # Stack stored unit vectors -> shape (|H_i|, D)
            M = torch.stack([v.float() for v in profile.gradient_memory])
            # Matrix-vector multiply for single-pass dot products
            sims = torch.mv(M, unit_candidate).numpy()
            sim_self_mean = float(np.mean(sims))
            sim_self_max = float(np.max(sims))
            med_sim = float(np.median(sims))
            sim_self_mad = float(np.median(np.abs(sims - med_sim)))

        # 3. Self Norm Deviation: Robust MAD-based normalized deviation
        norms = np.array(profile.norm_history)
        med_norm = float(np.median(norms))
        mad_norm = float(np.median(np.abs(norms - med_norm)))
        norm_deviation_self = float(np.abs(norm_raw - med_norm) / (1.4826 * mad_norm + 1e-6))

        return BehavioralEvidence(
            sim_self_mean=sim_self_mean,
            sim_self_max=sim_self_max,
            sim_self_mad=sim_self_mad,
            norm_deviation_self=norm_deviation_self,
            cadence_consistency=cadence_consistency,
            history_depth=depth,
            sim_anchor=sim_anchor,
            sim_frozen_anchor=sim_frozen_anchor,
            anchor_drift=anchor_drift,
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

    def get_state(self) -> dict:
        """Serializes full behavioral memory manager state for checkpointing."""
        return {
            "profiles": {k: v.get_state() for k, v in self.profiles.items()},
        }

    def load_state(self, state: dict) -> None:
        """Restores behavioral memory manager state from checkpoint."""
        self.profiles = {}
        profiles_data = state.get("profiles", state)
        for k, v in profiles_data.items():
            cid = int(k)
            prof = self.get_or_create_profile(cid)
            prof.load_state(v)
