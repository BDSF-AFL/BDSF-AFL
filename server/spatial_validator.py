from collections import deque
from typing import Optional, List

import torch
import numpy as np

from shared.types import AcceptedEntry, SpatialEvidence

EPS = 1e-9


class SpatialValidator:
    """Maintains the accepted gradient buffer, builds the Top-K trust-anchored
    reference vector, and performs cosine similarity checks and adaptive L2
    clipping against incoming gradients.

    Ablation flags:
      - top_k_ref (True)  → Top-K reference by I*P score (always used)
      - adaptive_clip (True)  → C_t = median(norms) * gamma_clip
                     (False) → static clip from config["static_clip_C"]
    """

    def __init__(self, config: dict) -> None:
        self.K_ref: int = config.get("K_ref", 10)
        self.M: int = config.get("M", 30)
        self.theta_cos: float = config.get("theta_cos", 0.1)
        self.gamma_clip: float = config.get("gamma_clip", 1.5)

        self.use_top_k_ref: bool = True
        self.use_adaptive_clip: bool = config.get("adaptive_clip_enabled", True)
        # Own buffer — AggregatorServer calls on_accept() to keep it in sync
        self._buffer: deque[AcceptedEntry] = deque(maxlen=self.M)

        # Keep config reference for static_clip_C fallback
        self.config: dict = config
        
        # Track the last computed cosine similarity for the borderline suspicion check
        self.last_sim: Optional[float] = None

    # ------------------------------------------------------------------
    # Helper: Positive Norm Extraction
    # ------------------------------------------------------------------

    def _get_positive_norms(self) -> List[float]:
        """Extracts finite positive Euclidean norms from the accepted buffer."""
        pos_norms = []
        for e in self._buffer:
            n = torch.norm(e.delta_W.flatten().float()).item()
            if np.isfinite(n) and n > EPS:
                pos_norms.append(n)
        return pos_norms

    # ------------------------------------------------------------------
    # Evidence Extraction
    # ------------------------------------------------------------------

    def extract_evidence(self, delta_W: torch.Tensor) -> SpatialEvidence:
        """Extracts continuous spatial evidence without modifying state."""
        ref, ref_count, coherence = self._build_reference_stats()
        spatial_mature = (ref is not None and ref_count >= self.K_ref)

        dW_flat = delta_W.flatten().float()
        norm_raw = torch.norm(dW_flat).item()

        if not spatial_mature or norm_raw <= EPS or not np.isfinite(norm_raw):
            sim_global = None
        else:
            ref_flat = ref.flatten().float()
            ref_norm = torch.norm(ref_flat).item()
            if ref_norm <= EPS or not np.isfinite(ref_norm):
                sim_global = None
            else:
                sim_global = float(torch.dot(dW_flat, ref_flat).item() / (norm_raw * ref_norm))

        # Static vs Adaptive warmup vs Adaptive mature
        if not self.use_adaptive_clip:
            C_t: Optional[float] = float(self.config.get("static_clip_C", 10.0))
            norm_ratio_median: Optional[float] = 1.0
            norm_clipped = min(norm_raw, C_t) if norm_raw > EPS and C_t > EPS else norm_raw
        else:
            pos_norms = self._get_positive_norms()
            if len(pos_norms) == 0:
                # Warmup: no mature adaptive bound yet
                C_t = None
                norm_ratio_median = None
                norm_clipped = norm_raw
            else:
                med_norm = float(np.median(pos_norms))
                if med_norm > EPS and np.isfinite(med_norm):
                    C_t = med_norm * self.gamma_clip
                    norm_ratio_median = norm_raw / med_norm
                    norm_clipped = min(norm_raw, C_t) if norm_raw > C_t and norm_raw > EPS else norm_raw
                else:
                    C_t = None
                    norm_ratio_median = None
                    norm_clipped = norm_raw

        return SpatialEvidence(
            sim_global=sim_global,
            norm_raw=norm_raw,
            norm_clipped=norm_clipped,
            norm_ratio_median=norm_ratio_median,
            dynamic_bound_C=C_t,
            reference_available=spatial_mature,
            spatial_mature=spatial_mature,
            spatial_reference_count=ref_count,
            spatial_coherence=coherence,
        )

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def on_accept(self, entry: AcceptedEntry) -> None:
        """Called by AggregatorServer every time an update is fully accepted
        (after Step 10 in the pipeline)."""
        self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Cosine similarity gate
    # ------------------------------------------------------------------

    def cosine_check(self, delta_W: torch.Tensor) -> bool:
        """Returns True if the gradient passes the spatial check (should be
        accepted), False if it fails (cosine similarity below theta_cos)."""

        ref = self._build_reference()

        # No reference yet — burn-in; accept everything
        if ref is None:
            self.last_sim = None
            return True

        dW_flat = delta_W.flatten().float()
        ref_flat = ref.flatten().float()

        dW_norm = torch.norm(dW_flat).item()
        ref_norm = torch.norm(ref_flat).item()

        # Zero gradient — don't reject
        if dW_norm <= EPS or ref_norm <= EPS or not np.isfinite(dW_norm) or not np.isfinite(ref_norm):
            self.last_sim = None
            return True

        sim = torch.dot(dW_flat, ref_flat).item() / (dW_norm * ref_norm)
        self.last_sim = sim
        return sim >= self.theta_cos

    # ------------------------------------------------------------------
    # Adaptive L2 clipping
    # ------------------------------------------------------------------

    def adaptive_clip(self, delta_W: torch.Tensor) -> torch.Tensor:
        """Clips gradient norm. Returns a (possibly clipped) *new* tensor.
        Never modifies the input in-place."""
        dW_flat = delta_W.flatten().float()
        norm_dW = torch.norm(dW_flat).item()

        # Case 1: Zero or near-zero incoming update - return unchanged
        if norm_dW <= EPS or not np.isfinite(norm_dW):
            return delta_W.clone()

        # Case 2: Static clipping mode
        if not self.use_adaptive_clip:
            C_t = float(self.config.get("static_clip_C", 10.0))
        else:
            # Case 3: Adaptive clipping warmup (insufficient positive history)
            pos_norms = self._get_positive_norms()
            if len(pos_norms) == 0:
                return delta_W.clone()

            # Case 4: Mature adaptive clipping
            med_norm = float(np.median(pos_norms))
            C_t = med_norm * self.gamma_clip

        if C_t > EPS and norm_dW > C_t:
            dW_clipped = dW_flat * (C_t / norm_dW)
        else:
            dW_clipped = dW_flat.clone()

        return dW_clipped.reshape(delta_W.shape)

    # ------------------------------------------------------------------
    # Reference vector construction (internal)
    # ------------------------------------------------------------------

    def _build_reference_stats(self) -> tuple[Optional[torch.Tensor], int, float]:
        """Builds Top-K reference vector, counts positive entries, and computes spatial coherence."""
        valid_entries = []
        for e in self._buffer:
            gnorm = torch.norm(e.delta_W.flatten().float()).item()
            if np.isfinite(gnorm) and gnorm > EPS:
                valid_entries.append(e)

        ref_count = len(valid_entries)
        if ref_count < self.K_ref:
            return None, ref_count, 0.0

        # Top-K by composite reputation score I*P (descending)
        ranked = sorted(
            valid_entries,
            key=lambda e: e.I_score * e.P_score,
            reverse=True,
        )
        top_k = ranked[: min(self.K_ref, len(ranked))]

        normed_grads = []
        for e in top_k:
            g = e.delta_W.flatten().float()
            gnorm = torch.norm(g).item()
            normed_grads.append(g / gnorm)

        r_raw = torch.stack(normed_grads).mean(dim=0)
        coherence = float(torch.norm(r_raw.flatten().float()).item())
        if not np.isfinite(coherence) or coherence <= EPS:
            return None, ref_count, 0.0

        ref = r_raw / (coherence + 1e-9)
        return ref, ref_count, min(1.0, coherence)

    def _build_reference(self) -> Optional[torch.Tensor]:
        """Builds the trust-anchored reference vector using Top-K reputation scores."""
        ref, _, _ = self._build_reference_stats()
        return ref

