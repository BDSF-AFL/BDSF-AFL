from collections import deque
from typing import Optional

import torch
import numpy as np

from shared.types import AcceptedEntry


class SpatialValidator:
    """Maintains the accepted gradient buffer, builds the Top-K trust-anchored
    reference vector, and performs cosine similarity checks and adaptive L2
    clipping against incoming gradients.

    Ablation flags:
      - top_k_ref (True)  → Top-K reference by I*P score
                  (False) → weighted mean of all buffer entries
      - adaptive_clip (True)  → C_t = median(norms) * gamma_clip
                     (False) → static clip from config["static_clip_C"]
    """

    def __init__(self, config: dict) -> None:
        self.K_ref: int = config.get("K_ref", 10)
        self.M: int = config.get("M", 30)
        self.theta_cos: float = config.get("theta_cos", 0.1)
        self.gamma_clip: float = config.get("gamma_clip", 1.5)

        self.use_top_k_ref: bool = config.get("top_k_ref", True)
        self.use_adaptive_clip: bool = config.get("adaptive_clip_enabled", True)

        # Own buffer — AggregatorServer calls on_accept() to keep it in sync
        self._buffer: deque[AcceptedEntry] = deque(maxlen=self.M)

        # Keep config reference for static_clip_C fallback
        self.config: dict = config

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
            return True

        dW_flat = delta_W.flatten().float()
        ref_flat = ref.flatten().float()

        dW_norm = torch.norm(dW_flat).item()
        ref_norm = torch.norm(ref_flat).item()

        # Zero gradient — don't reject
        if dW_norm < 1e-9 or ref_norm < 1e-9:
            return True

        sim = torch.dot(dW_flat, ref_flat).item() / (dW_norm * ref_norm)
        return sim >= self.theta_cos

    # ------------------------------------------------------------------
    # Adaptive L2 clipping
    # ------------------------------------------------------------------

    def adaptive_clip(self, delta_W: torch.Tensor) -> torch.Tensor:
        """Clips gradient norm. Returns a (possibly clipped) *new* tensor.
        Never modifies the input in-place."""

        if not self.use_adaptive_clip:
            # Ablation: static clip
            C_t = self.config.get("static_clip_C", 10.0)
        else:
            # Adaptive: C_t = median(accepted norms) * gamma_clip
            if len(self._buffer) == 0:
                return delta_W.clone()

            norms = [torch.norm(e.delta_W.flatten()).item() for e in self._buffer]
            C_t = float(np.median(norms)) * self.gamma_clip

        dW_flat = delta_W.flatten().float()
        norm_dW = torch.norm(dW_flat).item()

        if norm_dW > C_t and norm_dW > 1e-9:
            dW_clipped = dW_flat * (C_t / norm_dW)
        else:
            dW_clipped = dW_flat.clone()

        return dW_clipped.reshape(delta_W.shape)

    # ------------------------------------------------------------------
    # Reference vector construction (internal)
    # ------------------------------------------------------------------

    def _build_reference(self) -> Optional[torch.Tensor]:
        """Builds the Top-K trust-anchored reference vector from the accepted
        gradient buffer. Returns None if the buffer is empty."""

        if len(self._buffer) == 0:
            return None

        if self.use_top_k_ref:
            # Top-K by composite reputation score I*P (descending)
            ranked = sorted(
                self._buffer,
                key=lambda e: e.I_score * e.P_score,
                reverse=True,
            )
            top_k = ranked[: min(self.K_ref, len(ranked))]
            ref = torch.stack(
                [e.delta_W.flatten().float() for e in top_k]
            ).mean(dim=0)
        else:
            # Ablation: weighted mean of all entries
            weights = torch.tensor(
                [e.I_score * e.P_score for e in self._buffer]
            )
            weight_sum = weights.sum().item()

            if weight_sum < 1e-9:
                return None

            grads = torch.stack(
                [e.delta_W.flatten().float() for e in self._buffer]
            )
            ref = (weights.unsqueeze(1) * grads).sum(dim=0) / weight_sum

        return ref
