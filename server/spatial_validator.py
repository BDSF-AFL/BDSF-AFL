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
        self.spatial_warmup_rounds: int = int(config.get("spatial_warmup_rounds", 50))
        self._unique_accepted_clients: set = set()
        self._total_accepted_count: int = 0
        
        # Track the last computed cosine similarity for the borderline suspicion check
        self.last_sim: Optional[float] = None

        # Stores (client_id, residual_vector) tuples from recent accepted updates.
        # The residual is the component of delta_W orthogonal to the consensus reference.
        # Honest clients share structured residuals (same task); S2 mimicry has random residuals.
        self.prc_buffer_k: int = config.get("prc_buffer_k", 10)
        self._residual_buffer: deque[tuple[int, torch.Tensor]] = deque(maxlen=self.prc_buffer_k)
        # Per-client last residual vector for Temporal Residual Autocorrelation (TRA)
        self._client_last_residual: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helper: Positive Norm Extraction
    # ------------------------------------------------------------------

    def _get_positive_norms(self) -> List[float]:
        """Extracts finite positive Euclidean norms from the accepted buffer, skipping warmup entries."""
        pos_norms = []
        for e in self._buffer:
            if getattr(e, "is_warmup", False):
                continue
            n = torch.norm(e.delta_W.flatten().float()).item()
            if np.isfinite(n) and n > EPS:
                pos_norms.append(n)
        return pos_norms

    # ------------------------------------------------------------------
    # Pairwise Residual Coherence (PRC)
    # ------------------------------------------------------------------

    def _compute_residual(self, dW_flat: torch.Tensor, ref_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Computes the component of dW orthogonal to the reference direction.
        
        For honest clients training on overlapping classes, this residual captures
        class-specific gradient structure that is correlated across clients.
        For S2 mimicry, this residual is the random ε·v⊥ injected each round.
        """
        ref_norm_sq = torch.dot(ref_flat, ref_flat).item()
        if ref_norm_sq < EPS:
            return None
        # Project dW onto ref direction, then subtract to get orthogonal residual
        proj_coeff = torch.dot(dW_flat, ref_flat).item() / ref_norm_sq
        residual = dW_flat - proj_coeff * ref_flat
        res_norm = torch.norm(residual).item()
        if res_norm < EPS or not np.isfinite(res_norm):
            return None
        return residual / res_norm  # unit-normalize for cosine comparison

    def _compute_prc(self, dW_flat: torch.Tensor, ref_flat: torch.Tensor, 
                     client_id: int) -> Optional[float]:
        """Computes Pairwise Residual Coherence score for an incoming update.
        
        PRC = maximum cosine similarity between this update's orthogonal residual
        and the residuals of recent accepted updates from OTHER clients (nearest neighbor
        residual subspace alignment).
        
        Mathematical basis:
        - Honest non-IID: residual captures class-specific gradient structure that matches
          other clients sharing those classes → max cross-client residual similarity > 0.30.
        - S2 Mimicry: residual = ε·v⊥ where v⊥ is random in 11M dimensions → PRC ≈ 0.
        """
        residual = self._compute_residual(dW_flat, ref_flat)
        if residual is None:
            return None
        
        # Compare against cross-client residuals (exclude same client)
        cross_sims = []
        for buf_cid, buf_residual in self._residual_buffer:
            if buf_cid == client_id:
                continue  # Only compare against OTHER clients
            sim = torch.dot(residual, buf_residual).item()
            if np.isfinite(sim):
                cross_sims.append(sim)
        
        if len(cross_sims) < 3:  # Need minimum cross-client samples
            return None
        
        return float(np.max(cross_sims))

    def _compute_tra(self, client_id: int, residual: torch.Tensor) -> Optional[float]:
        """Computes Temporal Residual Autocorrelation (TRA) for the submitting client.
        
        TRA = cos(r_{i, t}, r_{i, t-1}).
        Honest clients exhibit TRA >= 0.60-0.95 (persistent class gradient structure).
        S2 Mimicry exhibits TRA ≈ 0.00-0.15 (independent random Gaussian noise).
        """
        if client_id in self._client_last_residual:
            last_r = self._client_last_residual[client_id]
            sim = torch.dot(residual, last_r).item()
            if np.isfinite(sim):
                return float(sim)
        return None

    def record_residual(self, client_id: int, delta_W: torch.Tensor) -> None:
        """Records the orthogonal residual of an accepted update into the PRC and TRA buffers.
        Called by AggregatorServer after an update is accepted/downweighted."""
        ref, ref_count, _ = self._build_reference_stats()
        if ref is None or ref_count < self.K_ref:
            return
        dW_flat = delta_W.flatten().float()
        ref_flat = ref.flatten().float()
        residual = self._compute_residual(dW_flat, ref_flat)
        if residual is not None:
            self._client_last_residual[client_id] = residual.clone()
            self._residual_buffer.append((client_id, residual))

    # ------------------------------------------------------------------
    # Evidence Extraction
    # ------------------------------------------------------------------

    def extract_evidence(self, delta_W: torch.Tensor, client_id: int = -1) -> SpatialEvidence:
        """Extracts continuous spatial evidence without modifying state."""
        ref, ref_count, coherence = self._build_reference_stats()
        n_unique = len(self._unique_accepted_clients)
        min_unique = max(self.K_ref, self.config.get("N_clients", 20) // 2)
        spatial_mature = (
            ref is not None
            and ref_count >= self.K_ref
            and n_unique >= min_unique
            and self._total_accepted_count >= self.spatial_warmup_rounds
        )

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

        # --- Pairwise Residual Coherence & Temporal Residual Autocorrelation ---
        prc_score = None
        tra_score = None
        if spatial_mature and ref is not None and norm_raw > EPS and np.isfinite(norm_raw):
            ref_flat = ref.flatten().float()
            prc_score = self._compute_prc(dW_flat, ref_flat, client_id)
            residual = self._compute_residual(dW_flat, ref_flat)
            if residual is not None and client_id >= 0:
                tra_score = self._compute_tra(client_id, residual)

        oer_score = (1.0 - sim_global ** 2) if sim_global is not None else None

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
            prc_score=prc_score,
            tra_score=tra_score,
            oer_score=oer_score,
        )

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def on_accept(self, entry: AcceptedEntry) -> None:
        """Called by AggregatorServer every time an update is fully accepted
        (after Step 10 in the pipeline)."""
        self._buffer.append(entry)
        if entry.client_id is not None:
            self._unique_accepted_clients.add(entry.client_id)
        self._total_accepted_count += 1

    def reset_buffer(self) -> None:
        """Flushes the sliding window buffer to break reference stagnation deadlocks."""
        self._buffer.clear()

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
        # Build valid entry set.
        # Deduplication (keep latest per client) is opt-in via config["deduplicate_spatial_ref"].
        # Default OFF to match debug branch — all recent valid entries contribute to the centroid.
        # Enabling this shifts centroid toward rare-class clients under extreme non-IID (alpha=0.1),
        # which can ease mimicry threshold satisfaction for S2_MIMICRY attackers.
        raw_entries = []
        for e in self._buffer:
            gnorm = torch.norm(e.delta_W.flatten().float()).item()
            if np.isfinite(gnorm) and gnorm > EPS:
                raw_entries.append(e)

        if self.config.get("deduplicate_spatial_ref", False):
            latest_per_client = {}
            for e in raw_entries:
                latest_per_client[e.client_id] = e
            valid_entries = list(latest_per_client.values())
        else:
            valid_entries = raw_entries
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

    def get_state(self) -> dict:
        """Serializes spatial validator state for checkpoint equivalence."""
        buffer_state = []
        for e in self._buffer:
            buffer_state.append({
                "delta_W": e.delta_W.clone().cpu(),
                "I_score": e.I_score,
                "P_score": e.P_score,
                "client_id": e.client_id,
                "is_warmup": e.is_warmup,
            })
        res_state = []
        for cid, res in self._residual_buffer:
            res_state.append((cid, res.clone().cpu()))
        last_res_state = {cid: res.clone().cpu() for cid, res in self._client_last_residual.items()}
        return {
            "buffer": buffer_state,
            "residual_buffer": res_state,
            "client_last_residual": last_res_state,
            "_unique_accepted_clients": list(self._unique_accepted_clients),
            "_total_accepted_count": self._total_accepted_count,
            "last_sim": self.last_sim,
        }

    def load_state(self, state: dict) -> None:
        """Restores spatial validator state from checkpoint."""
        self._buffer = deque(maxlen=self.M)
        buf_list = state.get("buffer", state.get("_buffer", []))
        for s in buf_list:
            entry = AcceptedEntry(
                delta_W=s["delta_W"].clone().cpu(),
                I_score=float(s.get("I_score", 1.0)),
                P_score=float(s.get("P_score", 1.0)),
                client_id=s.get("client_id"),
                is_warmup=bool(s.get("is_warmup", False)),
            )
            self._buffer.append(entry)
        self._residual_buffer = deque(maxlen=self.prc_buffer_k)
        for cid, res in state.get("residual_buffer", []):
            self._residual_buffer.append((cid, res.clone().cpu()))
        self._client_last_residual = {}
        for cid_str, res in state.get("client_last_residual", {}).items():
            self._client_last_residual[int(cid_str)] = res.clone().cpu()
        clients = state.get("_unique_accepted_clients", state.get("unique_accepted_clients", []))
        self._unique_accepted_clients = set(clients)
        self._total_accepted_count = int(state.get("_total_accepted_count", state.get("total_accepted_count", 0)))
        self.last_sim = state.get("last_sim")

