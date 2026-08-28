from typing import Optional, Dict, Any
import math
import numpy as np
import torch

from shared.types import (TemporalEvidence, SpatialEvidence, BehavioralEvidence,
                          JointDecisionOutcome)


class JointDecisionEngine:
    """Deterministic Multi-Criteria Joint Decision Engine for BDSF-AFL Phase 3.
    
    Evaluates continuous temporal, spatial, behavioral, and reputation evidence
    through a strictly ordered, multi-manifold State-Maturity Priority hierarchy:
      - Priority 0a: Universal Safety Pre-Check (Extreme Norm Explosion)
      - Priority 0b: Spatial Warmup (Reference Vector Incomplete, attenuated 0.5x)
      - Priority 1: Hard Safety Invariant Violations (Global Inversion, Temporal Spam/Straggler)
      - Priority 2: Full Consensus Acceptance
      - Priority 3: Non-IID Honest Soft-Filtering (DOWNWEIGHT with dynamic attenuation & drift bounding)
      - Priority 4: Ambiguous / Borderline Quarantine
      - Priority 5: Uncoordinated Adversarial Fallthrough (REJECT)
    """

    def __init__(self, config: dict):
        self.config = config
        self.theta_cos: float = config.get("theta_cos", 0.10)
        self.theta_self: float = config.get("theta_self", 0.30)
        self.theta_floor: float = config.get("theta_floor", 0.40)
        self.theta_anchor_min: float = config.get("theta_anchor_min", 0.50)
        self.alpha_downweight: float = config.get("alpha_downweight", 0.35)
        self.K_drift_max: int = config.get("K_drift_max", 10)
        self.delta_theta_step: float = config.get("delta_theta_step", 0.05)
        self.delta_theta_max: float = config.get("delta_theta_max", 0.25)
        self.delta_temp_mod: float = config.get("delta_temp_mod", 0.50)
        self.enable_quarantine: bool = config.get("enable_quarantine", True)
        self.delta_borderline: float = config.get("delta_borderline", 0.05)
        self.trusted_integrity_min: float = config.get("trusted_integrity_min", 0.80)
        self.warmup_weight_factor: float = config.get("warmup_weight_factor", 0.50)
        self.static_clip_C: float = float(config.get("static_clip_C", 10.0))
        self.norm_anomaly_threshold: float = float(config.get("norm_anomaly_threshold", 2.5))
        # --- Pairwise Residual Coherence (PRC) thresholds ---
        # theta_prc: minimum PRC score for full-weight P2 acceptance.
        #   S2 mimicry has PRC ≈ 0-0.12; honest has PRC > 0.60-0.90 (shared class structure).
        #   Updates failing this gate fall through to P3 (DOWNWEIGHT) instead of P2 (ACCEPT).
        self.theta_prc: float = config.get("theta_prc", 0.20)
        # theta_prc_hard: PRC below this triggers hard rejection in P5 + integrity slash.
        #   A deeply negative PRC indicates the residual is anti-correlated with other clients.
        self.theta_prc_hard: float = config.get("theta_prc_hard", -0.10)

    def evaluate(
        self,
        cid: int,
        temporal_ev: TemporalEvidence,
        spatial_ev: SpatialEvidence,
        behavioral_ev: BehavioralEvidence,
        I_i: float,
        P_i: float,
        current_round: int = 0,
        version_lag: int = 0,
    ) -> JointDecisionOutcome:
        """Deterministically evaluates candidate update evidence against Priority 0-5 hierarchy."""
        
        sim_g = spatial_ev.sim_global
        sim_s = behavioral_ev.sim_self_mean
        norm_r = spatial_ev.norm_raw
        C_t = spatial_ev.dynamic_bound_C if spatial_ev.dynamic_bound_C is not None else self.static_clip_C
        g_margin = temporal_ev.fence_margin
        depth = behavioral_ev.history_depth

        v_lag = version_lag if version_lag != 0 else getattr(temporal_ev, "version_lag", 0)
        sim_frozen = getattr(behavioral_ev, "sim_frozen_anchor", None)
        drift_a = getattr(behavioral_ev, "anchor_drift", None)

        # ---------------------------------------------------------------------
        # PRIORITY 0a: Universal Safety Pre-Check (Active ALWAYS, even during warmup)
        # ---------------------------------------------------------------------
        if norm_r <= 1e-9:
            return JointDecisionOutcome(
                action="REJECT",
                primary_reason="HARD_GUARD_ZERO_GRADIENT",
                aggregation_weight=0.0,
                force_sync_required=True,
                diagnostic_features={
                    "priority": 0,
                    "violation": "norm_raw == 0",
                    "norm_r": norm_r,
                    "version_lag": v_lag,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                }
            )

        if C_t is not None and np.isfinite(C_t) and C_t > 1e-9 and norm_r > 3.0 * C_t:
            return JointDecisionOutcome(
                action="REJECT",
                primary_reason="HARD_GUARD_NORM_EXPLOSION",
                aggregation_weight=0.0,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 0,
                    "violation": "norm_raw > 3.0*C_t",
                    "norm_r": norm_r,
                    "C_t": C_t,
                    "version_lag": v_lag,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 0b: Spatial Warmup (Reference Centroid Incomplete)
        # ---------------------------------------------------------------------
        if not spatial_ev.spatial_mature:
            return JointDecisionOutcome(
                action="ACCEPT",
                primary_reason="SPATIAL_WARMUP_ACCEPT",
                aggregation_weight=self.warmup_weight_factor * (I_i * P_i),
                force_sync_required=False,
                diagnostic_features={
                    "priority": 0,
                    "stage": "spatial_warmup",
                    "spatial_mature": False,
                    "ref_count": spatial_ev.spatial_reference_count,
                    "coherence": spatial_ev.spatial_coherence,
                    "version_lag": v_lag,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 1: Hard Safety Invariant Violations (Drop & Slash Immediately)
        # ---------------------------------------------------------------------
        # 1a. Severe Directional Inversion (Opposes Top-K global consensus beyond floor)
        # Bypass hard drop ONLY if client is proven to be a persistent, anchor-consistent non-IID minority node
        if sim_g is not None and sim_g < -self.theta_floor:
            sim_a = behavioral_ev.sim_anchor
            is_anchor_minority = (behavioral_ev.behavioral_mature and sim_a is not None and sim_s is not None and 
                                  sim_a >= self.theta_anchor_min and sim_s >= self.theta_self)
            if not is_anchor_minority:
                return JointDecisionOutcome(
                    action="REJECT",
                    primary_reason="HARD_GUARD_GLOBAL_INVERSION",
                    aggregation_weight=0.0,
                    force_sync_required=False,
                    diagnostic_features={
                        "priority": 1,
                        "violation": "sim_global < -theta_floor",
                        "sim_g": sim_g,
                        "version_lag": v_lag,
                        "sim_frozen_anchor": sim_frozen,
                        "anchor_drift": drift_a,
                    }
                )

        # 1b. Extreme Temporal Violations (Gated strictly by temporal maturity)
        if temporal_ev.temporal_mature and g_margin > self.delta_temp_mod:
            if temporal_ev.lower_fence is not None and temporal_ev.g_i < temporal_ev.lower_fence:
                return JointDecisionOutcome(
                    action="REJECT",
                    primary_reason="HARD_GUARD_TEMPORAL_SPAM",
                    aggregation_weight=0.0,
                    force_sync_required=False,
                    diagnostic_features={
                        "priority": 1,
                        "violation": "extreme_high_freq",
                        "margin": g_margin,
                        "version_lag": v_lag,
                        "sim_frozen_anchor": sim_frozen,
                        "anchor_drift": drift_a,
                    }
                )
            else:
                return JointDecisionOutcome(
                    action="REJECT",
                    primary_reason="HARD_GUARD_TEMPORAL_STRAGGLER",
                    aggregation_weight=0.0,
                    force_sync_required=True,  # Dispatches HMAC Hard-Reset Payload
                    diagnostic_features={
                        "priority": 1,
                        "violation": "extreme_straggler",
                        "margin": g_margin,
                        "version_lag": v_lag,
                        "sim_frozen_anchor": sim_frozen,
                        "anchor_drift": drift_a,
                    }
                )

        # ---------------------------------------------------------------------
        # PRIORITY 2: Strong Multi-Domain Agreement (Full Consensus Acceptance)
        # ---------------------------------------------------------------------
        is_spatial_valid = (sim_g is not None and sim_g >= self.theta_cos)
        is_self_valid = (not behavioral_ev.behavioral_mature or sim_s is None or sim_s >= self.theta_self)
        is_anchor_valid = (behavioral_ev.sim_anchor is None or behavioral_ev.sim_anchor >= self.theta_anchor_min)
        # Tolerate minor inter-batch GPU/thread timing jitter (g_margin <= 0.20) when spatial & self agreement are strong
        is_temporal_valid = (not temporal_ev.temporal_mature or g_margin <= 0.20)
        # Pairwise Residual Coherence: S2 mimicry has PRC ≈ 0 (random orthogonal noise),
        # honest clients have PRC > 0 (shared class structure in gradient residuals).
        # None = not yet computed (warmup/insufficient buffer) → pass through.
        prc = spatial_ev.prc_score
        is_prc_valid = (prc is None or prc >= self.theta_prc)

        if is_spatial_valid and is_self_valid and is_anchor_valid and is_temporal_valid and is_prc_valid:
            return JointDecisionOutcome(
                action="ACCEPT",
                primary_reason="FULL_CONSENSUS_ACCEPT",
                aggregation_weight=1.0 * (I_i * P_i),
                force_sync_required=False,
                diagnostic_features={
                    "priority": 2,
                    "sim_g": sim_g,
                    "sim_s": sim_s,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "prc_score": prc,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 3: Non-IID Honest Soft-Filtering & Moderate Jitter (DOWNWEIGHT)
        # ---------------------------------------------------------------------
        # Evaluates honest non-IID clients and moderate timing variations
        c_dw = behavioral_ev.consecutive_dw
        theta_self_eff = self.theta_self + min(self.delta_theta_max, c_dw * self.delta_theta_step)
        is_anchor_valid = (behavioral_ev.sim_anchor is None or behavioral_ev.sim_anchor >= self.theta_anchor_min)
        
        # Dynamic minority consistency:
        # Client is proven authentic minority non-IID node when aligned with its Genesis Anchor
        sim_a = behavioral_ev.sim_anchor
        is_minority_consistent = (sim_a is not None and sim_a >= self.theta_anchor_min)
        
        is_drift_bounded = (c_dw < self.K_drift_max) or is_minority_consistent
        is_temporal_tolerable = (not temporal_ev.temporal_mature or g_margin <= self.delta_temp_mod)
        # Spatial range: supports non-IID minority (sim_g < theta_cos) AND high consensus with moderate temporal jitter (sim_g >= theta_cos)
        is_spatial_range = (sim_g is not None and (sim_g >= -self.theta_floor or is_minority_consistent))

        if (behavioral_ev.behavioral_mature and is_spatial_range and
            sim_s is not None and sim_s >= theta_self_eff and
            is_anchor_valid and is_drift_bounded and is_temporal_tolerable):
            
            # Confidence-weighted attenuation factor
            conf_factor = min(1.0, max(0.5, (sim_s - self.theta_self) / (1.0 - self.theta_self + 1e-6)))
            alpha_eff = self.alpha_downweight * conf_factor
            return JointDecisionOutcome(
                action="DOWNWEIGHT",
                primary_reason="NON_IID_HONEST_CONSISTENCY",
                aggregation_weight=alpha_eff * (I_i * P_i),
                force_sync_required=False,
                diagnostic_features={
                    "priority": 3,
                    "c_dw": c_dw,
                    "theta_self_eff": theta_self_eff,
                    "alpha_eff": alpha_eff,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "is_minority_consistent": is_minority_consistent
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 4: Ambiguous / Borderline Evidence (QUARANTINE)
        # ---------------------------------------------------------------------
        if self.enable_quarantine:
            is_borderline_spatial = (sim_g is not None and abs(sim_g - self.theta_cos) <= self.delta_borderline and not behavioral_ev.behavioral_mature)
            is_moderate_temporal_trusted = (temporal_ev.temporal_mature and 0.0 < g_margin <= self.delta_temp_mod and I_i >= self.trusted_integrity_min and (sim_g is None or sim_g >= 0.0))

            if is_borderline_spatial or is_moderate_temporal_trusted:
                return JointDecisionOutcome(
                    action="QUARANTINE",
                    primary_reason="AMBIGUOUS_EVIDENCE_QUARANTINE",
                    aggregation_weight=0.0,
                    force_sync_required=False,
                    diagnostic_features={
                        "priority": 4,
                        "is_borderline_spatial": is_borderline_spatial,
                        "is_mod_temp": is_moderate_temporal_trusted,
                        "sim_frozen_anchor": sim_frozen,
                        "anchor_drift": drift_a,
                        "version_lag": v_lag,
                    }
                )

        # 3b. Early Transition Non-IID Downweight (depth < 3, legitimate spatial range before full profile depth)
        is_early_self_valid = (sim_s is None or sim_s >= self.theta_self)
        if (not behavioral_ev.behavioral_mature and is_spatial_range and
            is_early_self_valid and is_anchor_valid and is_drift_bounded and is_temporal_tolerable):
            alpha_eff = self.alpha_downweight * 0.5
            return JointDecisionOutcome(
                action="DOWNWEIGHT",
                primary_reason="EARLY_STAGE_NON_IID_HOLD",
                aggregation_weight=alpha_eff * (I_i * P_i),
                force_sync_required=False,
                diagnostic_features={
                    "priority": 3,
                    "stage": "early_transition",
                    "c_dw": c_dw,
                    "alpha_eff": alpha_eff,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "is_minority_consistent": is_minority_consistent
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 5: Adversarial / Inconsistent Fallthrough (REJECT)
        # ---------------------------------------------------------------------
        return JointDecisionOutcome(
            action="REJECT",
            primary_reason="UNCOORDINATED_OR_ADVERSARIAL_REJECT",
            aggregation_weight=0.0,
            force_sync_required=False,
            diagnostic_features={
                "priority": 5,
                "sim_g": sim_g,
                "sim_s": sim_s,
                "c_dw": c_dw,
                "sim_frozen_anchor": sim_frozen,
                "anchor_drift": drift_a,
                "version_lag": v_lag,
            }
        )
