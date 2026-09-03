from typing import Optional, Dict, Any
import math
import numpy as np
import torch

from shared.types import (TemporalEvidence, SpatialEvidence, BehavioralEvidence,
                          JointDecisionOutcome)


class JointDecisionEngine:
    """Deterministic Multi-Criteria Joint Decision Engine for BDSF-AFL.
    
    Evaluates continuous temporal, spatial, behavioral trajectory, and reputation evidence
    through a strictly ordered, multi-manifold State-Maturity Priority hierarchy:
      - Phase 0: Universal Hard Safety Pre-Checks (Zero norm, Norm Explosion > 3x C_t)
      - Phase 1: Observation Warmup Horizon (Rounds 0 - warmup_rounds / 300 updates)
      - Phase 2: Post-Warmup Active Defense:
          - Priority 1: Invariant Hard Violations (Global Directional Inversion, Temporal Spam/Straggler)
          - Priority 2: Full Consensus Acceptance (sim_g >= theta_cos, clean TRS & suspicion)
          - Priority 3: Trajectory Rigidity Rejection (TRS >= trs_reject_thresh — Primary Mimicry/Compound Defense)
          - Priority 4: Non-IID Honest Soft-Filtering (DOWNWEIGHT with dynamic attenuation & drift bounding)
          - Priority 5: Ambiguous / Borderline Quarantine
          - Priority 6: Adversarial Fallthrough (REJECT)
    """

    def __init__(self, config: dict):
        self.config = config
        self.theta_cos: float = config.get("theta_cos", 0.10)
        self.theta_self: float = config.get("theta_self", 0.30)
        self.theta_floor: float = config.get("theta_floor", 0.40)
        self.theta_anchor_min: float = config.get("theta_anchor_min", 0.25)
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
        self.norm_anomaly_threshold: float = float(config.get("norm_anomaly_threshold", 1.60))

        # --- Warmup rounds ---
        self.warmup_rounds: int = int(config.get("spatial_warmup_rounds", config.get("warmup_rounds", 300)))

        # --- Trajectory Rigidity & Variance thresholds ---
        self.trs_reject_thresh: float = float(config.get("trs_reject_thresh", 0.85))
        self.trs_warn_thresh: float = float(config.get("trs_warn_thresh", 0.80))
        self.trs_safe_thresh: float = float(config.get("trs_safe_thresh", 0.70))
        self.trs_min_depth: int = int(config.get("trs_min_depth", 5))
        self.trs_accept_thresh: float = float(config.get("trs_accept_thresh", self.trs_reject_thresh))

        # --- Pairwise Residual Coherence (PRC) & TRA ---
        self.theta_prc: float = config.get("theta_prc", 0.20)
        self.theta_prc_hard: float = config.get("theta_prc_hard", -0.10)
        self.theta_tra: float = float(config.get("theta_tra", 0.45))

        # --- Multi-Round Suspicion Accumulator ---
        self.suspicion_decay: float = float(config.get("suspicion_decay", 0.50))
        self.suspicion_step: float = float(config.get("suspicion_step", 0.15))
        self.suspicion_reject_thresh: float = float(config.get("suspicion_reject_thresh", 0.65))
        self.suspicion_scores: dict[int, float] = {}

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
        """Deterministically evaluates candidate update evidence against Priority 0-6 hierarchy."""
        
        sim_g = spatial_ev.sim_global
        sim_s_mean = behavioral_ev.sim_self_mean
        sim_s_max = behavioral_ev.sim_self_max
        if sim_s_mean is not None and sim_s_max is not None:
            sim_s = max(sim_s_mean, sim_s_max)
        elif sim_s_mean is not None:
            sim_s = sim_s_mean
        else:
            sim_s = sim_s_max
        norm_r = spatial_ev.norm_raw
        C_t = spatial_ev.dynamic_bound_C if spatial_ev.dynamic_bound_C is not None else self.static_clip_C
        g_margin = temporal_ev.fence_margin
        depth = behavioral_ev.history_depth

        v_lag = version_lag if version_lag != 0 else getattr(temporal_ev, "version_lag", 0)
        # Asynchronous Staleness Penalty Matrix: dampens variance window of delayed gradients
        staleness_factor = 1.0 / math.sqrt(1.0 + v_lag)
        sim_frozen = getattr(behavioral_ev, "sim_frozen_anchor", None)
        drift_a = getattr(behavioral_ev, "anchor_drift", None)
        gdv = getattr(behavioral_ev, "gdv_score", None)
        dbp = getattr(behavioral_ev, "dbp_score", None)
        trs = getattr(behavioral_ev, "trs_score", None)
        tra = spatial_ev.tra_score
        prc = spatial_ev.prc_score

        # Multi-Round Suspicion Accumulator
        # Suspicion accumulates ONLY when a client is out of consensus or exhibiting anomalous residuals
        is_consensus_aligned = (
            (sim_g is not None and sim_g >= self.theta_cos) and
            (prc is None or prc >= self.theta_prc) and
            (tra is None or tra >= self.theta_tra) and
            (trs is None or depth < self.trs_min_depth or trs < self.trs_warn_thresh)
        )
        if is_consensus_aligned:
            self.suspicion_scores[cid] = max(0.0, self.suspicion_scores.get(cid, 0.0) * self.suspicion_decay - 0.05)
        elif trs is not None and gdv is not None and trs >= self.trs_warn_thresh and gdv <= 0.10:
            # Out-of-consensus persistent rigid steering
            self.suspicion_scores[cid] = min(1.0, self.suspicion_scores.get(cid, 0.0) + self.suspicion_step)
        elif tra is not None and tra < self.theta_tra:
            self.suspicion_scores[cid] = min(1.0, self.suspicion_scores.get(cid, 0.0) + self.suspicion_step)
        elif prc is not None and prc < self.theta_prc:
            self.suspicion_scores[cid] = min(1.0, self.suspicion_scores.get(cid, 0.0) + self.suspicion_step)
        else:
            self.suspicion_scores[cid] = max(0.0, self.suspicion_scores.get(cid, 0.0) * self.suspicion_decay - 0.02)
        S_i = self.suspicion_scores.get(cid, 0.0)
        spatial_ev.suspicion_score = S_i

        # ---------------------------------------------------------------------
        # PHASE 0: Universal Safety Pre-Checks (Active ALWAYS, even during warmup)
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
        # PHASE 1: Warmup Horizon (Reference & Dossier Construction)
        # ---------------------------------------------------------------------
        if current_round < self.warmup_rounds or not spatial_ev.spatial_mature:
            return JointDecisionOutcome(
                action="ACCEPT",
                primary_reason="SPATIAL_WARMUP_ACCEPT",
                aggregation_weight=self.warmup_weight_factor * (I_i * P_i) * staleness_factor,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 0,
                    "stage": "spatial_warmup",
                    "spatial_mature": spatial_ev.spatial_mature,
                    "current_round": current_round,
                    "ref_count": spatial_ev.spatial_reference_count,
                    "coherence": spatial_ev.spatial_coherence,
                    "version_lag": v_lag,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "trs_score": trs,
                    "gdv_score": gdv,
                    "dbp_score": dbp,
                }
            )

        # ---------------------------------------------------------------------
        # PHASE 2: Post-Warmup Active Multi-Criteria Defense
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # PRIORITY 1: Invariant Hard Violations (Drop & Slash Immediately)
        # ---------------------------------------------------------------------
        # 1a. Severe Directional Inversion
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

        # 1b. Extreme Temporal Violations
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
                    force_sync_required=True,
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
        is_anchor_valid = (behavioral_ev.sim_anchor is None or behavioral_ev.sim_anchor >= self.theta_anchor_min or is_spatial_valid)
        is_temporal_valid = (not temporal_ev.temporal_mature or g_margin <= 0.20)
        is_suspicion_clean = (S_i < 0.30)
        is_prc_valid = (prc is None or prc >= self.theta_prc)
        is_trs_clean = (trs is None or depth < self.trs_min_depth or trs < self.trs_accept_thresh)

        if (is_spatial_valid and is_self_valid and is_anchor_valid and is_temporal_valid and
            is_suspicion_clean and is_prc_valid and is_trs_clean):
            return JointDecisionOutcome(
                action="ACCEPT",
                primary_reason="FULL_CONSENSUS_ACCEPT",
                aggregation_weight=1.0 * (I_i * P_i) * staleness_factor,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 2,
                    "sim_g": sim_g,
                    "sim_s": sim_s,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "trs_score": trs,
                    "gdv_score": gdv,
                    "dbp_score": dbp,
                    "prc_score": prc,
                    "tra_score": tra,
                    "suspicion_score": S_i,
                    "is_trs_clean": is_trs_clean,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 3: Trajectory Rigidity Rejection (Primary Mimicry/Compound Defense)
        # ---------------------------------------------------------------------
        # Triggers when update is out of consensus or anomalous AND exhibits rigid directional steering
        if trs is not None and trs >= self.trs_reject_thresh and depth >= self.trs_min_depth:
            return JointDecisionOutcome(
                action="REJECT",
                primary_reason="TRAJECTORY_RIGIDITY_REJECT",
                aggregation_weight=0.0,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 3,
                    "trs_score": trs,
                    "gdv_score": gdv,
                    "dbp_score": dbp,
                    "depth": depth,
                    "suspicion_score": S_i,
                    "sim_g": sim_g,
                    "version_lag": v_lag,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "is_trs_clean": False,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 4: Non-IID Honest Soft-Filtering & Moderate Jitter (DOWNWEIGHT)
        # ---------------------------------------------------------------------
        c_dw = behavioral_ev.consecutive_dw
        theta_self_eff = self.theta_self + min(self.delta_theta_max, c_dw * self.delta_theta_step)
        is_anchor_valid_p4 = (behavioral_ev.sim_anchor is None or behavioral_ev.sim_anchor >= self.theta_anchor_min or (sim_g is not None and sim_g >= self.theta_cos))
        sim_a = behavioral_ev.sim_anchor
        is_minority_consistent = (sim_a is not None and sim_a >= self.theta_anchor_min)
        is_drift_bounded = (c_dw < self.K_drift_max) or is_minority_consistent
        is_temporal_tolerable = (not temporal_ev.temporal_mature or g_margin <= self.delta_temp_mod)
        is_spatial_range = (sim_g is not None and (sim_g >= -self.theta_floor or is_minority_consistent))
        is_trs_tolerable = (trs is None or trs <= self.trs_warn_thresh or is_minority_consistent)

        if (behavioral_ev.behavioral_mature and is_spatial_range and
            sim_s is not None and sim_s >= theta_self_eff and
            is_anchor_valid_p4 and is_drift_bounded and is_temporal_tolerable and is_trs_tolerable):
            
            # Persistent multi-round suspicion triggers soft reject
            if S_i >= self.suspicion_reject_thresh and (trs is None or trs >= 0.72):
                primary_reason = "TEMPORAL_RESIDUAL_INCOHERENCE_REJECT" if trs is None else "PROGRESSIVE_SUSPICION_TRAJECTORY_REJECT"
                return JointDecisionOutcome(
                    action="REJECT",
                    primary_reason=primary_reason,
                    aggregation_weight=0.0,
                    force_sync_required=False,
                    diagnostic_features={
                        "priority": 5,
                        "trs_score": trs,
                        "gdv_score": gdv,
                        "dbp_score": dbp,
                        "suspicion_score": S_i,
                        "sim_g": sim_g,
                        "sim_s": sim_s,
                        "version_lag": v_lag,
                    }
                )

            conf_factor = min(1.0, max(0.2, (1.0 - S_i) * (sim_s - self.theta_self) / (1.0 - self.theta_self + 1e-6)))
            alpha_eff = self.alpha_downweight * conf_factor
            reason = "PROGRESSIVE_SUSPICION_DOWNWEIGHT" if S_i >= 0.30 else "NON_IID_HONEST_CONSISTENCY"
            return JointDecisionOutcome(
                action="DOWNWEIGHT",
                primary_reason=reason,
                aggregation_weight=alpha_eff * (I_i * P_i) * staleness_factor,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 4,
                    "c_dw": c_dw,
                    "theta_self_eff": theta_self_eff,
                    "alpha_eff": alpha_eff,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "is_minority_consistent": is_minority_consistent,
                    "trs_score": trs,
                    "gdv_score": gdv,
                    "dbp_score": dbp,
                    "tra_score": tra,
                    "suspicion_score": S_i,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 5: Ambiguous / Borderline Evidence (QUARANTINE)
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
                        "priority": 5,
                        "is_borderline_spatial": is_borderline_spatial,
                        "is_mod_temp": is_moderate_temporal_trusted,
                        "sim_frozen_anchor": sim_frozen,
                        "anchor_drift": drift_a,
                        "version_lag": v_lag,
                        "trs_score": trs,
                    }
                )

        # Early Transition Non-IID Downweight (depth < 3)
        is_early_self_valid = (sim_s is None or sim_s >= self.theta_self)
        if (not behavioral_ev.behavioral_mature and is_spatial_range and
            is_early_self_valid and is_anchor_valid_p4 and is_drift_bounded and is_temporal_tolerable and is_trs_tolerable):
            alpha_eff = self.alpha_downweight * 0.5
            return JointDecisionOutcome(
                action="DOWNWEIGHT",
                primary_reason="EARLY_STAGE_NON_IID_HOLD",
                aggregation_weight=alpha_eff * (I_i * P_i) * staleness_factor,
                force_sync_required=False,
                diagnostic_features={
                    "priority": 4,
                    "stage": "early_transition",
                    "c_dw": c_dw,
                    "alpha_eff": alpha_eff,
                    "sim_anchor": behavioral_ev.sim_anchor,
                    "sim_frozen_anchor": sim_frozen,
                    "anchor_drift": drift_a,
                    "version_lag": v_lag,
                    "is_minority_consistent": is_minority_consistent,
                    "trs_score": trs,
                    "tra_score": tra,
                    "suspicion_score": S_i,
                }
            )

        # ---------------------------------------------------------------------
        # PRIORITY 6: Adversarial / Inconsistent Fallthrough (REJECT & SLASH)
        # ---------------------------------------------------------------------
        if trs is not None and trs >= self.trs_reject_thresh:
            primary_reason = "TRAJECTORY_RIGIDITY_REJECT"
        elif S_i >= self.suspicion_reject_thresh:
            primary_reason = "PROGRESSIVE_SUSPICION_TRAJECTORY_REJECT"
        else:
            primary_reason = "UNCOORDINATED_OR_ADVERSARIAL_REJECT"

        return JointDecisionOutcome(
            action="REJECT",
            primary_reason=primary_reason,
            aggregation_weight=0.0,
            force_sync_required=False,
            diagnostic_features={
                "priority": 6,
                "sim_g": sim_g,
                "sim_s": sim_s,
                "c_dw": c_dw,
                "sim_frozen_anchor": sim_frozen,
                "anchor_drift": drift_a,
                "version_lag": v_lag,
                "trs_score": trs,
                "gdv_score": gdv,
                "dbp_score": dbp,
                "tra_score": tra,
                "prc_score": prc,
                "suspicion_score": S_i,
            }
        )

    def get_state(self) -> dict:
        """Serializes decision engine state for checkpoint equivalence."""
        return {
            "suspicion_scores": {int(k): float(v) for k, v in self.suspicion_scores.items()},
        }

    def load_state(self, state: dict) -> None:
        """Restores decision engine state from checkpoint."""
        self.suspicion_scores = {int(k): float(v) for k, v in state.get("suspicion_scores", {}).items()}
