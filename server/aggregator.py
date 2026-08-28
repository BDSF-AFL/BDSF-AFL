import time
import os
from collections import deque
from typing import Optional

import torch
import numpy as np

from shared.types import (UpdateSubmission, AcceptedEntry,
                          ForceSyncPayload, ClientRegistration)
from server.temporal_filter import TemporalFilter
from server.force_sync import ForceSyncDispatcher
from server.reputation_manager import ReputationManager
from server.spatial_validator import SpatialValidator
from server.behavioral_memory import BehavioralMemoryManager
from server.decision_engine import JointDecisionEngine
from server.quarantine_manager import QuarantineManager
from utils.logger import BDSFLogger


class AggregatorServer:
    """Central orchestrator for the BDSF-AFL defense pipeline.

    Receives ``UpdateSubmission`` objects from clients and runs the full
    12-step decision pipeline:
        temporal gate -> spatial cosine check -> adaptive clip ->
        reputation-weighted merge -> recovery -> log.

    Maintains global model weights, per-client registry, and the
    accepted gradient buffer shared with ``SpatialValidator``.
    """

    def __init__(
        self,
        config: dict,
        W_init: torch.Tensor,
        client_ids: list[int],
        logger: BDSFLogger,
    ) -> None:
        # Step 1: Store config and logger
        self.config = config
        self.logger = logger

        # Step 2: Current global model (flattened 1D float32)
        self.W_global: torch.Tensor = W_init.clone().float()
        self.server_momentum: float = float(config.get("server_momentum", 0.90))
        self.v_momentum: torch.Tensor = torch.zeros_like(self.W_global)

        # Step 3: Client IDs
        self.client_ids = client_ids

        # Step 4: Number of clients
        N = len(client_ids)

        # Step 5: Compute burn-in count
        self.N_burn: int = max(4 * N, config.get("K_base", 50))

        # Step 6: Create a local config copy with burn_in_count for TemporalFilter
        tf_config = dict(config)
        tf_config["burn_in_count"] = self.N_burn

        # Step 7: Instantiate temporal filter
        self.temporal_filter = TemporalFilter(tf_config)

        # Step 8: Instantiate reputation manager
        self.rep_manager = ReputationManager(client_ids, config)

        # Step 9: Buffer size
        M = config.get("M", 30)

        # Step 10: Instantiate spatial validator
        self.spatial_validator = SpatialValidator(config)

        # Step 10b: Instantiate behavioral memory manager (Phase 2)
        self.behavioral_memory = BehavioralMemoryManager(config)

        # Step 10c: Instantiate Phase 3 Joint Decision Engine & Quarantine Manager
        self.decision_mode: str = config.get("decision_mode", "joint")
        self.decision_engine = JointDecisionEngine(config)
        self.quarantine_manager = QuarantineManager(config)

        # Step 11: Accepted buffer (shared concept with SpatialValidator)
        self.accepted_buffer: deque[AcceptedEntry] = deque(maxlen=M)

        # Step 12: Instantiate force-sync dispatcher
        self.force_sync_dispatcher = ForceSyncDispatcher()
        self.total_eval_time = 0.0

        # Step 13: Build client registry
        self.registry: dict[int, ClientRegistration] = {}
        for cid in client_ids:
            session_key = os.urandom(32)
            self.registry[cid] = ClientRegistration(
                client_id=cid,
                session_key=session_key,
                last_update_time=self.get_virtual_time(),
                pull_time=self.get_virtual_time(),
                is_byzantine=False,
            )

        # Step 14: Total accepted updates counter
        self.update_counter: int = 0

        # Step 15: Round number (incremented on each accepted update)
        self.round_number: int = 0

        # Step 15b: Monotonic server model version counter
        self.model_version: int = 0

        # Step 16: Baseline aggregation configs
        self.aggregation = config.get("aggregation", "bdsf_afl")
        self.sync = config.get("sync", False)
        self.sync_accumulator = {}
        self.grad_history = {cid: [] for cid in client_ids}

        # Step 17: Deadlock-breaking watchdog
        self.consecutive_rejects: int = 0
        self.deadlock_threshold: int = config.get("deadlock_threshold", len(client_ids))

    def get_model_version(self) -> int:
        """Returns the current monotonic global model version."""
        return self.model_version

    # ------------------------------------------------------------------
    # Main entry point — the 12-step pipeline
    # ------------------------------------------------------------------

    def handle_update(self, submission: UpdateSubmission) -> dict:
        """Process a single client update through the selected aggregation pipeline."""
        cid = submission.client_id
        reg = self.registry[cid]
        t_now = submission.t_submit
        g_i = t_now - reg.last_update_time
        I_i, P_i = self.rep_manager.get(cid)
        pulled_version = getattr(submission, "model_version_at_pull", 0)
        version_lag = max(0, self.model_version - pulled_version)

        if self.aggregation in ("fedavg", "fedprox"):
            if self.sync:
                # Synchronous aggregation
                self.sync_accumulator[cid] = submission.delta_W.clone()
                
                if len(self.sync_accumulator) == len(self.client_ids):
                    reg.last_update_time = t_now
                    avg_delta = torch.stack(list(self.sync_accumulator.values())).mean(dim=0)
                    self.W_global = self.W_global + self.config.get("eta", 0.01) * avg_delta
                    self.sync_accumulator.clear()
                    
                    self._log_update(
                        round=self.round_number, client_id=cid,
                        g_i=g_i, I_i=I_i, P_i=P_i,
                        status="ACCEPT", reason="SYNC_ACCUMULATE",
                    )
                    ret_val = {
                        "status": "ACCEPT",
                        "reason": "SYNC_ACCUMULATE",
                        "force_sync": None,
                        "round": self.round_number,
                        "I_i": I_i,
                        "P_i": P_i,
                    }
                    self.update_counter += 1
                    self.round_number += 1
                    return ret_val
                else:
                    reg.last_update_time = t_now
                    self._log_update(
                        round=self.round_number, client_id=cid,
                        g_i=g_i, I_i=I_i, P_i=P_i,
                        status="ACCEPT", reason="SYNC_ACCUMULATE",
                    )
                    return {
                        "status": "ACCEPT",
                        "reason": "SYNC_ACCUMULATE",
                        "force_sync": None,
                        "round": self.round_number,
                        "I_i": I_i,
                        "P_i": P_i,
                    }
            else:
                # Asynchronous FedAvg
                reg.last_update_time = t_now
                self.W_global = self.W_global + self.config.get("eta", 0.01) * submission.delta_W
                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, I_i=I_i, P_i=P_i,
                    status="ACCEPT", reason="ASYNC_FEDAVG",
                )
                ret_val = {
                    "status": "ACCEPT",
                    "reason": "ASYNC_FEDAVG",
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }
                self.update_counter += 1
                self.round_number += 1
                return ret_val

        elif self.aggregation == "afl_unconstrained":
            reg.last_update_time = t_now
            self.W_global = self.W_global + self.config.get("eta", 0.01) * submission.delta_W
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=I_i, P_i=P_i,
                status="ACCEPT", reason="UNCONSTRAINED_AFL",
            )
            ret_val = {
                "status": "ACCEPT",
                "reason": "UNCONSTRAINED_AFL",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }
            self.update_counter += 1
            self.round_number += 1
            return ret_val

        elif self.aggregation == "static_delay_afl":
            tau_max = self.config.get("static_tau_max", 5.0)
            s_i = t_now - submission.tau
            if s_i > tau_max:
                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, I_i=I_i, P_i=P_i,
                    status="REJECT", reason="STATIC_DELAY_EXCEEDED",
                )
                return {
                    "status": "REJECT",
                    "reason": "STATIC_DELAY_EXCEEDED",
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }
            reg.last_update_time = t_now
            self.W_global = self.W_global + self.config.get("eta", 0.01) * submission.delta_W
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=I_i, P_i=P_i,
                status="ACCEPT", reason="STATIC_DELAY_AFL",
            )
            ret_val = {
                "status": "ACCEPT",
                "reason": "STATIC_DELAY_AFL",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }
            self.update_counter += 1
            self.round_number += 1
            return ret_val

        elif self.aggregation == "pure_cosine":
            passes_cosine = self.spatial_validator.cosine_check(submission.delta_W)
            spat_ev = self.spatial_validator.extract_evidence(submission.delta_W)
            if not passes_cosine:
                # Fix: same cascade-prevention fix as in the bdsf_afl pipeline.
                reg.last_update_time = t_now
                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, I_i=I_i, P_i=P_i,
                    status="REJECT", reason="SPATIAL_COSINE",
                    sim_global=spat_ev.sim_global,
                    norm_ratio=spat_ev.norm_ratio_median,
                    weight=0.0, action="REJECT",
                )
                return {
                    "status": "REJECT",
                    "reason": "SPATIAL_COSINE",
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }
            reg.last_update_time = t_now
            delta_W_clipped = self.spatial_validator.adaptive_clip(submission.delta_W)
            self.W_global = self.W_global + self.config.get("eta", 0.01) * delta_W_clipped
            entry = AcceptedEntry(
                delta_W=delta_W_clipped.clone(),
                I_score=1.0,
                P_score=1.0,
                client_id=cid,
            )
            self.accepted_buffer.append(entry)
            self.spatial_validator.on_accept(entry)
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=1.0, P_i=1.0,
                status="ACCEPT", reason="PURE_COSINE",
                sim_global=spat_ev.sim_global,
                norm_ratio=spat_ev.norm_ratio_median,
                weight=1.0, action="ACCEPT",
            )
            ret_val = {
                "status": "ACCEPT",
                "reason": "PURE_COSINE",
                "force_sync": None,
                "round": self.round_number,
                "I_i": 1.0,
                "P_i": 1.0,
            }
            self.update_counter += 1
            self.round_number += 1
            return ret_val

        elif self.aggregation == "foolsgold":
            self.grad_history[cid].append(submission.delta_W.clone())
            summed_hist = {}
            for c_id in self.client_ids:
                if self.grad_history[c_id]:
                    summed_hist[c_id] = torch.stack(self.grad_history[c_id]).sum(dim=0)
                else:
                    summed_hist[c_id] = torch.zeros_like(submission.delta_W)
            v_i = summed_hist[cid]
            norm_i = torch.norm(v_i).item()
            contrib_sim = 0.0
            if norm_i > 1e-9:
                for other_id in self.client_ids:
                    if other_id == cid:
                        continue
                    v_j = summed_hist[other_id]
                    norm_j = torch.norm(v_j).item()
                    if norm_j > 1e-9:
                        sim = torch.dot(v_i, v_j).item() / (norm_i * norm_j)
                        contrib_sim = max(contrib_sim, sim)
            multiplier = max(0.0, min(1.0, 1.0 - contrib_sim))
            reg.last_update_time = t_now
            self.W_global = self.W_global + self.config.get("eta", 0.01) * multiplier * submission.delta_W
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=multiplier, P_i=1.0,
                status="ACCEPT", reason="FOOLSGOLD",
            )
            ret_val = {
                "status": "ACCEPT",
                "reason": "FOOLSGOLD",
                "force_sync": None,
                "round": self.round_number,
                "I_i": multiplier,
                "P_i": 1.0,
            }
            self.update_counter += 1
            self.round_number += 1
            return ret_val

        # --- BDSF-AFL Proposed Pipeline (Step 1-12) ---
        if self.decision_mode == "joint":
            self.temporal_filter.step_seen()

        # --- Evidence Extraction (Observability Layer) ---
        temporal_evidence = self.temporal_filter.extract_evidence(g_i, cid, version_lag=version_lag)
        spatial_evidence = self.spatial_validator.extract_evidence(submission.delta_W)
        client_gaps = self.temporal_filter.client_gap_history.get(cid, [])
        behavioral_evidence = self.behavioral_memory.extract_evidence(
            client_id=cid,
            delta_W=submission.delta_W,
            g_i=g_i,
            client_gap_history=client_gaps,
        )

        # --------------------------------------------------------------
        # PHASE 3: JOINT DECISION PIPELINE (when decision_mode == "joint")
        # --------------------------------------------------------------
        if self.decision_mode == "joint":
            eff_round = self.round_number // max(1, self.config.get("N_clients", 20))
            outcome = self.decision_engine.evaluate(
                cid=cid,
                temporal_ev=temporal_evidence,
                spatial_ev=spatial_evidence,
                behavioral_ev=behavioral_evidence,
                I_i=I_i,
                P_i=P_i,
                current_round=eff_round,
            )

            delta_W_clipped = self.spatial_validator.adaptive_clip(submission.delta_W)
            eta = self.config.get("eta", 1.0)

            # 1. Handle ACCEPT Action (Full Consensus / Warm-Up)
            if outcome.action == "ACCEPT":
                weight = outcome.aggregation_weight
                self._apply_global_update(eta * weight * delta_W_clipped)

                is_warmup = (outcome.primary_reason in ["SPATIAL_WARMUP_ACCEPT", "BURN_IN_ACCEPT"])

                entry = AcceptedEntry(
                    delta_W=delta_W_clipped.clone(),
                    I_score=I_i,
                    P_score=P_i,
                    client_id=cid,
                    is_warmup=is_warmup,
                )
                self.accepted_buffer.append(entry)
                self.spatial_validator.on_accept(entry)
                self.temporal_filter.record_gap(g_i, cid)

                # Warmup updates build the client's historical trajectory & genesis anchor
                self.behavioral_memory.on_accept(cid, delta_W_clipped, is_downweight=False)
                if not is_warmup:
                    self.rep_manager.record_accepted_update(cid)
                    self.rep_manager.recover(cid)

                I_i, P_i = self.rep_manager.get(cid)
                reg.last_update_time = t_now

                # Re-evaluate any pending quarantined updates against fresh reference
                resolved_q = self.quarantine_manager.re_evaluate_pending(
                    self.round_number, self.spatial_validator._build_reference(), self.decision_engine.theta_cos
                )
                for q_entry, q_act, q_age_mult in resolved_q:
                    if q_act == "ACCEPT":
                        q_w = q_age_mult * (q_entry.reputation_at_entry[0] * q_entry.reputation_at_entry[1])
                        self._apply_global_update(eta * q_w * q_entry.delta_W_clipped)
                        self._log_update(
                            round=self.round_number, client_id=q_entry.client_id,
                            status="ACCEPT", reason="QUARANTINE_RELEASE_ACCEPT",
                            version_lag=version_lag,
                            weight=q_w, action="ACCEPT",
                            quarantine_depth=self.quarantine_manager.depth,
                        )
                    elif q_act == "REJECT":
                        self._log_update(
                            round=self.round_number, client_id=q_entry.client_id,
                            status="REJECT", reason="QUARANTINE_EXPIRED_REJECT",
                            version_lag=version_lag,
                            weight=None, action="REJECT",
                            quarantine_depth=self.quarantine_manager.depth,
                        )

                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                    status="ACCEPT", reason=outcome.primary_reason,
                    lower_fence=temporal_evidence.lower_fence,
                    upper_fence=temporal_evidence.upper_fence,
                    fence_margin=temporal_evidence.fence_margin,
                    client_z_score=temporal_evidence.client_z_score,
                    is_burn_in=temporal_evidence.is_burn_in,
                    spatial_mature=spatial_evidence.spatial_mature,
                    temporal_mature=temporal_evidence.temporal_mature,
                    behavioral_mature=behavioral_evidence.behavioral_mature,
                    spatial_ref_count=spatial_evidence.spatial_reference_count,
                    spatial_coherence=spatial_evidence.spatial_coherence,
                    sim_global=spatial_evidence.sim_global,
                    norm_raw=spatial_evidence.norm_raw,
                    norm_clipped=spatial_evidence.norm_clipped,
                    norm_ratio_median=spatial_evidence.norm_ratio_median,
                    dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                    reference_available=spatial_evidence.reference_available,
                    weight=weight, action="ACCEPT",
                    sim_self_mean=behavioral_evidence.sim_self_mean,
                    sim_self_max=behavioral_evidence.sim_self_max,
                    norm_deviation_self=behavioral_evidence.norm_deviation_self,
                    cadence_consistency=behavioral_evidence.cadence_consistency,
                    history_depth=behavioral_evidence.history_depth,
                    sim_anchor=behavioral_evidence.sim_anchor,
                    sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                    anchor_drift=behavioral_evidence.anchor_drift,
                    consecutive_dw=behavioral_evidence.consecutive_dw,
                    quarantine_depth=self.quarantine_manager.depth,
                )
                ret_val = {
                    "status": "ACCEPT",
                    "reason": outcome.primary_reason,
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }
                self.consecutive_rejects = 0
                self.update_counter += 1
                self.round_number += 1
                return ret_val

            # 2. Handle DOWNWEIGHT Action (Non-IID Honest Soft-Filtering)
            elif outcome.action == "DOWNWEIGHT":
                weight = outcome.aggregation_weight
                self._apply_global_update(eta * weight * delta_W_clipped)

                entry = AcceptedEntry(
                    delta_W=delta_W_clipped.clone(),
                    I_score=I_i,
                    P_score=P_i,
                    client_id=cid,
                )
                self.accepted_buffer.append(entry)
                self.spatial_validator.on_accept(entry)
                self.behavioral_memory.on_accept(cid, delta_W_clipped, is_downweight=True)
                self.temporal_filter.record_gap(g_i, cid)

                # Neutral Hold: Reset spatial rejection streak, no integrity slash, no additive recovery
                self.rep_manager.record_accepted_update(cid)
                I_i, P_i = self.rep_manager.get(cid)
                reg.last_update_time = t_now

                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                    status="DOWNWEIGHT", reason=outcome.primary_reason,
                    lower_fence=temporal_evidence.lower_fence,
                    upper_fence=temporal_evidence.upper_fence,
                    fence_margin=temporal_evidence.fence_margin,
                    client_z_score=temporal_evidence.client_z_score,
                    is_burn_in=temporal_evidence.is_burn_in,
                    spatial_mature=spatial_evidence.spatial_mature,
                    temporal_mature=temporal_evidence.temporal_mature,
                    behavioral_mature=behavioral_evidence.behavioral_mature,
                    spatial_ref_count=spatial_evidence.spatial_reference_count,
                    spatial_coherence=spatial_evidence.spatial_coherence,
                    sim_global=spatial_evidence.sim_global,
                    norm_raw=spatial_evidence.norm_raw,
                    norm_clipped=spatial_evidence.norm_clipped,
                    norm_ratio_median=spatial_evidence.norm_ratio_median,
                    dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                    reference_available=spatial_evidence.reference_available,
                    weight=weight, action="DOWNWEIGHT",
                    sim_self_mean=behavioral_evidence.sim_self_mean,
                    sim_self_max=behavioral_evidence.sim_self_max,
                    norm_deviation_self=behavioral_evidence.norm_deviation_self,
                    cadence_consistency=behavioral_evidence.cadence_consistency,
                    history_depth=behavioral_evidence.history_depth,
                    sim_anchor=behavioral_evidence.sim_anchor,
                    sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                    anchor_drift=behavioral_evidence.anchor_drift,
                    consecutive_dw=behavioral_evidence.consecutive_dw,
                    quarantine_depth=self.quarantine_manager.depth,
                )
                ret_val = {
                    "status": "DOWNWEIGHT",
                    "reason": outcome.primary_reason,
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }
                self.consecutive_rejects = 0
                self.update_counter += 1
                self.round_number += 1
                return ret_val

            # 3. Handle QUARANTINE Action (Ambiguous / Borderline Evidence)
            elif outcome.action == "QUARANTINE":
                self.quarantine_manager.enqueue(
                    client_id=cid,
                    delta_W_clipped=delta_W_clipped,
                    current_round=self.round_number,
                    virtual_time=t_now,
                    reputation=(I_i, P_i),
                    reason=outcome.primary_reason,
                )
                reg.last_update_time = t_now

                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                    status="QUARANTINE", reason=outcome.primary_reason,
                    lower_fence=temporal_evidence.lower_fence,
                    upper_fence=temporal_evidence.upper_fence,
                    fence_margin=temporal_evidence.fence_margin,
                    client_z_score=temporal_evidence.client_z_score,
                    is_burn_in=temporal_evidence.is_burn_in,
                    spatial_mature=spatial_evidence.spatial_mature,
                    temporal_mature=temporal_evidence.temporal_mature,
                    behavioral_mature=behavioral_evidence.behavioral_mature,
                    spatial_ref_count=spatial_evidence.spatial_reference_count,
                    spatial_coherence=spatial_evidence.spatial_coherence,
                    sim_global=spatial_evidence.sim_global,
                    norm_raw=spatial_evidence.norm_raw,
                    norm_clipped=spatial_evidence.norm_clipped,
                    norm_ratio_median=spatial_evidence.norm_ratio_median,
                    dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                    reference_available=spatial_evidence.reference_available,
                    weight=0.0, action="QUARANTINE",
                    sim_self_mean=behavioral_evidence.sim_self_mean,
                    sim_self_max=behavioral_evidence.sim_self_max,
                    norm_deviation_self=behavioral_evidence.norm_deviation_self,
                    cadence_consistency=behavioral_evidence.cadence_consistency,
                    history_depth=behavioral_evidence.history_depth,
                    sim_anchor=behavioral_evidence.sim_anchor,
                    sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                    anchor_drift=behavioral_evidence.anchor_drift,
                    consecutive_dw=behavioral_evidence.consecutive_dw,
                    quarantine_depth=self.quarantine_manager.depth,
                )
                return {
                    "status": "QUARANTINE",
                    "reason": outcome.primary_reason,
                    "force_sync": None,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }

            # 4. Handle REJECT Action (Hard Violations / Adversarial Fallthrough)
            else:
                fs_payload = None
                if outcome.force_sync_required or outcome.primary_reason == "HARD_GUARD_TEMPORAL_STRAGGLER":
                    self.rep_manager.reduce_pace(cid)
                    fs_payload = self.force_sync_dispatcher.build_payload(
                        cid, self.W_global, reg.session_key, self.get_virtual_time()
                    )
                    reg.last_update_time = fs_payload.timestamp
                elif outcome.primary_reason == "HARD_GUARD_TEMPORAL_SPAM":
                    self.rep_manager.reduce_pace(cid)
                else:
                    self.rep_manager.record_spatial_rejection(cid)
                    reg.last_update_time = t_now

                I_i, P_i = self.rep_manager.get(cid)
                self._log_update(
                    round=self.round_number, client_id=cid,
                    g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                    status="REJECT", reason=outcome.primary_reason,
                    lower_fence=temporal_evidence.lower_fence,
                    upper_fence=temporal_evidence.upper_fence,
                    fence_margin=temporal_evidence.fence_margin,
                    client_z_score=temporal_evidence.client_z_score,
                    is_burn_in=temporal_evidence.is_burn_in,
                    spatial_mature=spatial_evidence.spatial_mature,
                    temporal_mature=temporal_evidence.temporal_mature,
                    behavioral_mature=behavioral_evidence.behavioral_mature,
                    spatial_ref_count=spatial_evidence.spatial_reference_count,
                    spatial_coherence=spatial_evidence.spatial_coherence,
                    sim_global=spatial_evidence.sim_global,
                    norm_raw=spatial_evidence.norm_raw,
                    norm_clipped=spatial_evidence.norm_clipped,
                    norm_ratio_median=spatial_evidence.norm_ratio_median,
                    dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                    reference_available=spatial_evidence.reference_available,
                    weight=None, action="REJECT",
                    sim_self_mean=behavioral_evidence.sim_self_mean,
                    sim_self_max=behavioral_evidence.sim_self_max,
                    norm_deviation_self=behavioral_evidence.norm_deviation_self,
                    cadence_consistency=behavioral_evidence.cadence_consistency,
                    history_depth=behavioral_evidence.history_depth,
                    sim_anchor=behavioral_evidence.sim_anchor,
                    sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                    anchor_drift=behavioral_evidence.anchor_drift,
                    consecutive_dw=behavioral_evidence.consecutive_dw,
                    quarantine_depth=self.quarantine_manager.depth,
                )
                self.consecutive_rejects += 1
                if self.consecutive_rejects >= self.deadlock_threshold:
                    # Watchdog: deadlock prevention is now handled by correct threshold calibration
                    # (Fix F1 in decision_engine.py). reset_buffer() was an exploit surface —
                    # coordinated Byzantine clients could trigger N consecutive rejects to wipe
                    # the spatial reference and inject during the blind rebuild window.
                    # Counter is reset for metrics tracking only; buffer is NOT flushed.
                    self.consecutive_rejects = 0

                return {
                    "status": "REJECT",
                    "reason": outcome.primary_reason,
                    "force_sync": fs_payload,
                    "round": self.round_number,
                    "I_i": I_i,
                    "P_i": P_i,
                }

        # --------------------------------------------------------------
        # LEGACY BDSF-AFL PIPELINE (when decision_mode == "legacy")
        # --------------------------------------------------------------
        # --- Step 1: Compute behavioral gap g_i ---
        # --- Step 3: Run temporal gate ---
        temporal_result = self.temporal_filter.evaluate(g_i, cid)

        # --- Step 4: Handle REJECT_HIGH_FREQ ---
        if temporal_result == "REJECT_HIGH_FREQ":
            self.rep_manager.reduce_pace(cid)
            I_i, P_i = self.rep_manager.get(cid)
            # Fix (Livelock): Do not reset last_update_time on high-frequency rejection.
            # Letting the gap accumulate prevents the client from being trapped in a loop.
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                status="REJECT", reason="TEMPORAL_HIGH_FREQ",
                lower_fence=temporal_evidence.lower_fence,
                upper_fence=temporal_evidence.upper_fence,
                fence_margin=temporal_evidence.fence_margin,
                client_z_score=temporal_evidence.client_z_score,
                is_burn_in=temporal_evidence.is_burn_in,
                spatial_mature=spatial_evidence.spatial_mature,
                temporal_mature=temporal_evidence.temporal_mature,
                behavioral_mature=behavioral_evidence.behavioral_mature,
                spatial_ref_count=spatial_evidence.spatial_reference_count,
                spatial_coherence=spatial_evidence.spatial_coherence,
                sim_global=spatial_evidence.sim_global,
                norm_raw=spatial_evidence.norm_raw,
                norm_clipped=spatial_evidence.norm_clipped,
                norm_ratio_median=spatial_evidence.norm_ratio_median,
                dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                reference_available=spatial_evidence.reference_available,
                weight=None, action="REJECT",
                sim_self_mean=behavioral_evidence.sim_self_mean,
                sim_self_max=behavioral_evidence.sim_self_max,
                norm_deviation_self=behavioral_evidence.norm_deviation_self,
                cadence_consistency=behavioral_evidence.cadence_consistency,
                history_depth=behavioral_evidence.history_depth,
                sim_anchor=behavioral_evidence.sim_anchor,
                sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                anchor_drift=behavioral_evidence.anchor_drift,
                consecutive_dw=behavioral_evidence.consecutive_dw,
                quarantine_depth=self.quarantine_manager.depth,
            )
            return {
                "status": "REJECT",
                "reason": "TEMPORAL_HIGH_FREQ",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 5: Handle REJECT_STRAGGLER ---
        if temporal_result == "REJECT_STRAGGLER":
            self.rep_manager.reduce_pace(cid)
            I_i, P_i = self.rep_manager.get(cid)
            fs_payload = self.force_sync_dispatcher.build_payload(
                cid, self.W_global, reg.session_key, self.get_virtual_time()
            )
            # Fix: synchronise the server-side timeline with the force_sync
            # timestamp we are about to send to the client.  Without this,
            # every subsequent g_i is measured from the stale T_prev and
            # always exceeds U, trapping the client in a permanent
            # reject → force-sync → reject livelock.
            reg.last_update_time = fs_payload.timestamp
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                status="REJECT", reason="TEMPORAL_STRAGGLER",
                lower_fence=temporal_evidence.lower_fence,
                upper_fence=temporal_evidence.upper_fence,
                fence_margin=temporal_evidence.fence_margin,
                client_z_score=temporal_evidence.client_z_score,
                is_burn_in=temporal_evidence.is_burn_in,
                spatial_mature=spatial_evidence.spatial_mature,
                temporal_mature=temporal_evidence.temporal_mature,
                behavioral_mature=behavioral_evidence.behavioral_mature,
                spatial_ref_count=spatial_evidence.spatial_reference_count,
                spatial_coherence=spatial_evidence.spatial_coherence,
                sim_global=spatial_evidence.sim_global,
                norm_raw=spatial_evidence.norm_raw,
                norm_clipped=spatial_evidence.norm_clipped,
                norm_ratio_median=spatial_evidence.norm_ratio_median,
                dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                reference_available=spatial_evidence.reference_available,
                weight=None, action="REJECT",
                sim_self_mean=behavioral_evidence.sim_self_mean,
                sim_self_max=behavioral_evidence.sim_self_max,
                norm_deviation_self=behavioral_evidence.norm_deviation_self,
                cadence_consistency=behavioral_evidence.cadence_consistency,
                history_depth=behavioral_evidence.history_depth,
                sim_anchor=behavioral_evidence.sim_anchor,
                sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                anchor_drift=behavioral_evidence.anchor_drift,
                consecutive_dw=behavioral_evidence.consecutive_dw,
                quarantine_depth=self.quarantine_manager.depth,
            )
            return {
                "status": "REJECT",
                "reason": "TEMPORAL_STRAGGLER",
                "force_sync": fs_payload,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 7: Spatial cosine check ---
        passes_cosine = self.spatial_validator.cosine_check(submission.delta_W)
        if not passes_cosine:
            # Spatial Grace Counter: Increment streak and slash only if streak >= grace limit
            self.rep_manager.record_spatial_rejection(cid)
            I_i, P_i = self.rep_manager.get(cid)
            # Fix: advance last_update_time so the next submission's g_i is
            # measured from t_now (≈ one training round) rather than from the
            # previous accepted update.  Without this, repeated spatial
            # rejections accumulate gap until g_i > U, cascading into the
            # permanent TEMPORAL_STRAGGLER livelock.
            reg.last_update_time = t_now
            self._log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
                status="REJECT", reason="SPATIAL_COSINE",
                lower_fence=temporal_evidence.lower_fence,
                upper_fence=temporal_evidence.upper_fence,
                fence_margin=temporal_evidence.fence_margin,
                client_z_score=temporal_evidence.client_z_score,
                is_burn_in=temporal_evidence.is_burn_in,
                spatial_mature=spatial_evidence.spatial_mature,
                temporal_mature=temporal_evidence.temporal_mature,
                behavioral_mature=behavioral_evidence.behavioral_mature,
                spatial_ref_count=spatial_evidence.spatial_reference_count,
                spatial_coherence=spatial_evidence.spatial_coherence,
                sim_global=spatial_evidence.sim_global,
                norm_raw=spatial_evidence.norm_raw,
                norm_clipped=spatial_evidence.norm_clipped,
                norm_ratio_median=spatial_evidence.norm_ratio_median,
                dynamic_bound_C=spatial_evidence.dynamic_bound_C,
                reference_available=spatial_evidence.reference_available,
                weight=None, action="REJECT",
                sim_self_mean=behavioral_evidence.sim_self_mean,
                sim_self_max=behavioral_evidence.sim_self_max,
                norm_deviation_self=behavioral_evidence.norm_deviation_self,
                cadence_consistency=behavioral_evidence.cadence_consistency,
                history_depth=behavioral_evidence.history_depth,
                sim_anchor=behavioral_evidence.sim_anchor,
                sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
                anchor_drift=behavioral_evidence.anchor_drift,
                consecutive_dw=behavioral_evidence.consecutive_dw,
                quarantine_depth=self.quarantine_manager.depth,
            )
            return {
                "status": "REJECT",
                "reason": "SPATIAL_COSINE",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 8: Adaptive L2 clipping ---
        delta_W_clipped = self.spatial_validator.adaptive_clip(submission.delta_W)

        # --- Step 9: Reputation-weighted merge ---
        eta = self.config.get("eta", 0.01)
        weight = I_i * P_i
        self.W_global = self.W_global + eta * weight * delta_W_clipped

        # --- Step 10: Append to accepted_buffer & behavioral memory ---
        entry = AcceptedEntry(
            delta_W=delta_W_clipped.clone(),
            I_score=I_i,
            P_score=P_i,
            client_id=cid,
        )
        self.accepted_buffer.append(entry)
        self.spatial_validator.on_accept(entry)
        self.behavioral_memory.on_accept(cid, delta_W_clipped)

        # --- Step 11: Reputation recovery ---
        # Borderline Suspicion Counter: Perform suspicion check on the exposed cosine similarity
        sim = self.spatial_validator.last_sim
        self.rep_manager.record_borderline_check(cid, sim)
        # Spatial Grace Counter: Reset the rejection streak since update is fully accepted
        self.rep_manager.record_accepted_update(cid)
        # Perform normal recovery
        self.rep_manager.recover(cid)
        # Retrieve final scores after checking borderline/grace conditions
        I_i, P_i = self.rep_manager.get(cid)

        # --- Step 12: Log and return under current round, then increment ---
        reg.last_update_time = t_now
        self._log_update(
            round=self.round_number, client_id=cid,
            g_i=g_i, version_lag=version_lag, I_i=I_i, P_i=P_i,
            status="ACCEPT", reason="FULL_ACCEPT",
            lower_fence=temporal_evidence.lower_fence,
            upper_fence=temporal_evidence.upper_fence,
            fence_margin=temporal_evidence.fence_margin,
            client_z_score=temporal_evidence.client_z_score,
            is_burn_in=temporal_evidence.is_burn_in,
            spatial_mature=spatial_evidence.spatial_mature,
            temporal_mature=temporal_evidence.temporal_mature,
            behavioral_mature=behavioral_evidence.behavioral_mature,
            spatial_ref_count=spatial_evidence.spatial_reference_count,
            spatial_coherence=spatial_evidence.spatial_coherence,
            sim_global=spatial_evidence.sim_global,
            norm_raw=spatial_evidence.norm_raw,
            norm_clipped=spatial_evidence.norm_clipped,
            norm_ratio_median=spatial_evidence.norm_ratio_median,
            dynamic_bound_C=spatial_evidence.dynamic_bound_C,
            reference_available=spatial_evidence.reference_available,
            weight=weight, action="ACCEPT",
            sim_self_mean=behavioral_evidence.sim_self_mean,
            sim_self_max=behavioral_evidence.sim_self_max,
            norm_deviation_self=behavioral_evidence.norm_deviation_self,
            cadence_consistency=behavioral_evidence.cadence_consistency,
            history_depth=behavioral_evidence.history_depth,
            sim_anchor=behavioral_evidence.sim_anchor,
            sim_frozen_anchor=behavioral_evidence.sim_frozen_anchor,
            anchor_drift=behavioral_evidence.anchor_drift,
            consecutive_dw=behavioral_evidence.consecutive_dw,
            quarantine_depth=self.quarantine_manager.depth,
        )
        ret_val = {
            "status": "ACCEPT",
            "reason": "FULL_ACCEPT",
            "force_sync": None,
            "round": self.round_number,
            "I_i": I_i,
            "P_i": P_i,
        }
        self.update_counter += 1
        self.round_number += 1
        return ret_val

    # ------------------------------------------------------------------
    # Auxiliary public methods
    # ------------------------------------------------------------------

    def get_global_weights(self) -> torch.Tensor:
        """Returns a clone of the current global model weights."""
        return self.W_global.clone()

    def register_client_ground_truth(
        self, client_id: int, is_byzantine: bool,
    ) -> None:
        """Called by SimulationEnvironment after construction.
        Sets the ground truth label used only by metrics/logger."""
        self.registry[client_id].is_byzantine = is_byzantine

    def get_session_key(self, client_id: int) -> bytes:
        """Returns the HMAC session key for a given client.
        Called by SimulationEnvironment to initialise each ClientNode."""
        return self.registry[client_id].session_key

    def update_pull_time(self, client_id: int, pull_time: float) -> None:
        """Called when a client pulls W_global. Stores the pull timestamp."""
        self.registry[client_id].pull_time = pull_time

    def get_virtual_time(self) -> float:
        """Returns the current virtual timeline time (excluding blocking evaluation periods)."""
        return time.time() - self.total_eval_time

    def _apply_global_update(self, effective_delta: torch.Tensor) -> None:
        """Applies momentum-enhanced asynchronous aggregation and increments model version.
        effective_delta is already scaled by eta and decision weight (1.0 for ACCEPT, 0.5 for DW).
        """
        if self.server_momentum > 0.0:
            self.v_momentum = self.server_momentum * self.v_momentum + effective_delta
            self.W_global = self.W_global + self.v_momentum
        else:
            self.W_global = self.W_global + effective_delta
        self.model_version += 1

    def get_momentum_norm(self) -> float:
        """Returns the L2 norm of the server momentum velocity vector."""
        if self.v_momentum is not None:
            return float(torch.norm(self.v_momentum).item())
        return 0.0

    def _log_update(self, **kwargs):
        if "v_momentum_norm" not in kwargs:
            kwargs["v_momentum_norm"] = self.get_momentum_norm()
        self.logger.log_update(**kwargs)

    def get_state(self) -> dict:
        """Serializes full reproducible server state for atomic checkpointing."""
        return {
            "W_global": self.W_global.clone().cpu(),
            "v_momentum": self.v_momentum.clone().cpu() if self.v_momentum is not None else None,
            "round_number": self.round_number,
            "update_counter": self.update_counter,
            "model_version": self.model_version,
            "rep_scores": {cid: {"I": s["I"], "P": s["P"]} for cid, s in self.rep_manager.scores.items()},
            "rep_history": {cid: list(h) for cid, h in self.rep_manager._history.items()},
            "spatial_streaks": dict(self.rep_manager.spatial_reject_streak),
            "borderline_streaks": dict(self.rep_manager.borderline_streak),
            "temporal_state": self.temporal_filter.get_state(),
            "spatial_state": self.spatial_validator.get_state(),
            "behavioral_profiles": self.behavioral_memory.get_state(),
            "quarantine_state": self.quarantine_manager.get_state(),
        }

    def load_state(self, state: dict) -> None:
        """Restores full server state from a saved checkpoint."""
        if "W_global" in state:
            self.W_global.copy_(state["W_global"])
        if "v_momentum" in state and state["v_momentum"] is not None and self.v_momentum is not None:
            self.v_momentum.copy_(state["v_momentum"])
        self.round_number = state.get("round_number", self.round_number)
        self.update_counter = state.get("update_counter", self.update_counter)
        self.model_version = state.get("model_version", self.model_version)
        if "rep_scores" in state:
            for cid, s in state["rep_scores"].items():
                if cid in self.rep_manager.scores:
                    self.rep_manager.scores[cid]["I"] = s["I"]
                    self.rep_manager.scores[cid]["P"] = s["P"]
        if "spatial_streaks" in state:
            for cid, val in state["spatial_streaks"].items():
                if cid in self.rep_manager.spatial_reject_streak:
                    self.rep_manager.spatial_reject_streak[cid] = val
        if "borderline_streaks" in state:
            for cid, val in state["borderline_streaks"].items():
                if cid in self.rep_manager.borderline_streak:
                    self.rep_manager.borderline_streak[cid] = val
        temp_state = state.get("temporal_state") or state.get("temporal_filter")
        if temp_state:
            self.temporal_filter.load_state(temp_state)
        spat_state = state.get("spatial_state") or state.get("spatial_validator")
        if spat_state:
            self.spatial_validator.load_state(spat_state)
        behav_state = state.get("behavioral_profiles") or state.get("behavioral_memory")
        if behav_state:
            self.behavioral_memory.load_state(behav_state)
        quar_state = state.get("quarantine_state") or state.get("quarantine_manager")
        if quar_state:
            self.quarantine_manager.load_state(quar_state)

    def accumulate_eval_time(self, duration: float) -> None:
        """Accrues CPU execution time spent on blocking evaluations."""
        self.total_eval_time += duration
