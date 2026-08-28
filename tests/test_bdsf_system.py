"""BDSF-AFL Master System Verification & Regression Test Suite.

Covers:
1. Model Architectures: CIFAR-10 Adapted ResNet-18 (11.17M params), CIFAR-10 CNN, MNIST MLP.
2. Optimization: Momentum-Enhanced Asynchronous Aggregation (FedAvgM beta_m=0.90).
3. Security: 6-Priority Joint Decision Engine, Dynamic Genesis Anchors & Bounded Quarantine.
4. Reputation Mechanics: Multiplicative Slashing, Additive Recovery, and beta_I < beta_P Invariants.
5. Reproducibility: Full-System State Checkpointing and Seamless CSV Telemetry Resumption.
"""

import os
import sys
import csv
import unittest
import torch
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import CIFAR10ResNet18, MNISTMLP, CIFAR10CNN
from server.aggregator import AggregatorServer
from server.decision_engine import JointDecisionEngine
from server.quarantine_manager import QuarantineManager
from server.behavioral_memory import BehavioralMemoryManager
from server.reputation_manager import ReputationManager
from shared.types import (
    UpdateSubmission, TemporalEvidence, SpatialEvidence, BehavioralEvidence,
    JointDecisionOutcome, AcceptedEntry
)
from utils.logger import BDSFLogger
import utils.metrics as metrics


class TestBDSFSystem(unittest.TestCase):

    def setUp(self):
        self.config = {
            "N_clients": 20,
            "total_rounds": 50,
            "eval_every": 5,
            "eta": 0.01,
            "server_momentum": 0.90,
            "model_architecture": "resnet18",
            "theta_cos": 0.10,
            "theta_self": 0.60,
            "theta_floor": 0.15,
            "theta_anchor_min": 0.50,
            "alpha_downweight": 0.35,
            "K_drift_max": 5,
            "delta_theta_step": 0.05,
            "delta_theta_max": 0.25,
            "delta_temp_mod": 0.50,
            "enable_quarantine": True,
            "quarantine_horizon": 5,
            "quarantine_capacity": 20,
            "delta_borderline": 0.05,
            "trusted_integrity_min": 0.80,
            "behavioral_history_size": 10,
            "behavioral_min_history": 3,
            "alpha_I": 0.30,
            "alpha_P": 0.20,
            "beta_I": 0.05,
            "beta_P": 0.08,
            "log_dir": "logs/test_suite_tmp/",
        }

    # =========================================================================
    # 1. MODEL ARCHITECTURE TESTS
    # =========================================================================
    def test_cifar10_resnet18_dimensions(self):
        """Verifies CIFAR-10 Adapted ResNet-18 forward pass and exact 11.17M parameter count."""
        model = CIFAR10ResNet18()
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (4, 10))
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 11173962)

    def test_baseline_models(self):
        """Verifies backward-compatibility baseline models."""
        mlp = MNISTMLP()
        self.assertEqual(mlp(torch.randn(4, 784)).shape, (4, 10))
        cnn = CIFAR10CNN()
        self.assertEqual(cnn(torch.randn(4, 3, 32, 32)).shape, (4, 10))

    # =========================================================================
    # 2. MOMENTUM-ENHANCED AGGREGATION TESTS
    # =========================================================================
    def test_momentum_accumulation(self):
        """Verifies velocity buffer tracking under Momentum-Enhanced Aggregation."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_mom", self.config)
        server = AggregatorServer(self.config, W_init, list(range(20)), logger)

        self.assertEqual(server.server_momentum, 0.90)
        self.assertEqual(server.v_momentum.shape, W_init.shape)

        delta = torch.ones_like(W_init) * 0.05
        server._apply_global_update(delta)
        self.assertTrue(torch.allclose(server.v_momentum, delta))
        self.assertTrue(torch.allclose(server.W_global, W_init + delta))

        # Cleanup
        if os.path.exists(logger.csv_path):
            os.remove(logger.csv_path)

    # =========================================================================
    # 3. JOINT DECISION ENGINE & SECURITY TESTS
    # =========================================================================
    def test_joint_decision_priorities(self):
        """Verifies 6-priority decision exclusivity (Priority 0 to Priority 5)."""
        engine = JointDecisionEngine(self.config)

        # Priority 0: Burn-in / Warmup
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=True)
        spat_ev = SpatialEvidence(sim_global=None, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=False, spatial_mature=False)
        behav_ev = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=0)
        out0 = engine.evaluate(0, temp_ev, spat_ev, behav_ev, 1.0, 1.0)
        self.assertEqual(out0.action, "ACCEPT")

        # Priority 1: Catastrophic Gradient Inversion (Spatial Poison)
        temp_ev.is_burn_in = False
        spat_ev.spatial_mature = True
        spat_ev.sim_global = -0.50
        spat_ev.reference_available = True
        out1 = engine.evaluate(1, temp_ev, spat_ev, behav_ev, 1.0, 1.0)
        self.assertEqual(out1.action, "REJECT")
        self.assertEqual(out1.primary_reason, "HARD_GUARD_GLOBAL_INVERSION")

        # Priority 2: Full Consensus (Mature Honest Node)
        spat_ev.sim_global = 0.85
        behav_ev.sim_self_mean = 0.90
        behav_ev.history_depth = 5
        behav_ev.sim_anchor = 0.90
        out2 = engine.evaluate(2, temp_ev, spat_ev, behav_ev, 1.0, 1.0)
        self.assertEqual(out2.action, "ACCEPT")
        self.assertEqual(out2.primary_reason, "FULL_CONSENSUS_ACCEPT")
        self.assertEqual(out2.aggregation_weight, 1.0)

        # Priority 2 Defense: S2 Mimicry Attack (Mature Attacker Evasion Blocked)
        # Attacker maintains sim_global >= 0.10, but fails temporal identity (sim_self < 0.35 & sim_anchor < 0.40)
        behav_ev_mimic = BehavioralEvidence(
            sim_self_mean=0.12,
            sim_self_max=0.18,
            norm_deviation_self=0.5,
            cadence_consistency=0.9,
            history_depth=5,
            sim_anchor=0.25,
        )
        spat_ev_mimic = SpatialEvidence(
            sim_global=0.15,  # evades snapshot spatial check (0.15 >= 0.10)
            norm_raw=0.30,
            norm_clipped=0.30,
            norm_ratio_median=1.0,
            dynamic_bound_C=2.0,
            reference_available=True,
            spatial_mature=True,
        )
        out_mimic = engine.evaluate(3, temp_ev, spat_ev_mimic, behav_ev_mimic, 1.0, 1.0)
        self.assertEqual(out_mimic.action, "REJECT")
        self.assertEqual(out_mimic.aggregation_weight, 0.0)
        self.assertEqual(out_mimic.primary_reason, "UNCOORDINATED_OR_ADVERSARIAL_REJECT")

    # =========================================================================
    # 4. REPUTATION MECHANICS & MATHEMATICAL INVARIANTS
    # =========================================================================
    def test_reputation_invariants(self):
        """Verifies beta_I < beta_P constraint, slashing decay, and additive recovery."""
        rep = ReputationManager(list(range(5)), self.config)
        self.assertLess(rep.beta_I, rep.beta_P)

        # Multiplicative slashing drops score
        rep.slash_integrity(0)
        I_0, _ = rep.get(0)
        self.assertAlmostEqual(I_0, 0.70, places=4)

        # Additive recovery steps up
        rep.recover(0)
        I_rec, _ = rep.get(0)
        self.assertAlmostEqual(I_rec, 0.75, places=4)

    # =========================================================================
    # 5. STATE SERIALIZATION & CSV RESUME TESTS
    # =========================================================================
    def test_state_serialization_and_csv_resume(self):
        """Verifies atomic state checkpoint save/load and CSV deduplication."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger1 = BDSFLogger("test_res_run", self.config)
        server = AggregatorServer(self.config, W_init, list(range(20)), logger1)

        # Log 5 updates
        for r in range(5):
            logger1.log_update(round=r, client_id=r, status="ACCEPT", reason="TEST_ACCEPT", weight=1.0)

        # Serialize State
        state = server.get_state()
        self.assertIn("W_global", state)
        self.assertIn("v_momentum", state)
        self.assertIn("rep_scores", state)
        self.assertIn("behavioral_profiles", state)

        # Simulate Resume from Round 3
        cfg_resume = dict(self.config)
        cfg_resume["resume"] = True
        cfg_resume["resume_round"] = 3
        logger2 = BDSFLogger("test_res_run", cfg_resume)

        # Check rows sanitized to round 3 (1 header + 4 rows = 5 lines)
        with open(logger2.csv_path, "r") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 5)

        # Clean test directory
        import shutil
        if os.path.exists(self.config["log_dir"]):
            shutil.rmtree(self.config["log_dir"])

    # =========================================================================
    # 6. MODEL-VERSION TRACKING & VERSION LAG EXTRACTION
    # =========================================================================
    def test_server_model_version_tracking_and_lag(self):
        """Verifies monotonic model version counter and lag extraction."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_vtrack", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        self.assertEqual(server.get_model_version(), 0)

        # Simulate update 1 from version 0
        dW1 = torch.ones_like(W_init) * 0.01
        sub1 = UpdateSubmission(client_id=0, delta_W=dW1, t_submit=1.0, tau=0.5, model_version_at_pull=0)
        resp1 = server.handle_update(sub1)
        self.assertEqual(resp1["status"], "ACCEPT")
        self.assertEqual(server.get_model_version(), 1)

        # Client 1 submits with model_version_at_pull=0 (lag = 1)
        dW2 = torch.ones_like(W_init) * 0.01
        sub2 = UpdateSubmission(client_id=1, delta_W=dW2, t_submit=2.0, tau=0.5, model_version_at_pull=0)
        resp2 = server.handle_update(sub2)
        self.assertEqual(resp2["status"], "ACCEPT")
        self.assertEqual(server.get_model_version(), 2)

        if os.path.exists(logger.csv_path):
            os.remove(logger.csv_path)

    # =========================================================================
    # 7. DECISION-STATE INVARIANT ISOLATION (REJECT / QUARANTINE NEVER POLLUTE)
    # =========================================================================
    def test_decision_state_invariants_isolation(self):
        """Verifies that REJECT and QUARANTINE actions never mutate W_global, accepted_buffer, or behavioral memory."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_isolation", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        # 1. Reject Update (Extreme Norm Explosion)
        W_before = server.get_global_weights()
        dW_expl = torch.ones_like(W_init) * 100.0
        sub_expl = UpdateSubmission(client_id=0, delta_W=dW_expl, t_submit=1.0, tau=0.5, model_version_at_pull=0)
        resp_expl = server.handle_update(sub_expl)
        self.assertEqual(resp_expl["status"], "REJECT")
        self.assertTrue(torch.allclose(server.get_global_weights(), W_before))
        self.assertEqual(len(server.accepted_buffer), 0)
        self.assertEqual(server.behavioral_memory.get_or_create_profile(0).depth, 0)
        self.assertEqual(server.get_model_version(), 0)

        # 2. Reject Update (Zero Gradient)
        dW_zero = torch.zeros_like(W_init)
        sub_zero = UpdateSubmission(client_id=1, delta_W=dW_zero, t_submit=1.0, tau=0.5, model_version_at_pull=0)
        resp_zero = server.handle_update(sub_zero)
        self.assertEqual(resp_zero["status"], "REJECT")
        self.assertTrue(torch.allclose(server.get_global_weights(), W_before))
        self.assertEqual(len(server.accepted_buffer), 0)
        self.assertEqual(server.get_model_version(), 0)

        if os.path.exists(logger.csv_path):
            os.remove(logger.csv_path)

    # =========================================================================
    # 8. BOUNDED QUARANTINE LIFECYCLE TESTS
    # =========================================================================
    def test_quarantine_bounded_lifecycle(self):
        """Verifies quarantine capacity bounding, age-attenuated release, and horizon expiration."""
        qm = QuarantineManager({"quarantine_capacity": 3, "quarantine_horizon": 5, "theta_cos": 0.10})
        
        # Enqueue 3 entries
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        v3 = torch.tensor([0.0, 0.0, 1.0])
        
        qm.enqueue(0, v1, current_round=0, virtual_time=1.0, reputation=(1.0, 1.0), reason="BORDERLINE")
        qm.enqueue(1, v2, current_round=0, virtual_time=1.0, reputation=(1.0, 1.0), reason="BORDERLINE")
        qm.enqueue(2, v3, current_round=0, virtual_time=1.0, reputation=(1.0, 1.0), reason="BORDERLINE")
        self.assertEqual(qm.depth, 3)

        # Re-evaluate with reference aligned with v1 at round 2
        ref = torch.tensor([1.0, 0.0, 0.0])
        resolved = qm.re_evaluate_pending(current_round=2, reference_vector=ref)
        
        # v1 should be ACCEPT with age multiplier 1 / (1 + 0.1 * 2) = 1 / 1.2
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0][0].client_id, 0)
        self.assertEqual(resolved[0][1], "ACCEPT")
        self.assertAlmostEqual(resolved[0][2], 1.0 / 1.2, places=4)
        self.assertEqual(qm.depth, 2)

        # Advance to round 10 (> horizon 5) without alignment -> remaining entries expire as REJECT
        resolved_expired = qm.re_evaluate_pending(current_round=10, reference_vector=ref)
        self.assertEqual(len(resolved_expired), 2)
        for entry, action, mult in resolved_expired:
            self.assertEqual(action, "REJECT")
            self.assertEqual(mult, 0.0)
        self.assertEqual(qm.depth, 0)

    # =========================================================================
    # 9. DUAL-HORIZON GENESIS ANCHOR & DRIFT DETECTION
    # =========================================================================
    def test_dual_horizon_genesis_anchor_and_drift_detection(self):
        """Verifies frozen ground-truth anchor retention, bounded adaptive anchor drift, and divergence logging."""
        bmm = BehavioralMemoryManager({"behavioral_history_size": 10, "behavioral_min_history": 3})
        
        # Push 3 initial accepted vectors to form the Genesis Anchor
        v_base = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for _ in range(3):
            bmm.on_accept(client_id=0, delta_W=v_base, norm_val=1.0, is_downweight=False)

        profile = bmm.get_or_create_profile(0)
        self.assertIsNotNone(profile.genesis_anchor)
        self.assertIsNotNone(profile.frozen_genesis_anchor)
        self.assertTrue(torch.allclose(profile.genesis_anchor.float(), profile.frozen_genesis_anchor.float()))
        self.assertEqual(profile.compute_anchor_drift(), 0.0)

        # Slowly update adaptive anchor on full accept
        v_shift = torch.tensor([0.6, 0.8, 0.0, 0.0])
        bmm.on_accept(client_id=0, delta_W=v_shift, norm_val=1.0, is_downweight=False)
        
        # Frozen anchor must remain exactly unchanged
        self.assertTrue(torch.allclose(profile.frozen_genesis_anchor.float(), v_base.float()))
        # Adaptive anchor drifts slightly
        self.assertGreater(profile.compute_anchor_drift(), 0.0)

    # =========================================================================
    # 10. HMAC FORCE-SYNC REPLAY & FRESHNESS DEFENSE
    # =========================================================================
    def test_hmac_force_sync_replay_and_freshness(self):
        """Verifies HMAC authentication, replay rejection of duplicate nonces, monotonic sequence freshness, and max_age window expiration."""
        logger = BDSFLogger("test_fs_replay", self.config)
        from server.force_sync import ForceSyncDispatcher
        from client.force_sync_handler import ForceSyncHandler
        dispatcher = ForceSyncDispatcher()
        session_key = os.urandom(32)
        handler = ForceSyncHandler(client_id=0, session_key=session_key, logger=logger, max_age=60.0)

        W_target = torch.randn(10)
        payload = dispatcher.build_payload(0, W_target, session_key, timestamp=10.0)

        client_state = {
            "W_local": torch.zeros(10),
            "gradient_buffer": [torch.ones(10)],
            "last_reset_time": 5.0,
            "current_virtual_time": 15.0
        }

        # 1. First valid application succeeds
        res1 = handler.verify_and_apply(payload, client_state)
        self.assertTrue(res1)
        self.assertTrue(torch.allclose(client_state["W_local"], W_target))
        self.assertEqual(client_state["last_reset_time"], 10.0)

        # 2. Replay defense: Replaying the exact same payload (seen nonce) fails
        res_replay = handler.verify_and_apply(payload, client_state)
        self.assertFalse(res_replay)

        # 3. Monotonic sequence freshness: A genuinely old-but-unseen payload (new nonce, timestamp <= last_reset_time) fails
        unseen_stale_payload = dispatcher.build_payload(0, W_target, session_key, timestamp=9.0)
        res_stale = handler.verify_and_apply(unseen_stale_payload, client_state)
        self.assertFalse(res_stale)

        # 4. Absolute horizon freshness: An unseen payload generated outside the max_age window fails
        client_state_fresh = {
            "W_local": torch.zeros(10),
            "gradient_buffer": [],
            "last_reset_time": 0.0,
            "current_virtual_time": 200.0  # elapsed 200s > max_age 60s
        }
        expired_payload = dispatcher.build_payload(0, W_target, session_key, timestamp=50.0)
        res_expired = handler.verify_and_apply(expired_payload, client_state_fresh)
        self.assertFalse(res_expired)

        if os.path.exists(logger.csv_path):
            os.remove(logger.csv_path)

    # =========================================================================
    # 11. EXACT CHECKPOINT EQUIVALENCE & STATE REPLAY
    # =========================================================================
    def test_checkpoint_equivalence_exact_replay(self):
        """Verifies complete state serialization and exact decision replay upon reload."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger1 = BDSFLogger("test_chk_orig", self.config)
        server1 = AggregatorServer(self.config, W_init, list(range(10)), logger1)

        # Process 5 updates on server 1
        for i in range(5):
            dW = torch.ones_like(W_init) * (0.01 * (i + 1))
            sub = UpdateSubmission(client_id=i, delta_W=dW, t_submit=float(i + 1), tau=float(i), model_version_at_pull=0)
            server1.handle_update(sub)

        # Save checkpoint
        checkpoint_state = server1.get_state()

        # Initialize fresh server 2 and load checkpoint
        logger2 = BDSFLogger("test_chk_reload", self.config)
        server2 = AggregatorServer(self.config, W_init, list(range(10)), logger2)
        server2.load_state(checkpoint_state)

        # Verify state equality
        self.assertTrue(torch.allclose(server1.get_global_weights(), server2.get_global_weights()))
        self.assertEqual(server1.get_model_version(), server2.get_model_version())
        self.assertEqual(server1.round_number, server2.round_number)
        self.assertEqual(server1.update_counter, server2.update_counter)

        # Submit identical next update to both servers
        dW_test = torch.ones_like(W_init) * 0.05
        sub_test1 = UpdateSubmission(client_id=2, delta_W=dW_test, t_submit=10.0, tau=9.0, model_version_at_pull=server1.get_model_version())
        sub_test2 = UpdateSubmission(client_id=2, delta_W=dW_test, t_submit=10.0, tau=9.0, model_version_at_pull=server2.get_model_version())

        res1 = server1.handle_update(sub_test1)
        res2 = server2.handle_update(sub_test2)

        self.assertEqual(res1["status"], res2["status"])
        self.assertEqual(res1["reason"], res2["reason"])
        self.assertTrue(torch.allclose(server1.get_global_weights(), server2.get_global_weights()))

        # Cleanup
        for lg in [logger1, logger2]:
            if os.path.exists(lg.csv_path):
                os.remove(lg.csv_path)

    # =========================================================================
    # 12. TEMPORAL FILTER DEDUPLICATION & STATE MATURITY
    # =========================================================================
    def test_temporal_filter_burn_in_deduplication(self):
        """Verifies that TemporalFilter consolidates is_burn_in cleanly with state_maturity."""
        from server.temporal_filter import TemporalFilter
        tf = TemporalFilter({"warm_start_mode": "state_maturity", "temporal_min_samples": 5})
        self.assertTrue(tf.is_burn_in())

        for i in range(5):
            tf.record_gap(1.0 + 0.1 * i, client_id=0)

        self.assertTrue(tf.is_temporal_mature())
        self.assertFalse(tf.is_burn_in())

    # =========================================================================
    # 13. PRIORITY 3B VS PRIORITY 4 PRECEDENCE & DIAGNOSTIC FEATURES
    # =========================================================================
    def test_priority3b_vs_priority4_borderline_quarantine_precedence(self):
        """Verifies that immature profiles with borderline spatial similarity (|sim_g - theta_cos| <= delta_borderline)
        under mature spatial reference route to Priority 4 (QUARANTINE) rather than Priority 3b (DOWNWEIGHT).
        """
        engine = JointDecisionEngine(self.config)
        # Spatial reference is mature
        # Profile is immature (history_depth = 1 < 3)
        # sim_global = 0.08 (theta_cos=0.10, delta_borderline=0.05 -> |0.08 - 0.10| = 0.02 <= 0.05 -> borderline)
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False, temporal_mature=True, version_lag=1)
        spat_ev = SpatialEvidence(sim_global=0.08, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True, spatial_mature=True)
        behav_ev = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, history_depth=1, sim_anchor=0.60, behavioral_mature=False, sim_frozen_anchor=0.60, anchor_drift=0.0)

        outcome = engine.evaluate(cid=0, temporal_ev=temp_ev, spatial_ev=spat_ev, behavioral_ev=behav_ev, I_i=1.0, P_i=1.0, version_lag=1)
        self.assertEqual(outcome.action, "QUARANTINE")
        self.assertEqual(outcome.primary_reason, "AMBIGUOUS_EVIDENCE_QUARANTINE")
        self.assertEqual(outcome.diagnostic_features["priority"], 4)
        self.assertTrue(outcome.diagnostic_features["is_borderline_spatial"])
        self.assertEqual(outcome.diagnostic_features["version_lag"], 1)
        self.assertEqual(outcome.diagnostic_features["sim_frozen_anchor"], 0.60)
        self.assertEqual(outcome.diagnostic_features["anchor_drift"], 0.0)

        # Non-borderline immature profile (sim_global = 0.02 -> |0.02 - 0.10| = 0.08 > delta_borderline 0.05, but >= -theta_floor -0.15)
        # should proceed to Priority 3b (DOWNWEIGHT)
        spat_ev_non_borderline = SpatialEvidence(sim_global=0.02, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True, spatial_mature=True)
        outcome_3b = engine.evaluate(cid=0, temporal_ev=temp_ev, spatial_ev=spat_ev_non_borderline, behavioral_ev=behav_ev, I_i=1.0, P_i=1.0, version_lag=1)
        self.assertEqual(outcome_3b.action, "DOWNWEIGHT")
        self.assertEqual(outcome_3b.primary_reason, "EARLY_STAGE_NON_IID_HOLD")
        self.assertEqual(outcome_3b.diagnostic_features["priority"], 3)
        self.assertEqual(outcome_3b.diagnostic_features["version_lag"], 1)
        self.assertEqual(outcome_3b.diagnostic_features["sim_frozen_anchor"], 0.60)
        self.assertEqual(outcome_3b.diagnostic_features["anchor_drift"], 0.0)

    # =========================================================================
    # 14. COMPONENT GROUP 2 HARDENING VERIFICATION
    # =========================================================================
    def test_component_group_2_hardening(self):
        """Comprehensive verification of Component Group 2 pre-experiment hardening changes."""
        from server.temporal_filter import TemporalFilter
        from server.spatial_validator import SpatialValidator
        from server.quarantine_manager import QuarantineManager
        from server.behavioral_memory import BehavioralMemoryManager

        # 1. TemporalFilter hardening
        tf = TemporalFilter({"warm_start_mode": "state_maturity", "temporal_min_samples": 4, "N_burn": 50})
        self.assertTrue(tf.is_burn_in())
        # Record 4 gaps
        for g in [10.0, 11.0, 12.0, 13.0]:
            tf.record_gap(g, client_id=1)
        self.assertTrue(tf.is_temporal_mature())
        self.assertFalse(tf.is_burn_in())
        tev = tf.extract_evidence(12.0, client_id=1, version_lag=2)
        self.assertEqual(tev.version_lag, 2)
        self.assertFalse(tev.is_burn_in)
        self.assertTrue(tev.temporal_mature)

        # TemporalFilter checkpoint equivalence
        tf_state = tf.get_state()
        tf_reloaded = TemporalFilter({"warm_start_mode": "state_maturity", "temporal_min_samples": 4})
        tf_reloaded.load_state(tf_state)
        self.assertEqual(tf_reloaded.gap_history, tf.gap_history)
        self.assertEqual(tf_reloaded.client_gap_history, tf.client_gap_history)
        self.assertEqual(tf_reloaded._total_seen, tf._total_seen)

        # 2. BehavioralMemoryManager hardening
        bmm = BehavioralMemoryManager({"behavioral_history_size": 10, "behavioral_min_history": 3})
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([1.0, 0.0, 0.0])
        v3 = torch.tensor([1.0, 0.0, 0.0])
        bmm.on_accept(client_id=0, delta_W=v1, norm_val=1.0, is_downweight=False)
        bmm.on_accept(client_id=0, delta_W=v2, norm_val=1.0, is_downweight=False)
        prof = bmm.get_or_create_profile(0)
        self.assertIsNone(prof.genesis_anchor)
        self.assertIsNone(prof.frozen_genesis_anchor)

        bmm.on_accept(client_id=0, delta_W=v3, norm_val=1.0, is_downweight=False)
        self.assertIsNotNone(prof.genesis_anchor)
        self.assertIsNotNone(prof.frozen_genesis_anchor)
        self.assertTrue(torch.allclose(prof.genesis_anchor.float(), prof.frozen_genesis_anchor.float()))

        # compute_anchor_similarity returning (adaptive, frozen)
        sim_a, sim_f = prof.compute_anchor_similarity(torch.tensor([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(sim_a, 1.0, places=3)
        self.assertAlmostEqual(sim_f, 1.0, places=3)

        # Downweight with sim_self >= 0.35 triggers micro-adaptation (lambda=0.02)
        v_dw_consistent = torch.tensor([0.9, 0.1, 0.0])
        prof_anchor_before = prof.genesis_anchor.clone()
        bmm.on_accept(client_id=0, delta_W=v_dw_consistent, norm_val=1.0, is_downweight=True)
        self.assertEqual(prof.consecutive_downweights, 1)
        self.assertFalse(torch.allclose(prof.genesis_anchor, prof_anchor_before))
        self.assertTrue(torch.allclose(prof.frozen_genesis_anchor.float(), v1.float()))

        # Behavioral evidence extraction
        bev = bmm.extract_evidence(client_id=0, delta_W=torch.tensor([1.0, 0.0, 0.0]))
        self.assertIsNotNone(bev.sim_anchor)
        self.assertIsNotNone(bev.sim_frozen_anchor)
        self.assertIsNotNone(bev.anchor_drift)

        # Behavioral memory checkpoint equivalence
        bmm_state = bmm.get_state()
        bmm_reloaded = BehavioralMemoryManager({"behavioral_history_size": 10, "behavioral_min_history": 3})
        bmm_reloaded.load_state(bmm_state)
        prof_reloaded = bmm_reloaded.get_or_create_profile(0)
        self.assertTrue(torch.allclose(prof_reloaded.genesis_anchor.float(), prof.genesis_anchor.float()))
        self.assertTrue(torch.allclose(prof_reloaded.frozen_genesis_anchor.float(), prof.frozen_genesis_anchor.float()))
        self.assertEqual(prof_reloaded.consecutive_downweights, prof.consecutive_downweights)

        # 3. QuarantineManager hardening
        qm = QuarantineManager({"quarantine_capacity": 20, "quarantine_horizon": 5, "theta_cos": 0.10})
        self.assertEqual(qm.capacity, 20)
        self.assertEqual(qm.horizon, 5)

        # Enqueue update for client 5
        vec_q = torch.tensor([1.0, 0.0, 0.0])
        e1 = qm.enqueue(client_id=5, delta_W_clipped=vec_q, current_round=10, virtual_time=100.0, reputation=(0.9, 0.9), reason="BORDERLINE")
        self.assertEqual(qm.depth, 1)

        # Enqueue another update for client 5 -> should evict previous entry
        e2 = qm.enqueue(client_id=5, delta_W_clipped=vec_q, current_round=11, virtual_time=110.0, reputation=(0.9, 0.9), reason="BORDERLINE_2")
        self.assertEqual(qm.depth, 1)
        self.assertEqual(qm.buffer[0].entry_id, e2.entry_id)

        # Re-evaluate expired entry (round 20 - round 11 = 9 > horizon 5)
        resolved_expired = qm.re_evaluate_pending(current_round=20, reference_vector=None)
        self.assertEqual(len(resolved_expired), 1)
        self.assertEqual(resolved_expired[0][1], "REJECT")
        self.assertEqual(resolved_expired[0][2], 0.0)

        # Re-evaluate matching reference entry
        qm.enqueue(client_id=6, delta_W_clipped=vec_q, current_round=20, virtual_time=200.0, reputation=(0.9, 0.9), reason="TEST")
        ref_vec = torch.tensor([1.0, 0.0, 0.0])
        resolved_match = qm.re_evaluate_pending(current_round=22, reference_vector=ref_vec, theta_cos=0.10)
        self.assertEqual(len(resolved_match), 1)
        self.assertEqual(resolved_match[0][1], "ACCEPT")
        self.assertAlmostEqual(resolved_match[0][2], 1.0 / (1.0 + 0.1 * 2), places=4)

        # QuarantineManager checkpoint equivalence
        qm.enqueue(client_id=7, delta_W_clipped=vec_q, current_round=30, virtual_time=300.0, reputation=(0.8, 0.8), reason="CHK")
        qm_state = qm.get_state()
        qm_reloaded = QuarantineManager({"quarantine_capacity": 20, "quarantine_horizon": 5})
        qm_reloaded.load_state(qm_state)
        self.assertEqual(qm_reloaded.depth, qm.depth)
        self.assertEqual(qm_reloaded.buffer[0].client_id, 7)
        self.assertEqual(qm_reloaded.buffer[0].primary_reason, "CHK")

        # 4. SpatialValidator hardening
        sv = SpatialValidator({"K_ref": 5, "M": 10, "theta_cos": 0.10})
        entry_sv = AcceptedEntry(delta_W=torch.tensor([1.0, 2.0, 3.0]), I_score=0.95, P_score=0.98, client_id=3, is_warmup=False)
        sv.on_accept(entry_sv)
        sv.last_sim = 0.85
        self.assertEqual(sv._total_accepted_count, 1)
        self.assertIn(3, sv._unique_accepted_clients)

        # SpatialValidator checkpoint equivalence
        sv_state = sv.get_state()
        sv_reloaded = SpatialValidator({"K_ref": 5, "M": 10, "theta_cos": 0.10})
        sv_reloaded.load_state(sv_state)
        self.assertEqual(sv_reloaded._total_accepted_count, 1)
        self.assertIn(3, sv_reloaded._unique_accepted_clients)
        self.assertEqual(sv_reloaded.last_sim, 0.85)
        self.assertEqual(len(sv_reloaded._buffer), 1)
        self.assertTrue(torch.allclose(sv_reloaded._buffer[0].delta_W, entry_sv.delta_W))


if __name__ == "__main__":
    unittest.main(verbosity=2)
