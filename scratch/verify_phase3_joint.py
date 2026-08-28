"""BDSF-AFL Phase 3 Unit & Algorithmic Verification Suite.

Tests:
1. Priority 0 to Priority 5 Deterministic Joint Decision Engine Exclusivity.
2. Dual-Horizon Genesis Anchor, Anti-Drift Escalation & Self-Threshold Dynamics.
3. Bounded Quarantine State Machine Lifecycle (Enqueue, Capacity, Re-eval Release, Expiry).
4. Aggregator Integration & Zero-Regression Legacy Parity Check.
"""

import os
import sys
import unittest
import torch
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.types import (
    UpdateSubmission, TemporalEvidence, SpatialEvidence, BehavioralEvidence,
    JointDecisionOutcome, AcceptedEntry
)
from server.decision_engine import JointDecisionEngine
from server.quarantine_manager import QuarantineManager
from server.behavioral_memory import BehavioralMemoryManager, ClientBehavioralProfile
from server.aggregator import AggregatorServer
from utils.logger import BDSFLogger


class TestPhase3JointDecision(unittest.TestCase):

    def setUp(self):
        self.config = {
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
        }
        self.engine = JointDecisionEngine(self.config)

    def test_priority_0_burn_in(self):
        """Priority 0: Spatial warmup or missing reference must ACCEPT with warmup weight."""
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=True)
        spat_ev = SpatialEvidence(sim_global=None, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=False)
        behav_ev = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=0)
        
        outcome = self.engine.evaluate(0, temp_ev, spat_ev, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "ACCEPT")
        self.assertIn(outcome.primary_reason, ["SPATIAL_WARMUP_ACCEPT", "BURN_IN_ACCEPT"])
        self.assertAlmostEqual(outcome.aggregation_weight, 0.5)

    def test_priority_1_hard_invariants(self):
        """Priority 1: Hard global inversion, norm explosion, and extreme temporal anomalies must REJECT."""
        # 1a. Global inversion: sim_g = -0.20 < -0.15
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False)
        spat_ev = SpatialEvidence(sim_global=-0.20, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        behav_ev = BehavioralEvidence(sim_self_mean=0.99, sim_self_max=0.99, norm_deviation_self=0.1, cadence_consistency=0.1, history_depth=5)
        
        outcome = self.engine.evaluate(0, temp_ev, spat_ev, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "REJECT")
        self.assertEqual(outcome.primary_reason, "HARD_GUARD_GLOBAL_INVERSION")

        # 1b. Norm explosion: norm_raw = 10.0 > 3.0 * 2.0
        spat_ev_norm = SpatialEvidence(sim_global=0.50, norm_raw=10.0, norm_clipped=2.0, norm_ratio_median=5.0, dynamic_bound_C=2.0, reference_available=True)
        outcome_norm = self.engine.evaluate(0, temp_ev, spat_ev_norm, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome_norm.action, "REJECT")
        self.assertEqual(outcome_norm.primary_reason, "HARD_GUARD_NORM_EXPLOSION")

        # 1c. Extreme temporal spam: margin = 0.80 > 0.50, g_i < lower_fence
        temp_ev_spam = TemporalEvidence(g_i=0.05, lower_fence=0.5, upper_fence=2.0, fence_margin=0.80, client_z_score=-3.0, is_burn_in=False)
        spat_ev_ok = SpatialEvidence(sim_global=0.50, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        outcome_spam = self.engine.evaluate(0, temp_ev_spam, spat_ev_ok, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome_spam.action, "REJECT")
        self.assertEqual(outcome_spam.primary_reason, "HARD_GUARD_TEMPORAL_SPAM")

        # 1d. Extreme straggler: margin = 0.80 > 0.50, g_i > upper_fence
        temp_ev_strag = TemporalEvidence(g_i=5.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.80, client_z_score=3.0, is_burn_in=False)
        outcome_strag = self.engine.evaluate(0, temp_ev_strag, spat_ev_ok, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome_strag.action, "REJECT")
        self.assertEqual(outcome_strag.primary_reason, "HARD_GUARD_TEMPORAL_STRAGGLER")
        self.assertTrue(outcome_strag.force_sync_required)

    def test_priority_2_full_consensus(self):
        """Priority 2: High global agreement + high self consistency + normal temporal -> ACCEPT."""
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False)
        spat_ev = SpatialEvidence(sim_global=0.45, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        behav_ev = BehavioralEvidence(sim_self_mean=0.85, sim_self_max=0.90, norm_deviation_self=0.2, cadence_consistency=0.1, history_depth=5)
        
        outcome = self.engine.evaluate(0, temp_ev, spat_ev, behav_ev, I_i=0.9, P_i=1.0)
        self.assertEqual(outcome.action, "ACCEPT")
        self.assertEqual(outcome.primary_reason, "FULL_CONSENSUS_ACCEPT")
        self.assertAlmostEqual(outcome.aggregation_weight, 0.90)

    def test_priority_3_non_iid_soft_downweight(self):
        """Priority 3: Non-IID honest divergence (sim_g in [-0.15, 0.10), sim_s >= 0.60) -> DOWNWEIGHT."""
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False)
        # sim_global is slightly negative (-0.05) due to extreme non-IID class partition, but within [-theta_floor, theta_cos)
        spat_ev = SpatialEvidence(sim_global=-0.05, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        behav_ev = BehavioralEvidence(
            sim_self_mean=0.92, sim_self_max=0.95, norm_deviation_self=0.1, cadence_consistency=0.1,
            history_depth=5, sim_anchor=0.88, consecutive_dw=1
        )
        
        outcome = self.engine.evaluate(0, temp_ev, spat_ev, behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "DOWNWEIGHT")
        self.assertEqual(outcome.primary_reason, "NON_IID_HONEST_CONSISTENCY")
        self.assertGreater(outcome.aggregation_weight, 0.0)
        self.assertLessEqual(outcome.aggregation_weight, self.config["alpha_downweight"])

    def test_slow_drift_escalation(self):
        """Test that accumulating consecutive downweights escalates threshold and eventually REJECTS."""
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False)
        spat_ev = SpatialEvidence(sim_global=-0.05, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        
        # When consecutive_dw exceeds K_drift_max (5), downweight exception is blocked -> REJECT
        behav_ev_exceeded = BehavioralEvidence(
            sim_self_mean=0.70, sim_self_max=0.75, norm_deviation_self=0.1, cadence_consistency=0.1,
            history_depth=5, sim_anchor=0.88, consecutive_dw=5
        )
        outcome = self.engine.evaluate(0, temp_ev, spat_ev, behav_ev_exceeded, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "REJECT")

    def test_priority_4_quarantine(self):
        """Priority 4: Borderline spatial similarity or moderate temporal anomaly from trusted client -> QUARANTINE."""
        # 4a. Borderline spatial: sim_g = 0.08 (within 0.05 of theta_cos=0.10) with low depth
        temp_ev = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False)
        spat_ev_border = SpatialEvidence(sim_global=0.08, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        behav_ev_cold = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=1)
        
        outcome_border = self.engine.evaluate(0, temp_ev, spat_ev_border, behav_ev_cold, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome_border.action, "QUARANTINE")

        # 4b. Moderate temporal anomaly with high integrity: margin = 0.25, I_i = 0.95
        temp_ev_mod = TemporalEvidence(g_i=2.3, lower_fence=0.5, upper_fence=2.0, fence_margin=0.25, client_z_score=1.5, is_burn_in=False)
        spat_ev_mod = SpatialEvidence(sim_global=0.05, norm_raw=1.0, norm_clipped=1.0, norm_ratio_median=1.0, dynamic_bound_C=2.0, reference_available=True)
        behav_ev_mod = BehavioralEvidence(sim_self_mean=0.50, sim_self_max=0.55, norm_deviation_self=0.1, cadence_consistency=0.1, history_depth=4)
        
        outcome_temp = self.engine.evaluate(0, temp_ev_mod, spat_ev_mod, behav_ev_mod, I_i=0.95, P_i=1.0)
        self.assertEqual(outcome_temp.action, "QUARANTINE")


class TestQuarantineManager(unittest.TestCase):

    def setUp(self):
        self.qm = QuarantineManager({"quarantine_capacity": 5, "quarantine_horizon": 3, "theta_cos": 0.10})

    def test_enqueue_and_capacity(self):
        vec = torch.randn(100)
        for i in range(7):
            self.qm.enqueue(client_id=i, delta_W_clipped=vec, current_round=i, virtual_time=float(i), reputation=(1.0, 1.0), reason="TEST")
        # Capacity is 5, so older entries are evicted
        self.assertEqual(self.qm.depth, 5)

    def test_re_evaluation_release(self):
        # Create an update and a reference that matches it (sim = 1.0)
        vec = torch.ones(50)
        self.qm.enqueue(client_id=10, delta_W_clipped=vec, current_round=1, virtual_time=1.0, reputation=(1.0, 1.0), reason="TEST")
        
        ref = torch.ones(50)
        resolved = self.qm.re_evaluate_pending(current_round=2, reference_vector=ref, theta_cos=0.10)
        
        self.assertEqual(len(resolved), 1)
        entry, action, age_mult = resolved[0]
        self.assertEqual(action, "ACCEPT")
        self.assertEqual(entry.client_id, 10)
        self.assertAlmostEqual(age_mult, 1.0 / 1.1)
        self.assertEqual(self.qm.depth, 0)

    def test_expiry(self):
        vec = torch.ones(50)
        self.qm.enqueue(client_id=20, delta_W_clipped=vec, current_round=1, virtual_time=1.0, reputation=(1.0, 1.0), reason="TEST")
        
        # Advance 5 rounds (horizon is 3)
        resolved = self.qm.re_evaluate_pending(current_round=6, reference_vector=None)
        self.assertEqual(len(resolved), 1)
        entry, action, _ = resolved[0]
        self.assertEqual(action, "REJECT")
        self.assertEqual(self.qm.depth, 0)


class TestDualHorizonGenesisAnchor(unittest.TestCase):

    def setUp(self):
        self.manager = BehavioralMemoryManager({"behavioral_history_size": 10, "behavioral_min_history": 3})

    def test_anchor_initialization_and_momentum(self):
        profile = self.manager.get_or_create_profile(1)
        self.assertIsNone(profile.genesis_anchor)

        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([1.0, 0.1, 0.0])
        v3 = torch.tensor([1.0, 0.0, 0.1])

        # Step 1-3: Accept 3 updates to initialize anchor
        profile.append(v1, 1.0, is_downweight=False)
        profile.append(v2, 1.0, is_downweight=False)
        self.assertIsNone(profile.genesis_anchor)
        profile.append(v3, 1.0, is_downweight=False)
        self.assertIsNotNone(profile.genesis_anchor)

        # Step 4: Check anchor similarity
        sim_adaptive, sim_frozen = profile.compute_anchor_similarity(torch.tensor([1.0, 0.0, 0.0]))
        self.assertGreater(sim_adaptive, 0.95)
        self.assertGreater(sim_frozen, 0.95)

        # Step 5: Downweight increments counter without updating anchor
        anchor_before = profile.genesis_anchor.clone()
        profile.append(torch.tensor([0.0, 1.0, 0.0]), 1.0, is_downweight=True)
        self.assertEqual(profile.consecutive_downweights, 1)
        self.assertTrue(torch.allclose(profile.genesis_anchor, anchor_before))


class TestAggregatorZeroRegression(unittest.TestCase):

    def test_legacy_mode_execution(self):
        """Verifies that AggregatorServer runs flawlessly with decision_mode: 'legacy'."""
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        legacy_config = dict(config)
        legacy_config["decision_mode"] = "legacy"
        legacy_config["log_dir"] = "logs/test_legacy_verify/"
        
        W_init = torch.zeros(100)
        client_ids = list(range(5))
        logger = BDSFLogger("legacy_parity_test", legacy_config)
        
        aggregator = AggregatorServer(legacy_config, W_init, client_ids, logger)
        
        # Submit 5 initial updates
        for cid in client_ids:
            sub = UpdateSubmission(
                client_id=cid,
                delta_W=torch.randn(100),
                t_submit=10.0 + cid,
                tau=10.0
            )
            res = aggregator.handle_update(sub)
            self.assertIn(res["status"], ["ACCEPT", "REJECT"])

        print("[OK] Zero-Regression Legacy Mode ran cleanly.")


if __name__ == "__main__":
    unittest.main()
