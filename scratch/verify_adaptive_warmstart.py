import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np

from server.spatial_validator import SpatialValidator, EPS
from server.temporal_filter import TemporalFilter
from server.behavioral_memory import BehavioralMemoryManager
from server.decision_engine import JointDecisionEngine
from server.aggregator import AggregatorServer
from shared.types import UpdateSubmission, AcceptedEntry, TemporalEvidence, SpatialEvidence, BehavioralEvidence
from utils.logger import BDSFLogger


class TestAdaptiveWarmStart(unittest.TestCase):
    def setUp(self):
        self.config = {
            "N_clients": 5,
            "burn_in_count": 80,
            "K_base": 50,
            "K_ref": 10,
            "M": 20,
            "temporal_min_samples": 20,
            "behavioral_min_depth": 3,
            "warmup_weight_factor": 0.5,
            "theta_cos": 0.10,
            "theta_self": 0.60,
            "theta_floor": 0.15,
            "theta_anchor_min": 0.50,
            "gamma_clip": 1.5,
            "adaptive_clip_enabled": True,
            "static_clip_C": 10.0,
            "decision_mode": "joint",
            "log_dir": "logs/test_warmstart/",
        }
        self.engine = JointDecisionEngine(self.config)

    def test_1_universal_safety_pre_check_during_warmup(self):
        """TEST 1: Universal Safety Pre-Check rejects extreme norm explosion even during early warmup."""
        # Spatial is immature (0 entries)
        spat_ev_extreme = SpatialEvidence(
            sim_global=None,
            norm_raw=500.0,
            norm_clipped=10.0,
            norm_ratio_median=None,
            dynamic_bound_C=10.0,
            reference_available=False,
            spatial_mature=False,
            spatial_reference_count=0,
            spatial_coherence=0.0
        )
        temp_ev = TemporalEvidence(g_i=5.0, lower_fence=None, upper_fence=None, fence_margin=0.0, client_z_score=0.0, is_burn_in=True, temporal_mature=False)
        behav_ev = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=0, behavioral_mature=False)

        outcome = self.engine.evaluate(cid=0, temporal_ev=temp_ev, spatial_ev=spat_ev_extreme, behavioral_ev=behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "REJECT")
        self.assertEqual(outcome.primary_reason, "HARD_GUARD_NORM_EXPLOSION")
        self.assertEqual(outcome.aggregation_weight, 0.0)

    def test_2_warmup_attenuation_and_strict_isolation(self):
        """TEST 2: Warmup accepts benign updates with 0.5x weight and Option A zero-trust isolation."""
        W_init = torch.zeros(50)
        logger = BDSFLogger("test_warmup_iso", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        # Submit benign gradient during warmup (update 1)
        dW = torch.ones(50) * 0.1
        sub = UpdateSubmission(client_id=1, delta_W=dW, t_submit=10.0, tau=0.0)
        res = server.handle_update(sub)

        self.assertEqual(res["status"], "ACCEPT")
        self.assertEqual(res["reason"], "SPATIAL_WARMUP_ACCEPT")

        # 1. Verify weight attenuation (0.5 * 1.0 = 0.5)
        expected_W = W_init + 0.01 * 0.5 * dW
        self.assertTrue(torch.allclose(server.W_global, expected_W, atol=1e-5))

        # 2. Verify Option A Isolation: No reputation recovery
        I_1, P_1 = server.rep_manager.get(1)
        self.assertEqual(I_1, 1.0)
        self.assertEqual(P_1, 1.0)

        # 3. Verify Option A Isolation: No behavioral memory insertion
        profile = server.behavioral_memory.get_or_create_profile(1)
        self.assertEqual(profile.depth, 0, "Behavioral memory must remain empty during warmup")
        self.assertIsNone(profile.genesis_anchor, "Genesis anchor must not be initialized during warmup")

    def test_3_early_byzantine_interception_before_temporal_maturity(self):
        """TEST 3: Byzantine sign flip after spatial maturity (update 11) but before temporal maturity is caught."""
        # 10 valid entries in buffer -> spatial_mature=True, but temporal gap count=11 < 20 -> temporal_mature=False
        spat_ev_inv = SpatialEvidence(
            sim_global=-0.50, # Inverted gradient
            norm_raw=1.0,
            norm_clipped=1.0,
            norm_ratio_median=1.0,
            dynamic_bound_C=2.0,
            reference_available=True,
            spatial_mature=True,
            spatial_reference_count=10,
            spatial_coherence=0.95
        )
        temp_ev_immature = TemporalEvidence(g_i=10.0, lower_fence=None, upper_fence=None, fence_margin=0.0, client_z_score=0.0, is_burn_in=True, temporal_mature=False)
        behav_ev = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=0, behavioral_mature=False)

        outcome = self.engine.evaluate(cid=0, temporal_ev=temp_ev_immature, spatial_ev=spat_ev_inv, behavioral_ev=behav_ev, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "REJECT")
        self.assertEqual(outcome.primary_reason, "HARD_GUARD_GLOBAL_INVERSION")
        self.assertEqual(outcome.aggregation_weight, 0.0)

    def test_4_honest_jitter_immunity_before_temporal_maturity(self):
        """TEST 4: Honest update with arrival delay during temporal warmup is not falsely rejected."""
        spat_ev_good = SpatialEvidence(
            sim_global=0.60,
            norm_raw=1.0,
            norm_clipped=1.0,
            norm_ratio_median=1.0,
            dynamic_bound_C=2.0,
            reference_available=True,
            spatial_mature=True,
            spatial_reference_count=10,
            spatial_coherence=0.95
        )
        # Even with a 300s delay, because temporal is immature, fence_margin is neutral (0.0)
        temp_ev_immature = TemporalEvidence(g_i=300.0, lower_fence=None, upper_fence=None, fence_margin=0.0, client_z_score=0.0, is_burn_in=True, temporal_mature=False)
        behav_ev_cold = BehavioralEvidence(sim_self_mean=None, sim_self_max=None, norm_deviation_self=None, cadence_consistency=None, history_depth=0, behavioral_mature=False)

        outcome = self.engine.evaluate(cid=2, temporal_ev=temp_ev_immature, spatial_ev=spat_ev_good, behavioral_ev=behav_ev_cold, I_i=1.0, P_i=1.0)
        self.assertEqual(outcome.action, "ACCEPT")
        self.assertEqual(outcome.primary_reason, "FULL_CONSENSUS_ACCEPT")
        self.assertEqual(outcome.aggregation_weight, 1.0)

    def test_5_decoupled_dual_horizon_behavioral_activation(self):
        """TEST 5: Genesis anchor active at depth >= 1; k-NN trajectory consistency active at depth >= 3."""
        bmm = BehavioralMemoryManager(self.config)
        cid = 3
        d1 = torch.tensor([1.0, 0.0, 0.0])
        d2 = torch.tensor([0.9, 0.1, 0.0])
        d3 = torch.tensor([0.8, 0.2, 0.0])

        # Step 1: 1 accepted update -> depth=1
        bmm.on_accept(cid, d1)
        ev1 = bmm.extract_evidence(cid, d1)
        self.assertEqual(ev1.history_depth, 1)
        self.assertFalse(ev1.behavioral_mature, "depth=1 must not be trajectory mature")
        self.assertIsNone(ev1.sim_self_mean)
        self.assertIsNotNone(ev1.sim_anchor, "sim_anchor must be active at depth >= 1")
        self.assertAlmostEqual(ev1.sim_anchor, 1.0, places=5)

        # Step 2: 2 accepted updates -> depth=2
        bmm.on_accept(cid, d2)
        ev2 = bmm.extract_evidence(cid, d2)
        self.assertEqual(ev2.history_depth, 2)
        self.assertFalse(ev2.behavioral_mature, "depth=2 must not be trajectory mature")
        self.assertIsNone(ev2.sim_self_mean)
        self.assertIsNotNone(ev2.sim_anchor)

        # Step 3: 3 accepted updates -> depth=3 (trajectory mature!)
        bmm.on_accept(cid, d3)
        ev3 = bmm.extract_evidence(cid, d3)
        self.assertEqual(ev3.history_depth, 3)
        self.assertTrue(ev3.behavioral_mature, "depth=3 must be trajectory mature")
        self.assertIsNotNone(ev3.sim_self_mean)
        self.assertIsNotNone(ev3.sim_anchor)

    def test_6_scientific_spatial_coherence_metric(self):
        """TEST 6: Spatial coherence accurately measures Top-K reference consensus cohesion."""
        sv = SpatialValidator(self.config) # K_ref = 10

        # Case A: Identical unit vectors -> coherence = 1.0
        v_base = torch.randn(50)
        v_base = v_base / torch.norm(v_base)
        for i in range(10):
            sv.on_accept(AcceptedEntry(delta_W=v_base.clone(), I_score=1.0, P_score=1.0, client_id=i))

        _, count_a, coh_a = sv._build_reference_stats()
        self.assertEqual(count_a, 10)
        self.assertAlmostEqual(coh_a, 1.0, places=4, msg="Identical vectors must have coherence ~ 1.0")

        # Case B: Opposing vectors -> coherence ~ 0.0
        sv_opp = SpatialValidator(self.config)
        for i in range(5):
            sv_opp.on_accept(AcceptedEntry(delta_W=v_base.clone(), I_score=1.0, P_score=1.0, client_id=i))
            sv_opp.on_accept(AcceptedEntry(delta_W=-v_base.clone(), I_score=1.0, P_score=1.0, client_id=i+5))

        _, count_b, coh_b = sv_opp._build_reference_stats()
        self.assertEqual(count_b, 10)
        self.assertAlmostEqual(coh_b, 0.0, places=4, msg="Opposing vectors must cancel to coherence ~ 0.0")

    def test_7_warmup_byzantine_accumulation_resistance(self):
        """TEST 7: Unverified Byzantine updates during early rounds cannot farm reputation or poison memory."""
        W_init = torch.zeros(50)
        logger = BDSFLogger("test_warmup_resist", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        # Rounds 1-8: Byzantine clients submit unverified noise/poison
        for r in range(8):
            byz_dW = torch.randn(50) * 0.5
            sub = UpdateSubmission(client_id=0, delta_W=byz_dW, t_submit=1.0 + r, tau=0.0)
            res = server.handle_update(sub)
            self.assertEqual(res["status"], "ACCEPT")
            self.assertEqual(res["reason"], "SPATIAL_WARMUP_ACCEPT")

        # Verify Byzantine client 0 could not farm reputation
        I_0, P_0 = server.rep_manager.get(0)
        self.assertEqual(I_0, 1.0, "Byzantine client must not farm reputation above baseline")
        self.assertEqual(P_0, 1.0)

        # Verify Byzantine client 0 has 0 behavioral memory entries
        p0 = server.behavioral_memory.get_or_create_profile(0)
        self.assertEqual(p0.depth, 0, "No unverified updates may enter client behavioral profile")
        self.assertIsNone(p0.genesis_anchor, "Genesis anchor must remain clean")


if __name__ == "__main__":
    unittest.main()
