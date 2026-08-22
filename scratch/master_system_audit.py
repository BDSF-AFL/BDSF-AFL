import os
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import unittest

from server.reputation_manager import ReputationManager
from server.behavioral_memory import BehavioralMemoryManager
from server.decision_engine import JointDecisionEngine
from shared.types import TemporalEvidence, SpatialEvidence, BehavioralEvidence, JointDecisionOutcome

class MasterSystemAudit(unittest.TestCase):

    def setUp(self):
        self.config = {
            "alpha_I": 0.30,
            "alpha_P": 0.20,
            "beta_I": 0.05,
            "beta_P": 0.08,
            "theta_cos": 0.10,
            "theta_self": 0.60,
            "theta_floor": 0.15,
            "theta_anchor_min": 0.50,
            "alpha_downweight": 0.35,
            "K_drift_max": 5,
            "enable_quarantine": True,
            "spatial_grace_k": 2,
        }
        self.rep_mgr = ReputationManager(client_ids=list(range(5)), config=self.config)
        self.beh_mgr = BehavioralMemoryManager(config=self.config)
        self.dec_engine = JointDecisionEngine(config=self.config)

    def test_1_reputation_invariants_and_dynamics(self):
        """Audit 1: Verify Asymmetry Invariants and Recovery Curves."""
        # 1. Asymmetry assertions
        self.assertLess(self.rep_mgr.beta_I, self.rep_mgr.beta_P)
        self.assertGreaterEqual(self.rep_mgr.alpha_I, self.rep_mgr.alpha_P)
        
        # 2. Honest Node Transient Slash & Recovery
        # Initial state: 1.0, 1.0
        I_0, P_0 = self.rep_mgr.get(0)
        self.assertEqual(I_0, 1.0)
        self.assertEqual(P_0, 1.0)
        
        # Single slash: I drops to 0.70
        self.rep_mgr.slash_integrity(0)
        I_1, _ = self.rep_mgr.get(0)
        self.assertAlmostEqual(I_1, 0.70, places=4)
        
        # Recovery over 6 rounds
        for r in range(6):
            self.rep_mgr.recover(0)
        I_recovered, _ = self.rep_mgr.get(0)
        self.assertAlmostEqual(I_recovered, 1.0, places=4) # 0.70 + 6*0.05 = 1.0 (capped)
        
        # 3. Byzantine Exponential Lockout
        # 10 consecutive slashes
        for _ in range(10):
            self.rep_mgr.slash_integrity(1)
        I_byz, _ = self.rep_mgr.get(1)
        self.assertLess(I_byz, 0.03) # 0.70^10 = 0.0282 <= 0.03 (Neutralized)

    def test_2_borderline_accepted_updates_no_penalty(self):
        """Audit 2: Ensure borderline accepted updates never slash integrity."""
        # Client 2 submits 10 consecutive borderline updates (sim = 0.12 in [0.10, 0.20])
        for _ in range(10):
            self.rep_mgr.record_borderline_check(2, sim=0.12)
        I_2, P_2 = self.rep_mgr.get(2)
        self.assertEqual(I_2, 1.0) # Integrity perfectly preserved!

    def test_3_dynamic_manifold_anchor_consistency(self):
        """Audit 3: Verify that Anchor-Consistent Non-IID clients are protected."""
        cid = 3
        # Seed behavioral memory with 5 consistent updates
        base_vec = torch.randn(100)
        for _ in range(5):
            self.beh_mgr.get_or_create_profile(cid).append(base_vec + 0.01 * torch.randn(100), norm_val=1.0)
            
        candidate_dW = base_vec + 0.02 * torch.randn(100)
        bev = self.beh_mgr.extract_evidence(cid, candidate_dW)
        
        # Set consecutive downweights to 6 (exceeding K_drift_max = 5)
        bev.consecutive_dw = 6
        
        # Spatial evidence: Low global similarity (sim_global = 0.02 < 0.10)
        sev = SpatialEvidence(sim_global=0.02, norm_raw=1.0, norm_clipped=1.0, spatial_mature=True)
        tev = TemporalEvidence(g_i=15.0, lower_fence=10.0, upper_fence=25.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False, temporal_mature=True)
        
        outcome = self.dec_engine.evaluate(cid, tev, sev, bev, I_i=1.0, P_i=1.0)
        
        # Decision must be DOWNWEIGHT (Protected!) instead of UNCOORDINATED_REJECT
        self.assertEqual(outcome.action, "DOWNWEIGHT")
        self.assertEqual(outcome.primary_reason, "NON_IID_HONEST_CONSISTENCY")
        self.assertTrue(outcome.diagnostic_features["is_minority_consistent"])

    def test_4_adaptive_drift_attacker_trapped(self):
        """Audit 4: Verify that an adaptive drift attacker rotating away is caught and cut off."""
        cid = 4
        # Initial anchor seeded along X axis
        base_vec = torch.zeros(100)
        base_vec[0] = 1.0
        for _ in range(5):
            self.beh_mgr.get_or_create_profile(cid).append(base_vec, norm_val=1.0)
            
        # Attacker rotates to Y axis (sim_anchor -> 0.0)
        rotated_dW = torch.zeros(100)
        rotated_dW[1] = 1.0
        bev = self.beh_mgr.extract_evidence(cid, rotated_dW)
        bev.consecutive_dw = 6 # Exceeds K_drift_max
        
        sev = SpatialEvidence(sim_global=-0.05, norm_raw=1.0, norm_clipped=1.0, spatial_mature=True)
        tev = TemporalEvidence(g_i=15.0, lower_fence=10.0, upper_fence=25.0, fence_margin=0.0, client_z_score=0.0, is_burn_in=False, temporal_mature=True)
        
        outcome = self.dec_engine.evaluate(cid, tev, sev, bev, I_i=1.0, P_i=1.0)
        
        # Must be REJECTED!
        self.assertEqual(outcome.action, "REJECT")
        self.assertEqual(outcome.primary_reason, "UNCOORDINATED_OR_ADVERSARIAL_REJECT")

if __name__ == "__main__":
    unittest.main()
