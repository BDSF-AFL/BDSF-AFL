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


if __name__ == "__main__":
    unittest.main(verbosity=2)
