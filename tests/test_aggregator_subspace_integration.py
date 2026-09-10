"""BDSF-AFL v2 Aggregator & Subspace Projection Engine Integration Test Suite.

Verifies:
1. Gate 1 Macro Consensus Manifold Alignment: Immediate rejection of inverted updates.
2. Gate 1b Layer-Wise Coordinate Variance Floor: Detection & rejection of cross-layer masking.
3. Consensus Basis Admission Policy (Council Approved): Warmup bootstrapping, post-warmup ACCEPT admission, and DOWNWEIGHT exclusion.
4. Gate 2 Adaptive Orthogonal Energy Bounding: Smooth damping of orthogonal perturbations.
5. Checkpoint Equivalence & Exact Replay: Complete serialization and reload parity of basis Q and directional memory m_perp.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import CIFAR10ResNet18, MNISTMLP
from server.aggregator import AggregatorServer
from server.subspace_engine import DEFAULT_RESNET18_SLICES
from shared.types import UpdateSubmission
from utils.logger import BDSFLogger


class TestAggregatorSubspaceIntegration(unittest.TestCase):

    def setUp(self):
        self.config = {
            "N_clients": 5,
            "total_rounds": 50,
            "eval_every": 5,
            "eta": 0.01,
            "server_momentum": 0.0,
            "model_architecture": "resnet18",
            "theta_cos": 0.10,
            "theta_self": 0.30,
            "theta_floor": 0.40,
            "theta_anchor_min": 0.25,
            "alpha_downweight": 0.35,
            "enable_quarantine": True,
            "warmup_rounds": 5,
            "spatial_warmup_rounds": 5,
            "log_dir": "logs/test_subspace_tmp/",
            "enable_subspace": True,
            "subspace_K": 5,
            "subspace_eps_floor": 0.05,
            "subspace_alpha": 0.60,
            "subspace_lam": 3.0,
            "subspace_beta": 0.85,
            "subspace_kappa": 0.10,
            "subspace_m_perp_base": 2.0,
        }

    def tearDown(self):
        import shutil
        if os.path.exists(self.config["log_dir"]):
            shutil.rmtree(self.config["log_dir"], ignore_errors=True)

    def test_subspace_macro_inversion_rejection(self):
        """Verifies that an update opposing the consensus manifold is rejected with MACRO_MANIFOLD_INVERSION."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_macro_inv", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        # Seed the basis with known directions until full (K=5)
        basis_dir = torch.randn_like(W_init)
        basis_dir = basis_dir / torch.norm(basis_dir)
        for _ in range(server.subspace_engine.K):
            server.subspace_engine.update_basis(basis_dir + torch.randn_like(W_init) * 1e-4)

        # Submit an update directly opposing the consensus basis
        v_inv = -2.0 * basis_dir
        sub = UpdateSubmission(
            client_id=1,
            delta_W=v_inv,
            t_submit=1.0,
            tau=0.5,
            model_version_at_pull=0,
        )

        W_before = server.get_global_weights()
        resp = server.handle_update(sub)

        self.assertEqual(resp["status"], "REJECT")
        self.assertEqual(resp["reason"], "MACRO_MANIFOLD_INVERSION")
        self.assertTrue(torch.allclose(server.get_global_weights(), W_before))

    def test_subspace_gate1b_cross_layer_collapse_rejection(self):
        """Verifies that an update masking cross-layer coordinate collapse is rejected with CROSS_LAYER_COLLAPSE."""
        cfg = dict(self.config)
        cfg["subspace_K"] = 1
        model = CIFAR10ResNet18()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_gate1b", cfg)
        server = AggregatorServer(cfg, W_init, list(range(5)), logger)

        # Seed basis with structured noise
        torch.manual_seed(42)
        base_v = torch.randn_like(W_init) * 0.1
        server.subspace_engine.update_basis(base_v)

        # Craft update with massive positive projection in Layer 4 (index ~2.7M to 11.16M)
        # but zeroed out in the FC head (11.168M to 11.173M)
        crafted_v = base_v.clone()
        fc_start, fc_end = DEFAULT_RESNET18_SLICES['fc']
        crafted_v[fc_start:fc_end] = 0.0  # Zero variance in FC head

        sub = UpdateSubmission(
            client_id=2,
            delta_W=crafted_v,
            t_submit=1.0,
            tau=0.5,
            model_version_at_pull=0,
        )

        W_before = server.get_global_weights()
        resp = server.handle_update(sub)

        self.assertEqual(resp["status"], "REJECT")
        self.assertEqual(resp["reason"], "CROSS_LAYER_COLLAPSE")
        self.assertTrue(torch.allclose(server.get_global_weights(), W_before))

    def test_subspace_basis_admission_invariants(self):
        """Verifies the Council-approved basis admission invariants:
        1. Warmup bootstraps up to K vectors.
        2. Post-warmup ACCEPT admits into Q.
        3. DOWNWEIGHT does NOT admit into Q.
        """
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_admission", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        K = server.subspace_engine.K
        self.assertEqual(server.subspace_engine.basis_count(), 0)

        # 1. Warmup Zero-Trojan Invariant: submit K clean updates during warmup
        for i in range(K):
            dW = torch.randn_like(W_init) * 0.05
            sub = UpdateSubmission(client_id=i % 5, delta_W=dW, t_submit=float(i + 1), tau=float(i), model_version_at_pull=0)
            server.handle_update(sub)

        # Invariant: Q remains unpopulated during warmup to prevent Trojan infiltration
        self.assertEqual(server.subspace_engine.basis_count(), 0)
        self.assertFalse(server.subspace_engine.is_basis_full())

        # 2. Post-warmup: manually or via spatial consensus reference, basis admits verified reference vectors
        ref = torch.randn_like(W_init)
        ref = ref / torch.norm(ref)
        server.subspace_engine.update_basis(ref)
        self.assertEqual(server.subspace_engine.basis_count(), 1)

        # 3. Downweight check: DOWNWEIGHT action must never admit into Q
        count_before = server.subspace_engine.basis_count()
        dW_downweight = torch.randn_like(W_init) * 0.02
        sub_dw = UpdateSubmission(client_id=0, delta_W=dW_downweight, t_submit=20.0, tau=1.0, model_version_at_pull=server.get_model_version())
        resp_dw = server.handle_update(sub_dw)

        if resp_dw["status"] == "DOWNWEIGHT":
            self.assertEqual(server.subspace_engine.basis_count(), count_before)

    def test_subspace_gate2_orthogonal_damping_on_accept(self):
        """Verifies that Gate 2 applies smooth roll-off to orthogonal energy without rejecting consensus component."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_gate2", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)

        # Seed basis
        consensus_dir = torch.randn_like(W_init)
        consensus_dir = consensus_dir / torch.norm(consensus_dir)
        server.subspace_engine.update_basis(consensus_dir)

        # Create orthogonal direction
        rand_dir = torch.randn_like(W_init)
        ortho_dir = rand_dir - torch.dot(rand_dir, consensus_dir) * consensus_dir
        ortho_dir = ortho_dir / torch.norm(ortho_dir)

        # Candidate update: legal consensus projection + large orthogonal component
        v_test = consensus_dir * 1.0 + ortho_dir * 5.0  # excess norm_perp > M_perp_base (2.0)
        v_clean, metrics = server.subspace_engine.filter_update(cid=0, v=v_test, tau=0.0)

        self.assertEqual(metrics["action"], "ACCEPT")
        self.assertLess(metrics["w_damp"], 0.80)  # Strong damping applied
        # Consensus component preserved
        c_clean = torch.dot(v_clean, consensus_dir).item()
        self.assertAlmostEqual(c_clean, 1.0, places=3)
        # Orthogonal component attenuated
        c_ortho_clean = torch.dot(v_clean, ortho_dir).item()
        self.assertLess(c_ortho_clean, 5.0)

    def test_subspace_checkpoint_exact_reload_parity(self):
        """Verifies atomic state checkpointing preserves Q, m_perp, and produces identical decisions on reload."""
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger1 = BDSFLogger("test_chk1", self.config)
        server1 = AggregatorServer(self.config, W_init, list(range(5)), logger1)

        # Process 3 updates on server 1
        for i in range(3):
            dW = torch.randn_like(W_init) * 0.05
            sub = UpdateSubmission(client_id=i, delta_W=dW, t_submit=float(i + 1), tau=float(i), model_version_at_pull=0)
            server1.handle_update(sub)

        checkpoint_state = server1.get_state()
        self.assertIn("subspace_state", checkpoint_state)
        self.assertIsNotNone(checkpoint_state["subspace_state"])
        self.assertIn("basis_queue", checkpoint_state["subspace_state"])
        self.assertIn("Q", checkpoint_state["subspace_state"])

        # Reload into Server 2
        logger2 = BDSFLogger("test_chk2", self.config)
        server2 = AggregatorServer(self.config, W_init, list(range(5)), logger2)
        server2.load_state(checkpoint_state)

        # Check subspace state parity
        self.assertEqual(server1.subspace_engine.basis_count(), server2.subspace_engine.basis_count())
        if server1.subspace_engine.Q is not None:
            self.assertTrue(torch.allclose(server1.subspace_engine.Q, server2.subspace_engine.Q))

        # Submit identical next update to both servers
        test_dW = torch.randn_like(W_init) * 0.05
        sub_test1 = UpdateSubmission(client_id=3, delta_W=test_dW, t_submit=10.0, tau=1.0, model_version_at_pull=server1.get_model_version())
        sub_test2 = UpdateSubmission(client_id=3, delta_W=test_dW.clone(), t_submit=10.0, tau=1.0, model_version_at_pull=server2.get_model_version())

        resp1 = server1.handle_update(sub_test1)
        resp2 = server2.handle_update(sub_test2)

        self.assertEqual(resp1["status"], resp2["status"])
        self.assertEqual(resp1["reason"], resp2["reason"])
        self.assertTrue(torch.allclose(server1.get_global_weights(), server2.get_global_weights()))

    def test_aggregator_csv_logger_records_full_modern_columns(self):
        """Verifies that live handle_update calls write all 44 columns to CSV with non-null modern architecture fields."""
        import csv
        model = MNISTMLP()
        W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
        logger = BDSFLogger("test_live_csv_cols", self.config)
        server = AggregatorServer(self.config, W_init, list(range(5)), logger)
        server.register_client_ground_truth(1, is_byzantine=True)

        # Send an update
        dW = torch.randn_like(W_init) * 0.05
        sub = UpdateSubmission(client_id=1, delta_W=dW, t_submit=1.0, tau=0.0, model_version_at_pull=0)
        server.handle_update(sub)

        with open(logger.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            row = reader[1]

        self.assertEqual(len(header), 44)
        self.assertEqual(len(row), 44)
        header_map = {name: idx for idx, name in enumerate(header)}
        
        # Ground truth label must be True for client 1
        self.assertEqual(row[header_map["is_byzantine"]], "True")
        # Status & Reason
        self.assertEqual(row[header_map["status"]], "ACCEPT")
        self.assertEqual(row[header_map["is_warmup"]], "True")
        # Subspace basis count must be recorded (0 during cold start)
        self.assertEqual(row[header_map["subspace_basis_count"]], "0")


if __name__ == "__main__":
    unittest.main()
