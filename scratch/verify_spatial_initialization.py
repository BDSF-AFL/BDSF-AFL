import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
import math

from server.spatial_validator import SpatialValidator, EPS
from shared.types import AcceptedEntry


class TestSpatialInitialization(unittest.TestCase):
    def setUp(self):
        self.config = {
            "K_ref": 5,
            "M": 15,
            "theta_cos": 0.1,
            "gamma_clip": 1.5,
            "adaptive_clip_enabled": True,
            "static_clip_C": 10.0,
        }

    def test_a_zero_first_update(self):
        """TEST A: Zero first update must not collapse adaptive clipping or reference."""
        sv = SpatialValidator(self.config)
        zero_vec = torch.zeros(50)

        # 1. Evidence extraction on empty buffer
        ev0 = sv.extract_evidence(zero_vec)
        self.assertEqual(ev0.norm_raw, 0.0)
        self.assertEqual(ev0.norm_clipped, 0.0)
        self.assertIsNone(ev0.dynamic_bound_C, "dynamic_bound_C must be None during warmup, not 0.0")
        self.assertIsNone(ev0.norm_ratio_median, "norm_ratio_median must be None during warmup")
        self.assertFalse(ev0.reference_available)
        self.assertIsNone(ev0.sim_global)

        # 2. Adaptive clip on zero input
        clipped_zero = sv.adaptive_clip(zero_vec)
        self.assertEqual(torch.norm(clipped_zero).item(), 0.0)

        # 3. Add zero entry to accepted buffer
        sv.on_accept(AcceptedEntry(delta_W=zero_vec, I_score=1.0, P_score=1.0, client_id=0))

        # 4. Verify positive norms remain empty
        pos_norms = sv._get_positive_norms()
        self.assertEqual(len(pos_norms), 0, "Zero vector must not enter positive clipping history")

        # 5. Extract evidence again - bound must remain uninitialized (None)
        ev1 = sv.extract_evidence(zero_vec)
        self.assertIsNone(ev1.dynamic_bound_C, "dynamic_bound_C must remain None after zero update")
        self.assertFalse(ev1.reference_available)
        self.assertIsNone(ev1.sim_global)

    def test_b_first_nonzero_update_after_zero(self):
        """TEST B: First non-zero update after a zero update must not be clipped to zero."""
        sv = SpatialValidator(self.config)
        zero_vec = torch.zeros(50)
        sv.on_accept(AcceptedEntry(delta_W=zero_vec, I_score=1.0, P_score=1.0, client_id=0))

        # Candidate non-zero vector
        nonzero_vec = torch.ones(50) * 0.5  # norm = sqrt(50 * 0.25) ≈ 3.5355
        raw_norm = torch.norm(nonzero_vec).item()
        self.assertGreater(raw_norm, EPS)

        # In warmup state (only zero in history), update must pass unclipped
        clipped = sv.adaptive_clip(nonzero_vec)
        clipped_norm = torch.norm(clipped).item()
        self.assertAlmostEqual(clipped_norm, raw_norm, places=5, msg="First non-zero update must survive unclipped")

        # Extract evidence for the non-zero update
        ev = sv.extract_evidence(nonzero_vec)
        self.assertAlmostEqual(ev.norm_raw, raw_norm, places=5)
        self.assertAlmostEqual(ev.norm_clipped, raw_norm, places=5)
        self.assertIsNone(ev.dynamic_bound_C, "Bound must remain None until positive history exists")

        # Accept this non-zero update
        sv.on_accept(AcceptedEntry(delta_W=nonzero_vec, I_score=1.0, P_score=1.0, client_id=1))
        pos_norms = sv._get_positive_norms()
        self.assertEqual(len(pos_norms), 1)
        self.assertAlmostEqual(pos_norms[0], raw_norm, places=5)

        # Now adaptive bound must mature to positive value
        ev_after = sv.extract_evidence(nonzero_vec)
        self.assertIsNotNone(ev_after.dynamic_bound_C)
        self.assertGreater(ev_after.dynamic_bound_C, 0.0)
        expected_bound = raw_norm * 1.5
        self.assertAlmostEqual(ev_after.dynamic_bound_C, expected_bound, places=5)

    def test_c_mature_adaptive_clipping(self):
        """TEST C: Mature adaptive clipping correctly scales larger updates and preserves smaller updates."""
        sv = SpatialValidator(self.config)
        torch.manual_seed(42)

        # Add 10 positive gradient updates with typical norm ~ 2.0
        for i in range(10):
            g = torch.randn(50)
            g = g / torch.norm(g) * 2.0  # exact norm 2.0
            sv.on_accept(AcceptedEntry(delta_W=g, I_score=1.0, P_score=1.0, client_id=i))

        pos_norms = sv._get_positive_norms()
        self.assertEqual(len(pos_norms), 10)
        self.assertAlmostEqual(float(np.median(pos_norms)), 2.0, places=5)

        expected_C_t = 2.0 * 1.5  # median * gamma_clip = 3.0

        # Sub-test 1: Update below C_t (norm = 1.5)
        small_vec = torch.randn(50)
        small_vec = small_vec / torch.norm(small_vec) * 1.5
        clipped_small = sv.adaptive_clip(small_vec)
        self.assertAlmostEqual(torch.norm(clipped_small).item(), 1.5, places=5, msg="Small update should not be clipped")

        # Sub-test 2: Update above C_t (norm = 6.0)
        large_vec = torch.randn(50)
        large_vec = large_vec / torch.norm(large_vec) * 6.0
        clipped_large = sv.adaptive_clip(large_vec)
        self.assertAlmostEqual(torch.norm(clipped_large).item(), expected_C_t, places=5, msg="Large update should be clipped to C_t")
        self.assertGreater(torch.norm(clipped_large).item(), 0.0, msg="Clipped output must remain positive non-zero")

    def test_d_zero_entries_cannot_poison_reference(self):
        """TEST D: Zero entries cannot contribute to or poison Top-K spatial reference construction."""
        sv = SpatialValidator(self.config)  # K_ref = 5
        torch.manual_seed(123)

        # Add 10 zero vectors with high reputation
        for i in range(10):
            zero_vec = torch.zeros(50)
            sv.on_accept(AcceptedEntry(delta_W=zero_vec, I_score=1.0, P_score=1.0, client_id=i))

        # Reference should be None because valid positive entries < K_ref (0 < 5)
        ref_none = sv._build_reference()
        self.assertIsNone(ref_none, "Reference must be None when only zero vectors exist")

        # Add 4 valid positive vectors (still < K_ref=5)
        for i in range(4):
            g = torch.randn(50)
            g = g / torch.norm(g) * 1.0
            sv.on_accept(AcceptedEntry(delta_W=g, I_score=0.9, P_score=0.9, client_id=10 + i))

        self.assertIsNone(sv._build_reference(), "Reference must remain None when valid entries (4) < K_ref (5)")

        # Add 5th valid positive vector (now valid entries = 5 >= K_ref)
        g5 = torch.randn(50)
        g5 = g5 / torch.norm(g5) * 1.0
        sv.on_accept(AcceptedEntry(delta_W=g5, I_score=0.9, P_score=0.9, client_id=15))

        ref = sv._build_reference()
        self.assertIsNotNone(ref, "Reference must be built when valid positive entries >= K_ref")
        ref_norm = torch.norm(ref).item()
        self.assertGreater(ref_norm, EPS, "Constructed reference must have positive non-zero norm")
        self.assertTrue(np.isfinite(ref_norm), "Reference norm must be finite")

    def test_e_evidence_consistency(self):
        """TEST E: SpatialEvidence invariants during warmup and mature states."""
        sv = SpatialValidator(self.config)
        torch.manual_seed(999)

        # 1. Warmup state checks
        ev_warmup = sv.extract_evidence(torch.ones(50))
        self.assertFalse(ev_warmup.reference_available)
        self.assertIsNone(ev_warmup.sim_global)
        self.assertIsNone(ev_warmup.dynamic_bound_C)
        self.assertIsNone(ev_warmup.norm_ratio_median)
        self.assertEqual(ev_warmup.norm_clipped, ev_warmup.norm_raw)

        # 2. Populate K_ref valid entries
        for i in range(self.config["K_ref"]):
            g = torch.randn(50) + 1.0  # bias slightly positive
            sv.on_accept(AcceptedEntry(delta_W=g, I_score=1.0, P_score=1.0, client_id=i))

        # 3. Mature state checks
        test_vec = torch.randn(50) + 1.0
        ev_mature = sv.extract_evidence(test_vec)
        self.assertTrue(ev_mature.reference_available)
        self.assertIsNotNone(ev_mature.sim_global)
        self.assertTrue(math.isfinite(ev_mature.sim_global))
        self.assertGreaterEqual(ev_mature.sim_global, -1.0)
        self.assertLessEqual(ev_mature.sim_global, 1.0)

        self.assertIsNotNone(ev_mature.dynamic_bound_C)
        self.assertGreater(ev_mature.dynamic_bound_C, 0.0)

        self.assertIsNotNone(ev_mature.norm_ratio_median)
        self.assertTrue(math.isfinite(ev_mature.norm_ratio_median))
        self.assertGreater(ev_mature.norm_ratio_median, 0.0)
        self.assertLess(ev_mature.norm_ratio_median, 1000.0, "norm_ratio_median must not be inflated by zero-division")


if __name__ == "__main__":
    unittest.main()
