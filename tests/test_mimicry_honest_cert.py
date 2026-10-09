"""Regression tests for the S2/ADAPTIVE mimicry defence v2 and the honest-convergence certificate.

Covers (decision_engine.py):
  * Priority 1c-i  mass-point gate (fast boundary-pinning detection that survives unattacked dips)
  * Priority 1d    boundary probation (thin in-band record / suspects never get full weight)
  * Honest-convergence certificate (TRS-exempt only with manifold evidence AND no pinning evidence)
  * staleness exponent / floor knobs
  * decision-engine state round trip (raw-cosine ledgers survive a checkpoint)
Run:  python -m unittest tests.test_mimicry_honest_cert -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.decision_engine import JointDecisionEngine
from shared.types import TemporalEvidence, SpatialEvidence, BehavioralEvidence


def base_cfg(**over):
    cfg = {
        "theta_cos": 0.15, "theta_self": 0.30, "theta_floor": 0.40, "theta_anchor_min": 0.25,
        "stochastic_jitter_max": 0.0, "warmup_rounds": 0, "spatial_warmup_rounds": 0,
        "trs_reject_thresh": 0.85, "trs_warn_thresh": 0.80, "trs_min_depth": 5,
        "trs_consensus_exempt": False,        # isolate the certificate from the legacy raw-cosine exemption
        "bisection_window": 6, "bisection_variance_thresh": 1e-4, "bisection_band_margin": 0.10,
        "subspace_mu_floor": 0.10, "subspace_energy_floor": 0.40,
        # new behaviour ON (code defaults are OFF so legacy tests are untouched)
        "honest_cert_enabled": True, "pin_probation_enabled": True,
        "pin_mass_tol": 0.0015, "pin_mass_min": 3, "pin_probation_min_samples": 3,
        "pin_probation_weight": 0.15, "pin_suspect_nn": 0.008, "pin_suspect_min_hits": 1,
        "pin_strong_hits": 2,
    }
    cfg.update(over)
    return cfg


TEMP = TemporalEvidence(g_i=1.0, lower_fence=0.5, upper_fence=2.0, fence_margin=0.0,
                        client_z_score=0.0, is_burn_in=False, temporal_mature=True, version_lag=0)


def behav(trs=0.925):
    return BehavioralEvidence(sim_self_mean=0.9, sim_self_max=0.95, history_depth=10, sim_anchor=0.7,
                              behavioral_mature=True, gdv_score=0.02, dbp_score=0.944, trs_score=trs)


def spat(raw, filt=0.30, prc=0.30, tra=0.80):
    return SpatialEvidence(sim_global=filt, sim_global_raw=raw, norm_raw=1.0, norm_clipped=1.0,
                           spatial_mature=True, prc_score=prc, tra_score=tra)


def run(engine, cid, raws, rnd0=1, trs=0.925, mu=0.28, rho=0.85, **kw):
    out = None
    for i, r in enumerate(raws):
        out = engine.evaluate(cid, TEMP, spat(r, **kw), behav(trs), 1.0, 1.0, current_round=rnd0 + i,
                              subspace_mu=mu, subspace_rho_parallel=rho)
    return out


# in-band (0.13..0.25), well spread, no repeated values -> an honest client's in-band record
HONEST_SPREAD = [0.140, 0.165, 0.190, 0.230, 0.205]


class TestMassPointGate(unittest.TestCase):
    def test_exact_boundary_pinning_rejected_after_three_hits(self):
        eng = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False))
        outs = [run(eng, 0, [0.2000]) for _ in range(5)]
        self.assertEqual(outs[3].primary_reason, "BOUNDARY_PINNING_REJECT")
        self.assertEqual(outs[3].diagnostic_features["violation"], "boundary_mass_point")
        self.assertEqual(outs[3].action, "REJECT")

    def test_mass_point_survives_interleaved_unattacked_dips(self):
        """S2 submits honest updates unmodified when natural cos < target. Those dips used to break the
        variance gate (window var > 1e-4) so the attacker kept getting accepted."""
        cfg = base_cfg(honest_cert_enabled=False, pin_probation_enabled=False, bisection_variance_thresh=1e-4)
        eng = JointDecisionEngine(cfg)
        seq = [0.2000, 0.1630, 0.2000, 0.1840, 0.2000, 0.1500, 0.2000]
        outs = [run(eng, 0, [s]) for s in seq]
        # the legacy variance gate (needs 6 clean in-band samples, var<=1e-4) never fires on this sequence
        legacy = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False, pin_mass_min=0))
        legacy_outs = [run(legacy, 0, [s]) for s in seq]
        self.assertNotIn("BOUNDARY_PINNING_REJECT", [o.primary_reason for o in legacy_outs])
        self.assertEqual(outs[-1].primary_reason, "BOUNDARY_PINNING_REJECT")

    def test_honest_spread_is_never_mass_point_rejected(self):
        eng = JointDecisionEngine(base_cfg())
        for r in HONEST_SPREAD * 3:
            out = run(eng, 1, [r])
            self.assertNotEqual(out.primary_reason, "BOUNDARY_PINNING_REJECT")

    def test_mass_gate_can_be_disabled(self):
        eng = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False,
                                           pin_mass_min=0, bisection_variance_thresh=0.0))
        out = None
        for _ in range(8):
            out = run(eng, 0, [0.2000])
        self.assertNotEqual(out.primary_reason, "BOUNDARY_PINNING_REJECT")


class TestProbation(unittest.TestCase):
    def test_thin_in_band_record_gets_probation_weight_not_full_weight(self):
        eng = JointDecisionEngine(base_cfg())
        out = run(eng, 2, [0.200])
        self.assertEqual(out.action, "DOWNWEIGHT")
        self.assertEqual(out.primary_reason, "BOUNDARY_PROBATION_DOWNWEIGHT")
        self.assertAlmostEqual(out.aggregation_weight, 0.15)
        self.assertFalse(out.diagnostic_features["pin_suspect"])      # thin record only: not a suspect

    def test_repeated_value_is_suspect_and_strong_after_two_hits(self):
        eng = JointDecisionEngine(base_cfg(pin_mass_min=0))            # keep gate off to observe probation
        outs = [run(eng, 3, [0.2000]) for _ in range(4)]
        self.assertTrue(outs[1].diagnostic_features["pin_suspect"])
        self.assertFalse(outs[1].diagnostic_features["pin_strong"])
        self.assertTrue(outs[3].diagnostic_features["pin_strong"])

    def test_honest_with_established_spread_is_not_on_probation(self):
        eng = JointDecisionEngine(base_cfg())
        out = run(eng, 4, HONEST_SPREAD)
        self.assertNotEqual(out.primary_reason, "BOUNDARY_PROBATION_DOWNWEIGHT")

    def test_probation_off_by_default_in_code(self):
        eng = JointDecisionEngine({"theta_cos": 0.15, "warmup_rounds": 0, "spatial_warmup_rounds": 0,
                                   "stochastic_jitter_max": 0.0})
        self.assertFalse(eng.pin_probation_enabled)
        self.assertFalse(eng.cert_enabled)


class TestHonestConvergenceCertificate(unittest.TestCase):
    def test_legacy_rejects_honest_converging_client_on_rigidity(self):
        eng = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False))
        out = run(eng, 5, HONEST_SPREAD)
        self.assertEqual(out.primary_reason, "TRAJECTORY_RIGIDITY_REJECT")

    def test_certified_honest_converging_client_is_fully_accepted(self):
        eng = JointDecisionEngine(base_cfg())
        out = run(eng, 5, HONEST_SPREAD)
        self.assertEqual(out.action, "ACCEPT")
        self.assertEqual(out.primary_reason, "FULL_CONSENSUS_ACCEPT")
        self.assertAlmostEqual(out.aggregation_weight, 1.0)

    def test_certificate_needs_manifold_evidence(self):
        eng = JointDecisionEngine(base_cfg())
        out = run(eng, 5, HONEST_SPREAD, mu=0.01, rho=0.20)
        self.assertNotEqual(out.primary_reason, "FULL_CONSENSUS_ACCEPT")

    def test_certificate_needs_coherent_residuals(self):
        eng = JointDecisionEngine(base_cfg())
        out = run(eng, 5, HONEST_SPREAD, prc=0.02)
        self.assertNotEqual(out.primary_reason, "FULL_CONSENSUS_ACCEPT")

    def test_boundary_steering_with_good_manifold_evidence_is_not_certified(self):
        """The failure mode of 'mu>=0.05 and rho>=0.40 => full weight' on its own: a mimicry attacker has
        exactly that manifold evidence. It must never reach FULL_CONSENSUS_ACCEPT."""
        eng = JointDecisionEngine(base_cfg(trs=None) if False else base_cfg())
        reasons = []
        for _ in range(10):
            out = run(eng, 6, [0.2000], trs=0.30)       # TRS low: nothing else blocks it
            reasons.append(out.primary_reason)
        self.assertNotIn("FULL_CONSENSUS_ACCEPT", reasons)
        self.assertIn("BOUNDARY_PINNING_REJECT", reasons)


class TestStaleness(unittest.TestCase):
    def _w(self, **over):
        eng = JointDecisionEngine(base_cfg(**over))
        out = eng.evaluate(7, TEMP, spat(0.40), behav(0.5), 1.0, 1.0, current_round=1,
                           version_lag=15, subspace_mu=0.3, subspace_rho_parallel=0.8)
        return out.aggregation_weight

    def test_default_is_legacy_inverse_sqrt(self):
        self.assertAlmostEqual(self._w(), 1.0 / (16 ** 0.5), places=6)

    def test_exponent_and_floor(self):
        self.assertAlmostEqual(self._w(staleness_exponent=0.25), 16 ** -0.25, places=6)
        self.assertAlmostEqual(self._w(staleness_floor=0.5), 0.5, places=6)


class TestState(unittest.TestCase):
    def test_pinning_ledgers_survive_checkpoint(self):
        a = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False))
        for _ in range(3):
            run(a, 0, [0.2000])
        b = JointDecisionEngine(base_cfg(honest_cert_enabled=False, pin_probation_enabled=False))
        b.load_state(a.get_state())
        out = run(b, 0, [0.2000])      # 4th identical value, 3 priors restored from the checkpoint
        self.assertEqual(out.primary_reason, "BOUNDARY_PINNING_REJECT")

    def test_load_state_accepts_legacy_checkpoint(self):
        e = JointDecisionEngine(base_cfg())
        e.load_state({"suspicion_scores": {"1": 0.2}})
        self.assertEqual(e.suspicion_scores, {1: 0.2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
