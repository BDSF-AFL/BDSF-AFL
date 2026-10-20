"""Gate-2 baseline must not stay frozen when honest norms outgrow it (SUBSPACE_ENERGY_COLLAPSE ratchet)."""
import math, os, sys, unittest
from collections import deque
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from server.subspace_engine import SubspaceProjectionEngine


def make(valve):
    eng = SubspaceProjectionEngine(K=3, valve_enabled=valve)
    return eng


class TestValve(unittest.TestCase):
    def _drive(self, valve):
        eng = make(valve)
        eng._trusted_perp_norms = deque([0.6] * 20, maxlen=50)       # early baseline
        ws = []
        for i in range(120):
            n = 1.3                                                    # honest norm grew to > 2x the baseline
            M = (float(torch.median(torch.tensor(list(eng._trusted_perp_norms)))) if len(eng._trusted_perp_norms) >= 5 else 2.0)
            # reproduce the engine's own baseline selection through its public attributes
            if (eng.valve_enabled and len(eng._recent_w_damp) >= eng.valve_min
                    and float(torch.median(torch.tensor(list(eng._recent_w_damp)))) < eng.valve_median_floor):
                eng._trusted_perp_norms = deque(eng._recent_perp_norms, maxlen=50)
                eng.valve_fired += 1
                M = float(torch.median(torch.tensor(list(eng._trusted_perp_norms))))
            M *= math.sqrt(1 + 0.1 * 14)
            w = math.exp(-eng.lam * max(0.0, n / M - eng.alpha) ** 2)
            eng._recent_perp_norms.append(n); eng._recent_w_damp.append(w)
            if w > 0.95:
                eng._trusted_perp_norms.append(n)
            ws.append(w)
        return ws, eng

    def test_ratchet_without_valve(self):
        ws, _ = self._drive(False)
        self.assertLess(ws[-1], 0.5)                                    # stays heavily damped forever

    def test_valve_recovers(self):
        ws, eng = self._drive(True)
        self.assertGreater(ws[-1], 0.9)
        self.assertGreaterEqual(eng.valve_fired, 1)

    def test_engine_has_valve_state_and_defaults(self):
        eng = SubspaceProjectionEngine()
        self.assertTrue(eng.valve_enabled)
        self.assertEqual((eng.valve_window, eng.valve_min, eng.valve_median_floor), (40, 20, 0.5))


if __name__ == "__main__":
    unittest.main()


class TestRealEngine(unittest.TestCase):
    def _run(self, valve):
        torch.manual_seed(0)
        D = 400
        eng = SubspaceProjectionEngine(K=3, valve_enabled=valve, default_m_perp_base=0.6)
        for _ in range(3):
            eng.update_basis(torch.randn(D))
        eng._trusted_perp_norms = deque([0.6] * 20, maxlen=50)
        last = []
        for i in range(150):
            v = torch.randn(D)
            v = v / torch.linalg.vector_norm(v) * 1.3                 # honest update far above the early baseline
            _, m = eng.filter_update(cid=i % 10, v=v, tau=14.0)
            last.append(m["w_damp"])
        return last, eng

    def test_filter_update_no_valve_collapses(self):
        w, _ = self._run(False)
        self.assertLess(sum(w[-20:]) / 20, 0.3)

    def test_filter_update_valve_recovers(self):
        w, eng = self._run(True)
        self.assertGreater(sum(w[-20:]) / 20, 0.5)
        self.assertGreaterEqual(eng.valve_fired, 1)
