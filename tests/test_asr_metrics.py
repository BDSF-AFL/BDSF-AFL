"""ASR must count only ATTACKED submissions, post-warmup; poison share is weight-based."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:                                    # real torch has this; the offline numpy shim does not
    import torch.utils.data  # noqa: F401
except Exception:
    import types
    _u = types.ModuleType("torch.utils"); _d = types.ModuleType("torch.utils.data")
    _d.DataLoader = object; _u.data = _d
    sys.modules.setdefault("torch.utils", _u); sys.modules.setdefault("torch.utils.data", _d)
from utils import metrics

BYZ = {0, 1}


def row(cid, status, mod=None, w=None, reason="X", warm=False):
    r = {"client_id": cid, "status": status, "reason": "SPATIAL_WARMUP_ACCEPT" if warm else reason,
         "is_warmup": warm, "weight": w}
    if mod is not None:
        r["attack_modified"] = mod
    return r


class TestASR(unittest.TestCase):
    def test_warmup_rows_excluded(self):
        log = [row(0, "ACCEPT", warm=True)] * 10 + [row(0, "REJECT")] * 10
        self.assertEqual(metrics.compute_attack_success_rate(log, BYZ), 0.0)

    def test_unmodified_byzantine_updates_are_not_attacks(self):
        log = ([row(0, "ACCEPT", mod=False)] * 8 +          # honest-valued update from a Byzantine client
               [row(0, "REJECT", mod=True)] * 9 + [row(1, "DOWNWEIGHT", mod=True)] * 1)
        self.assertAlmostEqual(metrics.compute_attack_success_rate(log, BYZ), 0.1)

    def test_fallback_without_flag_uses_all_post_warmup_byzantine_rows(self):
        log = [row(0, "ACCEPT")] * 3 + [row(0, "REJECT")] * 1
        self.assertAlmostEqual(metrics.compute_attack_success_rate(log, BYZ), 0.75)

    def test_poison_weight_share_is_weight_based(self):
        log = [row(0, "DOWNWEIGHT", mod=True, w=0.5), row(0, "ACCEPT", mod=False, w=4.0),
               row(5, "ACCEPT", w=5.5)]
        self.assertAlmostEqual(metrics.compute_poison_weight_share(log, BYZ), 0.05)

    def test_no_byzantine_rows(self):
        self.assertEqual(metrics.compute_attack_success_rate([row(5, "ACCEPT")], BYZ), 0.0)
        self.assertEqual(metrics.compute_poison_weight_share([], BYZ), 0.0)


if __name__ == "__main__":
    unittest.main()
