"""LR-schedule unit regression: honest AND Byzantine clients must feed the trainer a ROUND index, not an update count."""
import os
import re
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.lr_schedule import lr_schedule_round


def srv(updates, **cfg):
    return types.SimpleNamespace(round_number=updates, config=cfg)


class TestLrScheduleRound(unittest.TestCase):
    def test_updates_converted_to_rounds(self):
        self.assertAlmostEqual(lr_schedule_round(srv(250, N_clients=20)), 12.5)
        self.assertAlmostEqual(lr_schedule_round(srv(5000, N_clients=20)), 250.0)

    def test_legacy_unit_update(self):
        self.assertEqual(lr_schedule_round(srv(250, N_clients=20, lr_schedule_unit="update")), 250)

    def test_missing_config_falls_back_to_raw(self):
        self.assertEqual(lr_schedule_round(types.SimpleNamespace(round_number=7)), 7)

    def test_every_trainer_call_site_uses_the_helper(self):
        """Guards the exact bug where only the Byzantine wrapper was converted."""
        offenders = []
        for sub in ("client", "simulation"):
            for fn in os.listdir(os.path.join(ROOT, sub)):
                if fn.endswith(".py"):
                    src = open(os.path.join(ROOT, sub, fn)).read()
                    if re.search(r'current_round\s*=\s*getattr\(\s*self\.server\s*,\s*"round_number"', src):
                        offenders.append(f"{sub}/{fn}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
