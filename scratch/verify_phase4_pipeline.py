"""BDSF-AFL Phase 4: Fast Pipeline Verification & Smoke Test Suite.

Runs four zero-round fast tests without launching heavy multi-round simulations:
  - Test 1: Synthetic Telemetry Generation, CSV Parsing & Consistency
  - Test 2: Algorithm & Ablation Configuration Mapping Validation
  - Test 3: Metric Calculators & Statistical Aggregation (AUC, FRR, ASR, Mean/Std)
  - Test 4: Full Artifact Generation & File Validation (PNG, PDF, TeX, MD)
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.metrics as metrics
from experiments.run_benchmarks import configure_algorithm
from experiments.run_ablation_matrix import configure_ablation_variant
from experiments.generate_paper_figures import generate_all_publication_figures
from experiments.generate_report_tables import generate_all_report_tables


class TestPhase4Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "logs/phase4_results/test_artifacts/"
        cls.fig_dir = os.path.join(cls.test_dir, "figures/")
        cls.tab_dir = os.path.join(cls.test_dir, "tables/")
        cls.summary_dir = os.path.join(cls.test_dir, "summaries/")
        os.makedirs(cls.fig_dir, exist_ok=True)
        os.makedirs(cls.tab_dir, exist_ok=True)
        os.makedirs(cls.summary_dir, exist_ok=True)

        cls.base_config = {
            "N_clients": 20,
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
            "log_dir": cls.test_dir,
        }

    def test_1_synthetic_csv_and_parsing(self):
        """TEST 1: Verify synthetic CSV generation, parsing, and dataframe schemas."""
        bench_records = []
        algos = ["FedAvg", "FedProx", "Unconstrained_AFL", "Static_Delay_AFL", "Pure_Cosine_AFL", "FoolsGold_AFL", "Legacy_BDSF_AFL", "Proposed_BDSF_AFL"]
        for a in algos:
            for s in [42, 123]:
                bench_records.append({
                    "run_id": f"{a}_COMPOUND_s{s}",
                    "algorithm": a,
                    "attack": "COMPOUND",
                    "alpha": "0.1",
                    "byzantine_fraction": 0.2,
                    "seed": s,
                    "final_accuracy": 0.88 if "Proposed" in a else 0.10,
                    "convergence_round": 32.0 if "Proposed" in a else float("inf"),
                    "wall_clock_seconds": 1000.0,
                    "FRR": 0.04 if "Proposed" in a else 0.26,
                    "ASR": 0.00 if "Proposed" in a else 0.12,
                    "rep_precision": 1.0,
                    "reputation_separation_auc": 0.98 if "Proposed" in a else 0.65,
                    "final_I_mean_honest": 0.98,
                    "final_P_mean_honest": 0.99,
                    "final_I_mean_byzantine": 0.05,
                    "final_P_mean_byzantine": 0.20,
                    "communication_count": 350,
                    "accept_count": 235,
                    "downweight_count": 10,
                    "quarantine_count": 25,
                    "reject_count": 80,
                    "accuracy_trajectory": [0.10, 0.25, 0.50, 0.75, 0.88],
                })

        df_bench = pd.DataFrame(bench_records)
        b_csv_path = os.path.join(self.summary_dir, "test_benchmark_summary.csv")
        df_bench.to_csv(b_csv_path, index=False)

        # Parse back
        loaded = pd.read_csv(b_csv_path)
        self.assertEqual(len(loaded), len(bench_records))
        self.assertIn("algorithm", loaded.columns)
        self.assertIn("FRR", loaded.columns)
        self.assertIn("ASR", loaded.columns)

    def test_2_algorithm_and_ablation_config_mapping(self):
        """TEST 2: Verify that all 8 algorithms and 8 ablation variants configure without errors."""
        # 1. Test 8 Benchmark algorithms
        algos = ["FedAvg", "FedProx", "Unconstrained_AFL", "Static_Delay_AFL", "Pure_Cosine_AFL", "FoolsGold_AFL", "Legacy_BDSF_AFL", "Proposed_BDSF_AFL"]
        for a in algos:
            cfg = configure_algorithm(self.base_config, a)
            self.assertEqual(cfg["algorithm_name"], a)
            if a == "Proposed_BDSF_AFL":
                self.assertEqual(cfg["decision_mode"], "joint")
                self.assertEqual(cfg["warm_start_mode"], "state_maturity")
                self.assertTrue(cfg["enable_quarantine"])
            elif a == "Legacy_BDSF_AFL":
                self.assertEqual(cfg["decision_mode"], "legacy")
                self.assertEqual(cfg["warm_start_mode"], "fixed_burn_in")

        # 2. Test 8 Ablation variants
        for i in range(8):
            v_dict = {"code": f"Abl-{i}", "name": f"Ablation_{i}"}
            if i == 1: v_dict["warm_start_mode"] = "fixed_burn_in"; v_dict["burn_in_count"] = 80
            elif i == 2: v_dict["decision_mode"] = "legacy"
            elif i == 3: v_dict["enable_quarantine"] = False
            elif i == 4: v_dict["theta_anchor_min"] = -1.0
            elif i == 5: v_dict["adaptive_clip_enabled"] = False
            elif i == 6: v_dict["asymmetric_rep"] = False
            elif i == 7: v_dict["warmup_weight_factor"] = 1.0

            cfg_abl = configure_ablation_variant(self.base_config, v_dict)
            self.assertEqual(cfg_abl["ablation_code"], f"Abl-{i}")

    def test_3_metrics_and_reputation_auc(self):
        """TEST 3: Verify metric calculators and reputation separation AUC."""
        class MockRepManager:
            def __init__(self):
                self.scores = {
                    0: (0.95, 1.0),
                    1: (0.98, 0.95),
                    2: (0.92, 0.90),
                    3: (0.05, 0.20),
                    4: (0.02, 0.10),
                }
            def get(self, cid):
                return self.scores.get(cid, (1.0, 1.0))

        mock_rep = MockRepManager()
        honest_ids = {0, 1, 2}
        byz_ids = {3, 4}

        auc = metrics.compute_reputation_separation_auc(mock_rep, honest_ids, byz_ids)
        self.assertAlmostEqual(auc, 1.0, places=4, msg="Perfect separation must yield AUC = 1.0")

        means = metrics.compute_reputation_means(mock_rep, honest_ids, byz_ids)
        self.assertGreater(means["final_I_mean_honest"], 0.90)
        self.assertLess(means["final_I_mean_byzantine"], 0.10)

    def test_4_publication_artifacts_generation(self):
        """TEST 4: Verify generation of all 7 PNG/PDF figures and all 4 TeX/MD tables."""
        # 1. Generate Figures
        generate_all_publication_figures(
            benchmark_summary_csv=None,
            ablation_summary_csv=None,
            output_dir=self.fig_dir
        )
        for i in range(1, 8):
            for ext in ["png", "pdf"]:
                f_path = os.path.join(self.fig_dir, f"fig{i}_*.{ext}")
                matching = [f for f in os.listdir(self.fig_dir) if f.startswith(f"fig{i}_") and f.endswith(f".{ext}")]
                self.assertTrue(len(matching) > 0, f"Missing Figure {i} with extension {ext}")
                full_p = os.path.join(self.fig_dir, matching[0])
                self.assertGreater(os.path.getsize(full_p), 100, f"Figure file {full_p} is empty or invalid")

        # 2. Generate Tables
        generate_all_report_tables(
            benchmark_summary_csv=None,
            ablation_summary_csv=None,
            output_dir=self.tab_dir
        )
        for i in range(1, 5):
            for ext in ["tex", "md"]:
                matching = [f for f in os.listdir(self.tab_dir) if f.startswith(f"table{i}_") and f.endswith(f".{ext}")]
                self.assertTrue(len(matching) > 0, f"Missing Table {i} with extension {ext}")
                full_p = os.path.join(self.tab_dir, matching[0])
                self.assertGreater(os.path.getsize(full_p), 50, f"Table file {full_p} is empty or invalid")


if __name__ == "__main__":
    unittest.main()
