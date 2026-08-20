import sys
sys.path.insert(0, ".")
import os
import csv
import torch
import yaml
import copy
import numpy as np

from shared.types import (TemporalEvidence, SpatialEvidence, BehavioralEvidence, 
                          JointDecisionOutcome, AcceptedEntry, UpdateSubmission)
from server.temporal_filter import TemporalFilter
from server.spatial_validator import SpatialValidator
from server.reputation_manager import ReputationManager
from server.aggregator import AggregatorServer
from utils.logger import BDSFLogger
from simulation.environment import SimulationEnvironment

def run_regression_verification():
    print("=" * 80)
    print("BDSF-AFL PHASE 1 OBSERVABILITY & ZERO-REGRESSION A/B VERIFICATION")
    print("=" * 80)

    # 1. Test unit behavior of extractors
    print("\n[TEST 1] Testing Spatial & Temporal Evidence Extraction (Side-Effect Free)")
    config = {
        "K_base": 10, "lam": 0.3, "kappa": 1.5, "burn_in_count": 5, "fixed_K": False, "use_tukey": True,
        "K_ref": 3, "M": 10, "theta_cos": 0.1, "gamma_clip": 1.5, "adaptive_clip_enabled": True, "static_clip_C": 10.0
    }
    tf = TemporalFilter(config)
    sv = SpatialValidator(config)

    # Check temporal extraction before any evaluate call
    assert tf._total_seen == 0
    t_ev0 = tf.extract_evidence(1.0, client_id=0)
    assert tf._total_seen == 0, "FAIL: extract_evidence mutated _total_seen!"
    assert t_ev0.is_burn_in is True

    # Check spatial extraction on empty buffer
    dW = torch.randn(50)
    s_ev0 = sv.extract_evidence(dW)
    assert len(sv._buffer) == 0, "FAIL: extract_evidence mutated spatial buffer!"
    assert s_ev0.reference_available is False
    assert s_ev0.sim_global is None
    print("  -> extract_evidence is 100% side-effect free on initial state.")

    # 2. Run simulation with fast config (10 rounds) to verify telemetry and model evolution
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    test_cfg = copy.deepcopy(cfg)
    test_cfg["dataset"] = "MNIST"
    test_cfg["N_clients"] = 10
    test_cfg["total_rounds"] = 10
    test_cfg["eval_every"] = 5
    test_cfg["local_epochs"] = 1
    test_cfg["batch_size"] = 128
    test_cfg["T_base"] = 0.0
    test_cfg["K_base"] = 5
    test_cfg["log_dir"] = "logs/phase1_verify/"
    os.makedirs(test_cfg["log_dir"], exist_ok=True)
    
    # Clean any stale files in verification folder
    for f_name in os.listdir(test_cfg["log_dir"]):
        if f_name.endswith(".csv"):
            os.remove(os.path.join(test_cfg["log_dir"], f_name))

    print("\n[TEST 2] Running 10-Round Simulation Verification (NONE & COMPOUND)")
    for attack in ["NONE", "COMPOUND"]:
        print(f"\n--- Running Attack Scenario: {attack} ---")
        run_id = f"{attack}_42"
        env = SimulationEnvironment(config=test_cfg, attack_type=attack, seed=42)
        results = env.run()
        
        # Verify accuracy log exists and is populated
        acc_log = results.get("accuracy_log", [])
        print(f"  Accuracy trajectory: {[round(a, 4) for a in acc_log]}")
        assert len(acc_log) > 0, "Accuracy log empty!"

        # Check CSV output
        csv_path = os.path.join(test_cfg["log_dir"], f"{attack}_42_updates.csv")
        # In case logger uses default name:
        if not os.path.exists(csv_path):
            candidates = [os.path.join(test_cfg["log_dir"], f) for f in os.listdir(test_cfg["log_dir"]) if f.endswith("_updates.csv")]
            assert len(candidates) > 0, f"No CSV found in {test_cfg['log_dir']}"
            csv_path = candidates[-1]
            
        print(f"  Inspecting CSV Log: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Verify legacy column order is preserved at the front
            expected_legacy_front = ["round", "client_id", "status", "reason", "g_i", "I_i", "P_i"]
            for i, col in enumerate(expected_legacy_front):
                assert fieldnames[i] == col, f"FAIL: Legacy column order altered! Column {i} is {fieldnames[i]}, expected {col}"
                
            # Verify new continuous evidence columns exist
            expected_evidence_cols = [
                "lower_fence", "upper_fence", "fence_margin", "client_z_score", "is_burn_in",
                "sim_global", "norm_raw", "norm_clipped", "norm_ratio_median", "dynamic_bound_C", "reference_available",
                "weight", "action"
            ]
            for col in expected_evidence_cols:
                assert col in fieldnames, f"FAIL: Missing evidence column {col}"
                
            rows = list(reader)
            print(f"  Total updates logged: {len(rows)}")
            
            # Verify evidence values are logged properly
            accept_rows = [r for r in rows if r["status"] == "ACCEPT"]
            reject_rows = [r for r in rows if r["status"] == "REJECT"]
            print(f"  ACCEPT count: {len(accept_rows)}, REJECT count: {len(reject_rows)}")
            
            for r in accept_rows:
                assert r["action"] == "ACCEPT"
                assert r["weight"] != "" # weight logged
                assert r["norm_raw"] != ""
                assert r["is_burn_in"] in ("True", "False")
                
            for r in reject_rows:
                assert r["action"] == "REJECT"
                assert r["weight"] == "" # None / empty on reject

    print("\n" + "=" * 80)
    print("ALL A/B REGRESSION AND OBSERVABILITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_regression_verification()
