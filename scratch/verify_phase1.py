import sys
sys.path.insert(0, ".")
import os
import csv
import torch
import yaml
import copy

from shared.types import (TemporalEvidence, SpatialEvidence, BehavioralEvidence, 
                          JointDecisionOutcome, AcceptedEntry, UpdateSubmission)
from server.temporal_filter import TemporalFilter
from server.spatial_validator import SpatialValidator
from server.reputation_manager import ReputationManager
from server.aggregator import AggregatorServer
from utils.logger import BDSFLogger
from simulation.environment import SimulationEnvironment

def test_evidence_schemas():
    print("=== 1. Testing Evidence Schemas in shared/types.py ===")
    te = TemporalEvidence(
        g_i=1.23,
        lower_fence=0.5,
        upper_fence=2.0,
        fence_margin=0.0,
        client_z_score=0.15,
        is_burn_in=False
    )
    assert te.g_i == 1.23
    assert te.lower_fence == 0.5
    assert te.upper_fence == 2.0
    assert te.fence_margin == 0.0
    assert te.client_z_score == 0.15
    assert te.is_burn_in is False
    print("  [PASS] TemporalEvidence dataclass validated")

    se = SpatialEvidence(
        sim_global=0.85,
        norm_raw=1.2,
        norm_clipped=1.0,
        norm_ratio_median=1.1,
        dynamic_bound_C=1.0,
        reference_available=True
    )
    assert se.sim_global == 0.85
    assert se.norm_raw == 1.2
    assert se.norm_clipped == 1.0
    assert se.reference_available is True
    print("  [PASS] SpatialEvidence dataclass validated")

    be = BehavioralEvidence(
        sim_self_mean=0.92,
        sim_self_max=0.98,
        norm_deviation_self=0.05,
        cadence_consistency=0.95,
        history_depth=10
    )
    assert be.sim_self_mean == 0.92
    assert be.history_depth == 10
    print("  [PASS] BehavioralEvidence dataclass validated")

    jd = JointDecisionOutcome(
        action="ACCEPT",
        primary_reason="FULL_ACCEPT",
        aggregation_weight=1.0,
        force_sync_required=False,
        diagnostic_features={"sim_global": 0.85}
    )
    assert jd.action == "ACCEPT"
    assert jd.aggregation_weight == 1.0
    print("  [PASS] JointDecisionOutcome dataclass validated")

    ae = AcceptedEntry(delta_W=torch.randn(10), I_score=1.0, P_score=1.0, client_id=4)
    assert ae.client_id == 4
    print("  [PASS] AcceptedEntry dataclass with client_id validated")
    print()

def test_temporal_extractor():
    print("=== 2. Testing TemporalFilter.extract_evidence ===")
    config = {
        "K_base": 10,
        "lam": 0.3,
        "kappa": 1.5,
        "burn_in_count": 5,
        "fixed_K": False,
        "use_tukey": True
    }
    tf = TemporalFilter(config)
    
    # Burn-in check
    ev_burn = tf.extract_evidence(1.0, client_id=0)
    assert ev_burn.is_burn_in is True
    assert ev_burn.g_i == 1.0
    print("  [PASS] Burn-in extraction correct")

    # Feed some gaps
    for _ in range(5):
        tf.evaluate(1.0, client_id=0)
        
    for _ in range(10):
        tf.evaluate(1.05, client_id=0)
        
    ev_normal = tf.extract_evidence(1.02, client_id=0)
    assert ev_normal.is_burn_in is False
    assert ev_normal.lower_fence is not None
    assert ev_normal.upper_fence is not None
    assert ev_normal.fence_margin == 0.0  # within bounds
    print(f"  [PASS] Post-burn-in extraction: L={ev_normal.lower_fence:.4f}, U={ev_normal.upper_fence:.4f}, z={ev_normal.client_z_score:.4f}")

    # Outlier gap test
    ev_outlier = tf.extract_evidence(5.0, client_id=0)
    assert ev_outlier.fence_margin > 0.0
    print(f"  [PASS] Outlier margin calculated: {ev_outlier.fence_margin:.4f}")
    print()

def test_spatial_extractor():
    print("=== 3. Testing SpatialValidator.extract_evidence ===")
    config = {
        "K_ref": 3,
        "M": 10,
        "theta_cos": 0.1,
        "gamma_clip": 1.5,
        "adaptive_clip_enabled": True
    }
    sv = SpatialValidator(config)
    dW = torch.randn(50)
    ev_empty = sv.extract_evidence(dW)
    assert ev_empty.reference_available is False
    assert ev_empty.sim_global is None
    assert ev_empty.norm_ratio_median is None
    print("  [PASS] Empty buffer spatial extraction correct")

    # Populate buffer
    ref_dir = torch.randn(50)
    ref_dir = ref_dir / torch.norm(ref_dir)
    for i in range(5):
        sv.on_accept(AcceptedEntry(delta_W=ref_dir * 2.0, I_score=1.0, P_score=1.0, client_id=i))

    ev_populated = sv.extract_evidence(ref_dir * 1.5)
    assert ev_populated.reference_available is True
    assert ev_populated.sim_global is not None
    assert abs(ev_populated.sim_global - 1.0) < 1e-4
    print(f"  [PASS] Populated spatial extraction: sim_global={ev_populated.sim_global:.4f}, norm_raw={ev_populated.norm_raw:.4f}, bound_C={ev_populated.dynamic_bound_C:.4f}")
    print()

def test_10_round_simulation():
    print("=== 4. Running 10-Round End-to-End Simulation Verification ===")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Adjust for ultra-fast local verification (10 rounds)
    test_cfg = copy.deepcopy(cfg)
    test_cfg["total_rounds"] = 10
    test_cfg["eval_every"] = 5
    test_cfg["local_epochs"] = 1
    test_cfg["batch_size"] = 128
    test_cfg["log_dir"] = "logs/phase1_test/"
    
    os.makedirs(test_cfg["log_dir"], exist_ok=True)
    
    print(f"  Config: total_rounds={test_cfg['total_rounds']}, dataset={test_cfg['dataset']}, num_clients={test_cfg['N_clients']}")
    env = SimulationEnvironment(config=test_cfg, attack_type="NONE", seed=42)
    results = env.run()
    
    print(f"  [PASS] 10 rounds completed successfully!")
    print(f"  Final test accuracy: {results['accuracy_log'][-1]:.4f}")
    
    # Verify CSV file output and columns
    csv_files = [f for f in os.listdir(test_cfg["log_dir"]) if f.endswith("_updates.csv")]
    assert len(csv_files) > 0, "No CSV updates log found!"
    csv_file_path = os.path.join(test_cfg["log_dir"], csv_files[0])
    
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"  CSV Headers: {fieldnames}")
        expected_fields = ["round", "client_id", "status", "reason", "g_i", "I_i", "P_i", "sim_global", "sim_self", "norm_ratio", "weight", "action"]
        for field in expected_fields:
            assert field in fieldnames, f"Missing header: {field}"
            
        row_count = 0
        populated_sim_count = 0
        for row in reader:
            row_count += 1
            if row["sim_global"] != "":
                populated_sim_count += 1
                
        print(f"  Logged {row_count} update submissions, {populated_sim_count} with continuous sim_global.")
        assert row_count > 0, "No rows written to CSV!"
    
    print("  [PASS] CSV structure and continuous telemetry validated successfully!")
    print()

if __name__ == "__main__":
    test_evidence_schemas()
    test_temporal_extractor()
    test_spatial_extractor()
    test_10_round_simulation()
    print("========================================")
    print("ALL PHASE 1 VERIFICATION TESTS PASSED!")
    print("========================================")
