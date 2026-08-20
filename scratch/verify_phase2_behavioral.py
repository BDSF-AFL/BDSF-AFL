import sys
sys.path.insert(0, ".")
import os
import csv
import copy
import yaml
import torch
import numpy as np

from server.behavioral_memory import BehavioralMemoryManager, ClientBehavioralProfile
from shared.types import BehavioralEvidence
from simulation.environment import SimulationEnvironment


def test_1_side_effect_free():
    print("\n[TEST 1] Testing Side-Effect-Free Behavioral Evidence Extraction...")
    config = {"behavioral_history_size": 10, "behavioral_min_history": 3}
    bm = BehavioralMemoryManager(config)

    # Add 4 updates for client 0
    base_v = torch.randn(100)
    for i in range(4):
        bm.on_accept(client_id=0, delta_W=base_v + 0.01 * torch.randn(100))

    profile = bm.get_or_create_profile(0)
    depth_before = profile.depth
    mem_lens = len(profile.gradient_memory)
    norm_lens = len(profile.norm_history)

    # Call extract_evidence multiple times
    ev1 = bm.extract_evidence(client_id=0, delta_W=torch.randn(100), g_i=1.0, client_gap_history=[1.0, 1.1, 0.9, 1.0])
    ev2 = bm.extract_evidence(client_id=0, delta_W=torch.randn(100), g_i=5.0, client_gap_history=[1.0, 1.1, 0.9, 1.0])

    assert profile.depth == depth_before, "FAIL: extract_evidence changed profile depth!"
    assert len(profile.gradient_memory) == mem_lens, "FAIL: extract_evidence mutated gradient_memory!"
    assert len(profile.norm_history) == norm_lens, "FAIL: extract_evidence mutated norm_history!"
    print("  -> PASSED: extract_evidence is 100% side-effect free.")


def test_2_synthetic_behavioral_signals():
    print("\n[TEST 2] Testing Synthetic Directional, Norm, and Cadence Signals...")
    config = {"behavioral_history_size": 10, "behavioral_min_history": 3}
    bm = BehavioralMemoryManager(config)

    # Seed client 1 with consistent direction and norm=1.0
    u = torch.randn(200)
    u = u / torch.norm(u)

    for i in range(5):
        bm.on_accept(client_id=1, delta_W=u * 1.0)

    # Case A: Matching direction & norm
    matching_v = u * 1.0
    ev_a = bm.extract_evidence(client_id=1, delta_W=matching_v, g_i=1.0, client_gap_history=[1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"  Case A (Matching update)   -> sim_self_mean: {ev_a.sim_self_mean:.4f}, norm_dev: {ev_a.norm_deviation_self:.4f}, cadence_dev: {ev_a.cadence_consistency:.4f}")
    assert ev_a.sim_self_mean > 0.99, f"FAIL: Expected sim_self_mean > 0.99, got {ev_a.sim_self_mean}"
    assert ev_a.sim_self_max > 0.99, f"FAIL: Expected sim_self_max > 0.99, got {ev_a.sim_self_max}"
    assert ev_a.norm_deviation_self < 0.05, f"FAIL: Expected low norm_dev, got {ev_a.norm_deviation_self}"

    # Case B: Orthogonal direction
    # Generate vector orthogonal to u
    rand_v = torch.randn(200)
    ortho_v = rand_v - torch.dot(rand_v, u) * u
    ortho_v = ortho_v / torch.norm(ortho_v)
    ev_b = bm.extract_evidence(client_id=1, delta_W=ortho_v, g_i=1.0, client_gap_history=[1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"  Case B (Orthogonal update) -> sim_self_mean: {ev_b.sim_self_mean:.4f}, sim_self_max: {ev_b.sim_self_max:.4f}")
    assert abs(ev_b.sim_self_mean) < 0.15, f"FAIL: Expected low cosine similarity, got {ev_b.sim_self_mean}"

    # Case C: Norm outlier (3.0 vs 1.0)
    outlier_norm_v = u * 3.0
    ev_c = bm.extract_evidence(client_id=1, delta_W=outlier_norm_v, g_i=1.0, client_gap_history=[1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"  Case C (Norm outlier)      -> norm_dev: {ev_c.norm_deviation_self:.4f}")
    assert ev_c.norm_deviation_self > 10.0, f"FAIL: Expected high norm_dev on 3x norm, got {ev_c.norm_deviation_self}"

    # Case D: Cadence outlier (g_i=10.0 vs history of 1.0s)
    ev_d = bm.extract_evidence(client_id=1, delta_W=matching_v, g_i=10.0, client_gap_history=[1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"  Case D (Cadence outlier)   -> cadence_dev: {ev_d.cadence_consistency:.4f}")
    assert ev_d.cadence_consistency > 10.0, f"FAIL: Expected high cadence_dev on 10x gap, got {ev_d.cadence_consistency}"
    print("  -> PASSED: Synthetic behavioral evidence accurately reflects directional, norm, and cadence shifts.")


def test_3_bounded_memory():
    print("\n[TEST 3] Testing Bounded Deque Memory Limit...")
    config = {"behavioral_history_size": 10, "behavioral_min_history": 3}
    bm = BehavioralMemoryManager(config)

    # Insert 35 updates
    for i in range(35):
        bm.on_accept(client_id=2, delta_W=torch.randn(50))

    profile = bm.get_or_create_profile(2)
    assert len(profile.gradient_memory) == 10, f"FAIL: Expected 10 items, got {len(profile.gradient_memory)}"
    assert len(profile.norm_history) == 10, f"FAIL: Expected 10 norms, got {len(profile.norm_history)}"
    assert profile.depth == 10
    assert profile.total_accepted == 35
    print("  -> PASSED: Memory remains strictly bounded at behavioral_history_size (10).")


def test_4_cpu_only_memory_safety():
    print("\n[TEST 4] Testing CPU-Only Storage & No Grad Retention...")
    config = {"behavioral_history_size": 10, "behavioral_min_history": 3}
    bm = BehavioralMemoryManager(config)

    # Create dummy tensor with grad
    v = torch.randn(100, requires_grad=True)
    if torch.cuda.is_available():
        v = v.cuda()

    bm.on_accept(client_id=3, delta_W=v)
    profile = bm.get_or_create_profile(3)
    stored_t = profile.gradient_memory[0]

    assert stored_t.device.type == "cpu", f"FAIL: Tensor stored on {stored_t.device}, expected CPU!"
    assert stored_t.requires_grad is False, "FAIL: requires_grad was True in stored tensor!"
    print(f"  -> PASSED: Stored tensor device: {stored_t.device}, requires_grad: {stored_t.requires_grad}.")


def test_5_simulation_and_csv_verification():
    print("\n[TEST 5] Running Simulation Verification for Phase 2 Telemetry...")
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
    test_cfg["behavioral_history_size"] = 10
    test_cfg["behavioral_min_history"] = 3
    test_cfg["log_dir"] = "logs/phase2_verify/"
    os.makedirs(test_cfg["log_dir"], exist_ok=True)

    # Clean old CSVs in verify directory
    for f_name in os.listdir(test_cfg["log_dir"]):
        if f_name.endswith(".csv"):
            os.remove(os.path.join(test_cfg["log_dir"], f_name))

    for attack in ["NONE", "COMPOUND"]:
        print(f"\n--- Running Phase 2 Simulation for {attack} ---")
        env = SimulationEnvironment(config=test_cfg, attack_type=attack, seed=42)
        results = env.run()

        csv_path = os.path.join(test_cfg["log_dir"], f"{attack}_42_updates.csv")
        assert os.path.exists(csv_path), f"FAIL: CSV {csv_path} was not created!"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Verify all 25 columns exist in exact required sequence
            expected_cols = [
                "round", "client_id", "status", "reason",
                "g_i", "I_i", "P_i",
                "lower_fence", "upper_fence", "fence_margin", "client_z_score", "is_burn_in",
                "sim_global", "norm_raw", "norm_clipped", "norm_ratio_median", "dynamic_bound_C", "reference_available",
                "weight", "action",
                "sim_self_mean", "sim_self_max", "norm_deviation_self", "cadence_consistency", "history_depth"
            ]
            for i, col in enumerate(expected_cols):
                assert fieldnames[i] == col, f"FAIL: Column {i} mismatch! Found {fieldnames[i]}, expected {col}"

            rows = list(reader)
            print(f"  Total logged updates: {len(rows)}")

            # Check that behavioral evidence is populated once depth >= 3
            populated_rows = [r for r in rows if r["sim_self_mean"] != ""]
            print(f"  Updates with populated behavioral memory (depth >= 3): {len(populated_rows)}")
            assert len(populated_rows) > 0, "FAIL: Behavioral evidence never populated!"

            for r in populated_rows:
                assert float(r["sim_self_mean"]) >= -1.0 and float(r["sim_self_mean"]) <= 1.0001
                assert int(r["history_depth"]) >= 3 and int(r["history_depth"]) <= 10

    print("\n" + "=" * 80)
    print("ALL PHASE 2 BEHAVIORAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_1_side_effect_free()
    test_2_synthetic_behavioral_signals()
    test_3_bounded_memory()
    test_4_cpu_only_memory_safety()
    test_5_simulation_and_csv_verification()
