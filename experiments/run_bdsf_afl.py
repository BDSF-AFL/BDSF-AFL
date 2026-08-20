"""Ultimate BDSF-AFL Experiment Runner & Benchmark Suite.

Executes asynchronous federated learning simulations under Phase 3 Joint Decision Engine
or Legacy Decision Mode across diverse attack vectors and non-IID heterogeneity partitions.

Saves detailed per-update continuous telemetry to `logs/phase3_verify/` and consolidated
metrics to `logs/phase3_verify/phase3_benchmark_summary.csv`.

Usage Examples:
    # Run default single experiment (COMPOUND attack, seed 42, joint mode)
    python experiments/run_bdsf_afl.py

    # Run quick 10-round verification
    python experiments/run_bdsf_afl.py --rounds 10 --attack COMPOUND

    # Run honest non-IID baseline (NONE) with 50 rounds
    python experiments/run_bdsf_afl.py --rounds 50 --attack NONE

    # Run legacy ablation comparison
    python experiments/run_bdsf_afl.py --rounds 50 --attack COMPOUND --mode legacy

    # Run full Phase 3 benchmark suite across all attacks
    python experiments/run_bdsf_afl.py --all --rounds 50
"""

import sys
import os
import argparse
import copy
import time
import yaml
import torch
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SimulationEnvironment
from utils.logger import BDSFLogger
import utils.metrics as metrics


def print_label_distribution(dataloaders, dataset_name: str):
    """Prints non-IID class distribution across clients for transparency."""
    print("\n" + "=" * 80)
    print(f"CLIENT CLASS DISTRIBUTION ({dataset_name})")
    print("=" * 80)
    for i, loader in enumerate(dataloaders):
        subset_indices = loader.dataset.indices
        if hasattr(loader.dataset.dataset, "targets"):
            targets = np.array(loader.dataset.dataset.targets)
            client_targets = [targets[idx] for idx in subset_indices]
            unique, counts = np.unique(client_targets, return_counts=True)
            hist = dict(zip(unique, counts))
            hist_str = ", ".join([f"{k}: {hist.get(k, 0):>3}" for k in range(10)])
            print(f"Client {i:>2} (total={len(client_targets):>4}) | {hist_str}")
    print("=" * 80 + "\n")


def run_single_experiment(config: dict, attack_type: str, seed: int, mode: str = "joint") -> dict:
    """Executes a single BDSF-AFL experiment and analyzes results in-depth."""
    exp_config = copy.deepcopy(config)
    exp_config["attack_type"] = attack_type
    exp_config["decision_mode"] = mode
    
    log_dir = exp_config.get("log_dir", "logs/phase3_verify/")
    os.makedirs(log_dir, exist_ok=True)
    exp_config["log_dir"] = log_dir

    print("\n" + "#" * 80)
    print(f"STARTING EXPERIMENT: Mode={mode.upper()} | Attack={attack_type} | Seed={seed}")
    print(f"Total Rounds={exp_config.get('total_rounds', 500)} | Clients={exp_config.get('N_clients', 20)} | Dirichlet Alpha={exp_config.get('dirichlet_alpha', 0.1)}")
    print(f"Target Log Directory: {log_dir}")
    print("#" * 80 + "\n")

    t_start = time.time()
    env = SimulationEnvironment(config=exp_config, attack_type=attack_type, seed=seed)
    results = env.run()
    elapsed = time.time() - t_start

    accuracy_log = results.get("accuracy_log", [])
    rejection_log = results.get("rejection_log", [])
    rep_manager = results.get("rep_manager")
    byz_ids = results.get("byz_ids", set())
    honest_ids = results.get("honest_ids", set())

    final_accuracy = accuracy_log[-1] if accuracy_log else 0.0
    convergence_time = metrics.compute_convergence_time(accuracy_log, target=0.85)
    ASR = metrics.compute_attack_success_rate(rejection_log, byz_ids)
    FRR = metrics.compute_false_rejection_rate(rejection_log, honest_ids)
    
    rep_precision = 1.0
    if rep_manager:
        rep_precision = metrics.compute_reputation_precision(rep_manager, byz_ids, threshold=0.5)

    # Convert rejection log into a DataFrame for detailed breakdown
    df = pd.DataFrame(rejection_log)
    action_col = "action" if "action" in df.columns else "status"

    print("\n" + "=" * 80)
    print(f"EXPERIMENT RESULTS: Mode={mode.upper()} | Attack={attack_type} | Seed={seed}")
    print("=" * 80)
    print(f"  Execution Time           : {elapsed:.2f}s")
    print(f"  Final Test Accuracy      : {final_accuracy * 100:.2f}%")
    print(f"  False Rejection Rate (FRR): {FRR * 100:.2f}% (Target: < 5% on honest)")
    print(f"  Attack Success Rate (ASR) : {ASR * 100:.2f}% (Target: 0% on Byzantine)")
    print(f"  True Positive Rate (TPR)  : {(1.0 - ASR) * 100:.2f}%")
    print(f"  Reputation Precision      : {rep_precision * 100:.2f}%")
    print(f"  Convergence Round         : {convergence_time if convergence_time != float('inf') else 'N/A'}")

    if not df.empty and action_col in df.columns:
        sub_df = df[df["status"].isin(["ACCEPT", "DOWNWEIGHT", "QUARANTINE", "REJECT"])]
        print("\n--- MULTI-ACTION SUMMARY (TOTAL SUBMISSIONS) ---")
        action_counts = sub_df[action_col].value_counts()
        for act, cnt in action_counts.items():
            pct = (cnt / len(sub_df)) * 100.0 if len(sub_df) > 0 else 0
            print(f"  {act:<14}: {cnt:>5} ({pct:>5.1f}%)")

        if len(honest_ids) > 0:
            honest_df = sub_df[sub_df["client_id"].isin(honest_ids)]
            if not honest_df.empty:
                h_accept = sum(honest_df[action_col] == "ACCEPT")
                h_dw = sum(honest_df[action_col] == "DOWNWEIGHT")
                h_quar = sum(honest_df[action_col] == "QUARANTINE")
                h_rej = sum(honest_df[action_col] == "REJECT")
                print(f"\n--- HONEST CLIENTS (Submissions: {len(honest_df)}, N={len(honest_ids)}) ---")
                print(f"  Full Accepts       : {h_accept:>4} ({h_accept/len(honest_df)*100:.1f}%)")
                print(f"  Soft Downweights   : {h_dw:>4} ({h_dw/len(honest_df)*100:.1f}%)")
                print(f"  Quarantined Entries: {h_quar:>4} ({h_quar/len(honest_df)*100:.1f}%)")
                print(f"  Rejected (False)   : {h_rej:>4} ({h_rej/len(honest_df)*100:.1f}%)")

        if len(byz_ids) > 0:
            byz_df = sub_df[sub_df["client_id"].isin(byz_ids)]
            if not byz_df.empty:
                b_accept = sum(byz_df[action_col] == "ACCEPT")
                b_dw = sum(byz_df[action_col] == "DOWNWEIGHT")
                b_quar = sum(byz_df[action_col] == "QUARANTINE")
                b_rej = sum(byz_df[action_col] == "REJECT")
                print(f"\n--- BYZANTINE CLIENTS (Submissions: {len(byz_df)}, N={len(byz_ids)}) ---")
                print(f"  Slipped Accepts    : {b_accept:>4} ({b_accept/len(byz_df)*100:.1f}%) [Burn-in: {b_accept}]")
                print(f"  Slipped Downweights: {b_dw:>4} ({b_dw/len(byz_df)*100:.1f}%)")
                print(f"  Quarantined        : {b_quar:>4} ({b_quar/len(byz_df)*100:.1f}%)")
                print(f"  Detected & Rejected: {b_rej:>4} ({b_rej/len(byz_df)*100:.1f}%)")

    print("=" * 80 + "\n")

    return {
        "mode": mode,
        "attack_type": attack_type,
        "seed": seed,
        "final_accuracy": final_accuracy,
        "FRR": FRR,
        "ASR": ASR,
        "TPR": 1.0 - ASR,
        "rep_precision": rep_precision,
        "convergence_time": convergence_time,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Ultimate BDSF-AFL Phase 3 Experiment Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--attack", type=str, default=None, help="Attack type (NONE, COMPOUND, S1_POISON, S2_MIMICRY, T1_HIGH_FREQ, T2_STRAGGLER, ADAPTIVE)")
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "legacy"], help="Decision mode (joint or legacy)")
    parser.add_argument("--rounds", type=int, default=None, help="Override total rounds (e.g. 10, 50, 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--log_dir", type=str, default="logs/phase3_verify/", help="Output directory for logs")
    parser.add_argument("--dataset", type=str, default=None, choices=["MNIST", "CIFAR10"], help="Dataset to use")
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite across all attacks")
    args = parser.parse_args()

    # Load baseline configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.rounds is not None:
        config["total_rounds"] = args.rounds
    if args.dataset is not None:
        config["dataset"] = args.dataset
    config["log_dir"] = args.log_dir
    config["decision_mode"] = args.mode

    os.makedirs(args.log_dir, exist_ok=True)

    if args.all:
        print("\n" + "=" * 80)
        print("RUNNING FULL BENCHMARK SUITE (ALL ATTACKS)")
        print("=" * 80)
        ATTACKS = ["NONE", "COMPOUND", "S1_POISON", "S2_MIMICRY", "T1_HIGH_FREQ", "T2_STRAGGLER"]
        summary_records = []
        for atk in ATTACKS:
            res = run_single_experiment(config, attack_type=atk, seed=args.seed, mode=args.mode)
            summary_records.append(res)
        
        summary_df = pd.DataFrame(summary_records)
        summary_csv = os.path.join(args.log_dir, f"phase3_benchmark_{args.mode}_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n>> Benchmark Complete! Consolidated summary saved to: {summary_csv}\n")
        print(summary_df.to_string(index=False))
    else:
        attack = args.attack if args.attack is not None else config.get("attack_type", "COMPOUND")
        res = run_single_experiment(config, attack_type=attack, seed=args.seed, mode=args.mode)
        
        # Save single run summary
        summary_df = pd.DataFrame([res])
        summary_csv = os.path.join(args.log_dir, f"{args.mode}_{attack}_{args.seed}_summary.csv")
        summary_df.to_csv(summary_csv, index=False)


if __name__ == "__main__":
    main()
