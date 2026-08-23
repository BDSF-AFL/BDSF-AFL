"""BDSF-AFL Phase 4: Standardized Baseline Benchmark Runner.

Executes and benchmarks:
  1. FedAvg (Synchronous Federated Averaging)
  2. FedProx (Synchronous with proximal regularization mu=0.01)
  3. Unconstrained_AFL (Asynchronous FedAvg, no defenses)
  4. Static_Delay_AFL (Asynchronous with staleness dampening)
  5. Pure_Cosine_AFL (Geometric cosine threshold only)
  6. FoolsGold_AFL (Sybil/Poisoning historical cosine defense)
  7. Legacy_BDSF_AFL (Fixed burn-in, hard sequential binary gates)
  8. Proposed_BDSF_AFL (State-Maturity Gating, Joint Decision Engine, CPU Quarantine, Genesis Anchor)
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
from typing import Dict, Any, List, Optional

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SimulationEnvironment
import utils.metrics as metrics


def configure_algorithm(base_config: dict, algo_name: str) -> dict:
    """Configures environment parameters corresponding to the target algorithm."""
    cfg = copy.deepcopy(base_config)
    cfg["algorithm_name"] = algo_name

    if algo_name == "FedAvg":
        cfg["aggregation"] = "fedavg"
        cfg["sync"] = True
        cfg["decision_mode"] = "legacy"
    elif algo_name == "FedProx":
        cfg["aggregation"] = "fedavg"
        cfg["sync"] = True
        cfg["fedprox_mu"] = 0.01
        cfg["decision_mode"] = "legacy"
    elif algo_name == "Unconstrained_AFL":
        cfg["aggregation"] = "afl_unconstrained"
        cfg["sync"] = False
        cfg["decision_mode"] = "legacy"
        cfg["use_tukey"] = False
        cfg["top_k_ref"] = False
        cfg["adaptive_clip_enabled"] = False
    elif algo_name == "Static_Delay_AFL":
        cfg["aggregation"] = "static_delay_afl"
        cfg["sync"] = False
        cfg["decision_mode"] = "legacy"
        cfg["use_tukey"] = False
        cfg["top_k_ref"] = False
    elif algo_name == "Pure_Cosine_AFL":
        cfg["aggregation"] = "pure_cosine"
        cfg["sync"] = False
        cfg["decision_mode"] = "legacy"
        cfg["use_tukey"] = False
        cfg["top_k_ref"] = True
    elif algo_name == "FoolsGold_AFL":
        cfg["aggregation"] = "foolsgold"
        cfg["sync"] = False
        cfg["decision_mode"] = "legacy"
        cfg["use_tukey"] = False
    elif algo_name == "Legacy_BDSF_AFL":
        cfg["sync"] = False
        cfg["decision_mode"] = "legacy"
        cfg["warm_start_mode"] = "fixed_burn_in"
        cfg["burn_in_count"] = cfg.get("burn_in_count", 80)
        cfg["use_tukey"] = True
        cfg["top_k_ref"] = True
        cfg["adaptive_clip_enabled"] = True
    elif algo_name == "Proposed_BDSF_AFL":
        cfg["sync"] = False
        cfg["decision_mode"] = "joint"
        cfg["warm_start_mode"] = "state_maturity"
        cfg["enable_quarantine"] = True
        cfg["use_tukey"] = True
        cfg["top_k_ref"] = True
        cfg["adaptive_clip_enabled"] = cfg.get("adaptive_clip_enabled", False)
        cfg["warmup_weight_factor"] = 0.5
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return cfg


def run_benchmark_trial(
    config: dict,
    algo_name: str,
    attack_type: str,
    alpha_val: Any,
    byz_fraction: float,
    seed: int,
    target_accuracy: float = 0.85,
    log_dir: str = "logs/phase4_results/benchmarks/"
) -> dict:
    """Executes a single benchmark run and extracts all evaluation metrics."""
    os.makedirs(log_dir, exist_ok=True)
    exp_cfg = configure_algorithm(config, algo_name)
    exp_cfg["attack_type"] = attack_type
    exp_cfg["byz_fraction"] = byz_fraction
    exp_cfg["log_dir"] = log_dir

    if alpha_val == "iid" or alpha_val is None:
        exp_cfg["dirichlet_alpha"] = None
        alpha_label = "iid"
    else:
        exp_cfg["dirichlet_alpha"] = float(alpha_val)
        alpha_label = str(alpha_val)

    run_id = f"{algo_name}_{attack_type}_a{alpha_label}_f{int(byz_fraction*100)}_s{seed}"
    exp_cfg["run_id"] = run_id

    t_start = time.time()
    env = SimulationEnvironment(config=exp_cfg, attack_type=attack_type, seed=seed)
    results = env.run()
    elapsed = time.time() - t_start

    accuracy_log = results.get("accuracy_log", [])
    rejection_log = results.get("rejection_log", [])
    rep_manager = results.get("rep_manager")
    byz_ids = results.get("byz_ids", set())
    honest_ids = results.get("honest_ids", set())

    final_accuracy = accuracy_log[-1] if accuracy_log else 0.0
    conv_round = metrics.compute_convergence_time(accuracy_log, target=target_accuracy)
    ASR = metrics.compute_attack_success_rate(rejection_log, byz_ids)
    FRR = metrics.compute_false_rejection_rate(rejection_log, honest_ids)
    rep_precision = metrics.compute_reputation_precision(rep_manager, byz_ids, 0.5) if rep_manager else 1.0
    rep_auc = metrics.compute_reputation_separation_auc(rep_manager, honest_ids, byz_ids) if rep_manager else 1.0
    rep_means = metrics.compute_reputation_means(rep_manager, honest_ids, byz_ids) if rep_manager else {
        "final_I_mean_honest": 1.0, "final_P_mean_honest": 1.0, "final_I_mean_byzantine": 0.0, "final_P_mean_byzantine": 0.0
    }

    # Action distribution
    df_rej = pd.DataFrame(rejection_log) if rejection_log else pd.DataFrame()
    act_col = "action" if "action" in df_rej.columns else ("status" if "status" in df_rej.columns else None)
    
    accept_count = int((df_rej[act_col] == "ACCEPT").sum()) if act_col and not df_rej.empty else 0
    dw_count = int((df_rej[act_col] == "DOWNWEIGHT").sum()) if act_col and not df_rej.empty else 0
    quarantine_count = int((df_rej[act_col] == "QUARANTINE").sum()) if act_col and not df_rej.empty else 0
    reject_count = int((df_rej[act_col] == "REJECT").sum()) if act_col and not df_rej.empty else 0

    return {
        "run_id": run_id,
        "algorithm": algo_name,
        "attack": attack_type,
        "alpha": alpha_label,
        "byzantine_fraction": byz_fraction,
        "seed": seed,
        "final_accuracy": final_accuracy,
        "convergence_round": conv_round,
        "wall_clock_seconds": elapsed,
        "FRR": FRR,
        "ASR": ASR,
        "rep_precision": rep_precision,
        "reputation_separation_auc": rep_auc,
        "final_I_mean_honest": rep_means["final_I_mean_honest"],
        "final_P_mean_honest": rep_means["final_P_mean_honest"],
        "final_I_mean_byzantine": rep_means["final_I_mean_byzantine"],
        "final_P_mean_byzantine": rep_means["final_P_mean_byzantine"],
        "communication_count": len(rejection_log),
        "accept_count": accept_count,
        "downweight_count": dw_count,
        "quarantine_count": quarantine_count,
        "reject_count": reject_count,
        "accuracy_trajectory": accuracy_log,
    }


def run_benchmarks_from_manifest(
    manifest_path: str,
    algorithms: Optional[List[str]] = None,
    attacks: Optional[List[str]] = None,
    alphas: Optional[List[Any]] = None,
    byz_fractions: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    rounds: Optional[int] = None,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    early_stopping: Optional[bool] = None,
    target_accuracy: Optional[float] = None,
    patience: Optional[int] = None,
    save_checkpoints: Optional[bool] = None,
    checkpoint_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Executes the benchmark matrix specified in a manifest YAML file."""
    # Load global defaults from config.yaml if available
    config_yaml_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    base_cfg = {}
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)["experiment"]

    base_cfg.update(copy.deepcopy(manifest))
    if rounds:
        base_cfg["total_rounds"] = rounds
    else:
        base_cfg["total_rounds"] = manifest.get("rounds", 50)

    if dataset:
        base_cfg["dataset"] = dataset
    if model:
        base_cfg["model_architecture"] = model.lower()
    if early_stopping is not None:
        base_cfg["early_stopping"] = early_stopping
    if patience is not None:
        base_cfg["early_stopping_patience"] = patience
    if checkpoint_dir:
        base_cfg["checkpoint_dir"] = checkpoint_dir
    for k, v in kwargs.items():
        base_cfg[k] = v

    target_algos = algorithms or manifest.get("algorithms", ["Legacy_BDSF_AFL", "Proposed_BDSF_AFL"])
    target_attacks = attacks or manifest.get("attacks", ["COMPOUND"])
    target_target_acc = target_accuracy if target_accuracy is not None else manifest.get("target_accuracy", 0.85)
    
    # Parse heterogeneity
    raw_het = alphas or [h["alpha"] if isinstance(h, dict) else h for h in manifest.get("heterogeneity", [0.1])]
    target_byz = byz_fractions or manifest.get("byzantine_fractions", [0.2])
    target_seeds = seeds or manifest.get("seeds", [42])

    records = []
    total_runs = len(target_algos) * len(target_attacks) * len(raw_het) * len(target_byz) * len(target_seeds)
    print(f"\n=======================================================")
    print(f"STARTING PHASE 4 BENCHMARK SUITE: {total_runs} Total Executions")
    print(f"Algorithms : {target_algos}")
    print(f"Attacks    : {target_attacks}")
    print(f"Alphas     : {raw_het}")
    print(f"Byz Fracs  : {target_byz}")
    print(f"Seeds      : {target_seeds}")
    print(f"Model Arch : {base_cfg.get('model_architecture', 'resnet18')}")
    print(f"Early Stop : {base_cfg.get('early_stopping', False)} (Patience={base_cfg.get('early_stopping_patience', 5)})")
    print(f"=======================================================\n")

    run_idx = 0
    for algo in target_algos:
        for attack in target_attacks:
            for alpha in raw_het:
                for byz_f in target_byz:
                    for seed in target_seeds:
                        run_idx += 1
                        print(f"[{run_idx}/{total_runs}] Running {algo} | Attack={attack} | alpha={alpha} | byz={byz_f} | seed={seed}...")
                        res = run_benchmark_trial(
                            config=base_cfg,
                            algo_name=algo,
                            attack_type=attack,
                            alpha_val=alpha,
                            byz_fraction=byz_f,
                            seed=seed,
                            target_accuracy=target_target_acc,
                            log_dir=base_cfg.get("log_dir", "logs/phase4_results/benchmarks/")
                        )
                        records.append(res)
                        print(f"       --> Acc={res['final_accuracy']*100:.2f}% | FRR={res['FRR']*100:.2f}% | ASR={res['ASR']*100:.2f}% | Time={res['wall_clock_seconds']:.2f}s")

    df = pd.DataFrame(records)
    summary_path = "logs/phase4_results/summaries/benchmark_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Benchmark runs complete. Saved summary to: {summary_path}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDSF-AFL Phase 4 Benchmark Runner")
    parser.add_argument("--manifest", type=str, default="experiments/manifests/benchmark_matrix.yaml")
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    run_benchmarks_from_manifest(
        manifest_path=args.manifest,
        algorithms=args.algorithms,
        attacks=args.attacks,
        alphas=args.alphas,
        seeds=args.seeds,
        rounds=args.rounds,
    )
