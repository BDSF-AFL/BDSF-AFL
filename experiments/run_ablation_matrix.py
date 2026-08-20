"""BDSF-AFL Phase 4: 7-Way Component Ablation Matrix Runner.

Systematically evaluates and isolates individual architectural components:
  - Abl-0: Full BDSF-AFL (All features active)
  - Abl-1: Remove State-Maturity Gating (Fixed N_burn=80)
  - Abl-2: Remove Joint Decision Engine (Legacy hard sequential binary gates)
  - Abl-3: Remove CPU Quarantine (Immediate rejection of borderline updates)
  - Abl-4: Remove Genesis Anchor (Flat rolling memory only)
  - Abl-5: Remove Adaptive Clipping (Static norm bound C=10.0)
  - Abl-6: Remove Asymmetric Reputation (Symmetric beta_I = beta_P)
  - Abl-7: Remove Warmup Hardening (Unverified early updates accepted with full weight)
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


def configure_ablation_variant(base_config: dict, variant: dict) -> dict:
    """Configures environment parameters corresponding to the target ablation variant."""
    cfg = copy.deepcopy(base_config)
    cfg["ablation_code"] = variant.get("code", "Abl-0")
    cfg["ablation_name"] = variant.get("name", "Full_BDSF_AFL")

    # Apply variant-specific overrides
    for k, v in variant.items():
        if k not in ["code", "name", "description"]:
            cfg[k] = v

    if not variant.get("asymmetric_rep", True):
        cfg["beta_I"] = cfg.get("beta_P", 0.05)

    return cfg


def run_ablation_trial(
    config: dict,
    variant: dict,
    attack_type: str,
    alpha_val: Any,
    byz_fraction: float,
    seed: int,
    target_accuracy: float = 0.85,
    log_dir: str = "logs/phase4_results/ablations/"
) -> dict:
    """Executes a single ablation run and extracts all evaluation metrics."""
    os.makedirs(log_dir, exist_ok=True)
    exp_cfg = configure_ablation_variant(config, variant)
    exp_cfg["attack_type"] = attack_type
    exp_cfg["byz_fraction"] = byz_fraction
    exp_cfg["log_dir"] = log_dir

    if alpha_val == "iid" or alpha_val is None:
        exp_cfg["dirichlet_alpha"] = None
        alpha_label = "iid"
    else:
        exp_cfg["dirichlet_alpha"] = float(alpha_val)
        alpha_label = str(alpha_val)

    run_id = f"{variant['code']}_{variant['name']}_{attack_type}_a{alpha_label}_s{seed}"
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

    # Action distribution and warmup Byzantine counts
    df_rej = pd.DataFrame(rejection_log) if rejection_log else pd.DataFrame()
    act_col = "action" if "action" in df_rej.columns else ("status" if "status" in df_rej.columns else None)
    
    accept_count = int((df_rej[act_col] == "ACCEPT").sum()) if act_col and not df_rej.empty else 0
    dw_count = int((df_rej[act_col] == "DOWNWEIGHT").sum()) if act_col and not df_rej.empty else 0
    quarantine_count = int((df_rej[act_col] == "QUARANTINE").sum()) if act_col and not df_rej.empty else 0
    reject_count = int((df_rej[act_col] == "REJECT").sum()) if act_col and not df_rej.empty else 0

    # Early warmup Byzantine acceptance metric
    warmup_byz_accepts = 0
    if not df_rej.empty and "client_id" in df_rej.columns and "reason" in df_rej.columns:
        byz_mask = df_rej["client_id"].isin(byz_ids)
        warmup_mask = df_rej["reason"].isin(["SPATIAL_WARMUP_ACCEPT", "BURN_IN_ACCEPT"])
        warmup_byz_accepts = int((byz_mask & warmup_mask).sum())

    return {
        "run_id": run_id,
        "variant_code": variant.get("code", "Abl-0"),
        "variant_name": variant.get("name", "Full_BDSF_AFL"),
        "description": variant.get("description", ""),
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
        "warmup_byz_accepts": warmup_byz_accepts,
        "communication_count": len(rejection_log),
        "accept_count": accept_count,
        "downweight_count": dw_count,
        "quarantine_count": quarantine_count,
        "reject_count": reject_count,
        "accuracy_trajectory": accuracy_log,
    }


def run_ablations_from_manifest(
    manifest_path: str,
    variants: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    rounds: Optional[int] = None,
) -> pd.DataFrame:
    """Executes the ablation matrix specified in a manifest YAML file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)["experiment"]

    base_cfg = copy.deepcopy(manifest)
    if rounds:
        base_cfg["total_rounds"] = rounds
    else:
        base_cfg["total_rounds"] = manifest.get("rounds", 50)

    all_variants = manifest.get("variants", [])
    if variants:
        target_variants = [v for v in all_variants if v.get("code") in variants or v.get("name") in variants]
    else:
        target_variants = all_variants

    target_seeds = seeds or manifest.get("seeds", [42])
    attack_type = manifest.get("attack_type", "COMPOUND")
    alpha_val = manifest.get("dirichlet_alpha", 0.1)
    byz_f = manifest.get("byz_fraction", 0.2)

    records = []
    total_runs = len(target_variants) * len(target_seeds)
    print(f"\n=======================================================")
    print(f"STARTING PHASE 4 ABLATION MATRIX: {total_runs} Total Executions")
    print(f"Variants : {[v.get('code') for v in target_variants]}")
    print(f"Attack   : {attack_type} | Alpha={alpha_val} | Byz={byz_f}")
    print(f"Seeds    : {target_seeds}")
    print(f"=======================================================\n")

    run_idx = 0
    for variant in target_variants:
        for seed in target_seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {variant['code']} ({variant['name']}) | Seed={seed}...")
            res = run_ablation_trial(
                config=base_cfg,
                variant=variant,
                attack_type=attack_type,
                alpha_val=alpha_val,
                byz_fraction=byz_f,
                seed=seed,
                target_accuracy=manifest.get("target_accuracy", 0.85),
                log_dir=manifest.get("log_dir", "logs/phase4_results/ablations/")
            )
            records.append(res)
            print(f"       --> Acc={res['final_accuracy']*100:.2f}% | FRR={res['FRR']*100:.2f}% | ASR={res['ASR']*100:.2f}% | WarmupByz={res['warmup_byz_accepts']}")

    df = pd.DataFrame(records)
    summary_path = "logs/phase4_results/summaries/ablation_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\n[SUCCESS] Ablation runs complete. Saved summary to: {summary_path}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDSF-AFL Phase 4 Ablation Runner")
    parser.add_argument("--manifest", type=str, default="experiments/manifests/ablation_matrix.yaml")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    run_ablations_from_manifest(
        manifest_path=args.manifest,
        variants=args.variants,
        seeds=args.seeds,
        rounds=args.rounds,
    )
