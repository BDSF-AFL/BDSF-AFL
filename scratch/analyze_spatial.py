import sys
sys.path.insert(0, ".")
import os
import csv
import yaml
import copy
import torch
import numpy as np
import asyncio
import time
from typing import List

from utils.device_utils import resolve_device, device_name, XLA_AVAILABLE

from simulation.data_partitioner import DataPartitioner
from simulation.environment import SimulationEnvironment, _build_model
from server.spatial_validator import SpatialValidator
from shared.types import AcceptedEntry
from utils.logger import BDSFLogger

# Global variables for patched tracking
client_stats = {}
global_config = {}
global_update_counter = 0

prev_ref = None
prev_client_grad = {}
rejection_encountered = {1: False, 6: False, 16: False}

drift_csv_path = ""
current_run_ref_drift_angles = []
current_run_client_similarities = {1: [], 6: [], 16: []}
current_run_honest_similarities = []

# Monkeypatch AcceptedEntry to track client_id dynamically
original_entry_init = AcceptedEntry.__init__

def patched_entry_init(self, *args, **kwargs):
    original_entry_init(self, *args, **kwargs)
    import inspect
    frame = inspect.currentframe()
    client_id = "Unknown"
    while frame:
        if frame.f_code.co_name == "handle_update":
            submission = frame.f_locals.get("submission")
            if submission:
                client_id = submission.client_id
            break
        frame = frame.f_back
    self.client_id = client_id

AcceptedEntry.__init__ = patched_entry_init

# Hook / Patch BDSFLogger to write outputs with "phase2_{ref_mode}_" prefix
original_logger_init = BDSFLogger.__init__

def patched_logger_init(self, run_id, config):
    ref_mode = config.get("ref_mode", "topk")
    # Prefix run_id so the original init uses f"phase2_{ref_mode}_{run_id}_updates.csv" and correctly writes headers
    original_logger_init(self, f"phase2_{ref_mode}_{run_id}", config)

BDSFLogger.__init__ = patched_logger_init

def print_label_histograms(dataloaders, dataset_name):
    print("\n" + "=" * 80)
    print(f"CLIENT LABEL HISTOGRAMS ({dataset_name})")
    print("=" * 80)
    for i, loader in enumerate(dataloaders):
        subset_indices = loader.dataset.indices
        if dataset_name == "MNIST":
            targets = loader.dataset.dataset.targets.numpy()
        else: # CIFAR10
            targets = np.array(loader.dataset.dataset.targets)
        
        client_targets = [targets[idx] for idx in subset_indices]
        unique, counts = np.unique(client_targets, return_counts=True)
        hist = dict(zip(unique, counts))
        
        hist_str = ", ".join([f"{k}: {hist.get(k, 0):>4}" for k in range(10)])
        total = len(client_targets)
        print(f"Client {i:>2} (total={total:>4}) | {hist_str}")
    print("=" * 80 + "\n")

def compute_cosine_and_angle(A, B):
    if A is None or B is None:
        return 1.0, 0.0
    A_flat = A.flatten().float()
    B_flat = B.flatten().float()
    A_norm = torch.norm(A_flat).item()
    B_norm = torch.norm(B_flat).item()
    if A_norm < 1e-9 or B_norm < 1e-9:
        return 1.0, 0.0
    sim = torch.dot(A_flat, B_flat).item() / (A_norm * B_norm)
    angle = np.degrees(np.arccos(np.clip(sim, -1.0, 1.0)))
    return sim, angle

# Hook / Patch for cosine check
original_cosine_check = SpatialValidator.cosine_check

def patched_cosine_check(self, delta_W: torch.Tensor) -> bool:
    global global_update_counter, prev_ref, prev_client_grad, rejection_encountered
    global global_config, drift_csv_path, current_run_ref_drift_angles, current_run_client_similarities
    global current_run_honest_similarities
    
    global_update_counter += 1
    
    import inspect
    frame = inspect.currentframe()
    client_id = "Unknown"
    I_val = 1.0
    P_val = 1.0
    self_server = None
    
    # Climb stack to find handle_update frame
    while frame:
        if frame.f_code.co_name == "handle_update":
            submission = frame.f_locals.get("submission")
            if submission:
                client_id = submission.client_id
                self_server = frame.f_locals.get("self")
                if self_server:
                    I_val, P_val = self_server.rep_manager.get(client_id)
            break
        frame = frame.f_back
        
    ref = self._build_reference()
    sim = None
    angle = "N/A"
    if ref is not None:
        sim, angle = compute_cosine_and_angle(delta_W, ref)
            
    res = original_cosine_check(self, delta_W)
    
    # Check if client is honest
    is_honest = True
    if self_server is not None and client_id != "Unknown" and client_id in self_server.registry:
        is_honest = not self_server.registry[client_id].is_byzantine

    # Track metrics for stats calculation
    if ref is not None and prev_ref is not None:
        _, ref_drift_angle = compute_cosine_and_angle(ref, prev_ref)
        current_run_ref_drift_angles.append(ref_drift_angle)
        
    if client_id in [1, 6, 16] and sim is not None:
        current_run_client_similarities[client_id].append(sim)
        
    if is_honest and sim is not None:
        current_run_honest_similarities.append(sim)
    
    # Track statistics for drift analysis
    if client_id in [1, 6, 16] and not rejection_encountered[client_id]:
        grad_norm = torch.norm(delta_W).item()
        ref_norm = torch.norm(ref).item() if ref is not None else 0.0
        
        # A) Reference drift rate
        ref_drift_sim, ref_drift_angle = 1.0, 0.0
        if ref is not None and prev_ref is not None:
            ref_drift_sim, ref_drift_angle = compute_cosine_and_angle(ref, prev_ref)
            
        # B) Client drift rate
        client_drift_sim, client_drift_angle = 1.0, 0.0
        if client_id in prev_client_grad:
            client_drift_sim, client_drift_angle = compute_cosine_and_angle(delta_W, prev_client_grad[client_id])
            
        # Check Top-K reference contributors
        contributed = False
        contributors = []
        if ref is not None:
            ranked = sorted(
                self._buffer,
                key=lambda e: e.I_score * e.P_score,
                reverse=True,
            )
            top_k = ranked[: min(self.K_ref, len(ranked))]
            contributors = [getattr(e, "client_id", "?") for e in top_k]
            contributed = client_id in contributors
            
        # Print diagnostic log
        status_str = "ACCEPT" if res else "REJECT"
        sim_str = f"{sim:.4f}" if sim is not None else "N/A"
        angle_str = f"{angle:.1f}°" if isinstance(angle, float) else str(angle)
        ref_drift_sim_str = f"{ref_drift_sim:.4f}"
        ref_drift_angle_str = f"{ref_drift_angle:.1f}°"
        client_drift_sim_str = f"{client_drift_sim:.4f}" if client_id in prev_client_grad else "N/A"
        client_drift_angle_str = f"{client_drift_angle:.1f}°" if client_id in prev_client_grad else "N/A"
        contrib_str = "YES" if contributed else "NO"
        
        stats = client_stats.get(client_id, {"size": "N/A", "skew": "N/A", "class": "N/A"})
        size = stats.get("size", "N/A")
        skew = stats.get("skew", "N/A")
        dominant_class = stats.get("class", "N/A")
        skew_str = f"{skew:.1f}% (Class {dominant_class})" if isinstance(skew, float) else str(skew)
        local_epochs = global_config.get("local_epochs", "N/A")
        margin = (sim - self.theta_cos) if sim is not None else 999.0
        margin_str = f"{margin:+.4f}" if sim is not None else "N/A"
        
        # 1. Save to custom drift CSV
        if drift_csv_path:
            with open(drift_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_update_counter,
                    self_server.round_number if self_server else "N/A",
                    client_id,
                    status_str,
                    size,
                    skew,
                    dominant_class,
                    local_epochs,
                    sim if sim is not None else "",
                    angle if isinstance(angle, float) else "",
                    self.theta_cos,
                    margin if sim is not None else "",
                    grad_norm,
                    I_val,
                    P_val,
                    ref_drift_sim,
                    ref_drift_angle,
                    client_drift_sim,
                    client_drift_angle,
                    contrib_str,
                    str(contributors)
                ])
            
        if not res:
            rejection_encountered[client_id] = True
            
    # Update trackers
    if res:
        prev_client_grad[client_id] = delta_W.clone()
    if ref is not None:
        prev_ref = ref.clone()
        
    return res

SpatialValidator.cosine_check = patched_cosine_check

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def compute_metrics(csv_path, byz_ids, honest_ids, K_ref=10, n_clients=20):
    decisions = {}
    
    if not os.path.exists(csv_path):
        return 0.0, 0.0, 0.0
        
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row["status"]
            if status == "INFO":
                continue
            round_val = int(row["round"])
            cid = int(row["client_id"])
            key = (round_val, cid)
            if key not in decisions:
                decisions[key] = status
                
    byz_accepted = 0
    byz_rejected = 0
    honest_accepted = 0
    honest_rejected = 0
    
    for (round_val, cid), status in decisions.items():
        if cid in byz_ids:
            if status == "REJECT":
                byz_rejected += 1
            else:
                byz_accepted += 1
        elif cid in honest_ids:
            if status == "REJECT":
                honest_rejected += 1
            else:
                honest_accepted += 1
                
    total_honest = honest_accepted + honest_rejected
    total_byz = byz_accepted + byz_rejected
    
    frr = honest_rejected / total_honest if total_honest > 0 else 0.0
    tpr = byz_rejected / total_byz if total_byz > 0 else 0.0

    # Calculate TPR excluding burn-in
    accepted_count = 0
    burn_in_rounds = set()
    rounds_sorted = sorted(list(set(r for r, c in decisions.keys())))
    
    for r in rounds_sorted:
        for c in range(n_clients):
            if (r, c) in decisions and decisions[(r, c)] == "ACCEPT":
                accepted_count += 1
                if accepted_count <= K_ref:
                    burn_in_rounds.add(r)
                    
    cutoff_round = max(burn_in_rounds) if burn_in_rounds else -1
    
    byz_accepted_ex = 0
    byz_rejected_ex = 0
    for (round_val, cid), status in decisions.items():
        if round_val > cutoff_round:
            if cid in byz_ids:
                if status == "REJECT":
                    byz_rejected_ex += 1
                else:
                    byz_accepted_ex += 1
                    
    total_byz_ex = byz_accepted_ex + byz_rejected_ex
    tpr_ex = byz_rejected_ex / total_byz_ex if total_byz_ex > 0 else 0.0

    return frr, tpr, tpr_ex

def main():
    global client_stats, global_config, global_update_counter
    global prev_ref, prev_client_grad, rejection_encountered
    global drift_csv_path, current_run_ref_drift_angles, current_run_client_similarities
    global current_run_honest_similarities
    
    base_config = load_config()
    
    os.makedirs("logs", exist_ok=True)
    
    # --- Device selection: TPU > CUDA > CPU ---
    device = resolve_device(base_config)
    print("\n" + "=" * 80)
    print(f"DEVICE: {device_name(device)}")
    print("=" * 80)
    
    # 1. Setup base data partitions for stats
    # (Removed base_config["dataset"] = "MNIST" to strictly use config.yaml)
    partitioner = DataPartitioner(base_config)
    dataloaders = partitioner.partition()
    print_label_histograms(dataloaders, base_config.get("dataset", "MNIST"))
    
    dataset_name = base_config.get("dataset", "MNIST")
    for i, loader in enumerate(dataloaders):
        subset_indices = loader.dataset.indices
        if dataset_name == "MNIST":
            targets = loader.dataset.dataset.targets.numpy()
        else:
            targets = np.array(loader.dataset.dataset.targets)
        client_targets = [targets[idx] for idx in subset_indices]
        unique, counts = np.unique(client_targets, return_counts=True)
        skew = 100.0 * max(counts) / len(client_targets) if len(client_targets) > 0 else 0.0
        dominant_class = unique[np.argmax(counts)] if len(client_targets) > 0 else "N/A"
        client_stats[i] = {"size": len(client_targets), "skew": skew, "class": dominant_class}

    modes = ["topk"]
    attacks = ["NONE", "T1_HIGH_FREQ", "T2_STRAGGLER", "S1_POISON", "S2_MIMICRY", "ADAPTIVE", "COMPOUND"]
    
    print("\n" + "=" * 80)
    print("SPATIAL ANALYSIS CONFIGURATIONS")
    print("=" * 80)
    display_config = copy.deepcopy(base_config)
    
    important_keys = ["dataset", "num_clients", "total_rounds", "batch_size", "local_epochs", "learning_rate", "theta_cos", "spatial_grace_k", "device"]
    important_config = {k: display_config[k] for k in important_keys if k in display_config}
    
    print(yaml.dump(important_config, default_flow_style=False).strip())
    print(f"Modes: {modes}")
    print(f"Attacks: {attacks}")
    print("=" * 80 + "\n")
    
    ablation_results = {mode: {} for mode in modes}
    
    results_csv_path = "logs/phase2_results.csv"
    with open(results_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "attack_type", "mode", "tpr", "tpr_exclude_burnin", "frr", 
            "final_accuracy", "max_accuracy", "honest_integrity_avg", "honest_pace_avg",
            "byz_integrity_avg", "byz_pace_avg", "ref_drift_avg", "honest_cosine_avg", "runtime"
        ])
    
    for attack in attacks:
        print(f"\n" + "=" * 95)
        print(f"RUNNING COMPARISON BENCHMARKS FOR ATTACK TYPE: {attack}")
        print("=" * 95)
        
        for mode in modes:
            print(f"\nSTARTING BENCHMARK: ref_mode = {mode.upper()} | attack = {attack}")
            print("-" * 60)
            
            config = copy.deepcopy(base_config)
            config["ref_mode"] = mode
            
            if attack == "NONE":
                config["byz_fraction"] = 0.0
            # else:
            #     config["byz_fraction"] = 0.2
                
            global_config = config
            
            # Reset globals
            global_update_counter = 0
            prev_ref = None
            prev_client_grad = {}
            rejection_encountered = {1: False, 6: False, 16: False}
            current_run_ref_drift_angles = []
            current_run_client_similarities = {1: [], 6: [], 16: []}
            current_run_honest_similarities = []
            
            drift_csv_path = f"logs/phase2_{mode}_{attack}_drift.csv"
            with open(drift_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "update_number", "round", "client_id", "status", "size", "skew", "dominant_class",
                    "local_epochs", "sim", "angle", "theta", "margin", "norm", "I_i", "P_i",
                    "ref_drift_sim", "ref_drift_angle", "client_drift_sim", "client_drift_angle",
                    "contributed", "contributors"
                ])
                
            run_start = time.time()
            env = SimulationEnvironment(config=config, attack_type=attack, seed=42)
            res = env.run()
            elapsed_time = time.time() - run_start
            
            # Read CSV outputs for metrics calculation
            csv_path = f"logs/phase2_{mode}_{attack}_42_updates.csv"
            N = config.get("N_clients", 20)
            frr, tpr, tpr_ex = compute_metrics(csv_path, res["byz_ids"], res["honest_ids"], K_ref=config.get("K_ref", 10), n_clients=N)
            final_acc = res["accuracy_log"][-1] if res["accuracy_log"] else 0.0
            
            avg_ref_drift = np.mean(current_run_ref_drift_angles) if current_run_ref_drift_angles else 0.0
            avg_honest_sim = np.mean(current_run_honest_similarities) if current_run_honest_similarities else 0.0
            
            # Extract reputation metrics and max accuracy
            rep_manager = res["rep_manager"]
            byz_ids = res["byz_ids"]
            honest_ids = res["honest_ids"]
            
            honest_i_vals = [rep_manager.get(cid)[0] for cid in honest_ids]
            honest_p_vals = [rep_manager.get(cid)[1] for cid in honest_ids]
            avg_honest_i = np.mean(honest_i_vals) if honest_i_vals else 1.0
            avg_honest_p = np.mean(honest_p_vals) if honest_p_vals else 1.0
            
            if byz_ids:
                byz_i_vals = [rep_manager.get(cid)[0] for cid in byz_ids]
                byz_p_vals = [rep_manager.get(cid)[1] for cid in byz_ids]
                avg_byz_i = np.mean(byz_i_vals)
                avg_byz_p = np.mean(byz_p_vals)
            else:
                avg_byz_i = 0.0
                avg_byz_p = 0.0
                
            max_acc = max(res["accuracy_log"]) if res["accuracy_log"] else 0.0
            
            # Capture minority trajectories (only for NONE run to see drift behavior clearly)
            sim_traj = {}
            if attack == "NONE":
                for cid in [1, 6, 16]:
                    traj = current_run_client_similarities[cid]
                    if len(traj) >= 3:
                        sim_traj[cid] = f"{traj[0]:.2f} -> {traj[len(traj)//2]:.2f} -> {traj[-1]:.2f}"
                    elif len(traj) > 0:
                        sim_traj[cid] = f"{traj[0]:.2f} -> {traj[-1]:.2f}"
                    else:
                        sim_traj[cid] = "N/A"
            
            ablation_results[mode][attack] = {
                "FRR": frr,
                "TPR": tpr,
                "TPR_EX": tpr_ex,
                "acc": final_acc,
                "max_acc": max_acc,
                "honest_i": avg_honest_i,
                "honest_p": avg_honest_p,
                "byz_i": avg_byz_i,
                "byz_p": avg_byz_p,
                "ref_drift": avg_ref_drift,
                "honest_sim": avg_honest_sim,
                "runtime": elapsed_time,
                "traj": sim_traj if attack == "NONE" else None
            }

        # Print Incrementally when this attack finishes
        if attack == "NONE":
            print("\n" + "=" * 95)
            print("HONEST EXPERIMENTS SUMMARY TABLE (NONE ATTACK)")
            print("=" * 95)
            print(f"| {'Mode':<15} | {'FRR (Honest)':<12} | {'Avg Honest Cos':<15} | {'Ref Drift':<10} | {'Acc':<8} | {'Runtime':<10} |")
            print(f"|{'-'*17}|{'-'*14}|{'-'*17}|{'-'*11}|{'-'*10}|{'-'*12}|")
            for m in modes:
                r = ablation_results[m]["NONE"]
                print(f"| {m:<15} | {r['FRR']:>11.2%} | {r['honest_sim']:>14.4f} | {r['ref_drift']:>9.1f}° | {r['acc']:>7.2%} | {r['runtime']:>8.1f}s |")
            print("=" * 95 + "\n")
            
            print("MINORITY SIMILARITY TRAJECTORIES (HONEST RUN)")
            print("=" * 95)
            for m in modes:
                print(f"ref_mode = {m.upper()}:")
                traj = ablation_results[m]["NONE"]["traj"]
                print(f"  - Client 1  (Class 8): {traj[1]}")
                print(f"  - Client 6  (Class 3): {traj[6]}")
                print(f"  - Client 16 (Class 8): {traj[16]}")
            print("=" * 95 + "\n")
        else:
            print("\n" + "=" * 115)
            print(f"ADVERSARIAL SUMMARY TABLE FOR {attack} ATTACK")
            print("=" * 115)
            print(f"| {'Mode':<15} | {'TPR (Overall)':<15} | {'TPR (Post Burn-In)':<18} | {'FRR (Under Attack)':<18} | {'Accuracy':<8} | {'Runtime':<10} |")
            print(f"|{'-'*17}|{'-'*17}|{'-'*20}|{'-'*20}|{'-'*10}|{'-'*12}|")
            for m in modes:
                r = ablation_results[m][attack]
                tpr_str = f"{r['TPR']:>14.2%}"
                tpr_ex_str = f"{r['TPR_EX']:>17.2%}"
                print(f"| {m:<15} | {tpr_str} | {tpr_ex_str} | {r['FRR']:>17.2%} | {r['acc']:>7.2%} | {r['runtime']:>8.1f}s |")
            print("=" * 115 + "\n")
            
        # Append incremental results for this attack to phase2_results.csv
        with open(results_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for m in modes:
                r = ablation_results[m][attack]
                tpr_str = f"{r['TPR']:.6f}" if attack != "NONE" else "N/A"
                tpr_ex_str = f"{r['TPR_EX']:.6f}" if attack != "NONE" else "N/A"
                frr_str = f"{r['FRR']:.6f}"
                byz_i_str = f"{r['byz_i']:.6f}" if attack != "NONE" else "N/A"
                byz_p_str = f"{r['byz_p']:.6f}" if attack != "NONE" else "N/A"
                
                writer.writerow([
                    attack,
                    m,
                    tpr_str,
                    tpr_ex_str,
                    frr_str,
                    f"{r['acc']:.6f}",
                    f"{r['max_acc']:.6f}",
                    f"{r['honest_i']:.6f}",
                    f"{r['honest_p']:.6f}",
                    byz_i_str,
                    byz_p_str,
                    f"{r['ref_drift']:.6f}",
                    f"{r['honest_sim']:.6f}",
                    f"{r['runtime']:.2f}"
                ])
        print(f"Results for attack scenario {attack} appended to {results_csv_path}")

if __name__ == "__main__":
    main()
