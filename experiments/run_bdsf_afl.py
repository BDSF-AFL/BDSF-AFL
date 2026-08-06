import time
import copy
import pandas as pd
from simulation.environment import SimulationEnvironment
from utils.metrics import (compute_accuracy, compute_attack_success_rate,
                           compute_false_rejection_rate, compute_reputation_precision,
                           compute_convergence_time)
from utils.logger import BDSFLogger

def run_experiment(config: dict, seed: int, attack_type: str) -> dict:
    env = SimulationEnvironment(config=config, attack_type=attack_type, seed=seed)
    results = env.run()
    
    accuracy_log = results.get("accuracy_log", [])
    rejection_log = results.get("rejection_log", [])
    rep_manager = results.get("rep_manager")
    byz_ids = results.get("byz_ids", set())
    honest_ids = results.get("honest_ids", set())
    
    final_accuracy = accuracy_log[-1] if accuracy_log else 0.0
    convergence_time = compute_convergence_time(accuracy_log, target=0.85)
    ASR = compute_attack_success_rate(rejection_log, byz_ids)
    FRR = compute_false_rejection_rate(rejection_log, honest_ids)
    
    rep_precision = 1.0
    if rep_manager:
        rep_precision = compute_reputation_precision(rep_manager, byz_ids, threshold=0.5)
        
    return {
        "seed": seed,
        "attack_type": attack_type,
        "final_accuracy": final_accuracy,
        "convergence_time": convergence_time,
        "ASR": ASR,
        "FRR": FRR,
        "rep_precision": rep_precision,
    }

def run_all(config: dict) -> dict:
    SEEDS = [42, 123, 256, 789, 1001]
    ATTACK_TYPES = ["T1_HIGH_FREQ", "T2_STRAGGLER", "S1_POISON",
                    "S2_MIMICRY", "ADAPTIVE", "COMPOUND"]
    BYZ_FRACS = [0.1, 0.2, 0.3]
    ALPHAS = [0.1, 0.5, float("inf")]
    
    results_dict = {}
    records = []
    
    for alpha in ALPHAS:
        for byz_frac in BYZ_FRACS:
            for attack_type in ATTACK_TYPES:
                for seed in SEEDS:
                    cfg = copy.deepcopy(config)
                    cfg["dirichlet_alpha"] = alpha
                    cfg["byz_fraction"] = byz_frac
                    
                    res = run_experiment(cfg, seed, attack_type)
                    key = (alpha, byz_frac, attack_type, seed)
                    results_dict[key] = res
                    
                    records.append({
                        "alpha": alpha,
                        "byz_frac": byz_frac,
                        "attack_type": attack_type,
                        "seed": seed,
                        "final_accuracy": res["final_accuracy"],
                        "convergence_time": res["convergence_time"],
                        "ASR": res["ASR"],
                        "FRR": res["FRR"],
                        "rep_precision": res["rep_precision"]
                    })
                    
    df = pd.DataFrame(records)
    df.to_csv(config.get("log_dir", "logs/") + "bdsf_afl_results.csv", index=False)
    
    return results_dict
