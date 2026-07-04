import copy
import pandas as pd
from dataclasses import dataclass
from simulation.environment import SimulationEnvironment
from utils.metrics import (compute_accuracy, compute_attack_success_rate,
                           compute_false_rejection_rate, compute_convergence_time)

@dataclass
class AblationConfig:
    name:           str
    use_tukey:      bool
    adaptive_K:     bool
    top_k_ref:      bool
    asymmetric_rep: bool
    adaptive_clip:  bool

ABLATION_VARIANTS = [
    AblationConfig("Ablation-1_raw_Q1Q3",   use_tukey=False, adaptive_K=True,
                    top_k_ref=True, asymmetric_rep=True, adaptive_clip=True),
    AblationConfig("Ablation-2_fixed_K",     use_tukey=True,  adaptive_K=False,
                    top_k_ref=True, asymmetric_rep=True, adaptive_clip=True),
    AblationConfig("Ablation-3_weighted_ref",use_tukey=True,  adaptive_K=True,
                    top_k_ref=False, asymmetric_rep=True, adaptive_clip=True),
    AblationConfig("Ablation-4_symmetric_rep",use_tukey=True, adaptive_K=True,
                    top_k_ref=True, asymmetric_rep=False, adaptive_clip=True),
    AblationConfig("Ablation-5_static_clip", use_tukey=True,  adaptive_K=True,
                    top_k_ref=True, asymmetric_rep=True, adaptive_clip=False),
    AblationConfig("Full_BDSF_AFL",          use_tukey=True,  adaptive_K=True,
                    top_k_ref=True, asymmetric_rep=True, adaptive_clip=True),
]

def build_ablation_config(base_config: dict, variant: AblationConfig) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["use_tukey"] = variant.use_tukey
    cfg["fixed_K"] = not variant.adaptive_K
    cfg["top_k_ref"] = variant.top_k_ref
    if not variant.asymmetric_rep:
        cfg["beta_I"] = cfg.get("beta_P", 0.05)
    cfg["adaptive_clip_enabled"] = variant.adaptive_clip
    return cfg

def run_ablation(base_config: dict) -> dict:
    SEEDS = [42, 123, 256, 789, 1001]
    
    cfg = copy.deepcopy(base_config)
    cfg["dirichlet_alpha"] = 0.1
    cfg["byz_fraction"] = 0.3
    
    results = {}
    records = []
    
    for variant in ABLATION_VARIANTS:
        for seed in SEEDS:
            var_cfg = build_ablation_config(cfg, variant)
            env = SimulationEnvironment(config=var_cfg, attack_type="COMPOUND", seed=seed)
            res = env.run()
            
            accuracy_log = res.get("accuracy_log", [])
            rejection_log = res.get("rejection_log", [])
            byz_ids = res.get("byz_ids", set())
            honest_ids = res.get("honest_ids", set())
            
            final_acc = accuracy_log[-1] if accuracy_log else 0.0
            frr = compute_false_rejection_rate(rejection_log, honest_ids)
            asr = compute_attack_success_rate(rejection_log, byz_ids)
            conv_time = compute_convergence_time(accuracy_log, 0.85)
            
            key = (variant.name, seed)
            results[key] = {
                "final_accuracy": final_acc,
                "FRR": frr,
                "ASR": asr,
                "convergence_time": conv_time
            }
            
            records.append({
                "variant": variant.name,
                "seed": seed,
                "final_accuracy": final_acc,
                "FRR": frr,
                "ASR": asr,
                "convergence_time": conv_time
            })
            
    df = pd.DataFrame(records)
    df.to_csv(base_config.get("log_dir", "logs/") + "ablation_results.csv", index=False)
    
    return results
