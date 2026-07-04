import torch
import numpy as np
import copy
from simulation.environment import SimulationEnvironment
import utils.metrics as M

def _extract_results(env_res):
    acc_log = env_res.get("accuracy_log", [])
    rej_log = env_res.get("rejection_log", [])
    rep_mgr = env_res.get("rep_manager")
    byz_ids = env_res.get("byz_ids", set())
    hon_ids = env_res.get("honest_ids", set())
    
    return {
        "final_accuracy": acc_log[-1] if acc_log else 0.0,
        "convergence_time": M.compute_convergence_time(acc_log, target=0.85),
        "ASR": M.compute_attack_success_rate(rej_log, byz_ids),
        "FRR": M.compute_false_rejection_rate(rej_log, hon_ids),
        "rep_precision": M.compute_reputation_precision(rep_mgr, byz_ids, 0.5) if rep_mgr else 1.0
    }

def run_fedavg(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "fedavg"
    cfg["sync"] = True
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_unconstrained_afl(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "afl_unconstrained"
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_static_delay_afl(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "static_delay_afl"
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_pure_cosine(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "pure_cosine"
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_fedprox(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "fedavg"
    cfg["sync"] = True
    cfg["fedprox_mu"] = 0.01
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_foolsgold(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["aggregation"] = "foolsgold"
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)

def run_bdsf_afl_base(config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["use_tukey"] = False
    cfg["fixed_K"] = True
    cfg["top_k_ref"] = False
    cfg["beta_I"] = cfg.get("beta_P", 0.05)
    cfg["adaptive_clip_enabled"] = False
    
    env = SimulationEnvironment(cfg, attack_type="NONE", seed=seed)
    res = env.run()
    return _extract_results(res)
