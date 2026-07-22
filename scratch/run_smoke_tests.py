import sys
sys.path.insert(0, ".")
import yaml
import copy
import os
import torch
import numpy as np
import asyncio
from simulation.environment import SimulationEnvironment, AttackInjectorWrapper, _build_model
from client.client_node import ClientNode
from server.aggregator import AggregatorServer

# Optimize PyTorch CPU threading for i5-6402P (4 cores / 4 threads) and 8GB RAM
torch.set_num_threads(1)

# Instrument the simulation to print progress at client and round level
class ProgressTracker:
    def __init__(self, N, total_rounds):
        self.N = N
        self.total_rounds = total_rounds
        self.started_clients = set()
        self.finished_clients = set()
        self.current_round = 0
        self.lock = asyncio.Lock()

    async def on_client_start(self, client_id):
        async with self.lock:
            if not self.started_clients:
                self.current_round += 1
                print(f"\n--- Round {self.current_round}/{self.total_rounds} Started ---", flush=True)
            self.started_clients.add(client_id)
            print(f"  [Client {client_id}] training...", flush=True)

    async def on_client_finish(self, client_id, status):
        async with self.lock:
            self.finished_clients.add(client_id)
            if len(self.finished_clients) == self.N:
                print(f"--- Round {self.current_round}/{self.total_rounds} Finished (All clients aggregated) ---", flush=True)
                self.started_clients.clear()
                self.finished_clients.clear()

_progress_tracker = None

# Dictionary to track client completed rounds
client_completed_rounds = {}

def check_early_termination():
    global _active_server, client_completed_rounds
    if _active_server is not None:
        total_rounds = _active_server.config.get("total_rounds", 5)
        N = len(_active_server.client_ids)
        byz_ids = {cid for cid, reg in _active_server.registry.items() if reg.is_byzantine}
        honest_ids = {cid for cid in _active_server.client_ids if cid not in byz_ids}
        
        if honest_ids:
            if all(client_completed_rounds.get(cid, 0) >= total_rounds for cid in honest_ids):
                total_updates = total_rounds * N
                if _active_server.update_counter < total_updates:
                    _active_server.update_counter = total_updates

_orig_client_run = ClientNode.run_one_round
async def custom_client_run(self):
    global _progress_tracker, client_completed_rounds
    if _progress_tracker is not None:
        await _progress_tracker.on_client_start(self.client_id)
    res = await _orig_client_run(self)
    client_completed_rounds[self.client_id] = client_completed_rounds.get(self.client_id, 0) + 1
    check_early_termination()
    if _progress_tracker is not None:
        await _progress_tracker.on_client_finish(self.client_id, res.get("status", "NONE"))
    return res
ClientNode.run_one_round = custom_client_run

_orig_wrapper_run = AttackInjectorWrapper.run_one_round
async def custom_wrapper_run(self):
    global _progress_tracker, client_completed_rounds
    client_id = self.client.client_id
    if _progress_tracker is not None:
        await _progress_tracker.on_client_start(client_id)
    res = await _orig_wrapper_run(self)
    client_completed_rounds[client_id] = client_completed_rounds.get(client_id, 0) + 1
    check_early_termination()
    if _progress_tracker is not None:
        await _progress_tracker.on_client_finish(client_id, res.get("status", "NONE"))
    return res
AttackInjectorWrapper.run_one_round = custom_wrapper_run

_active_server = None

_orig_server_init = AggregatorServer.__init__
def custom_server_init(self, config, W_init, client_ids, logger):
    global _active_server
    _orig_server_init(self, config, W_init, client_ids, logger)
    _active_server = self
AggregatorServer.__init__ = custom_server_init

def get_global_weights_patched(self):
    global _active_server
    if _active_server is not None:
        return _active_server.get_global_weights()
    return None
SimulationEnvironment.get_global_weights = get_global_weights_patched

_orig_env_run = SimulationEnvironment.run
def custom_env_run(self):
    global _progress_tracker, _active_server, client_completed_rounds
    _active_server = None
    client_completed_rounds.clear()
    N = self.config.get("N_clients", 10)
    total_rounds = self.config.get("total_rounds", 25)
    _progress_tracker = ProgressTracker(N, total_rounds)
    
    print(f"[Simulation] Starting environment run on {self.config.get('device', 'cpu')}...", flush=True)
    res = _orig_env_run(self)
    print(f"[Simulation] Environment run complete.", flush=True)
    _progress_tracker = None
    return res
SimulationEnvironment.run = custom_env_run


def load_base_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_config_1():
    print("\n=============================================")
    print("RUNNING SMOKE TEST CONFIGURATION 1")
    print("=============================================")
    
    config = load_base_config()
    config["N_clients"] = 10
    config["total_rounds"] = 25
    # config["T_base"] = 0.0  # Run instantly
    # config["eval_every"] = 1
    # config["dataset"] = "MNIST"
    # config["batch_size"] = 128  # Speed up loading/training
    # config["local_epochs"] = 1
    
    print(
        f"N={config['N_clients']}, "
        f"Rounds={config['total_rounds']}, "
        f"Epochs={config['local_epochs']}, "
        f"Dataset={config['dataset']}, "
        f"Batch={config['batch_size']}"
    )
    
    # We run SimulationEnvironment with NONE attack
    env = SimulationEnvironment(config=config, attack_type="NONE", seed=42)
    
    _, W_before = _build_model(config)
    results = env.run()
    W_after = env.get_global_weights()
    
    assert W_after is not None, "Global weights should not be None"
    assert not torch.allclose(W_before, W_after), "Global weights did not update!"
    
    print("Smoke Test 1 Finished Successfully!")
    print(f"accuracy_log: {results.get('accuracy_log')}")
    print(f"rejection_log length: {len(results.get('rejection_log'))}")
    
    # Verify weight file
    run_id = f"NONE_42"
    csv_file = os.path.join(config.get("log_dir", "logs/"), f"{run_id}_updates.csv")
    print(f"CSV file path: {csv_file}")
    if os.path.exists(csv_file):
        print(f"CSV file exists: Yes. Size: {os.path.getsize(csv_file)} bytes")
        with open(csv_file, "r") as f:
            lines = f.readlines()
            print(f"CSV lines count: {len(lines)}")
            print("First 3 lines of CSV:")
            for line in lines[:3]:
                print(f"  {line.strip()}")
    else:
        print("CSV file exists: No")

def run_config_2():
    print("\n=============================================")
    print("RUNNING SMOKE TEST CONFIGURATION 2")
    print("=============================================")
    
    config = load_base_config()
    config["N_clients"] = 10
    config["total_rounds"] = 25
    config["T_base"] = 0.0  # Run instantly
    config["eval_every"] = 1
    # config["dataset"] = "MNIST"
    config["batch_size"] = 128
    config["local_epochs"] = 1
    
    print(
        f"N={config['N_clients']}, "
        f"Rounds={config['total_rounds']}, "
        f"Epochs={config['local_epochs']}, "
        f"Dataset={config['dataset']}, "
        f"Batch={config['batch_size']}"
    )
    
    env = SimulationEnvironment(config=config, attack_type="NONE", seed=42)
    
    _, W_before = _build_model(config)
    results = env.run()
    W_after = env.get_global_weights()
    
    assert W_after is not None, "Global weights should not be None"
    assert not torch.allclose(W_before, W_after), "Global weights did not update!"
    
    print("Smoke Test 2 Finished Successfully!")
    print(f"accuracy_log: {results.get('accuracy_log')}")
    print(f"rejection_log length: {len(results.get('rejection_log'))}")

def _run_attacks(attacks: list, label: str) -> None:
    """Shared helper: runs a list of attack scenarios with burn-in fix applied.

    Sets K_base=5 in the smoke-test config only. With N=10 clients:
        N_burn = max(4*N, K_base) = max(40, 5) = 40
    Over 5 rounds (50 updates total) the temporal filter exits burn-in after
    update 40, so the final round (updates 41-50) is evaluated by the
    Tukey-fence gate.  This is sufficient to produce TEMPORAL_HIGH_FREQ /
    TEMPORAL_STRAGGLER rejections without changing production config.yaml.
    """
    print(f"\n=============================================")
    print(f"ATTACKS: {', '.join(attacks)}  [{label}]")
    print(f"K_base overridden to 5 (smoke-test only)")
    print(f"=============================================")

    for attack in attacks:
        print(f"\n--- Running Attack: {attack} ---")
        config = load_base_config()
        config["N_clients"] = 10
        config["total_rounds"] = 25
        config["T_base"] = 0.0
        config["eval_every"] = 1
        # config["dataset"] = "MNIST"
        config["batch_size"] = 32
        config["local_epochs"] = 1
        # --- Burn-in fix (smoke-test only) ----------------------------
        # Production K_base=50 causes N_burn=50 which equals total updates
        # (5 rounds * 10 clients), so the temporal filter never exits burn-in.
        # Lowering K_base here does NOT touch config.yaml.
        config["K_base"] = 5
        # --------------------------------------------------------------

        print(
            f"N={config['N_clients']}, "
            f"Rounds={config['total_rounds']}, "
            f"K_base={config['K_base']} (smoke override), "
            f"N_burn=max(4*{config['N_clients']},{config['K_base']})="
            f"{max(4 * config['N_clients'], config['K_base'])}, "
            f"Epochs={config['local_epochs']}, "
            f"Dataset={config['dataset']}"
        )

        env = SimulationEnvironment(config=config, attack_type=attack, seed=42)

        _, W_before = _build_model(config)
        results = env.run()
        W_after = env.get_global_weights()

        assert W_after is not None, "Global weights should not be None"
        assert not torch.allclose(W_before, W_after), "Global weights did not update!"

        # --- Summarise rejection reasons from this run ----------------
        rej_log = results.get("rejection_log", [])
        reason_counts: dict = {}
        for entry in rej_log:
            r = entry.get("reason", "UNKNOWN")
            reason_counts[r] = reason_counts.get(r, 0) + 1
        print(f"Attack {attack} finished. Rejection breakdown: {reason_counts}")


def run_config_3():
    """Phase-1 temporal validation: T1 and T2 only.

    Expected output:
        T1_HIGH_FREQ -> TEMPORAL_HIGH_FREQ dominates
        T2_STRAGGLER -> TEMPORAL_STRAGGLER dominates
    """
    _run_attacks(
        attacks=["T1_HIGH_FREQ", "T2_STRAGGLER"],
        label="Phase-1: Temporal filter validation",
    )


def run_config_3_all():
    """Phase-2 full profile: all 6 attack types.

    Expected qualitative behaviour:
        NONE         -> mostly FULL_ACCEPT
        T1_HIGH_FREQ -> mostly TEMPORAL_HIGH_FREQ
        T2_STRAGGLER -> mostly TEMPORAL_STRAGGLER
        S1_POISON    -> mostly SPATIAL_COSINE
        S2_MIMICRY   -> mix of SPATIAL_COSINE / FULL_ACCEPT
        ADAPTIVE     -> TEMPORAL_HIGH_FREQ early, SPATIAL_COSINE later
        COMPOUND     -> mix of TEMPORAL_HIGH_FREQ + SPATIAL_COSINE
    """
    _run_attacks(
        attacks=[
            "T1_HIGH_FREQ",
            "T2_STRAGGLER",
            "S1_POISON",
            "S2_MIMICRY",
            "ADAPTIVE",
            "COMPOUND",
        ],
        label="Phase-2: Full attack profile",
    )


if __name__ == "__main__":
    run_config_1()
    run_config_2()
    # Phase-1: validate temporal filter fires on T1 / T2
    run_config_3()
    # Phase-2: full six-attack profile with burn-in fix applied
    run_config_3_all()
