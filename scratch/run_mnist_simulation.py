import sys
import os
import yaml
import torch
import asyncio
import time
import random
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, ".")

from simulation.environment import SimulationEnvironment, _build_model

def main():
    # 1. Load the base configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Please run from the project root directory.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Apply customized parameters for the MNIST run
    config["N_clients"] = 20
    config["total_rounds"] = 100
    config["dataset"] = "MNIST"
    config["T_base"] = 0.0  # Set to 0.0 to disable artificial/simulated training delays
    config["batch_size"] = 128  # Speed up loading and training on CPU
    config["local_epochs"] = 5  # Realistic training epochs per round

    # On Colab, uncomment the line below to save logs directly to Google Drive:
    # config["log_dir"] = "/content/drive/MyDrive/BDSF_results/logs/"

    # Use GPU if available for faster training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Choose attack simulation type:
    # Options: "NONE", "T1_HIGH_FREQ", "T2_STRAGGLER", "S1_POISON", "S2_MIMICRY", "ADAPTIVE", "COMPOUND"
    # Starting with "NONE" for Run 1 sanity check
    attack_type = "NONE"
    
    # Guarantee the baseline is truly all honest when attack_type is NONE
    if attack_type == "NONE":
        config["byz_fraction"] = 0.0

    print("=" * 60)
    print("RUNNING MNIST FL SIMULATION")
    print("=" * 60)
    print(f"Clients:        {config['N_clients']}")
    print(f"Total Rounds:   {config['total_rounds']}")
    print(f"Dataset:        {config['dataset']}")
    print(f"Device:         {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"Batch Size:     {config['batch_size']}")
    print(f"Local Epochs:   {config['local_epochs']}")
    print(f"Attack Type:    {attack_type}")
    print(f"Byzantine Frac: {config['byz_fraction']}")
    print(f"Random Seed:    {SEED}")
    print("=" * 60)

    # 3. Instantiate the simulation environment
    env = SimulationEnvironment(config=config, attack_type=attack_type, seed=SEED)

    # 4. Run and time the simulation
    print("Starting environment simulation run...")
    start_time = time.time()
    results = env.run()
    elapsed_time = time.time() - start_time
    print("Simulation finished successfully!")
    print(f"Elapsed Time:   {elapsed_time/60:.2f} minutes")
    print("=" * 60)

    # 5. Summarize Results
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    
    accuracy_log = results.get("accuracy_log", [])
    if accuracy_log:
        print(f"Evaluations:      {len(accuracy_log)}")
        print(f"Initial Accuracy: {accuracy_log[0]:.4f}")
        print(f"Final Accuracy:   {accuracy_log[-1]:.4f}")
        print(f"Best Accuracy:    {max(accuracy_log):.4f}")
    
    rej_log = results.get("rejection_log", [])
    reason_counts = {}
    for entry in rej_log:
        reason = entry.get("reason", "UNKNOWN")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print(f"Total Updates Rejected: {len(rej_log)}")
    for reason, count in reason_counts.items():
        print(f"  - {reason}: {count}")
    print(f"Logs directory:         {config.get('log_dir', 'logs/')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
