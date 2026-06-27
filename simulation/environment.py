import asyncio
import time
import random
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from shared.types import UpdateSubmission, ForceSyncPayload

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    # Check if CUDA is available, though we default to CPU in config
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def _build_model(config: dict) -> Tuple[nn.Module, torch.Tensor]:
    """Builds the local model architecture and returns (model, W_init_flat)."""
    dataset_name = config.get("dataset", "CIFAR10")
    if dataset_name == "MNIST":
        model = MNISTMLP()
    elif dataset_name == "FEMNIST":
        model = MNISTMLP()  # FEMNIST has 784 inputs, same as MNIST
    else:  # CIFAR10
        model = CIFAR10CNN()
        
    W_init = torch.cat([p.data.flatten() for p in model.parameters()]).float()
    return model, W_init

class AggregatorServerStub:
    """Stub of AggregatorServer that accepts everything. Used in Phase 0 skeleton."""
    def __init__(self, W_init: torch.Tensor):
        self.W_global = W_init.clone()
        self.round_number = 0

    def get_global_weights(self) -> torch.Tensor:
        return self.W_global.clone()

    def update_pull_time(self, client_id: int, pull_time: float) -> None:
        pass

    def handle_update(self, submission: UpdateSubmission) -> dict:
        self.round_number += 1
        return {
            "status": "ACCEPT",
            "reason": "FULL_ACCEPT",
            "force_sync": None,
            "round": self.round_number,
            "I_i": 1.0,
            "P_i": 1.0,
        }

class DummyClientNode:
    """Dummy client node that prints logs. Used in Phase 0 skeleton."""
    def __init__(self, client_id: int, server: AggregatorServerStub):
        self.client_id = client_id
        self.server = server

    async def run(self):
        print(f"[DummyClientNode {self.client_id}] Pulling global weights...")
        tau = time.time()
        W_global = self.server.get_global_weights()
        self.server.update_pull_time(self.client_id, tau)
        
        # Simulate local work
        print(f"[DummyClientNode {self.client_id}] Training local model...")
        await asyncio.sleep(0.5)
        
        # Build mock update submission (zero gradients for dummy run)
        delta_W = torch.zeros_like(W_global)
        submission = UpdateSubmission(
            client_id=self.client_id,
            delta_W=delta_W,
            t_submit=time.time(),
            tau=tau
        )
        
        # Push submission
        print(f"[DummyClientNode {self.client_id}] Pushing updates to server...")
        response = self.server.handle_update(submission)
        print(f"[DummyClientNode {self.client_id}] Received server response: {response}")

class SimulationEnvironment:
    def __init__(self, config: dict, attack_type: str, seed: int):
        self.config = config
        self.attack_type = attack_type
        self.seed = seed

    def run(self) -> dict:
        """Runs the dummy skeleton simulation."""
        set_seed(self.seed)
        
        print("[SimulationEnvironment] Building model...")
        model, W_init = _build_model(self.config)
        print(f"[SimulationEnvironment] Model initialized with weight dimension: {W_init.shape[0]}")
        
        server = AggregatorServerStub(W_init)
        client = DummyClientNode(client_id=0, server=server)
        
        print("[SimulationEnvironment] Running async client loop...")
        asyncio.run(client.run())
        print("[SimulationEnvironment] Async loop finished successfully.")
        
        # Return mock results
        return {
            "accuracy_log": [0.1],
            "rejection_log": [],
            "reputation_log": [],
            "byz_ids": set(),
            "honest_ids": {0},
            "rep_manager": None
        }
