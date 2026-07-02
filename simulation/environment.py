import asyncio
import time
import copy
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional

from shared.types import UpdateSubmission, ForceSyncPayload
from simulation.data_partitioner import DataPartitioner
from simulation.attack_injector import AttackInjector
from server.aggregator import AggregatorServer
from client.client_node import ClientNode
from client.local_trainer import LocalTrainer
from client.force_sync_handler import ForceSyncHandler
from utils.logger import BDSFLogger
import utils.metrics as metrics

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

class AttackInjectorWrapper:
    """Wrapper that intercepts ClientNode.run_one_round() for Byzantine clients
    and applies timing/gradient modifications using AttackInjector.
    """
    def __init__(self, client: ClientNode, injector: AttackInjector, server: AggregatorServer):
        self.client = client
        self.injector = injector
        self.server = server
        self.last_update_time = time.time()  # Track locally to compute honest_g_i

    async def run_one_round(self) -> dict:
        # 1. Pull global weights
        tau = time.time()
        W_global = self.server.get_global_weights()
        self.server.update_pull_time(self.client.client_id, tau)
        self.client._state["W_local"] = W_global.clone()

        # 2. Simulate compute delay
        await self.client._simulate_delay()

        # 3. Train locally to get honest gradient
        honest_delta_W = self.client.trainer.train(W_global)
        t_submit_honest = time.time()
        
        # Calculate honest gap g_i
        honest_g_i = t_submit_honest - self.last_update_time

        # 4. Prepare context for injector
        # get median_g
        history = self.server.temporal_filter.gap_history
        median_g = float(np.median(history)) if history else None
        
        # get ref_delta_W
        ref_delta_W = self.server.spatial_validator._build_reference()
        
        # get own_P_i
        _, own_P_i = self.server.rep_manager.get(self.client.client_id)
        
        context = {
            "honest_g_i": honest_g_i,
            "median_g": median_g,
            "W_global": W_global,
            "ref_delta_W": ref_delta_W,
            "theta_cos": self.server.spatial_validator.theta_cos,
            "own_P_i": own_P_i,
        }

        # 5. Inject attack
        modified_dW, modified_g = self.injector.inject(honest_delta_W, context)

        # Build submission with modified delta_W and modified timing (t_submit = last_update_time + modified_g)
        t_submit_modified = self.last_update_time + modified_g
        
        submission = UpdateSubmission(
            client_id=self.client.client_id,
            delta_W=modified_dW,
            t_submit=t_submit_modified,
            tau=tau,
        )

        # 6. Push to server
        response = self.server.handle_update(submission)

        # 7. Handle force_sync if present
        if response.get("force_sync") is not None:
            self.client.fs_handler.verify_and_apply(response["force_sync"], self.client._state)
            # Reset last update time to the force sync timestamp
            self.last_update_time = response["force_sync"].timestamp

        # Update last update time if the update was NOT rejected by the temporal gate.
        if response.get("reason") not in ("TEMPORAL_HIGH_FREQ", "TEMPORAL_STRAGGLER"):
            self.last_update_time = t_submit_modified

        # 8. Log
        self.client.logger.log_update(
            round=response["round"], 
            client_id=self.client.client_id,
            status=response["status"], 
            reason=response["reason"],
            I_i=response["I_i"], 
            P_i=response["P_i"]
        )
        return response

class SimulationEnvironment:
    def __init__(self, config: dict, attack_type: str, seed: int):
        self.config = config
        self.attack_type = attack_type
        self.seed = seed

    def run(self) -> dict:
        """Runs the complete async federated learning simulation loop."""
        set_seed(self.seed)
        
        # 1. Initialize model
        model, W_init = _build_model(self.config)
        
        # 2. Partition dataset
        partitioner = DataPartitioner(self.config)
        dataloaders = partitioner.partition()
        test_loader = partitioner.get_test_loader()
        
        # 3. Create Logger
        run_id = f"{self.attack_type}_{self.seed}"
        logger = BDSFLogger(run_id=run_id, config=self.config)
        
        # 4. Construct AggregatorServer
        N = self.config.get("N_clients", 20)
        server = AggregatorServer(self.config, W_init, list(range(N)), logger)
        
        # 5. Designate Byzantine clients
        byz_fraction = self.config.get("byz_fraction", 0.2)
        byz_count = int(N * byz_fraction)
        byz_ids = set(range(byz_count))
        honest_ids = set(range(byz_count, N))
        
        # 6. Register ground truth labels
        for cid in byz_ids:
            server.register_client_ground_truth(cid, is_byzantine=True)
            
        # 7. Create ClientNode and ForceSyncHandler instances
        clients = []
        for i in range(N):
            local_model = copy.deepcopy(model)
            trainer = LocalTrainer(local_model, dataloaders[i], self.config)
            session_key = server.get_session_key(i)
            fs_handler = ForceSyncHandler(i, session_key, logger)
            client_node = ClientNode(i, trainer, server, fs_handler, self.config, logger)
            
            if i in byz_ids:
                injector = AttackInjector(self.attack_type, i, self.config)
                wrapper = AttackInjectorWrapper(client_node, injector, server)
                clients.append(wrapper)
            else:
                clients.append(client_node)
                
        # 8. Run async training loop
        accuracy_log = []
        total_rounds = self.config.get("total_rounds", 500)
        eval_every = self.config.get("eval_every", 10)
        device = self.config.get("device", "cpu")
        
        # Initial accuracy
        init_acc = metrics.compute_accuracy(model, test_loader, server.get_global_weights(), device=device)
        accuracy_log.append(init_acc)
        logger.log_metric(round=0, metric_name="test_accuracy", value=init_acc)
        
        async def run_loop():
            for r in range(total_rounds):
                # Run one concurrent round for all clients
                await asyncio.gather(*[c.run_one_round() for c in clients])
                
                # Evaluation
                if (r + 1) % eval_every == 0:
                    acc = metrics.compute_accuracy(model, test_loader, server.get_global_weights(), device=device)
                    accuracy_log.append(acc)
                    logger.log_metric(round=r + 1, metric_name="test_accuracy", value=acc)
                    
                    # Also log reputation snapshots
                    for cid in range(N):
                        I_val, P_val = server.rep_manager.get(cid)
                        is_byz = cid in byz_ids
                        logger.log_reputation(round=r + 1, client_id=cid, I_i=I_val, P_i=P_val, is_byzantine=is_byz)
        
        # Run async loop
        asyncio.run(run_loop())
        
        # Return results
        return {
            "accuracy_log": accuracy_log,
            "rejection_log": logger.get_rejection_log(),
            "reputation_log": logger.get_reputation_log(),
            "byz_ids": byz_ids,
            "honest_ids": honest_ids,
            "rep_manager": server.rep_manager,
        }
