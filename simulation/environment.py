import asyncio
import time
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.device_utils import resolve_device, mark_step, set_xla_seed
import utils.metrics as metrics

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_xla_seed(seed)
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
    """Deeper CNN with GroupNorm for CIFAR-10.

    GroupNorm is used instead of BatchNorm because the FL weight
    serialisation pipeline (model.parameters()) would silently drop
    BN running-statistic buffers.  GroupNorm parameters are regular
    learnable weights and are fully compatible.
    """
    def __init__(self):
        super().__init__()
        # Block 1: 3 -> 64, 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(8, 64)

        # Block 2: 64 -> 128, 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3   = nn.GroupNorm(8, 128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gn4   = nn.GroupNorm(8, 128)

        # Block 3: 128 -> 256, 8x8 -> 4x4
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn5   = nn.GroupNorm(8, 256)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(256 * 4 * 4, 256)
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))

        # Block 2: 16x16 -> 8x8
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool(F.relu(self.gn4(self.conv4(x))))

        # Block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.gn5(self.conv5(x))))

        # Classifier
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
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
        self.last_update_time = self.server.get_virtual_time()  # Track locally to compute honest_g_i
 
    async def run_one_round(self) -> dict:
        # 1. Pull global weights or use force-synced weights
        if self.client._state.get("force_sync_applied", False):
            W_global = self.client._state["W_local"].clone()
            tau = self.client._state.get("last_reset_time", self.server.get_virtual_time())
            self.server.update_pull_time(self.client.client_id, tau)
            self.client._state["force_sync_applied"] = False
        else:
            tau = self.server.get_virtual_time()
            W_global = self.server.get_global_weights()
            self.server.update_pull_time(self.client.client_id, tau)
            self.client._state["W_local"] = W_global.clone()
 
        # 2. Simulate compute delay
        await self.client._simulate_delay()
 
        # 3. Train locally to get honest gradient
        honest_delta_W = self.client.trainer.train(W_global)
        t_submit_honest = self.server.get_virtual_time()
        
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
        if self.attack_type == "NONE":
            byz_fraction = 0.0
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
                
        # 8. True async training loop
        #
        # Fix (Critical — Audit Bug 1): The old implementation used
        #   for r in range(total_rounds): await asyncio.gather(...)
        # which is a hard synchronisation barrier — every client had to finish
        # its round before the next could begin.  That is Synchronous FL, not AFL.
        #
        # Real AFL: each client runs as an independent continuous coroutine.
        # The server processes updates as they arrive; fast clients are never
        # blocked waiting for slow stragglers.
        #
        # Termination: total_updates = total_rounds * N accepted updates.
        # This preserves the interface: total_rounds=5, N=10 → 50 accepted updates,
        # identical to the old synchronous behaviour in terms of total work done.
        #
        # Evaluation (Fix — Audit Bug 3): rep_manager.log_round() is called here,
        # once per eval cycle, NOT inside AggregatorServer.handle_update().
        # That removes the O(N²) reputation history growth.
        accuracy_log = []
        total_rounds = self.config.get("total_rounds", 500)
        eval_every   = self.config.get("eval_every", 10)
        device       = resolve_device(self.config)  # resolves and stores device object in config

        # Total accepted updates to process (re-interprets total_rounds as per-client rounds)
        total_updates      = total_rounds * N
        # Evaluate accuracy every eval_every "effective global rounds" worth of updates
        eval_every_updates = eval_every * N

        loop_start = time.time()

        # Initial accuracy
        t_start = time.time()
        init_acc = metrics.compute_accuracy(model, test_loader, server.get_global_weights(), device=device)
        t_end = time.time()
        server.accumulate_eval_time(t_end - t_start)
        accuracy_log.append(init_acc)
        logger.log_metric(round=0, metric_name="test_accuracy", value=init_acc)

        async def run_loop():
            stop_event = asyncio.Event()

            async def client_task(client):
                """Independent per-client coroutine — runs until stop_event fires."""
                # Startup jitter to scramble arrival order at the server
                await asyncio.sleep(random.uniform(0.0, 0.1))
                while not stop_event.is_set():
                    await client.run_one_round()
                    await asyncio.sleep(0.01)  # Yield thread to allow other tasks' timers to resolve

            # Launch every client as an independent background task in random order
            shuffled_clients = list(clients)
            random.shuffle(shuffled_clients)
            tasks = [asyncio.create_task(client_task(c)) for c in shuffled_clients]

            # Monitor accepted update count and trigger evaluation.
            next_eval_at = eval_every_updates
            last_progress_at = -1  # track last round we printed progress
            while server.update_counter < total_updates:
                u = server.update_counter  # snapshot
                eff_round = u // N

                # --- Lightweight per-round progress (every effective round) ---
                if eff_round > last_progress_at:
                    last_progress_at = eff_round
                    elapsed = time.time() - loop_start
                    n_rejected = sum(1 for e in logger.get_rejection_log() if e.get("status") == "REJECT")
                    pct = 100.0 * u / total_updates
                    print(
                        f"  Round {eff_round:>3}/{total_rounds} "
                        f"| updates={u:>5}/{total_updates} ({pct:5.1f}%) "
                        f"| rejected={n_rejected:>4} "
                        f"| elapsed={elapsed:6.1f}s",
                        flush=True,
                    )

                if u >= next_eval_at:
                    u = server.update_counter   # snapshot for consistent logging
                    t_start = time.time()
                    acc = metrics.compute_accuracy(
                        model, test_loader, server.get_global_weights(), device=device
                    )
                    t_end = time.time()
                    server.accumulate_eval_time(t_end - t_start)
                    accuracy_log.append(acc)
                    logger.log_metric(round=u, metric_name="test_accuracy", value=acc)
                    elapsed = time.time() - loop_start
                    print(
                        f">>> Eval @ round {u // N}/{total_rounds} "
                        f"| Test Accuracy: {acc:.4f} "
                        f"| elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                    # Reputation snapshots — once per eval cycle (Bug 3 fix)
                    for cid in range(N):
                        I_val, P_val = server.rep_manager.get(cid)
                        is_byz = cid in byz_ids
                        logger.log_reputation(
                            round=u, client_id=cid,
                            I_i=I_val, P_i=P_val, is_byzantine=is_byz,
                        )
                    server.rep_manager.log_round(u)
                    next_eval_at += eval_every_updates

                await asyncio.sleep(0)  # yield to event loop so client tasks can run

            # Signal all client tasks to stop after their current round completes.
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run the true-async loop
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
