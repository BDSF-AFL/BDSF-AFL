import asyncio
import time
import copy
import random
import concurrent.futures
import multiprocessing
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
from utils.device_utils import resolve_device, resolve_all_devices, gpu_count, mark_step, set_xla_seed
import utils.metrics as metrics

def _run_trainer_in_process(model: nn.Module, dataloader, config: dict, W_global: torch.Tensor) -> torch.Tensor:
    """Standalone worker function executed inside ProcessPoolExecutor for true GPU parallelism."""
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    trainer = LocalTrainer(model, dataloader, config)
    return trainer.train(W_global)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_xla_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

import hashlib
from models.resnet import CIFAR10ResNet18, MNISTMLP, CIFAR10CNN

def _build_model(config: dict) -> Tuple[nn.Module, torch.Tensor]:
    """Builds the local model architecture and returns (model, W_init_flat)."""
    dataset_name = config.get("dataset", "CIFAR10")
    arch = config.get("model_architecture", "resnet18" if dataset_name == "CIFAR10" else "mlp").lower()
    
    if dataset_name in ["MNIST", "FEMNIST"]:
        model = MNISTMLP()
    else:  # CIFAR10
        if arch == "cnn":
            model = CIFAR10CNN()
        else:
            model = CIFAR10ResNet18()
        
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
        if getattr(self.client, "pool", None) is not None:
            loop = asyncio.get_running_loop()
            honest_delta_W = await loop.run_in_executor(
                self.client.pool, _run_trainer_in_process, self.client.local_model, self.client.dataloader, self.client.config, W_global
            )
        else:
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
        mode = self.config.get("decision_mode", "joint")
        run_id = f"{mode}_{self.attack_type}_{self.seed}"
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
        #    Distribute clients round-robin across all available GPUs so that
        #    on a dual-T4 Kaggle instance 10 clients train on cuda:0 and 10 on cuda:1.
        all_devices = resolve_all_devices()
        n_gpus = len(all_devices)
        print(f"  >> Distributing {N} clients across {n_gpus} device(s): "
              f"{[str(d) for d in all_devices]}")

        # Multi-process pool for parallel GPU training across cuda:0 and cuda:1
        use_pool = torch.cuda.is_available() and n_gpus > 1
        mp_ctx = multiprocessing.get_context("spawn") if use_pool else None
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_gpus, mp_context=mp_ctx) if use_pool else None

        clients = []
        for i in range(N):
            # Round-robin device assignment
            client_device = all_devices[i % n_gpus]
            client_config = dict(self.config)       # shallow copy is safe — no mutable nested dicts are written
            client_config["device"] = client_device

            local_model = copy.deepcopy(model)
            trainer = LocalTrainer(local_model, dataloaders[i], client_config)
            session_key = server.get_session_key(i)
            fs_handler = ForceSyncHandler(i, session_key, logger)
            client_node = ClientNode(i, trainer, server, fs_handler, client_config, logger, local_model=local_model, dataloader=dataloaders[i], pool=pool)
            
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
        # Primary device for server-side ops (eval, aggregation weight vectors)
        device       = resolve_device(self.config)  # resolves and stores primary device in config

        # Total accepted updates to process (re-interprets total_rounds as per-client rounds)
        total_updates      = total_rounds * N
        # Evaluate accuracy every eval_every "effective global rounds" worth of updates
        eval_every_updates = eval_every * N
        
        # Checkpointing & Convergence Configuration
        log_dir = self.config.get("log_dir", "logs/")
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        run_id = self.config.get("run_id", f"run_{int(time.time())}")
        config_hash = hashlib.sha256(str(sorted(self.config.items())).encode("utf-8")).hexdigest()[:16]
        
        patience = int(self.config.get("early_stopping_patience", 5))
        evals_without_improvement = 0
        best_score = -float("inf")
        best_acc = 0.0
        best_round = 0
        
        # Resume Checkpoint if requested
        if self.config.get("resume", False):
            ckpt_path = self.config.get("resume_checkpoint_path", os.path.join(ckpt_dir, f"{run_id}_latest.pt"))
            if os.path.exists(ckpt_path):
                print(f"[CHECKPOINT] Resuming from checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                saved_hash = ckpt.get("config_hash")
                if saved_hash and saved_hash != config_hash:
                    print(f"[WARNING] Checkpoint config_hash ({saved_hash}) differs from current config ({config_hash})")
                server.load_state(ckpt.get("server_state", {}))
                accuracy_log = list(ckpt.get("accuracy_log", []))
                best_score = ckpt.get("best_score", best_score)
                best_acc = ckpt.get("best_accuracy", best_acc)
                best_round = ckpt.get("best_round", best_round)

        def save_checkpoint(path: str, is_best: bool = False):
            state = {
                "config_hash": config_hash,
                "model_arch": self.config.get("model_architecture", "resnet18"),
                "round": server.round_number,
                "update_counter": server.update_counter,
                "best_accuracy": best_acc,
                "best_score": best_score,
                "best_round": best_round,
                "accuracy_log": list(accuracy_log),
                "server_state": server.get_state(),
                "rng_state": torch.get_rng_state(),
            }
            torch.save(state, path)

        loop_start = time.time()
        if not self.config.get("resume", False):
            t_start = time.time()
            init_acc, init_loss = metrics.compute_evaluation_metrics(model, test_loader, server.get_global_weights(), device=device)
            t_end = time.time()
            server.accumulate_eval_time(t_end - t_start)
            accuracy_log.append(init_acc)
            logger.log_metric(round=0, metric_name="test_accuracy", value=init_acc)
            logger.log_metric(round=0, metric_name="val_loss", value=init_loss)

        async def run_loop():
            # Dedicated background loop: drains any pending forced-sync queues
            # without blocking client local training tasks.
            drain_task = asyncio.create_task(server.drain_force_sync_queue(clients))
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
            nonlocal best_score, best_acc, best_round, evals_without_improvement
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
                    acc, val_loss = metrics.compute_evaluation_metrics(
                        model, test_loader, server.get_global_weights(), device=device
                    )
                    t_end = time.time()
                    server.accumulate_eval_time(t_end - t_start)
                    accuracy_log.append(acc)
                    logger.log_metric(round=u, metric_name="test_accuracy", value=acc)
                    logger.log_metric(round=u, metric_name="val_loss", value=val_loss)

                    # Generalization Gap & Composite Convergence Score
                    gen_gap = max(0.0, val_loss - 0.5)
                    score = acc - 0.05 * gen_gap

                    if score > best_score:
                        best_score = score
                        best_acc = acc
                        best_round = u // N
                        evals_without_improvement = 0
                        save_checkpoint(os.path.join(ckpt_dir, f"{run_id}_best.pt"), is_best=True)
                    else:
                        evals_without_improvement += 1

                    # Save latest checkpoint
                    save_checkpoint(os.path.join(ckpt_dir, f"{run_id}_latest.pt"), is_best=False)

                    elapsed = time.time() - loop_start
                    print(
                        f">>> Eval @ round {u // N}/{total_rounds} "
                        f"| Test Acc: {acc:.4f} "
                        f"| Loss: {val_loss:.4f} "
                        f"| Score: {score:.4f} (Best: {best_acc:.4f} @ R{best_round}) "
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

                    # Early stopping check based on consecutive evaluation checkpoints
                    if self.config.get("early_stopping", False) and evals_without_improvement >= patience:
                        print(
                            f"\n[EARLY STOPPING] Convergence plateau reached: no score gain for {patience} evals "
                            f"({patience * (eval_every_updates // N)} rounds). Optimal model saved: Best Acc = {best_acc:.4f} at Round {best_round}.",
                            flush=True,
                        )
                        break

                await asyncio.sleep(0)  # yield to event loop so client tasks can run

            # Signal all client tasks to stop after their current round completes.
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            drain_task.cancel()

        # Run the true-async loop
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is not None and running_loop.is_running():
                # Running inside Jupyter / IPython / Kaggle notebook kernel
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, run_loop())
                    future.result()
            else:
                asyncio.run(run_loop())
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        
        # Return results
        return {
            "accuracy_log": accuracy_log,
            "rejection_log": logger.get_rejection_log(),
            "reputation_log": logger.get_reputation_log(),
            "byz_ids": byz_ids,
            "honest_ids": honest_ids,
            "rep_manager": server.rep_manager,
        }
