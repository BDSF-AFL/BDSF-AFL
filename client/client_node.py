import asyncio
import time
import torch
import numpy as np
from shared.types import UpdateSubmission, ForceSyncPayload
from client.local_trainer import LocalTrainer
from client.force_sync_handler import ForceSyncHandler
from utils.logger import BDSFLogger

class ClientNode:
    def __init__(self, client_id: int, trainer: LocalTrainer, server: object, force_sync_handler: ForceSyncHandler, config: dict, logger: BDSFLogger, local_model = None, dataloader = None, pool = None):
        self.client_id = client_id
        self.trainer = trainer
        self.server = server
        self.fs_handler = force_sync_handler
        self.config = config
        self.logger = logger
        self.local_model = local_model
        self.dataloader = dataloader
        self.pool = pool
        self._state = {"W_local": None, "gradient_buffer": [], "last_reset_time": 0.0}
        self._mu_delay = config.get("lognormal_mu", 0.5)
        self._sigma_delay = config.get("lognormal_sigma", 1.0)
        self._T_base = config.get("T_base", 1.0)

    async def run_one_round(self) -> dict:
        # 1. Pull global weights or use force-synced weights
        if self._state.get("force_sync_applied", False):
            W_global = self._state["W_local"].clone()
            tau = self._state.get("last_reset_time", self.server.get_virtual_time())
            self.server.update_pull_time(self.client_id, tau)
            self._state["force_sync_applied"] = False
        else:
            tau = self.server.get_virtual_time()
            W_global = self.server.get_global_weights()
            self.server.update_pull_time(self.client_id, tau)
            self._state["W_local"] = W_global.clone()
 
        # 2. Simulate compute delay
        await self._simulate_delay()
 
        # 3. Train locally
        if self.pool is not None:
            from simulation.environment import _run_trainer_in_process
            loop = asyncio.get_running_loop()
            delta_W = await loop.run_in_executor(
                self.pool, _run_trainer_in_process, self.local_model, self.dataloader, self.config, W_global
            )
        else:
            delta_W = self.trainer.train(W_global)
 
        # 4. Build submission
        t_submit = self.server.get_virtual_time()
        submission = UpdateSubmission(
            client_id=self.client_id,
            delta_W=delta_W,
            t_submit=t_submit,
            tau=tau,
        )

        # 5. Push to server
        response = self.server.handle_update(submission)

        # 6. Handle force_sync if present
        if response.get("force_sync") is not None:
            self.fs_handler.verify_and_apply(response["force_sync"], self._state)

        return response

    async def _simulate_delay(self) -> None:
        X = float(np.random.lognormal(mean=self._mu_delay, sigma=self._sigma_delay))
        delay = self._T_base * (1.0 + X)
        await asyncio.sleep(delay)
