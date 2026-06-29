import time
import os
from collections import deque
from typing import Optional

import torch
import numpy as np

from shared.types import (UpdateSubmission, AcceptedEntry,
                          ForceSyncPayload, ClientRegistration)
from server.temporal_filter import TemporalFilter
from server.force_sync import ForceSyncDispatcher
from server.reputation_manager import ReputationManager
from server.spatial_validator import SpatialValidator
from utils.logger import BDSFLogger


class AggregatorServer:
    """Central orchestrator for the BDSF-AFL defense pipeline.

    Receives ``UpdateSubmission`` objects from clients and runs the full
    12-step decision pipeline:
        temporal gate -> spatial cosine check -> adaptive clip ->
        reputation-weighted merge -> recovery -> log.

    Maintains global model weights, per-client registry, and the
    accepted gradient buffer shared with ``SpatialValidator``.
    """

    def __init__(
        self,
        config: dict,
        W_init: torch.Tensor,
        client_ids: list[int],
        logger: BDSFLogger,
    ) -> None:
        # Step 1: Store config and logger
        self.config = config
        self.logger = logger

        # Step 2: Current global model (flattened 1D float32)
        self.W_global: torch.Tensor = W_init.clone().float()

        # Step 3: Client IDs
        self.client_ids = client_ids

        # Step 4: Number of clients
        N = len(client_ids)

        # Step 5: Compute burn-in count
        self.N_burn: int = max(4 * N, config.get("K_base", 50))

        # Step 6: Create a local config copy with burn_in_count for TemporalFilter
        tf_config = dict(config)
        tf_config["burn_in_count"] = self.N_burn

        # Step 7: Instantiate temporal filter
        self.temporal_filter = TemporalFilter(tf_config)

        # Step 8: Instantiate reputation manager
        self.rep_manager = ReputationManager(client_ids, config)

        # Step 9: Buffer size
        M = config.get("M", 30)

        # Step 10: Instantiate spatial validator
        self.spatial_validator = SpatialValidator(config)

        # Step 11: Accepted buffer (shared concept with SpatialValidator)
        self.accepted_buffer: deque[AcceptedEntry] = deque(maxlen=M)

        # Step 12: Instantiate force-sync dispatcher
        self.force_sync_dispatcher = ForceSyncDispatcher()

        # Step 13: Build client registry
        self.registry: dict[int, ClientRegistration] = {}
        for cid in client_ids:
            session_key = os.urandom(32)
            self.registry[cid] = ClientRegistration(
                client_id=cid,
                session_key=session_key,
                last_update_time=time.time(),
                pull_time=time.time(),
                is_byzantine=False,
            )

        # Step 14: Total accepted updates counter
        self.update_counter: int = 0

        # Step 15: Round number (incremented on each accepted update)
        self.round_number: int = 0

    # ------------------------------------------------------------------
    # Main entry point — the 12-step pipeline
    # ------------------------------------------------------------------

    def handle_update(self, submission: UpdateSubmission) -> dict:
        """Process a single client update through the full BDSF-AFL
        decision pipeline.

        Returns a response dict with keys: ``status``, ``reason``,
        ``force_sync``, ``round``, ``I_i``, ``P_i``.
        """

        # --- Step 1: Compute behavioral gap g_i ---
        cid = submission.client_id
        reg = self.registry[cid]
        t_now = submission.t_submit
        g_i = t_now - reg.last_update_time

        # --- Step 2: Get current reputation scores ---
        I_i, P_i = self.rep_manager.get(cid)

        # --- Step 3: Run temporal gate ---
        temporal_result = self.temporal_filter.evaluate(g_i)

        # --- Step 4: Handle REJECT_HIGH_FREQ ---
        if temporal_result == "REJECT_HIGH_FREQ":
            self.rep_manager.slash_integrity(cid)
            I_i, P_i = self.rep_manager.get(cid)
            self.logger.log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=I_i, P_i=P_i,
                status="REJECT", reason="TEMPORAL_HIGH_FREQ",
            )
            return {
                "status": "REJECT",
                "reason": "TEMPORAL_HIGH_FREQ",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 5: Handle REJECT_STRAGGLER ---
        if temporal_result == "REJECT_STRAGGLER":
            self.rep_manager.reduce_pace(cid)
            I_i, P_i = self.rep_manager.get(cid)
            fs_payload = self.force_sync_dispatcher.build_payload(
                cid, self.W_global, reg.session_key,
            )
            self.logger.log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=I_i, P_i=P_i,
                status="REJECT", reason="TEMPORAL_STRAGGLER",
            )
            return {
                "status": "REJECT",
                "reason": "TEMPORAL_STRAGGLER",
                "force_sync": fs_payload,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 6: Temporal PASS — update last_update_time ---
        reg.last_update_time = t_now

        # --- Step 7: Spatial cosine check ---
        passes_cosine = self.spatial_validator.cosine_check(submission.delta_W)
        if not passes_cosine:
            self.rep_manager.slash_integrity(cid)
            I_i, P_i = self.rep_manager.get(cid)
            self.logger.log_update(
                round=self.round_number, client_id=cid,
                g_i=g_i, I_i=I_i, P_i=P_i,
                status="REJECT", reason="SPATIAL_COSINE",
            )
            return {
                "status": "REJECT",
                "reason": "SPATIAL_COSINE",
                "force_sync": None,
                "round": self.round_number,
                "I_i": I_i,
                "P_i": P_i,
            }

        # --- Step 8: Adaptive L2 clipping ---
        delta_W_clipped = self.spatial_validator.adaptive_clip(submission.delta_W)

        # --- Step 9: Reputation-weighted merge ---
        eta = self.config.get("eta", 0.01)
        weight = I_i * P_i
        self.W_global = self.W_global + eta * weight * delta_W_clipped

        # --- Step 10: Append to accepted_buffer ---
        entry = AcceptedEntry(
            delta_W=delta_W_clipped.clone(),
            I_score=I_i,
            P_score=P_i,
        )
        self.accepted_buffer.append(entry)
        self.spatial_validator.on_accept(entry)

        # --- Step 11: Reputation recovery ---
        self.rep_manager.recover(cid)
        I_i, P_i = self.rep_manager.get(cid)

        # --- Step 12: Increment counters and return ---
        self.update_counter += 1
        self.round_number += 1
        self.logger.log_update(
            round=self.round_number, client_id=cid,
            g_i=g_i, I_i=I_i, P_i=P_i,
            status="ACCEPT", reason="FULL_ACCEPT",
        )
        return {
            "status": "ACCEPT",
            "reason": "FULL_ACCEPT",
            "force_sync": None,
            "round": self.round_number,
            "I_i": I_i,
            "P_i": P_i,
        }

    # ------------------------------------------------------------------
    # Auxiliary public methods
    # ------------------------------------------------------------------

    def get_global_weights(self) -> torch.Tensor:
        """Returns a clone of the current global model weights."""
        return self.W_global.clone()

    def register_client_ground_truth(
        self, client_id: int, is_byzantine: bool,
    ) -> None:
        """Called by SimulationEnvironment after construction.
        Sets the ground truth label used only by metrics/logger."""
        self.registry[client_id].is_byzantine = is_byzantine

    def get_session_key(self, client_id: int) -> bytes:
        """Returns the HMAC session key for a given client.
        Called by SimulationEnvironment to initialise each ClientNode."""
        return self.registry[client_id].session_key

    def update_pull_time(self, client_id: int, pull_time: float) -> None:
        """Called when a client pulls W_global. Stores the pull timestamp."""
        self.registry[client_id].pull_time = pull_time
