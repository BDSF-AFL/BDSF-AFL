from collections import deque
import hmac
import hashlib
import io
import torch
from shared.types import ForceSyncPayload
from server.force_sync import ForceSyncDispatcher
from utils.logger import BDSFLogger


class ForceSyncHandler:
    """Client-side handler for force-sync payloads.

    Receives a ``ForceSyncPayload``, verifies replay freshness and HMAC,
    and if valid, hard-resets the client's local state (weights + gradient buffer).
    """

    def __init__(self, client_id: int, session_key: bytes, logger: BDSFLogger, max_age: float = 60.0):
        self.client_id = client_id
        self.session_key = session_key
        self.logger = logger
        self.max_age = max_age
        self._deserializer = ForceSyncDispatcher()
        self.seen_nonces: set = set()
        self.nonce_history: deque = deque(maxlen=1000)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def verify_and_apply(
        self, payload: ForceSyncPayload, client_state: dict
    ) -> bool:
        """Called by ``ClientNode`` when a ``force_sync`` field is present
        in the server response.

        ``client_state`` is a **mutable** dict with keys:
        - ``"W_local"``:  ``torch.Tensor`` — current local weights (overwritten on success)
        - ``"gradient_buffer"``:  ``list`` — local gradient accumulator (cleared on success)
        - ``"last_reset_time"``:  ``float`` — set to ``payload.timestamp`` on success
        - ``"current_virtual_time"``: ``float`` — current timeline timestamp (optional)
        - ``"force_sync_applied"``: ``bool`` — set to ``True`` on success

        Returns ``True`` if the payload was authentic and the reset was
        applied, ``False`` otherwise.
        """
        # Step 1: Check replay freshness (nonce duplicate)
        if payload.nonce in self.seen_nonces:
            self.logger.log_update(
                round=0,
                client_id=self.client_id,
                status="WARN",
                reason="FORCE_SYNC_REPLAY_NONCE",
            )
            return False

        # Step 2: Check monotonic timestamp freshness against last reset
        if payload.timestamp <= client_state.get("last_reset_time", 0.0):
            self.logger.log_update(
                round=0,
                client_id=self.client_id,
                status="WARN",
                reason="FORCE_SYNC_STALE_TIMESTAMP",
            )
            return False

        # Step 3: Check absolute age freshness window against current virtual time
        curr_time = client_state.get("current_virtual_time")
        if curr_time is not None and (curr_time - payload.timestamp) > self.max_age:
            self.logger.log_update(
                round=0,
                client_id=self.client_id,
                status="WARN",
                reason="FORCE_SYNC_EXPIRED_TIMESTAMP",
            )
            return False

        # Step 2: Reconstruct expected HMAC and compare in constant time
        msg = (
            payload.weights
            + str(payload.timestamp).encode("utf-8")
            + payload.nonce
        )
        expected_mac = hmac.new(
            self.session_key, msg, hashlib.sha256
        ).digest()

        # Step 2: Constant-time comparison.
        if not hmac.compare_digest(expected_mac, payload.hmac_digest):
            self.logger.log_update(
                round=0,
                client_id=self.client_id,
                status="WARN",
                reason="FORCE_SYNC_HMAC_FAIL",
            )
            return False

        # Step 3: Verification passed — record nonce and apply hard reset.
        if len(self.nonce_history) >= self.nonce_history.maxlen:
            evicted = self.nonce_history.popleft()
            self.seen_nonces.discard(evicted)
        self.seen_nonces.add(payload.nonce)
        self.nonce_history.append(payload.nonce)

        W_new = self._deserializer.deserialize_weights(payload.weights)
        client_state["W_local"] = W_new.clone()
        client_state["gradient_buffer"] = []
        client_state["last_reset_time"] = payload.timestamp
        client_state["force_sync_applied"] = True

        self.logger.log_update(
            round=0,
            client_id=self.client_id,
            status="INFO",
            reason="FORCE_SYNC_APPLIED",
        )
        return True
