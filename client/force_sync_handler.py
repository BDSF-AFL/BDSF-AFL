import hmac
import hashlib
import io
import torch
from shared.types import ForceSyncPayload
from server.force_sync import ForceSyncDispatcher
from utils.logger import BDSFLogger


class ForceSyncHandler:
    """Client-side handler for force-sync payloads.

    Receives a ``ForceSyncPayload``, verifies its HMAC, and if valid,
    hard-resets the client's local state (weights + gradient buffer).
    """

    def __init__(self, client_id: int, session_key: bytes, logger: BDSFLogger):
        self.client_id = client_id
        self.session_key = session_key
        self.logger = logger
        self._deserializer = ForceSyncDispatcher()

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

        Returns ``True`` if the payload was authentic and the reset was
        applied, ``False`` otherwise.
        """
        # Step 1: Reconstruct the expected HMAC.
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

        # Step 3: Verification passed — apply hard reset.
        W_new = self._deserializer.deserialize_weights(payload.weights)
        client_state["W_local"] = W_new.clone()
        client_state["gradient_buffer"] = []
        client_state["last_reset_time"] = payload.timestamp

        self.logger.log_update(
            round=0,
            client_id=self.client_id,
            status="INFO",
            reason="FORCE_SYNC_APPLIED",
        )
        return True
