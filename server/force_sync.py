import hmac
import hashlib
import os
import time
import io
from typing import Optional
import torch
from shared.types import ForceSyncPayload


class ForceSyncDispatcher:
    """Builds HMAC-SHA256-signed ``ForceSyncPayload`` objects.

    Server calls this when a straggler is detected (g_i > U).
    Stateless utility class — no instance state beyond the methods.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build_payload(
        self,
        client_id: int,
        W_global: torch.Tensor,
        session_key: bytes,
        timestamp: Optional[float] = None,
    ) -> ForceSyncPayload:
        """Called by AggregatorServer when a straggler is detected.

        Returns a fully-signed ``ForceSyncPayload`` ready for dispatch.
        """
        weight_bytes = self._serialize_weights(W_global)
        if timestamp is None:
            timestamp = time.time()
        nonce = os.urandom(8)

        # HMAC message = weight_bytes || timestamp_str || nonce
        msg = weight_bytes + str(timestamp).encode("utf-8") + nonce
        mac = hmac.new(session_key, msg, hashlib.sha256).digest()

        return ForceSyncPayload(
            client_id=client_id,
            weights=weight_bytes,
            timestamp=timestamp,
            nonce=nonce,
            hmac_digest=mac,
        )

    def deserialize_weights(self, weight_bytes: bytes) -> torch.Tensor:
        """Reconstructs a ``torch.Tensor`` from serialised bytes.

        Also called by ``ForceSyncHandler`` on the client side.
        """
        buf = io.BytesIO(weight_bytes)
        return torch.load(buf, map_location="cpu")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _serialize_weights(self, W: torch.Tensor) -> bytes:
        """Serialises a tensor to raw bytes via ``torch.save``."""
        buf = io.BytesIO()
        torch.save(W.cpu(), buf)
        return buf.getvalue()
