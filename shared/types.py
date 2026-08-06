from dataclasses import dataclass, field
import torch

@dataclass
class UpdateSubmission:
    """Sent from ClientNode to AggregatorServer on each push."""
    client_id:  int
    delta_W:    torch.Tensor   # flattened 1D: W_local - W_global_at_pull
    t_submit:   float          # time.time() at push moment
    tau:        float          # time.time() when client pulled W_global
                               # s_i (staleness) = t_submit - tau  [NOT used by BDSF-AFL directly]

@dataclass
class AcceptedEntry:
    """Stored in accepted_buffer after a gradient passes all gates."""
    delta_W:  torch.Tensor
    I_score:  float            # I_i at time of acceptance
    P_score:  float            # P_i at time of acceptance

@dataclass
class ForceSyncPayload:
    """HMAC-signed server-to-client hard-reset payload."""
    client_id:  int
    weights:    bytes          # torch.save → BytesIO → bytes
    timestamp:  float
    nonce:      bytes          # 8 random bytes
    hmac_digest: bytes         # HMAC-SHA256

@dataclass
class ClientRegistration:
    """Server-side per-client state."""
    client_id:       int
    session_key:     bytes     # 32-byte random key; established at registration
    last_update_time: float    # time of last ACCEPTED update (not just any submit)
    pull_time:       float     # time of last W_global pull by this client
    is_byzantine:    bool      # ground truth label — used ONLY by metrics/logger
