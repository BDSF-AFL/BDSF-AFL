from dataclasses import dataclass, field
from typing import Optional, Dict, Any
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
    delta_W:   torch.Tensor
    I_score:   float           # I_i at time of acceptance
    P_score:   float           # P_i at time of acceptance
    client_id: Optional[int] = None  # Client ID of submitting client

@dataclass
class TemporalEvidence:
    """Continuous temporal features extracted on update submission."""
    g_i: float                         # Raw behavioral gap in seconds
    lower_fence: Optional[float]       # Current lower Tukey fence L
    upper_fence: Optional[float]       # Current upper Tukey fence U
    fence_margin: float                # Normalized distance to nearest fence (0.0 if within [L, U])
    client_z_score: float              # Gap normalized against client's local historical mean and std
    is_burn_in: bool                   # Deprecated legacy alias: Whether temporal manifold is in warmup
    temporal_mature: Optional[bool] = None  # True when >= temporal_min_samples & valid statistics exist

    def __post_init__(self):
        if self.temporal_mature is None:
            self.temporal_mature = not self.is_burn_in

@dataclass
class SpatialEvidence:
    """Continuous spatial features extracted on update submission."""
    sim_global: Optional[float]                    # Cosine similarity to Top-K reference vector (None if unavailable)
    norm_raw: float                                # Unclipped Euclidean norm of delta_W
    norm_clipped: float                            # Clipped Euclidean norm
    norm_ratio_median: Optional[float] = None      # ||delta_W|| / median(||delta_W_buffer||)
    dynamic_bound_C: Optional[float] = None        # Dynamic clipping threshold C_t (None during warmup)
    reference_available: bool = False              # Whether Top-K reference vector was available
    spatial_mature: Optional[bool] = None          # True when >= K_ref valid vectors exist
    spatial_reference_count: int = 0               # Count of positive contributors in buffer
    spatial_coherence: float = 0.0                 # Consensus coherence: ||(1/K)*sum(g_hat)||_2 in [0, 1]

    def __post_init__(self):
        if self.spatial_mature is None:
            self.spatial_mature = self.reference_available

@dataclass
class BehavioralEvidence:
    """Continuous historical behavioral consistency features."""
    sim_self_mean: Optional[float]        # Mean cosine similarity against client's own history (None if depth < min_history)
    sim_self_max: Optional[float]         # Max (nearest-neighbor) cosine similarity (None if depth < min_history)
    sim_self_mad: Optional[float] = None  # MAD dispersion of client's own historical trajectory
    norm_deviation_self: Optional[float] = None  # Robust MAD-based normalized deviation from client's historical norm
    cadence_consistency: Optional[float] = None  # Robust MAD-based normalized deviation of arrival gap
    history_depth: int = 0                # Number of entries in client's memory buffer
    sim_anchor: Optional[float] = None    # Cosine similarity against Genesis Anchor (active when depth >= 1)
    consecutive_dw: int = 0               # Active consecutive downweight streak
    behavioral_mature: Optional[bool] = None  # True when history_depth >= behavioral_min_depth (depth >= 3)

    def __post_init__(self):
        if self.behavioral_mature is None:
            self.behavioral_mature = (self.history_depth >= 3)

@dataclass
class QuarantinedEntry:
    """Represents a suspicious or borderline update held in quarantine."""
    entry_id: str                          # Unique identifier
    client_id: int                         # Submitting client
    delta_W_clipped: torch.Tensor          # Detached float32 CPU tensor
    entry_round: int                       # Global server round when quarantined
    entry_virtual_time: float              # Virtual timestamp at quarantine entry
    reputation_at_entry: tuple             # (I_i, P_i) snapshot at entry
    primary_reason: str                    # Quarantine trigger reason

@dataclass
class JointDecisionOutcome:
    """Outcome returned by joint decision engine."""
    action: str                        # "ACCEPT" | "DOWNWEIGHT" | "QUARANTINE" | "REJECT"
    primary_reason: str                # Diagnostic reason string
    aggregation_weight: float          # Multiplier for global model merge (e.g. 1.0, 0.35, 0.0)
    force_sync_required: bool          # Whether HMAC force-sync payload should be dispatched
    diagnostic_features: Dict[str, Any] = field(default_factory=dict)

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
