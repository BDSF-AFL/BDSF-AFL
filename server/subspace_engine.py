import math
from collections import deque
from typing import Dict, Tuple, Optional, Any, List

import torch
import torch.nn as nn


# Default stage slices matching CIFAR10ResNet18 (11,173,962 params)
DEFAULT_RESNET18_SLICES: Dict[str, Tuple[int, int]] = {
    'stem':   (0, 1856),
    'stage1': (1856, 149824),
    'stage2': (149824, 675392),
    'stage3': (675392, 2775104),
    'stage4': (2775104, 11168832),
    'fc':     (11168832, 11173962),
}


def build_stage_slices_from_model(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """Dynamically builds stage slice offsets from any PyTorch model."""
    stage_prefixes = {
        'stem': ['conv1', 'bn1', 'stem'],
        'stage1': ['layer1', 'block1'],
        'stage2': ['layer2', 'block2'],
        'stage3': ['layer3', 'block3'],
        'stage4': ['layer4', 'block4'],
        'fc': ['fc', 'classifier', 'linear', 'head'],
    }
    offsets: Dict[str, List[int]] = {}
    curr = 0
    for name, p in model.named_parameters():
        num = p.numel()
        matched = 'stem'
        for s_name, prefixes in stage_prefixes.items():
            if any(name.startswith(pfx) for pfx in prefixes):
                matched = s_name
                break
        if matched not in offsets:
            offsets[matched] = [curr, curr + num]
        else:
            offsets[matched][1] = curr + num
        curr += num
    
    if not offsets:
        return {'all': (0, curr)}
    return {k: (v[0], v[1]) for k, v in offsets.items()}


class SubspaceProjectionEngine:
    """Layer-Stratified K-Dimensional Subspace Projection Engine for BDSF-AFL v2.
    
    Decomposes incoming client updates v in R^D against a rolling orthonormal
    consensus basis Q in R^{D x K} into:
        v = v_parallel + v_perp
    where v_parallel in span(Q) represents consensus manifold learning and
    v_perp in Q^perp represents local non-IID novelty.
    
    Implements:
      - Gate 1: Macro Consensus Manifold Alignment (c = Q^T v)
      - Gate 1b: Layer-Wise Coordinate Variance Floor (eliminates cross-layer arbitrage)
      - Gate 2: Adaptive Orthogonal Energy Bounding with Two-Tier Damping:
          - Tier 1: Smooth Continuous Quadratic Roll-Off
          - Tier 2: Temporal EWMA Unit-Directional Suppression in Q^perp
    """

    def __init__(
        self,
        K: int = 10,
        eps_floor: float = 0.05,
        c_min_floor: float = 0.30,
        macro_floor: float = 0.25,
        alpha: float = 0.6,
        lam: float = 3.0,
        beta: float = 0.85,
        kappa: float = 0.10,
        stage_slices: Optional[Dict[str, Tuple[int, int]]] = None,
        default_m_perp_base: float = 2.0,
    ):
        self.K = K
        self.eps_floor = eps_floor
        self.c_min_floor = c_min_floor
        self.macro_floor = macro_floor
        self.alpha = alpha
        self.lam = lam
        self.beta = beta
        self.kappa = kappa
        self.stage_slices = stage_slices if stage_slices is not None else DEFAULT_RESNET18_SLICES
        self.default_m_perp_base = default_m_perp_base

        # Rolling consensus history and orthonormal basis
        self._basis_queue: deque[torch.Tensor] = deque(maxlen=self.K)
        self.Q: Optional[torch.Tensor] = None  # [D, K'] where K' <= K
        self.consensus_dir: Optional[torch.Tensor] = None  # [D], normalized centroid of admitted consensus vectors

        # Client directional momentum tracking in Q^perp: cid -> Tensor (on CPU to save VRAM)
        self.m_perp: Dict[int, torch.Tensor] = {}

        # Running median of trusted orthogonal norms
        self._trusted_perp_norms: deque[float] = deque(maxlen=50)

    # -------------------------------------------------------------------------
    # Basis Maintenance
    # -------------------------------------------------------------------------

    def update_basis(self, v_accepted: torch.Tensor) -> bool:
        """Admits a strictly verified consensus update into the rolling basis Q.
        
        Guards against dead, blank, uninitialized, or zero-norm parameter vectors
        that would produce degenerate matrix structures during Modified Gram-Schmidt.
        Returns True if admitted, False if skipped.
        """
        if v_accepted is None:
            return False
        flat_v = v_accepted.flatten().detach()
        if flat_v.numel() == 0 or not torch.all(torch.isfinite(flat_v)):
            return False

        norm_v = torch.linalg.vector_norm(flat_v).item()
        # Explicit cold-start dead / blank / zero-vector guard
        if norm_v < 1e-6:
            return False

        # Degenerate collinearity check: if new vector adds negligible orthogonal energy (< 0.01% norm),
        # do not append redundant vector that would collapse Gram-Schmidt rank
        if self.Q is not None and self.Q.shape[1] > 0:
            Q_dev = self.Q.to(flat_v.device)
            proj = torch.matmul(Q_dev, torch.matmul(Q_dev.T, flat_v))
            residual = flat_v - proj
            res_norm = torch.linalg.vector_norm(residual).item()
            if res_norm / (norm_v + 1e-8) < 1e-4:
                return False

        self._basis_queue.append((flat_v / norm_v).cpu())
        self._recompute_orthonormal_basis()
        return True

    def _recompute_orthonormal_basis(self) -> None:
        """Recomputes thin orthonormal basis Q via Modified Gram-Schmidt and consensus mean direction."""
        if not self._basis_queue:
            self.Q = None
            self.consensus_dir = None
            return

        # Stack into [D, num_vectors] on CPU
        R = torch.stack(list(self._basis_queue), dim=1)  # [D, N]
        D, N = R.shape
        target_device = self.Q.device if self.Q is not None else torch.device("cpu")

        # 1. Consensus mean direction (unorthogonalized normalized centroid of admitted consensus vectors)
        u_mean = torch.mean(R, dim=1)
        norm_u = torch.linalg.vector_norm(u_mean).item()
        if norm_u > 1e-6:
            self.consensus_dir = (u_mean / norm_u).to(target_device)
        else:
            self.consensus_dir = None

        # 2. Modified Gram-Schmidt
        Q_cols = []
        for i in range(N):
            q = R[:, i].clone()
            for q_prev in Q_cols:
                proj = torch.dot(q, q_prev)
                q = q - proj * q_prev
            norm_q = torch.linalg.vector_norm(q).item()
            if norm_q > 1e-6:
                Q_cols.append(q / norm_q)

        if Q_cols:
            self.Q = torch.stack(Q_cols, dim=1).to(target_device)  # [D, K']
        else:
            self.Q = None

    # -------------------------------------------------------------------------
    # Gate 1b: Layer-Wise Coordinate Variance Floor
    # -------------------------------------------------------------------------

    def check_rule_1b(self, v: torch.Tensor, v_parallel: torch.Tensor) -> bool:
        """Gate 1b: Rejects updates exhibiting cross-layer coordinate collapse or masking.
        
        Grounded directly against v_parallel = Q c, ensuring the floor cannot be manipulated
        by colluding adversaries in uncoordinated async FL rounds.
        """
        for stage, (start, end) in self.stage_slices.items():
            if start >= v.numel():
                continue
            actual_end = min(end, v.numel())
            block_v = v[start:actual_end]
            block_p = v_parallel[start:actual_end]

            if block_v.numel() > 1:
                tau_var = self.kappa * torch.var(block_p, unbiased=False)
                if torch.var(block_v, unbiased=False) < tau_var:
                    return False
        return True

    # -------------------------------------------------------------------------
    # Primary Filter Pipeline
    # -------------------------------------------------------------------------

    def filter_update(
        self,
        cid: int,
        v: torch.Tensor,
        tau: float = 0.0,
        M_perp_base: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decomposes, validates, and filters an incoming client update v.
        
        Returns:
            v_clean: Filtered gradient tensor with toxic energy excised.
            metrics: Dictionary of diagnostic evaluation metrics.
        """
        flat_v = v.flatten()
        device = flat_v.device
        norm_v = torch.linalg.vector_norm(flat_v).item()

        metrics: Dict[str, Any] = {
            "action": "ACCEPT",
            "reason": "CLEAN_MANIFOLD",
            "c_min": 0.0,
            "c_sum": 0.0,
            "norm_perp": 0.0,
            "M_perp": 0.0,
            "w_damp": 1.0,
            "w_temporal": 1.0,
            "rho": 0.0,
            "rule_1b_passed": True,
        }

        # Cold start fallback if basis is not yet populated
        if self.Q is None or self.Q.shape[1] == 0:
            metrics["reason"] = "COLD_START_UNCONSTRAINED"
            return v, metrics

        # Ensure Q is on matching device
        if self.Q.device != device:
            self.Q = self.Q.to(device)

        # 1. Gate 1: Macro Consensus Manifold Alignment (Unit-Normalized O(KD))
        norm_v = torch.linalg.vector_norm(flat_v).item()
        if norm_v < 1e-8:
            metrics["reason"] = "ZERO_NORM_PASS"
            return v, metrics

        v_unit = flat_v / norm_v

        # Directional consensus manifold alignment
        if self.consensus_dir is not None:
            if self.consensus_dir.device != device:
                self.consensus_dir = self.consensus_dir.to(device)
            mu = torch.dot(v_unit, self.consensus_dir).item()
        else:
            mu = 1.0

        c_hat = torch.matmul(self.Q.T, v_unit)  # [K'], coordinates
        c_hat_min = c_hat.min().item() if c_hat.numel() > 0 else 0.0
        c_hat_sum = c_hat.sum().item() if c_hat.numel() > 0 else 0.0
        metrics["c_min"] = c_hat_min
        metrics["c_sum"] = c_hat_sum
        metrics["mu"] = mu

        # Gate 1 check: Macro Manifold Inversion
        # Rejects if the update actively opposes the consensus manifold learning trajectory
        if mu < -self.macro_floor:
            metrics["action"] = "REJECT"
            metrics["reason"] = "MACRO_MANIFOLD_INVERSION"
            return torch.zeros_like(v), metrics

        # 2. Decompose into manifold and orthogonal residual (exact unnormalized coordinates)
        c = c_hat * norm_v
        v_parallel = torch.matmul(self.Q, c)
        v_perp = flat_v - v_parallel
        norm_perp = torch.linalg.vector_norm(v_perp).item()
        metrics["norm_perp"] = norm_perp

        # 3. Gate 1b: Layer-Wise Coordinate Variance Floor
        if not self.check_rule_1b(flat_v, v_parallel):
            metrics["action"] = "REJECT"
            metrics["reason"] = "CROSS_LAYER_COLLAPSE"
            metrics["rule_1b_passed"] = False
            return torch.zeros_like(v), metrics

        # 4. Gate 2: Adaptive Orthogonal Energy Bounding with Two-Tier Damping
        if M_perp_base is None:
            if len(self._trusted_perp_norms) >= 5:
                M_perp_base = float(torch.median(torch.tensor(list(self._trusted_perp_norms))))
            else:
                M_perp_base = self.default_m_perp_base

        M_perp = M_perp_base * math.sqrt(1.0 + 0.1 * float(max(0.0, tau)))
        metrics["M_perp"] = M_perp

        # Tier 1: Smooth Continuous Boundary Roll-Off
        ratio = norm_perp / (M_perp + 1e-8)
        excess = max(0.0, ratio - self.alpha)
        w_damp = math.exp(-self.lam * (excess ** 2))
        metrics["w_damp"] = w_damp

        # Tier 2: Temporal Directional Suppression in Q^perp (Unit-Directional Tracking)
        w_temporal = 1.0
        rho = 0.0

        if norm_perp > 1e-8:
            u_perp = (v_perp / norm_perp).detach().cpu()  # Store on CPU to avoid GPU OOM
            prev_m = self.m_perp.get(cid, None)

            if prev_m is not None:
                norm_m = torch.linalg.vector_norm(prev_m).item()
                if norm_m > 1e-8:
                    cos_sim = (torch.dot(u_perp, prev_m).item()) / (norm_m + 1e-8)
                    rho = max(0.0, cos_sim)
                    w_temporal = 1.0 - (rho ** 2)

                # Accumulate unit direction (convex combination naturally bounded <= 1.0)
                self.m_perp[cid] = (self.beta * prev_m + (1.0 - self.beta) * u_perp).detach()
            else:
                self.m_perp[cid] = u_perp.clone()
        else:
            prev_m = self.m_perp.get(cid, None)
            if prev_m is not None:
                self.m_perp[cid] = (self.beta * prev_m).detach()

        metrics["rho"] = rho
        metrics["w_temporal"] = w_temporal

        # Reconstruct safe, filtered update
        v_perp_scaled = v_perp * (w_damp * w_temporal)
        v_clean_flat = v_parallel + v_perp_scaled
        v_clean = v_clean_flat.reshape(v.shape)

        # Track trusted orthogonal norm if clean update passed
        if w_damp > 0.95 and w_temporal > 0.95:
            self._trusted_perp_norms.append(norm_perp)

        return v_clean, metrics

    # -------------------------------------------------------------------------
    # State Inspection & Serialization
    # -------------------------------------------------------------------------

    def basis_count(self) -> int:
        """Returns the current number of basis vectors in the rolling queue."""
        return len(self._basis_queue)

    def is_basis_full(self) -> bool:
        """Returns True if the rolling basis has accumulated K vectors."""
        return len(self._basis_queue) >= self.K

    def get_state(self) -> Dict[str, Any]:
        """Serializes full state for atomic checkpointing and exact replay."""
        return {
            "basis_queue": [v.clone().cpu() for v in self._basis_queue],
            "Q": self.Q.clone().cpu() if self.Q is not None else None,
            "consensus_dir": self.consensus_dir.clone().cpu() if self.consensus_dir is not None else None,
            "m_perp": {cid: m.clone().cpu() for cid, m in self.m_perp.items()},
            "trusted_perp_norms": list(self._trusted_perp_norms),
        }

    def load_state(self, state: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        """Restores state from checkpoint, automatically projecting active basis Q onto the target device."""
        self._basis_queue.clear()
        for v in state.get("basis_queue", []):
            self._basis_queue.append(v.clone().cpu())
        target_dev = device if device is not None else torch.device("cpu")
        if state.get("Q") is not None:
            self.Q = state["Q"].clone().to(target_dev)
        else:
            self.Q = None
        if state.get("consensus_dir") is not None:
            self.consensus_dir = state["consensus_dir"].clone().to(target_dev)
        elif self._basis_queue:
            u_mean = torch.mean(torch.stack(list(self._basis_queue), dim=1), dim=1)
            norm_u = torch.linalg.vector_norm(u_mean).item()
            self.consensus_dir = (u_mean / norm_u).to(target_dev) if norm_u > 1e-6 else None
        else:
            self.consensus_dir = None
        self.m_perp = {int(cid): m.clone().cpu() for cid, m in state.get("m_perp", {}).items()}
        self._trusted_perp_norms = deque(state.get("trusted_perp_norms", []), maxlen=50)

