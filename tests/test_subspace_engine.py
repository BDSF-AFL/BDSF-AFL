import math
import os
import sys
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import CIFAR10ResNet18
from server.subspace_engine import (
    SubspaceProjectionEngine,
    DEFAULT_RESNET18_SLICES,
    build_stage_slices_from_model,
)


def test_stage_slices_cifar10_resnet18():
    model = CIFAR10ResNet18()
    slices = build_stage_slices_from_model(model)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 11173962
    assert 'stage4' in slices
    assert 'fc' in slices
    assert slices['fc'][1] == total_params


def test_subspace_cold_start():
    engine = SubspaceProjectionEngine(K=5)
    v = torch.randn(1000)
    v_clean, metrics = engine.filter_update(cid=1, v=v, tau=0.0)
    assert metrics["reason"] == "COLD_START_UNCONSTRAINED"
    assert torch.allclose(v_clean, v)


def test_basis_update_and_orthonormality():
    d = 5000
    K = 8
    engine = SubspaceProjectionEngine(K=K)

    # Admit K linearly independent vectors
    for _ in range(K):
        vec = torch.randn(d)
        engine.update_basis(vec)

    assert engine.Q is not None
    assert engine.Q.shape == (d, K)

    # Check orthonormality Q^T Q == I_K
    gram = torch.matmul(engine.Q.T, engine.Q)
    eye = torch.eye(K, device=gram.device)
    assert torch.allclose(gram, eye, atol=1e-4)


def test_gate1_macro_inversion_rejection():
    d = 2000
    engine = SubspaceProjectionEngine(K=3, eps_floor=0.05)

    base_dir = torch.randn(d)
    base_dir = base_dir / torch.linalg.vector_norm(base_dir)
    engine.update_basis(base_dir)

    # S1 Sign-Flip: Directly inverted direction
    s1_poison = -10.0 * base_dir
    v_clean, metrics = engine.filter_update(cid=1, v=s1_poison)
    assert metrics["action"] == "REJECT"
    assert metrics["reason"] == "MACRO_MANIFOLD_INVERSION"
    assert torch.all(v_clean == 0)


def test_gate1b_layer_wise_cross_layer_masking():
    # Use small custom slices for fast unit test
    custom_slices = {
        'stem': (0, 100),
        'layer4': (100, 900),
        'fc': (900, 1000),
    }
    engine = SubspaceProjectionEngine(K=3, eps_floor=0.05, kappa=0.1, stage_slices=custom_slices)

    # Construct consensus basis
    consensus_v = torch.randn(1000)
    engine.update_basis(consensus_v)

    # Construct cross-layer attacker:
    # Highly positive in Layer 4 (80% of parameters), but collapsed/zeroed in FC head
    adv_v = consensus_v.clone()
    adv_v[900:1000] = 0.0  # Zero out classification head to mask variance

    v_clean, metrics = engine.filter_update(cid=2, v=adv_v)
    assert metrics["action"] == "REJECT"
    assert metrics["reason"] == "CROSS_LAYER_COLLAPSE"
    assert metrics["rule_1b_passed"] is False
    assert torch.all(v_clean == 0)


def test_gate2_tier1_smooth_rolloff():
    d = 2000
    engine = SubspaceProjectionEngine(K=3, alpha=0.6, lam=3.0)

    basis = torch.randn(d)
    basis = basis / torch.linalg.vector_norm(basis)
    engine.update_basis(basis)

    M_perp = 2.0

    # 1. Update with low orthogonal norm (0.5 * M_perp <= 0.6 * M_perp) -> w_damp should be 1.0
    v_parallel = basis.clone()
    v_perp_raw = torch.randn(d)
    v_perp_ortho = v_perp_raw - torch.dot(v_perp_raw, basis) * basis
    v_perp_low = v_perp_ortho / torch.linalg.vector_norm(v_perp_ortho) * 1.0
    v_low = v_parallel + v_perp_low

    _, m_low = engine.filter_update(cid=10, v=v_low, M_perp_base=M_perp)
    assert pytest.approx(m_low["w_damp"], rel=1e-3) == 1.0

    # 2. Boundary-hugging update (0.95 * M_perp) -> excess = 0.35, w_damp ~ exp(-3 * 0.35^2) ~ 0.69
    v_perp_high = v_perp_ortho / torch.linalg.vector_norm(v_perp_ortho) * 1.90
    v_high = v_parallel + v_perp_high

    _, m_high = engine.filter_update(cid=11, v=v_high, M_perp_base=M_perp)
    assert m_high["w_damp"] < 0.75
    assert m_high["w_damp"] > 0.50

    # 3. Massive overshoot (3.0 * M_perp) -> w_damp ~ 0
    v_perp_massive = v_perp_ortho / torch.linalg.vector_norm(v_perp_ortho) * 6.0
    v_massive = v_parallel + v_perp_massive

    _, m_massive = engine.filter_update(cid=12, v=v_massive, M_perp_base=M_perp)
    assert m_massive["w_damp"] < 1e-4


def test_gate2_tier2_temporal_ewma_suppression():
    d = 2000
    engine = SubspaceProjectionEngine(K=2, beta=0.85)

    basis = torch.randn(d)
    basis = basis / torch.linalg.vector_norm(basis)
    engine.update_basis(basis)

    # Fixed orthogonal poison direction
    u_adv = torch.randn(d)
    u_adv = u_adv - torch.dot(u_adv, basis) * basis
    u_adv = u_adv / torch.linalg.vector_norm(u_adv)

    # Attacker submits along identical u_adv across 10 rounds
    attacker_temporal_weights = []
    for _ in range(10):
        v = basis + 1.0 * u_adv
        _, m = engine.filter_update(cid=666, v=v, M_perp_base=5.0)
        attacker_temporal_weights.append(m["w_temporal"])

    # Round 0 has w_temporal == 1.0; by round 10 it should drop near zero
    assert attacker_temporal_weights[0] == 1.0
    assert attacker_temporal_weights[-1] < 0.10

    # Honest client with changing random orthogonal noise
    honest_temporal_weights = []
    for _ in range(10):
        u_honest = torch.randn(d)
        u_honest = u_honest - torch.dot(u_honest, basis) * basis
        u_honest = u_honest / torch.linalg.vector_norm(u_honest)
        v = basis + 1.0 * u_honest
        _, m = engine.filter_update(cid=101, v=v, M_perp_base=5.0)
        honest_temporal_weights.append(m["w_temporal"])

    # Honest random walk maintains high temporal weight (> 0.85 on average)
    avg_honest_weight = sum(honest_temporal_weights) / len(honest_temporal_weights)
    assert avg_honest_weight > 0.85


def test_staleness_aware_dilation():
    d = 1000
    engine = SubspaceProjectionEngine(K=2)
    basis = torch.randn(d)
    engine.update_basis(basis)

    v = basis + 0.5 * torch.randn(d)
    _, m_tau0 = engine.filter_update(cid=1, v=v, tau=0.0, M_perp_base=2.0)
    _, m_tau10 = engine.filter_update(cid=2, v=v, tau=10.0, M_perp_base=2.0)

    # M_perp(tau=10) = 2.0 * sqrt(1 + 0.1 * 10) = 2.0 * sqrt(2.0) ~ 2.828
    assert pytest.approx(m_tau0["M_perp"], rel=1e-3) == 2.0
    assert pytest.approx(m_tau10["M_perp"], rel=1e-3) == 2.0 * math.sqrt(2.0)


def test_gate1_full_basis_honest_non_iid_cascade_free():
    """Verifies that when basis is full (K=10), honest non-IID clients with class
    variance are never rejected with MACRO_MANIFOLD_INVERSION, while true inversions are rejected."""
    d = 10000
    K = 10
    engine = SubspaceProjectionEngine(K=K, macro_floor=0.25)

    base = torch.randn(d)
    base = base / torch.linalg.vector_norm(base)

    # Populate full basis with K=10 consensus vectors
    for _ in range(K):
        noise = torch.randn(d)
        noise = noise - torch.dot(noise, base) * base
        noise = noise / torch.linalg.vector_norm(noise)
        vec = 0.85 * base + math.sqrt(1 - 0.85**2) * noise
        engine.update_basis(vec)

    assert engine.is_basis_full()
    assert engine.consensus_dir is not None

    # Test 20 honest non-IID clients with varying class alignment
    for cid in range(20):
        target_cos = 0.10 + (cid % 10) * 0.05  # Cosines from 0.10 to 0.55
        noise = torch.randn(d)
        noise = noise - torch.dot(noise, base) * base
        noise = noise / torch.linalg.vector_norm(noise)
        v_honest = (target_cos * base + math.sqrt(1 - target_cos**2) * noise) * 5.0

        v_clean, metrics = engine.filter_update(cid=cid, v=v_honest)
        assert metrics["action"] == "ACCEPT", f"Client {cid} falsely rejected: {metrics}"
        assert metrics["reason"] != "MACRO_MANIFOLD_INVERSION"
        assert metrics["mu"] >= -0.25

    # True macro inversion must still be caught
    v_inverted = -10.0 * base
    v_clean_inv, metrics_inv = engine.filter_update(cid=999, v=v_inverted)
    assert metrics_inv["action"] == "REJECT"
    assert metrics_inv["reason"] == "MACRO_MANIFOLD_INVERSION"
    assert torch.all(v_clean_inv == 0)

