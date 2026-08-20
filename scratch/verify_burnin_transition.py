import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
from server.aggregator import AggregatorServer
from shared.types import UpdateSubmission
from utils.logger import BDSFLogger


def test_burnin_boundary_transition():
    print("=== Testing End-to-End Burn-in Boundary Transition ===")
    config = {
        "N_clients": 5,
        "burn_in_count": 5,
        "K_base": 5,
        "K_ref": 3,
        "M": 10,
        "theta_cos": 0.1,
        "theta_self": 0.6,
        "theta_floor": 0.15,
        "gamma_clip": 1.5,
        "adaptive_clip_enabled": True,
        "static_clip_C": 10.0,
        "decision_mode": "joint",
        "log_dir": "logs/test_boundary/",
    }

    W_init = torch.randn(100)
    logger = BDSFLogger("test_boundary_run", config)
    server = AggregatorServer(config, W_init, list(range(5)), logger)
    server.N_burn = 5
    server.temporal_filter.N_burn = 5

    # First update: Zero vector from client 0 (simulates edge case)
    zero_sub = UpdateSubmission(client_id=0, delta_W=torch.zeros(100), t_submit=1.0, tau=0)
    res0 = server.handle_update(zero_sub)
    print(f"Update 0 (Zero vector): status={res0['status']}, reason={res0['reason']}")
    assert res0['status'] == "ACCEPT"
    assert res0['reason'] in ["SPATIAL_WARMUP_ACCEPT", "BURN_IN_ACCEPT"]

    # Updates 1..4: Positive non-zero updates during burn-in
    ref_direction = torch.randn(100)
    ref_direction = ref_direction / torch.norm(ref_direction)

    for i in range(1, 5):
        # Slightly noisy version of ref_direction with norm ~ 2.0
        dW = (ref_direction + 0.1 * torch.randn(100)) * 2.0
        sub = UpdateSubmission(client_id=i, delta_W=dW, t_submit=1.0 + i * 2.0, tau=i)
        res = server.handle_update(sub)
        print(f"Update {i} (Early submission): status={res['status']}, reason={res['reason']}")
        assert res['status'] == "ACCEPT"
        if i <= 3:
            assert res['reason'] in ["SPATIAL_WARMUP_ACCEPT", "BURN_IN_ACCEPT"]
        else:
            # Adaptive warm-start activates immediately once K_ref=3 valid vectors accumulate!
            assert res['reason'] == "FULL_CONSENSUS_ACCEPT"

    print("-> State-Maturity spatial reference active.")

    # Update 5: First post-burn-in submission (Honest matching direction)
    dW_post1 = (ref_direction + 0.05 * torch.randn(100)) * 2.0
    sub_post1 = UpdateSubmission(client_id=1, delta_W=dW_post1, t_submit=12.0, tau=5)
    res_post1 = server.handle_update(sub_post1)
    print(f"Update 5 (Post-burn-in honest): status={res_post1['status']}, reason={res_post1['reason']}")
    
    # Must NOT be UNCOORDINATED_OR_ADVERSARIAL_REJECT!
    assert res_post1['status'] == "ACCEPT", f"Expected ACCEPT but got {res_post1['status']}"
    assert res_post1['reason'] == "FULL_CONSENSUS_ACCEPT", f"Expected FULL_CONSENSUS_ACCEPT but got {res_post1['reason']}"

    # Update 6: Non-IID honest client with moderate divergence (sim_g ~ 0.0, sim_s ~ 0.9)
    # Give client 2 consistent personal history in behavioral memory
    dW_personal = torch.randn(100)
    dW_personal = dW_personal / torch.norm(dW_personal) * 2.0
    # Manually populate client 2 behavioral memory with its personal direction
    for _ in range(3):
        server.behavioral_memory.on_accept(2, dW_personal)

    # Submitting personal direction
    sub_post2 = UpdateSubmission(client_id=2, delta_W=dW_personal, t_submit=14.0, tau=6)
    res_post2 = server.handle_update(sub_post2)
    print(f"Update 6 (Post-burn-in non-IID honest): status={res_post2['status']}, reason={res_post2['reason']}")
    assert res_post2['status'] in ["ACCEPT", "DOWNWEIGHT"]

    # Verify W_global actually moved
    diff = torch.norm(server.W_global - W_init).item()
    print(f"Total ||W_global - W_init|| movement: {diff:.4f}")
    assert diff > 0.0, "Model weights must advance and not stay frozen at W_init"

    print("\n[SUCCESS] Burn-in boundary test PASSED with valid spatial reference and model weight updates!")


if __name__ == "__main__":
    test_burnin_boundary_transition()
