import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.device_utils import mark_step

def compute_accuracy(model: torch.nn.Module, test_loader: DataLoader, W_global: torch.Tensor, device = "cpu") -> float:
    model = model.to(device)
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.copy_(W_global[offset:offset + numel].reshape(param.shape).to(device))
            offset += numel

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
        # Flush any pending XLA ops after eval loop (no-op on CUDA/CPU)
        mark_step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return correct / total if total > 0 else 0.0

def compute_attack_success_rate(rejection_log: list[dict], byzantine_ids: set[int]) -> float:
    byz_submissions = [r for r in rejection_log if r.get("client_id") in byzantine_ids]
    if not byz_submissions: return 0.0
    accepted = sum(1 for r in byz_submissions if r.get("status") == "ACCEPT")
    return accepted / len(byz_submissions)

def compute_false_rejection_rate(rejection_log: list[dict], honest_ids: set[int]) -> float:
    honest_submissions = [r for r in rejection_log if r.get("client_id") in honest_ids]
    if not honest_submissions: return 0.0
    rejected = sum(1 for r in honest_submissions if r.get("status") == "REJECT")
    return rejected / len(honest_submissions)

def compute_reputation_precision(rep_manager, byzantine_ids: set[int], threshold: float = 0.5) -> float:
    all_client_ids = list(rep_manager.scores.keys())
    low_rep = [cid for cid in all_client_ids if rep_manager.get(cid)[0] < threshold]
    if not low_rep: return 1.0
    true_positives = sum(1 for cid in low_rep if cid in byzantine_ids)
    return true_positives / len(low_rep)

def compute_convergence_time(accuracy_log: list[float], target: float = 0.85) -> float:
    for i, acc in enumerate(accuracy_log):
        if acc >= target:
            return float(i)
    return float("inf")

def compute_comm_overhead_per_node(model_dim: int, include_hmac: bool = True) -> dict:
    gradient_bytes = model_dim * 4
    hmac_bytes = 32 + 8 + 8
    total = gradient_bytes + (hmac_bytes if include_hmac else 0)
    return {"gradient_bytes": gradient_bytes, "overhead_bytes": hmac_bytes, "total": total}
