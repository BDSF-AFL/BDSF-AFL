import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.device_utils import mark_step

def compute_accuracy(model: torch.nn.Module, test_loader: DataLoader, W_global: torch.Tensor, device = "cpu") -> float:
    acc, _ = compute_evaluation_metrics(model, test_loader, W_global, device=device)
    return acc

def compute_evaluation_metrics(model: torch.nn.Module, test_loader: DataLoader, W_global: torch.Tensor, device = "cpu"):
    """Computes both test accuracy and cross-entropy loss on evaluation dataset."""
    model = model.to(device)
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.copy_(W_global[offset:offset + numel].reshape(param.shape).to(device))
            offset += numel

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        mark_step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss

def _post_warmup_byz_rows(rejection_log: list[dict], byzantine_ids: set[int]) -> list[dict]:
    return [r for r in rejection_log
            if r.get("client_id") in byzantine_ids
            and r.get("status") in ("ACCEPT", "DOWNWEIGHT", "QUARANTINE", "REJECT")
            and r.get("reason") != "SPATIAL_WARMUP_ACCEPT"
            and not r.get("is_warmup")]


def _is_true(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def _has_modified_flag(rows: list[dict]) -> bool:
    return any(str(r.get("attack_modified", "")).strip().lower() in ("true", "false") for r in rows)


def compute_attack_success_rate(rejection_log: list[dict], byzantine_ids: set[int]) -> float:
    """Share of ATTACKED submissions that reached the model (ACCEPT or DOWNWEIGHT), post-warmup.

    Two problems of the previous definition are fixed:
      * warmup accepts (the defence accepts everything by design) are excluded;
      * a Byzantine client's UNMODIFIED update (S2 sends the honest update untouched when its natural cosine is
        already below the target) is an honest gradient, not an attack, so it is not counted. This needs the
        ``attack_modified`` ground-truth flag the simulator now logs; for older logs without it the function falls
        back to all post-warmup Byzantine submissions (``compute_detection_breakdown['influence_rate']``).
    """
    rows = _post_warmup_byz_rows(rejection_log, byzantine_ids)
    if _has_modified_flag(rows):
        rows = [r for r in rows if _is_true(r.get("attack_modified"))]
    if not rows:
        return 0.0
    admitted = sum(1 for r in rows if r.get("status") in ("ACCEPT", "DOWNWEIGHT"))
    return admitted / len(rows)


def compute_poison_weight_share(rejection_log: list[dict], byzantine_ids: set[int]) -> float:
    """Aggregation weight carried by ATTACKED updates / total aggregation weight, post-warmup (impact, not count).
    Falls back to all Byzantine weight when the ``attack_modified`` flag is absent."""
    post = [r for r in rejection_log
            if r.get("status") in ("ACCEPT", "DOWNWEIGHT")
            and r.get("reason") != "SPATIAL_WARMUP_ACCEPT" and not r.get("is_warmup")]
    byz = [r for r in post if r.get("client_id") in byzantine_ids]
    if _has_modified_flag(_post_warmup_byz_rows(rejection_log, byzantine_ids)):
        byz = [r for r in byz if _is_true(r.get("attack_modified"))]
    total = sum(float(r.get("weight") or 0.0) for r in post)
    return (sum(float(r.get("weight") or 0.0) for r in byz) / total) if total > 0 else 0.0

def compute_detection_breakdown(rejection_log: list[dict], byzantine_ids: set[int],
                                exclude_warmup: bool = True) -> dict:
    """Per-action breakdown for Byzantine submissions plus a REJECT-reason histogram.

    ``compute_attack_success_rate`` lumps QUARANTINE in with "caught" and cannot say WHICH
    defense fired. Use this to check each defense is alive: e.g. if
    ``reject_reasons`` has no ``BOUNDARY_PINNING_REJECT`` for S2/ADAPTIVE, the pinning gate is
    not firing. Warmup accepts are excluded by default (they accept everything by design).
    """
    rows = [r for r in rejection_log
            if r.get("client_id") in byzantine_ids
            and r.get("status") in ("ACCEPT", "DOWNWEIGHT", "QUARANTINE", "REJECT")
            and not (exclude_warmup and r.get("reason") == "SPATIAL_WARMUP_ACCEPT")]
    n = len(rows)
    out = {"n_submissions": n}
    for s in ("ACCEPT", "DOWNWEIGHT", "QUARANTINE", "REJECT"):
        c = sum(1 for r in rows if r.get("status") == s)
        out[s.lower()] = c
        out[s.lower() + "_frac"] = (c / n) if n else 0.0
    reasons: dict = {}
    for r in rows:
        if r.get("status") == "REJECT":
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
    out["reject_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    # "Influence" rate: share of post-warmup Byzantine submissions that actually reached the model
    out["influence_rate"] = ((out["accept"] + out["downweight"]) / n) if n else 0.0
    return out

def compute_false_rejection_rate(rejection_log: list[dict], honest_ids: set[int]) -> float:
    honest_submissions = [r for r in rejection_log if r.get("client_id") in honest_ids and r.get("status") in ["ACCEPT", "DOWNWEIGHT", "QUARANTINE", "REJECT"]]
    if not honest_submissions: return 0.0
    rejected = sum(1 for r in honest_submissions if r.get("status") == "REJECT")
    return rejected / len(honest_submissions)

def compute_reputation_precision(rep_manager, byzantine_ids: set[int], threshold: float = 0.5,
                                 use_composite: bool = False) -> float:
    """Fraction of low-reputation clients that are truly Byzantine.

    By default only the integrity score I is thresholded (original behaviour). Pass
    ``use_composite=True`` to threshold I*P instead, which also surfaces clients that were only
    penalised on pace (e.g. COMPOUND's T1 half before the leaky-rejection fix).
    """
    all_client_ids = list(rep_manager.scores.keys())
    def _score(cid):
        I, P = rep_manager.get(cid)
        return I * P if use_composite else I
    low_rep = [cid for cid in all_client_ids if _score(cid) < threshold]
    if not low_rep: return 1.0
    true_positives = sum(1 for cid in low_rep if cid in byzantine_ids)
    return true_positives / len(low_rep)

def compute_convergence_time(accuracy_log: list[float], target: float = 0.85) -> float:
    for i, acc in enumerate(accuracy_log):
        if acc >= target:
            return float(i)
    return float("inf")

def compute_reputation_separation_auc(rep_manager, honest_ids: set[int], byzantine_ids: set[int]) -> float:
    """Computes Wilcoxon-Mann-Whitney rank AUC of reputation separation between honest and byzantine clients."""
    if not rep_manager or not honest_ids or not byzantine_ids:
        return 1.0
    honest_scores = [rep_manager.get(cid)[0] for cid in honest_ids if cid in rep_manager.scores]
    byz_scores = [rep_manager.get(cid)[0] for cid in byzantine_ids if cid in rep_manager.scores]
    if not honest_scores or not byz_scores:
        return 1.0
    pairs = 0
    wins = 0.0
    for h in honest_scores:
        for b in byz_scores:
            pairs += 1
            if h > b:
                wins += 1.0
            elif h == b:
                wins += 0.5
    return wins / pairs if pairs > 0 else 1.0

def compute_reputation_means(rep_manager, honest_ids: set[int], byzantine_ids: set[int]) -> dict:
    """Computes mean Integrity and Performance reputation for honest vs byzantine cohorts."""
    if not rep_manager:
        return {
            "final_I_mean_honest": 1.0, "final_P_mean_honest": 1.0,
            "final_I_mean_byzantine": 0.0, "final_P_mean_byzantine": 0.0
        }
    h_I = [rep_manager.get(cid)[0] for cid in honest_ids if cid in rep_manager.scores]
    h_P = [rep_manager.get(cid)[1] for cid in honest_ids if cid in rep_manager.scores]
    b_I = [rep_manager.get(cid)[0] for cid in byzantine_ids if cid in rep_manager.scores]
    b_P = [rep_manager.get(cid)[1] for cid in byzantine_ids if cid in rep_manager.scores]
    return {
        "final_I_mean_honest": float(np.mean(h_I)) if h_I else 1.0,
        "final_P_mean_honest": float(np.mean(h_P)) if h_P else 1.0,
        "final_I_mean_byzantine": float(np.mean(b_I)) if b_I else 0.0,
        "final_P_mean_byzantine": float(np.mean(b_P)) if b_P else 0.0,
    }

def compute_comm_overhead_per_node(model_dim: int, include_hmac: bool = True) -> dict:
    gradient_bytes = model_dim * 4
    hmac_bytes = 32 + 8 + 8
    total = gradient_bytes + (hmac_bytes if include_hmac else 0)
    return {"gradient_bytes": gradient_bytes, "overhead_bytes": hmac_bytes, "total": total}
