import os
import csv
import numpy as np

def compute_metrics(csv_path, byz_ids, honest_ids, K_ref=3):
    if not os.path.exists(csv_path):
        return None

    decisions = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("round") or not row.get("client_id") or not row.get("status"):
                continue
            if row["status"] in ("INFO", "WARN"):
                continue
            decisions.append({
                "round": int(row["round"]),
                "client_id": int(row["client_id"]),
                "status": row["status"],
                "reason": row["reason"],
                "sim_global": float(row["sim_global"]) if row.get("sim_global") else None,
                "norm_raw": float(row["norm_raw"]) if row.get("norm_raw") else None,
                "g_i": float(row["g_i"]) if row.get("g_i") else None,
                "is_burn_in": row.get("is_burn_in") == "True"
            })

    byz_accepted = 0
    byz_rejected = 0
    honest_accepted = 0
    honest_rejected = 0

    for d in decisions:
        cid = d["client_id"]
        status = d["status"]
        if cid in byz_ids:
            if status == "REJECT":
                byz_rejected += 1
            else:
                byz_accepted += 1
        elif cid in honest_ids:
            if status == "REJECT":
                honest_rejected += 1
            else:
                honest_accepted += 1

    total_honest = honest_accepted + honest_rejected
    total_byz = byz_accepted + byz_rejected

    frr = honest_rejected / total_honest if total_honest > 0 else 0.0
    tpr = byz_rejected / total_byz if total_byz > 0 else 0.0

    # Post burn-in calculations
    post_burn_decisions = [d for d in decisions if not d["is_burn_in"]]
    byz_accepted_post = sum(1 for d in post_burn_decisions if d["client_id"] in byz_ids and d["status"] != "REJECT")
    byz_rejected_post = sum(1 for d in post_burn_decisions if d["client_id"] in byz_ids and d["status"] == "REJECT")
    total_byz_post = byz_accepted_post + byz_rejected_post
    tpr_post_burn = byz_rejected_post / total_byz_post if total_byz_post > 0 else 0.0

    honest_sims = [d["sim_global"] for d in decisions if d["client_id"] in honest_ids and d["sim_global"] is not None]
    avg_honest_sim = np.mean(honest_sims) if honest_sims else 0.0

    byz_sims = [d["sim_global"] for d in decisions if d["client_id"] in byz_ids and d["sim_global"] is not None]
    avg_byz_sim = np.mean(byz_sims) if byz_sims else 0.0

    return {
        "total_updates": len(decisions),
        "total_honest": total_honest,
        "honest_accepted": honest_accepted,
        "honest_rejected": honest_rejected,
        "frr": frr,
        "total_byz": total_byz,
        "byz_accepted": byz_accepted,
        "byz_rejected": byz_rejected,
        "tpr": tpr,
        "tpr_post_burn": tpr_post_burn,
        "avg_honest_sim": avg_honest_sim,
        "avg_byz_sim": avg_byz_sim
    }

def main():
    log_dir = "logs/phase1_verify/"
    N = 10
    byz_count = 2 # 20% byzantine
    byz_ids = set(range(byz_count))
    honest_ids = set(range(byz_count, N))
    honest_all_ids = set(range(N))

    print("=" * 85)
    print("PHASE 1 OBSERVABILITY: FRR / TPR & TELEMETRY METRIC EVALUATION")
    print(f"Directory: {log_dir}")
    print("=" * 85)

    # 1. NONE attack scenario
    none_csv = os.path.join(log_dir, "NONE_42_updates.csv")
    if os.path.exists(none_csv):
        m_none = compute_metrics(none_csv, byz_ids=set(), honest_ids=honest_all_ids)
        print(f"\n[SCENARIO 1] NONE Attack (100% Honest)")
        print(f"  CSV File: {none_csv}")
        print(f"  Total Submissions: {m_none['total_updates']}")
        print(f"  Honest Accepted:   {m_none['honest_accepted']}")
        print(f"  Honest Rejected:   {m_none['honest_rejected']}")
        print(f"  FRR (False Rejection Rate): {m_none['frr']:.2%}")
        print(f"  Avg Honest Cosine Similarity: {m_none['avg_honest_sim']:.4f}")
    else:
        print(f"\n[SCENARIO 1] NONE_42_updates.csv not found in {log_dir}")

    # 2. COMPOUND attack scenario
    compound_csv = os.path.join(log_dir, "COMPOUND_42_updates.csv")
    if os.path.exists(compound_csv):
        m_comp = compute_metrics(compound_csv, byz_ids=byz_ids, honest_ids=honest_ids)
        print(f"\n[SCENARIO 2] COMPOUND Attack (20% Byzantine: Clients {list(byz_ids)})")
        print(f"  CSV File: {compound_csv}")
        print(f"  Total Submissions: {m_comp['total_updates']}")
        print(f"  Honest Submissions: {m_comp['total_honest']} (Accepted: {m_comp['honest_accepted']}, Rejected: {m_comp['honest_rejected']})")
        print(f"  Byzantine Submissions: {m_comp['total_byz']} (Detected/Rejected: {m_comp['byz_rejected']}, Slipped/Accepted: {m_comp['byz_accepted']})")
        print(f"  FRR (Honest under attack): {m_comp['frr']:.2%}")
        print(f"  TPR (Overall Detection Rate): {m_comp['tpr']:.2%}")
        print(f"  TPR (Post Burn-In Detection): {m_comp['tpr_post_burn']:.2%}")
        print(f"  Avg Honest Cosine Similarity: {m_comp['avg_honest_sim']:.4f}")
        print(f"  Avg Byzantine Cosine Similarity: {m_comp['avg_byz_sim']:.4f}")
    else:
        print(f"\n[SCENARIO 2] COMPOUND_42_updates.csv not found in {log_dir}")

    print("\n" + "=" * 85)

if __name__ == "__main__":
    main()
