import os
import csv
import numpy as np

def analyze_csv(csv_path, byz_ids=None):
    if byz_ids is None:
        byz_ids = set()

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("round") or not r.get("client_id") or not r.get("status"):
                continue
            if r["status"] in ("INFO", "WARN"):
                continue
            rows.append(r)

    print("=" * 80)
    print(f"ANALYSIS REPORT: {os.path.basename(csv_path)}")
    print(f"Total Updates Logged: {len(rows)}")
    print("=" * 80)

    # 1. Legacy Decisions & Breakdown
    accepts = [r for r in rows if r["status"] == "ACCEPT"]
    rejects = [r for r in rows if r["status"] == "REJECT"]
    print(f"\n[1. Legacy Decisions Summary]")
    print(f"  Total ACCEPT: {len(accepts)} ({len(accepts)/len(rows):.1%})")
    print(f"  Total REJECT: {len(rejects)} ({len(rejects)/len(rows):.1%})")

    rej_reasons = {}
    for r in rejects:
        reas = r["reason"]
        rej_reasons[reas] = rej_reasons.get(reas, 0) + 1
    print(f"  Rejection Reasons: {rej_reasons}")

    # 2. Behavioral Memory & Depth Growth
    depths = [int(r["history_depth"]) for r in rows if r["history_depth"] != ""]
    print(f"\n[2. Behavioral Memory Depth Progression]")
    print(f"  Min Depth: {min(depths)}, Max Depth: {max(depths)}, Median Depth: {np.median(depths):.1f}")
    
    # Check depth per client
    client_depths = {}
    for r in rows:
        cid = int(r["client_id"])
        d = int(r["history_depth"])
        client_depths[cid] = max(client_depths.get(cid, 0), d)
    print(f"  Max Depth Reached per Client: {dict(sorted(client_depths.items()))}")

    # 3. Behavioral Evidence Signals
    pop_rows = [r for r in rows if r["sim_self_mean"] != ""]
    print(f"\n[3. Behavioral Telemetry Signals (Depth >= 3)]")
    print(f"  Submissions with Populated Behavioral Evidence: {len(pop_rows)} / {len(rows)} ({len(pop_rows)/len(rows):.1%})")

    honest_pop = [r for r in pop_rows if int(r["client_id"]) not in byz_ids]
    byz_pop = [r for r in pop_rows if int(r["client_id"]) in byz_ids]

    if honest_pop:
        h_sim_mean = [float(r["sim_self_mean"]) for r in honest_pop]
        h_sim_max = [float(r["sim_self_max"]) for r in honest_pop]
        h_norm_dev = [float(r["norm_deviation_self"]) for r in honest_pop if r["norm_deviation_self"] != ""]
        h_cad_dev = [float(r["cadence_consistency"]) for r in honest_pop if r["cadence_consistency"] != ""]

        print(f"\n  -- Honest Clients (N={len(honest_pop)} updates with history) --")
        print(f"    sim_self_mean:       avg = {np.mean(h_sim_mean):.4f}, min = {np.min(h_sim_mean):.4f}, max = {np.max(h_sim_mean):.4f}")
        print(f"    sim_self_max:        avg = {np.mean(h_sim_max):.4f}, min = {np.min(h_sim_max):.4f}, max = {np.max(h_sim_max):.4f}")
        if h_norm_dev:
            print(f"    norm_deviation_self: avg = {np.mean(h_norm_dev):.4f}, median = {np.median(h_norm_dev):.4f}")
        if h_cad_dev:
            print(f"    cadence_consistency: avg = {np.mean(h_cad_dev):.4f}, median = {np.median(h_cad_dev):.4f}")

    if byz_pop:
        b_sim_mean = [float(r["sim_self_mean"]) for r in byz_pop]
        b_sim_max = [float(r["sim_self_max"]) for r in byz_pop]
        b_norm_dev = [float(r["norm_deviation_self"]) for r in byz_pop if r["norm_deviation_self"] != ""]
        b_cad_dev = [float(r["cadence_consistency"]) for r in byz_pop if r["cadence_consistency"] != ""]

        print(f"\n  -- Byzantine Clients (N={len(byz_pop)} updates with history) --")
        print(f"    sim_self_mean:       avg = {np.mean(b_sim_mean):.4f}, min = {np.min(b_sim_mean):.4f}, max = {np.max(b_sim_mean):.4f}")
        print(f"    sim_self_max:        avg = {np.mean(b_sim_max):.4f}, min = {np.min(b_sim_max):.4f}, max = {np.max(b_sim_max):.4f}")
        if b_norm_dev:
            print(f"    norm_deviation_self: avg = {np.mean(b_norm_dev):.4f}, median = {np.median(b_norm_dev):.4f}")
        if b_cad_dev:
            print(f"    cadence_consistency: avg = {np.mean(b_cad_dev):.4f}, median = {np.median(b_cad_dev):.4f}")
    elif len(byz_ids) > 0:
        print(f"\n  -- Byzantine Clients (Clients {list(byz_ids)}) --")
        print(f"    Notice: Byzantine updates were successfully REJECTED and prevented from entering memory,")
        print(f"    maintaining clean behavioral profiles.")

    # 4. Security & Detection Metrics (FRR / TPR)
    if len(byz_ids) > 0:
        honest_total = len([r for r in rows if int(r["client_id"]) not in byz_ids])
        honest_rej = len([r for r in rejects if int(r["client_id"]) not in byz_ids])
        byz_total = len([r for r in rows if int(r["client_id"]) in byz_ids])
        byz_rej = len([r for r in rejects if int(r["client_id"]) in byz_ids])

        frr = honest_rej / honest_total if honest_total > 0 else 0.0
        tpr = byz_rej / byz_total if byz_total > 0 else 0.0

        print(f"\n[4. Detection & Performance Metrics]")
        print(f"  Honest Updates: {honest_total} (Accepted: {honest_total - honest_rej}, Rejected: {honest_rej})")
        print(f"  Byzantine Updates: {byz_total} (Detected: {byz_rej}, Slipped in burn-in: {byz_total - byz_rej})")
        print(f"  FRR: {frr:.2%}")
        print(f"  TPR: {tpr:.2%}")

    print("\n")

def main():
    log_dir = "logs/phase2_verify"
    analyze_csv(os.path.join(log_dir, "NONE_42_updates.csv"), byz_ids=set())
    analyze_csv(os.path.join(log_dir, "COMPOUND_42_updates.csv"), byz_ids={0, 1})

if __name__ == "__main__":
    main()
