"""BDSF-AFL Phase 3 Results Analyzer.

Computes:
- False Rejection Rate (FRR) on Honest Clients
- True Positive Rate (TPR / Detection Rate) on Byzantine Clients
- Precision, Recall, F1 Score
- Multi-Action Breakdown (ACCEPT, DOWNWEIGHT, QUARANTINE, REJECT)
- Quarantine Resolution Statistics
- Average Anchor Similarity and Consecutive Downweight Streaks
"""

import sys
import os
import pandas as pd
import numpy as np


def analyze_phase3_csv(csv_path: str, byz_clients: set = None):
    if not os.path.exists(csv_path):
        print(f"Error: Log file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n=======================================================")
    print(f"PHASE 3 LOG ANALYSIS: {os.path.basename(csv_path)}")
    print(f"Total Logged Updates: {len(df)}")
    print(f"=======================================================")

    # Determine Byzantine clients if not provided
    # By default in BDSF-AFL 20 clients with 20% byz => [16, 17, 18, 19]
    if byz_clients is None:
        if "COMPOUND" in csv_path or "byz" in csv_path or "ATTACK" in csv_path:
            byz_clients = {16, 17, 18, 19}
        else:
            byz_clients = set()

    all_clients = set(df["client_id"].unique())
    honest_clients = all_clients - byz_clients

    # 1. Action Distribution
    print("\n--- ACTION BREAKDOWN ---")
    action_col = "action" if "action" in df.columns else "status"
    action_counts = df[action_col].value_counts()
    for act, cnt in action_counts.items():
        pct = (cnt / len(df)) * 100.0
        print(f"  {act:<12}: {cnt:>5} ({pct:>5.1f}%)")

    # 2. Honest Clients Metrics
    honest_df = df[df["client_id"].isin(honest_clients)]
    if len(honest_df) > 0:
        honest_rejected = honest_df[honest_df[action_col] == "REJECT"]
        honest_dw = honest_df[honest_df[action_col] == "DOWNWEIGHT"]
        honest_accept = honest_df[honest_df[action_col] == "ACCEPT"]
        honest_quarantine = honest_df[honest_df[action_col] == "QUARANTINE"]
        
        frr = (len(honest_rejected) / len(honest_df)) * 100.0
        dw_rate = (len(honest_dw) / len(honest_df)) * 100.0
        
        print("\n--- HONEST CLIENT PERFORMANCE ---")
        print(f"  Total Honest Submissions : {len(honest_df)}")
        print(f"  Full Accepts             : {len(honest_accept):>4} ({len(honest_accept)/len(honest_df)*100:.1f}%)")
        print(f"  Soft Downweights         : {len(honest_dw):>4} ({dw_rate:.1f}%)")
        print(f"  Quarantined              : {len(honest_quarantine):>4} ({len(honest_quarantine)/len(honest_df)*100:.1f}%)")
        print(f"  Rejections (False Slashing): {len(honest_rejected):>4} ({frr:.1f}%)")
        print(f"  False Rejection Rate (FRR): {frr:.2f}%")

    # 3. Byzantine Clients Metrics
    if len(byz_clients) > 0:
        byz_df = df[df["client_id"].isin(byz_clients)]
        if len(byz_df) > 0:
            byz_rejected = byz_df[byz_df[action_col] == "REJECT"]
            byz_quarantine = byz_df[byz_df[action_col] == "QUARANTINE"]
            byz_passed = byz_df[byz_df[action_col].isin(["ACCEPT", "DOWNWEIGHT"])]
            
            tpr = (len(byz_rejected) / len(byz_df)) * 100.0
            asr = (len(byz_passed) / len(byz_df)) * 100.0
            
            print("\n--- BYZANTINE DEFENSE PERFORMANCE ---")
            print(f"  Total Byzantine Submissions: {len(byz_df)}")
            print(f"  Detected & Rejected        : {len(byz_rejected):>4} ({tpr:.1f}%)")
            print(f"  Quarantined / Neutralized  : {len(byz_quarantine):>4} ({len(byz_quarantine)/len(byz_df)*100:.1f}%)")
            print(f"  Attack Success Rate (ASR)  : {asr:.2f}%")
            print(f"  True Positive Rate (TPR)   : {tpr:.2f}%")

    # 4. Behavioral & Anchor Diagnostics
    if "sim_anchor" in df.columns:
        valid_anchor = df["sim_anchor"].dropna()
        if len(valid_anchor) > 0:
            print("\n--- DUAL-HORIZON GENESIS ANCHOR DIAGNOSTICS ---")
            print(f"  Mean Genesis Anchor Similarity: {valid_anchor.mean():.4f}")
            print(f"  Min Genesis Anchor Similarity : {valid_anchor.min():.4f}")
            print(f"  Max Consecutive Downweights   : {df['consecutive_dw'].max() if 'consecutive_dw' in df.columns else 'N/A'}")

    print("=======================================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Check latest updates csv in logs/
        csv_file = "logs/phase3_verify/fix_topk_COMPOUND_42_updates.csv"
        if not os.path.exists(csv_file):
            import glob
            files = glob.glob("logs/*_updates.csv")
            csv_file = files[0] if files else "logs/test_updates.csv"

    analyze_phase3_csv(csv_file)
