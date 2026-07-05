import os
import csv

def compute_metrics(csv_path, byz_ids, honest_ids):
    decisions = {}
    
    if not os.path.exists(csv_path):
        return None, None
        
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("round") or not row.get("client_id") or not row.get("status"):
                continue
            
            if row["status"] == "INFO":
                continue
                
            round_val = int(row["round"])
            cid = int(row["client_id"])
            status = row["status"]
            
            key = (round_val, cid)
            if key not in decisions:
                decisions[key] = status
                
    byz_accepted = 0
    byz_rejected = 0
    honest_accepted = 0
    honest_rejected = 0
    
    for (round_val, cid), status in decisions.items():
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
    return frr, tpr

def main():
    modes = ["topk"]
    attacks = ["NONE", "S1_POISON", "S2_MIMICRY", "ADAPTIVE", "COMPOUND"]
    N = 20

    print("=" * 85)
    print("METRICS FOR ORIGINAL (PRE-FIX) LOGS")
    print("=" * 85)
    orig_path = "logs/NONE_42_updates.csv"
    if os.path.exists(orig_path):
        byz_ids = set()
        honest_ids = set(range(N))
        frr, tpr = compute_metrics(orig_path, byz_ids, honest_ids)
        print(f"Original NONE Run (NONE_42_updates.csv):")
        print(f"  FRR (False Rejection Rate): {frr:.2%}")
    else:
        print("Original NONE_42_updates.csv not found.")
    
    print("\n" + "=" * 85)
    print("METRICS FOR PHASE 2 LOGS (PREFIX: phase2_)")
    print("=" * 85)
    
    for attack in attacks:
        if attack == "NONE":
            byz_fraction = 0.0
        else:
            byz_fraction = 0.2
            
        byz_count = int(N * byz_fraction)
        byz_ids = set(range(byz_count))
        honest_ids = set(range(byz_count, N))
        
        print(f"\nADVERSARIAL SUMMARY TABLE FOR {attack} ATTACK")
        print("-" * 70)
        if attack == "NONE":
            print(f"| {'Mode':<15} | {'FRR (Honest)':<15} |")
            print(f"|{'-'*17}|{'-'*17}|")
        else:
            print(f"| {'Mode':<15} | {'Detection (TPR)':<15} | {'FRR (Under Attack)':<20} |")
            print(f"|{'-'*17}|{'-'*17}|{'-'*22}|")
            
        for mode in modes:
            csv_path = f"logs/phase2_{mode}_{attack}_42_updates.csv"
            frr, tpr = compute_metrics(csv_path, byz_ids, honest_ids)
            
            if frr is not None:
                if attack == "NONE":
                    print(f"| {mode:<15} | {frr:>14.2%} |")
                else:
                    print(f"| {mode:<15} | {tpr:>14.2%} | {frr:>19.2%} |")
            else:
                if attack == "NONE":
                    print(f"| {mode:<15} | {'N/A (Missing)':>14} |")
                else:
                    print(f"| {mode:<15} | {'N/A (Missing)':>14} | {'N/A (Missing)':>19} |")
        print("-" * 70)

if __name__ == "__main__":
    main()
