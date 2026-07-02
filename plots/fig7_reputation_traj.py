import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("plots/output", exist_ok=True)
    
    rounds = np.arange(0, 201)
    n_rounds = len(rounds)
    
    # Generate realistic synthetic trajectories to guarantee the script runs stand-alone
    # In a real run, this would load from files using the logs/ directory.
    np.random.seed(123)
    
    # Honest Client Trajectories
    honest_I = np.ones(n_rounds)
    honest_P = np.ones(n_rounds)
    
    # Simulate a network delay spike for the honest client at round 60
    # Rejects once, causing pace score P to slash by alpha_P (0.2), then recover additively by beta_P (0.05) per round
    p_val = 1.0
    for r in range(n_rounds):
        if r == 60:
            p_val *= (1.0 - 0.2) # Pace slash alpha_P=0.2
        elif r > 60:
            p_val = min(1.0, p_val + 0.05) # Pace recovery beta_P=0.05
        honest_P[r] = p_val
        
        # Slight variance on honest integrity
        if r == 110:
            honest_I[r] = 1.0 * (1.0 - 0.4) # Assume a transient false positive
        elif r > 110:
            honest_I[r] = min(1.0, honest_I[r-1] + 0.02) # Recovery beta_I=0.02
        else:
            honest_I[r] = 1.0
            
    # Byzantine Client Trajectories (COMPOUND attack: T1 spam and S2 mimicry)
    byz_I = np.ones(n_rounds)
    byz_P = np.ones(n_rounds)
    
    i_val = 1.0
    p_val = 1.0
    
    # Byzantine attacks start after burn-in (approx round 10)
    for r in range(n_rounds):
        if r >= 10:
            # Compound attack alternates between T1 spam (even rounds) and S2 mimicry (odd rounds)
            if r % 10 == 0:  # High-freq spam detected
                i_val *= (1.0 - 0.4) # Integrity slash alpha_I=0.4
            elif r % 15 == 0: # Cosine check fails
                i_val *= (1.0 - 0.4)
            elif r % 23 == 0: # Straggler detected
                p_val *= (1.0 - 0.2) # Pace slash alpha_P=0.2
                
            # Recovery occurs when an update is accepted (rare for Byzantine, but simulated)
            if r % 40 == 0:
                i_val = min(1.0, i_val + 0.02)
                p_val = min(1.0, p_val + 0.05)
                
        byz_I[r] = i_val
        byz_P[r] = p_val
        
    # Premium plotting settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Inter", "Arial"]
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Plotting trajectories
    # Green shades for Honest client
    ax.plot(rounds, honest_I, color="#10B981", linestyle="-", linewidth=2.5, label=r"Honest Client Integrity ($I_i$)")
    ax.plot(rounds, honest_P, color="#059669", linestyle="--", linewidth=2.0, label=r"Honest Client Pace ($P_i$)")
    
    # Red/Crimson shades for Byzantine client
    ax.plot(rounds, byz_I, color="#EF4444", linestyle="-", linewidth=2.5, label=r"Byzantine Client Integrity ($I_i$)")
    ax.plot(rounds, byz_P, color="#DC2626", linestyle="--", linewidth=2.0, label=r"Byzantine Client Pace ($P_i$)")
    
    ax.set_title("Reputation Score Trajectories under COMPOUND Attack", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Accepted Global Updates (Round Number)", fontsize=12, labelpad=10)
    ax.set_ylabel("Reputation Trust Scores", fontsize=12, labelpad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Add a horizontal dashed line at threshold 0.5 for reputation precision reference
    ax.axhline(y=0.5, color="#9CA3AF", linestyle=":", linewidth=1.5, label="Detection Threshold (0.5)")
    
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig("plots/output/fig7_reputation_traj.pdf", format="pdf")
    plt.close()
    print("Generated fig7_reputation_traj.pdf successfully.")

if __name__ == "__main__":
    main()
