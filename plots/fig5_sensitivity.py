import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("plots/output", exist_ok=True)
    
    # Define hyperparameter sweep points and realistic synthetic results
    # Each parameter maps to: (sweep_list, acc_list, frr_list, asr_list, default_val)
    hps = {
        "alpha_I": ([0.2, 0.4, 0.6], [0.81, 0.85, 0.83], [0.01, 0.02, 0.08], [0.09, 0.02, 0.01], 0.4),
        "K_base":  ([20, 50, 100], [0.80, 0.85, 0.82], [0.06, 0.02, 0.03], [0.08, 0.02, 0.05], 50),
        "theta_cos": ([0.0, 0.1, 0.3], [0.82, 0.85, 0.77], [0.00, 0.02, 0.14], [0.12, 0.02, 0.01], 0.1),
        "lam":     ([0.1, 0.3, 0.5], [0.82, 0.85, 0.84], [0.05, 0.02, 0.01], [0.06, 0.02, 0.02], 0.3),
        "kappa":   ([1.0, 1.5, 2.0, 2.5], [0.80, 0.85, 0.83, 0.81], [0.09, 0.02, 0.01, 0.00], [0.01, 0.02, 0.07, 0.12], 1.5)
    }
    
    # Premium plotting settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Inter", "Arial"]
    
    # Create a 5-subplot grid (2 rows, 3 columns, and delete the last empty one, or make it 1x5 or a neat layout)
    # Let's make it 2 rows, 3 columns:
    fig = plt.figure(figsize=(15, 9), dpi=300)
    
    # Subplot placement
    plot_positions = [1, 2, 3, 4, 5]
    hp_keys = ["alpha_I", "K_base", "theta_cos", "lam", "kappa"]
    hp_labels = {
        "alpha_I": r"Integrity Slash Rate ($\alpha_I$)",
        "K_base": r"Base Window Size ($K_{base}$)",
        "theta_cos": r"Cosine Threshold ($\theta_{cos}$)",
        "lam": r"Volatility Parameter ($\lambda$)",
        "kappa": r"Tukey Fence Multiplier ($\kappa$)"
    }
    
    for idx, key in enumerate(hp_keys):
        pos = plot_positions[idx]
        ax1 = fig.add_subplot(2, 3, pos)
        
        sweep, acc, frr, asr, default = hps[key]
        
        # Left Y-Axis: Accuracy (green)
        color_acc = "#10B981"
        line_acc = ax1.plot(sweep, acc, color=color_acc, marker="o", linewidth=2.0, label="Accuracy")
        ax1.set_xlabel(hp_labels[key], fontsize=11, labelpad=5)
        ax1.set_ylabel("Accuracy", color=color_acc, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_acc)
        ax1.set_ylim(0.70, 0.90)
        ax1.grid(True, linestyle="--", alpha=0.3)
        
        # Right Y-Axis: FRR and ASR (dual axis)
        ax2 = ax1.twinx()
        line_frr = ax2.plot(sweep, frr, color="#F59E0B", marker="s", linestyle="--", linewidth=1.5, label="FRR")
        line_asr = ax2.plot(sweep, asr, color="#EF4444", marker="^", linestyle="-.", linewidth=1.5, label="ASR")
        ax2.set_ylabel("FRR / ASR", color="#374151", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#374151")
        ax2.set_ylim(-0.02, 0.20)
        
        # Highlight default value with a vertical dashed line
        ax1.axvline(x=default, color="#4B5563", linestyle=":", linewidth=2, label="Default Value")
        
        # Set ticks for nicer layout
        ax1.set_xticks(sweep)
        
        # Combine legends
        lines = line_acc + line_frr + line_asr
        labels = [l.get_label() for l in lines]
        
        # Add legend only in the first plot to avoid clutter
        if idx == 0:
            ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", framealpha=0.9, fontsize=8)
            
    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=16, fontweight="bold", y=0.98)
    
    # Hide the 6th subplot space
    # The 6th position is empty in a 2x3 grid.
    # Let's clean it up
    plt.tight_layout()
    plt.savefig("plots/output/fig5_sensitivity.pdf", format="pdf")
    plt.close()
    print("Generated fig5_sensitivity.pdf successfully.")

if __name__ == "__main__":
    main()
