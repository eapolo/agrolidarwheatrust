import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Load & Prepare Data ==========
file_path = "data-rust-net.csv"
df = pd.read_csv(file_path, delimiter=";", decimal=",")

# Convert to numeric where necessary
for col in ["Nog", "Noe", "DAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# If you have columns named with spaces, rename them (optional):
df = df.rename(columns=lambda x: x.replace(" ", "_"))

# ========== 2. Define the Metrics & Labels ==========
metrics = ["Nog", "Noe"]
y_labels = ["Number of Grains (Nog)", "Number of Ears (Noe)"]

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12

# ========== 3. Create Subplots (2x2) ==========
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharex=True)

# ========== 3a. Add Row Annotations ==========
axes[0, 0].annotate(
    "Cultivars non-inoculated", xy=(0.5, 1.08), xycoords='axes fraction',
    ha='center', va='bottom', fontsize=12
)
axes[1, 0].annotate(
    "Cultivars inoculated", xy=(0.5, 1.08), xycoords='axes fraction',
    ha='center', va='bottom', fontsize=12
)

bar_width = 0.8

# ========== 4. Plot Loop ==========
for col_index, (metric, y_label) in enumerate(zip(metrics, y_labels)):
    for row_index, inoculated_value in enumerate([0, 1]): 
        # 0 = Non-Inoculated, 1 = Inoculated
        ax = axes[row_index, col_index]
        
        # Filter for current inoculation status
        subset = df[df["Inoculate"] == inoculated_value]
        
        # Compute mean & std error
        group_stats = (
            subset.groupby(["DAI", "variety"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        group_stats["SE"] = group_stats["std"] / np.sqrt(group_stats["count"])
        
        # Pivot for grouped bar plotting
        mean_table = group_stats.pivot(index="DAI", columns="variety", values="mean")
        se_table   = group_stats.pivot(index="DAI", columns="variety", values="SE")

        # Plot grouped bars
        bars = mean_table.plot(
            kind="bar", ax=ax, alpha=0.8, legend=False,
            width=bar_width, edgecolor="black", colormap="Set2"
        )

        # Add error bars
        for bar_group, errors in zip(bars.containers, se_table.values.T):
            for bar, err in zip(bar_group, errors):
                if not np.isnan(err):
                    ax.errorbar(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        yerr=err, fmt='none', ecolor='gray',
                        elinewidth=1, capsize=4, capthick=1
                    )
        
        # Y-axis label
        ax.set_ylabel(y_label)
        
        # Only put x-axis label on the bottom row
        if row_index == 1:
            ax.set_xlabel("Days After Inoculation (DAI)")
        else:
            ax.set_xlabel("")  # Hide label for top row

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=0)
        ax.grid(False)

# ========== 5. Unified Legend (No Title) ==========
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", ncol=len(labels),
    fontsize=11, frameon=False
)

# ========== 6. Final Layout Adjustments & Save ==========
plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for the legend

# Save as high-resolution PNG (300 DPI)
save_path = "Nog_Noe_Figure.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
print(f"✅ Saved figure as {save_path}")

# Show the figure
plt.show()
