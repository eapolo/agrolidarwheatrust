import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Load & Prepare Data ==========
file_path = "data-rust-net.csv"
df = pd.read_csv(file_path, delimiter=";", decimal=",")

# Convert numeric columns
for col in ["Aph", "Biomass", "Gw", "DAI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rename columns (optional)
df = df.rename(columns=lambda x: x.replace(" ", "_"))

# ========== 2. Set Plotting Parameters ==========
metrics = ["Aph", "Biomass", "Gw"]
y_labels = ["Actual Plant Height (cm)", "Biomass (g)", "Grain Weight (g)"]

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12

# ========== 3. Create Subplots ==========
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)

# -- Add row titles (Non-Inoculated / Inoculated) --
axes[0, 0].annotate(
    "Cultivars non-inoculated", xy=(0.5, 1.08), xycoords='axes fraction',
    ha='center', va='bottom', fontsize=12
)
axes[1, 0].annotate(
    "Cultivars inoculated", xy=(0.5, 1.08), xycoords='axes fraction',
    ha='center', va='bottom', fontsize=12
)

# ========== 4. Plot Each Metric for Each Row (Inoculation Status) ==========
bar_width = 0.8

for col_index, (metric, y_label) in enumerate(zip(metrics, y_labels)):
    for row_index, inoculated in enumerate([0, 1]):  # 0 = Non-Inoculated, 1 = Inoculated
        ax = axes[row_index, col_index]
        
        # -- Filter data for the current inoculation status --
        subset = df[df["Inoculate"] == inoculated]
        
        # -- Compute mean & standard error for each DAI×variety group --
        group_stats = subset.groupby(["DAI", "variety"])[metric].agg(["mean", "std", "count"]).reset_index()
        group_stats["SE"] = group_stats["std"] / np.sqrt(group_stats["count"])
        
        # -- Pivot table to prepare for grouped bar plot --
        mean_table = group_stats.pivot(index="DAI", columns="variety", values="mean")
        se_table   = group_stats.pivot(index="DAI", columns="variety", values="SE")
        
        # -- Plot the grouped bar chart --
        bars = mean_table.plot(
            kind="bar", ax=ax, alpha=0.8, legend=False,
            width=bar_width, edgecolor="black", colormap="Set2"
        )
        
        # -- Add error bars --
        for bar_group, errors in zip(bars.containers, se_table.values.T):
            for bar, err in zip(bar_group, errors):
                if not np.isnan(err):
                    ax.errorbar(
                        bar.get_x() + bar.get_width() / 2,  # X-center of bar
                        bar.get_height(),                  # Bar top
                        yerr=err, fmt='none', ecolor='gray',
                        elinewidth=1, capsize=4, capthick=1
                    )

        # -- Y-axis label --
        ax.set_ylabel(y_label)
        
        # -- Only label the x-axis on the bottom row --
        if row_index == 1:
            ax.set_xlabel("Days After Inoculation (DAI)")
        else:
            ax.set_xlabel("")  # Hide top row X label

        # -- Rotate x-axis labels slightly if desired --
        ax.tick_params(axis='x', rotation=0)
        
        # -- Clean up the grid --
        ax.grid(False)

# ========== 5. Create a Single Legend (no title) ==========
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center",
    ncol=len(labels), fontsize=12, frameon=False
)

# ========== 6. Tighten Layout & Save in High Resolution ==========
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as high-resolution PNG (300 DPI)
save_path = "Updated_Figure.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality image
print(f"✅ Saved figure as {save_path}")

# Show the figure
plt.show()
