import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# ========== 1. Load & Prepare Data ==========
file_path = "data-rust-net.csv"  # Update path if necessary
df = pd.read_csv(file_path, delimiter=";", decimal=",")

# Select relevant columns
correlation_columns = [
    "Int50", "Int60", "Int70", "Int80", "Int90", 
    "Severity", "type_wheat", "variety", "Inoculate"
]
df_selected = df[correlation_columns].copy()

# Convert numeric columns to numeric dtype
for col in ["Int50", "Int60", "Int70", "Int80", "Int90", "Severity"]:
    df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

# Drop rows with missing values in selected columns
df_selected = df_selected.dropna(subset=["Int50", "Int60", "Int70", "Int80", "Int90", "Severity", "Inoculate"])

# Filter only for non-inoculated samples (Inoculate = 1)
df_non_inoculated = df_selected[df_selected["Inoculate"] == 1]

# Get unique wheat types (e.g., "Durum" & "Bread")
wheat_types = df_non_inoculated["type_wheat"].unique()

# Define custom colormap (brown-to-teal gradient)
custom_cmap = sns.color_palette("BrBG", as_cmap=True)

# Set global font settings (optional, adjust as you like)
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Palatino Linotype"

# ========== 2. Generate Grouped Heatmaps ==========
for wheat in wheat_types:
    # Get unique cultivars for the current wheat type, sorted alphabetically
    cultivars = sorted(df_non_inoculated[df_non_inoculated["type_wheat"] == wheat]["variety"].unique())

    # Determine subplot layout (group in rows of 3)
    num_cultivars = len(cultivars)
    num_cols = 3
    num_rows = math.ceil(num_cultivars / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    # If there's only 1 row & 1 col, axes might not be an array
    axes = axes.flatten() if num_cultivars > 1 else [axes]

    for idx, cultivar in enumerate(cultivars):
        subset = df_non_inoculated[
            (df_non_inoculated["type_wheat"] == wheat) &
            (df_non_inoculated["variety"] == cultivar)
        ]

        if subset.empty:
            continue  # Skip if no data

        # Select only numeric columns for correlation
        numeric_cols = ["Int50", "Int60", "Int70", "Int80", "Int90", "Severity"]
        corr_matrix = subset[numeric_cols].corr(method="pearson")

        # Determine dynamic min/max for the heatmap color scale
        vmin, vmax = corr_matrix.min().min(), corr_matrix.max().max()

        # Plot heatmap
        ax = axes[idx]
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f",
            cmap=custom_cmap, vmin=vmin, vmax=vmax,
            linewidths=1, linecolor='white', annot_kws={"size": 10}, 
            ax=ax
        )
        ax.set_title(f"{cultivar}")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=12, fontfamily="Palatino Linotype", rotation=90
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=12, rotation=0,fontfamily="Palatino Linotype"
        )

    # Remove empty subplots if the number of cultivars < num_rows * num_cols
    for idx in range(num_cultivars, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure (300 DPI) as PNG
    save_path = f"Pearson_Correlation_Heatmap_{wheat}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved heatmap for {wheat} wheat as {save_path}")

    plt.show()
