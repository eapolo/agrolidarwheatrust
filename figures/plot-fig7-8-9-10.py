import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# ========== 1. Load & Prepare Data ==========
file_path = "data-rust-net.csv"  # Update path if necessary
df = pd.read_csv(file_path, delimiter=";", decimal=",")

# Select relevant columns
correlation_columns = ["Noe", "Aph", "Biomass", "Gw", "Nog", "type_wheat", "variety", "Severity", "Inoculate"]
df_selected = df[correlation_columns].copy()

# Convert to numeric where necessary
for col in ["Severity", "Noe", "Aph", "Biomass", "Gw", "Nog"]:
    df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

# Drop rows with missing values in selected columns
df_selected = df_selected.dropna()

# Filter only for non-Inoculate samples (Inoculate = 1)
df_non_Inoculate = df_selected[df_selected["Inoculate"] == 1] #Change for non-inoculate to 0

# Get unique wheat types (Durum & Bread)
wheat_types = df_non_Inoculate["type_wheat"].unique()

# Define custom colormap (brown-to-teal gradient)
custom_cmap = sns.color_palette("BrBG", as_cmap=True)

# Set global font settings
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Palatino Linotype"

# ========== 2. Generate Grouped Heatmaps ==========
for wheat in wheat_types:
    # Get unique cultivars for the current wheat type
    cultivars = sorted(df_non_Inoculate[df_non_Inoculate["type_wheat"] == wheat]["variety"].unique())  # Sorted alphabetically

    # Determine subplot layout (group in rows of 3)
    num_cultivars = len(cultivars)
    num_cols = 3
    num_rows = math.ceil(num_cultivars / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten() if num_cultivars > 1 else [axes]  # Ensure axes is iterable

    for idx, cultivar in enumerate(cultivars):
        subset = df_non_Inoculate[
            (df_non_Inoculate["type_wheat"] == wheat) & 
            (df_non_Inoculate["variety"] == cultivar)
        ]

        if subset.empty:
            continue  # Skip if no data

        # Compute correlation matrix
        corr_matrix = subset[["Severity", "Noe", "Aph", "Biomass", "Gw", "Nog"]].corr(method="pearson")

        # Determine min and max dynamically
        vmin, vmax = corr_matrix.min().min(), corr_matrix.max().max()

        # Plot heatmap in the correct subplot
        ax = axes[idx]
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap=custom_cmap, 
            vmin=vmin, vmax=vmax, linewidths=1, linecolor='white', ax=ax
        )
        ax.set_title(f"{cultivar}")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontfamily="Palatino Linotype", rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontfamily="Palatino Linotype")

    # Remove empty subplots if the number of cultivars is not a multiple of 3
    for idx in range(num_cultivars, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in high resolution (300 DPI) as PNG
    save_path = f"Pearson_Correlation_Heatmap_{wheat}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High resolution & no extra whitespace
    print(f"✅ Saved heatmap for {wheat} wheat as {save_path}")

    plt.show()
