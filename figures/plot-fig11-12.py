import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.patches as mpatches

# --------------------------
# 1. Load and prepare data
# --------------------------
file_path = "data-rust-net.csv"  # Update with your actual dataset path
df = pd.read_csv(file_path, delimiter=";", decimal=",")

df["Aph"] = pd.to_numeric(df["Aph"], errors="coerce")
df["Eph"] = pd.to_numeric(df["Eph"], errors="coerce")

# Rename column to remove trailing space
df = df.rename(columns={"type_wheat ": "tipo_de_trigo"})

# --------------------------
# 2. Define colors
# --------------------------
cultivar_palette = {
    "Amilcar": "#a6d8a4",       # Light green
    "Arthur Nick": "#f4a582",   # Light orange
    "Califa": "#df9be5",        # Light pink
    "Conil": "#b3e2cd",         # Light mint green
    "Kiko Nick": "#bababa",     # Light gray
    "Don Ricardo": "#e5d8af"    # Beige
}

# Optional: use a clean white Seaborn style
sns.set_theme(style="white")

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12

# --------------------------
# 3. Unique wheat types
# --------------------------
wheat_types = df["type_wheat"].unique()

# --------------------------
# 4. Loop over each wheat type
# --------------------------
for wheat_type in wheat_types:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)

    # Subset for this wheat type
    df_wheat = df[df["type_wheat"] == wheat_type]

    # For storing R² results
    model_stats = {0: {}, 1: {}}

    # We'll store cultivars we actually plot, for building patches in the legend
    cultivars_plotted = []

    # --------------------------
    # 5. Subplots: Non-Inoc (0) vs. Inoc (1)
    # --------------------------
    for col_idx, inoc_status in enumerate([0, 1]):
        ax = axes[col_idx]

        # Filter data for this subplot
        df_inoc = df_wheat[df_wheat["Inoculate"] == inoc_status].dropna(subset=["Eph", "Aph"])

        # Scatter + regression for each cultivar
        for cultivar in df_inoc["variety"].unique():
            subset = df_inoc[df_inoc["variety"] == cultivar]
            if len(subset) > 2:
                # Pearson correlation r²
                pearson_corr = np.corrcoef(subset["Eph"], subset["Aph"])[0, 1]
                pearson_r2 = pearson_corr**2
                model_stats[inoc_status][cultivar] = pearson_r2

                # Scatter points
                ax.scatter(
                    subset["Eph"],
                    subset["Aph"],
                    color=cultivar_palette.get(cultivar, "gray"),
                    s=100,
                    edgecolor="black",
                    marker="o"
                )

                # Regression line
                sns.regplot(
                    data=subset,
                    x="Eph",
                    y="Aph",
                    scatter=False,
                    color=cultivar_palette.get(cultivar, "gray"),
                    ci=None,
                    line_kws={"linewidth": 2},
                    ax=ax
                )

                # Keep track of which cultivars were plotted
                if cultivar not in cultivars_plotted:
                    cultivars_plotted.append(cultivar)

        # Titles, labels
        inoc_label = "non-inoculated" if inoc_status == 0 else "inoculated"
        ax.set_title(f"Cultivars {inoc_label}", fontsize=12)
        ax.set_xlabel("Estimated Plant Height (cm)", fontsize=12)
        ax.set_ylabel("Actual Plant Height (cm)", fontsize=12)
        ax.grid(False)

        # Display R² text at top-left
        for i, (cult, r2) in enumerate(model_stats[inoc_status].items()):
            ax.text(
                0.05, 0.95 - (i * 0.05),
                f"{cult} R²: {r2:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                fontstyle="italic",
                color=cultivar_palette.get(cult, "gray"),
                ha="left",
                va="top"
            )

    # --------------------------
    # 6. Create a legend with colored squares
    # --------------------------
    patch_list = []
    for cultivar in cultivars_plotted:
        color = cultivar_palette.get(cultivar, "gray")
        patch_list.append(mpatches.Patch(color=color, label=cultivar))

    # Single legend at the top
    fig.legend(
        handles=patch_list,
        loc="upper center",
        ncol=len(patch_list),
        bbox_to_anchor=(0.5, 1),
        frameon=False,
        fontsize=12,
        #title="Cultivar"
    )

    # --------------------------
    # 7. Save the Figure in High Resolution (300 DPI)
    # --------------------------
    save_path = f"Scatter_Plot_{wheat_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution PNG
    print(f"✅ Saved figure as {save_path}")

    # --------------------------
    # 8. Show Plot
    # --------------------------
    plt.show()
