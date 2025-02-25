import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Define custom colors for each cultivar
cultivar_colors = {
    "Amilcar": "#76A089",
    "Arthur Nick": "#DA8D7B",
    "Califa": "#E3A5C2",
    "Conil": "#B0D878",
    "Don Ricardo": "#E3D3A4",
    "Kiko Nick": "#A8A8A8"
}

# Read the CSV file
df = pd.read_csv('data-rust-net.csv', delimiter=';', encoding='utf-8')

# Convert necessary columns to numeric, handling errors
df["Severity"] = pd.to_numeric(df["Severity"], errors='coerce')
df["Inoculate"] = pd.to_numeric(df["Inoculate"].astype(str).str.replace(",", "."), errors='coerce')
df["DAI"] = pd.to_numeric(df["DAI"], errors='coerce')

# Filter for inoculated plants
df_filtered = df[df["Inoculate"] == 1]

# Remove rows where Severity is NaN
df_filtered = df_filtered.dropna(subset=["Severity"])

# Define DAIs and check valid values
dais = [3, 16, 31, 39]
valid_dais = df_filtered[df_filtered["DAI"].isin(dais)].groupby("DAI")["Severity"].count()
dais = valid_dais[valid_dais > 0].index.tolist()  # Keep only DAIs with valid data

# Compute means and standard errors
severity_stats = df_filtered.groupby(["DAI", "variety"])["Severity"].agg(["mean", "sem"]).reset_index()

# Ensure there are valid cultivars
cultivars = severity_stats["variety"].unique()
if len(cultivars) == 0:
    raise ValueError("No valid data available for inoculated plants with severity measurements.")

# Perform ANOVA for each valid DAI
anova_results = {}
tukey_results = {}
significant_dais = []

for dai in dais:
    subset = df_filtered[df_filtered["DAI"] == dai]
    
    if len(subset["variety"].unique()) > 1:  # Ensure multiple groups for ANOVA
        groups = [subset[subset["variety"] == cultivar]["Severity"].dropna() for cultivar in subset["variety"].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        anova_results[dai] = p_value
        
        if p_value < 0.05:
            significant_dais.append(dai)
            tukey = pairwise_tukeyhsd(subset["Severity"], subset["variety"], alpha=0.05)
            tukey_results[dai] = tukey

# Create figure
plt.figure(figsize=(8, 6))
ax = plt.gca()
bar_width = 0.15
x = np.arange(len(dais))

# Plot bars
for i, cultivar in enumerate(cultivars):
    subset = severity_stats[severity_stats["variety"] == cultivar]
    means = [subset[subset["DAI"] == d]["mean"].values[0] if d in subset["DAI"].values else 0 for d in dais]
    errors = [subset[subset["DAI"] == d]["sem"].values[0] if d in subset["DAI"].values else 0 for d in dais]
    ax.bar(x + i * bar_width, means, bar_width, yerr=np.array(errors), capsize=5, edgecolor="black", alpha=0.8,
           label=cultivar, color=cultivar_colors.get(cultivar, "gray"))
    
    for j in range(len(dais)):
        if dais[j] in significant_dais and means[j] > 0:
            plt.text(x[j] + i * bar_width, means[j] + 2, '*', ha='center', fontsize=14, fontweight='bold')

# Formatting
ax.set_xlabel("Days After Inoculation (DAI)")
ax.set_ylabel("Severity (%)")
ax.set_ylim(0, 100)
ax.set_xticks(x + bar_width * (len(cultivars) / 2))
ax.set_xticklabels(dais)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(cultivars), frameon=False)
plt.grid(False)
plt.tight_layout()

# Save figure
save_path = "Severity_Bar_Plot_ANOVA.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved figure as {save_path}")

# Show plot
plt.show()

# Print ANOVA and Tukey results
print("\n### ANOVA Results ###")
for dai, p_val in anova_results.items():
    print(f"DAI {dai}: p-value = {p_val:.5f} {'(Significant)' if p_val < 0.05 else ''}")

# Print Tukey results if available
for dai, tukey in tukey_results.items():
    print(f"\n### Tukey's HSD Results for DAI {dai} ###")
    print(tukey)

