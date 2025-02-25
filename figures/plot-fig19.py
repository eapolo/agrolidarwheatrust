import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
file_path = "data-rust-net.csv"  # Update path if needed
df = pd.read_csv(file_path, delimiter=";", decimal=",")

# 2. Convert Intensities and Inoculada to Numeric & Drop Missing
int_cols = ["Int90", "Int80", "Int70", "Int60", "Int50"]
for col in int_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Inoculada"] = pd.to_numeric(df["Inoculada"], errors="coerce")

# Drop rows with missing intensity or inoculation data
df = df.dropna(subset=int_cols + ["Inoculada"])

# 3. Filter for Inoculated == 1
df_inoculated = df[df["Inoculada"] == 1]

# 4. Melt Data (wide → long format)
df_melted = df_inoculated.melt(
    id_vars="Inoculada",       # Keep "Inoculada" column
    value_vars=int_cols,       # Columns to melt
    var_name="IntensityLevel", # New col with the intensity type
    value_name="Intensity"     # Numeric intensity value
)

# Convert "IntensityLevel" into a categorical type with a specified order (Int10 → Int50)
order = ["Int90", "Int80", "Int70", "Int60", "Int50"]
df_melted["IntensityLevel"] = pd.Categorical(
    df_melted["IntensityLevel"], 
    categories=order, 
    ordered=True
)

# 5. Create a Line Plot (Mean ± SD) for Inoculated=1
plt.figure(figsize=(8, 6))

sns.lineplot(
    data=df_melted,
    x="IntensityLevel",
    y="Intensity",
    estimator="mean",  # plot the mean for each intensity level
    ci="sd",           # shaded area = ±1 standard deviation
    sort=False         # use our categorical order instead of sorting alphabetically
)

# Customize labels and layout
plt.xlabel("Height‐percentile intensity classes")
plt.ylabel("Mean intensity (± SD)")
plt.tight_layout()

# Save the figure in high resolution before showing
plt.savefig("my_intensity_plot.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
