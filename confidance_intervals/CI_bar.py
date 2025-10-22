import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("/Users/salimalmandhari/Documents/Project/ArabicNLP_2025-extended/qwk_ci_summary.csv")

# Clean and standardize column names if needed
df.columns = [col.strip() for col in df.columns]
df = df.rename(columns={"Trait": "Trait", "QWK": "QWK", "95% CI Lower": "CI_Lower", "95% CI Upper": "CI_Upper"})

# Create combined column for plotting
df["Model_Level"] = df["Model"] + " - " + df["Prompt Level"]

# List of traits to plot
traits = df["Trait"].unique()

for trait in traits:
    trait_df = df[df["Trait"] == trait].copy()

    # Sort values for better visualization
    trait_df = trait_df.sort_values(by="QWK", ascending=True)

    # Calculate asymmetric error bars
    yerr = [
        trait_df["QWK"] - trait_df["CI_Lower"],
        trait_df["CI_Upper"] - trait_df["QWK"]
    ]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(
        y=trait_df["Model_Level"],
        width=trait_df["QWK"],
        xerr=yerr,
        align="center",
        color="skyblue",
        edgecolor="black",
        capsize=5
    )
    plt.xlabel("QWK Score")
    plt.title(f"QWK with 95% CI for {trait} Trait")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Save figure
    filename = f"{trait}_qwk_ci_plot.png".replace(" ", "_")
    plt.savefig(filename)
    plt.close()
