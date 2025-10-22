import pandas as pd
import matplotlib.pyplot as plt

# Load the summary CSV
df = pd.read_csv("/Users/salimalmandhari/Documents/Project/ArabicNLP_2025-extended/qwk_ci_summary.csv")

# Clean column names
df.columns = [col.strip() for col in df.columns]
df = df.rename(columns={
    "95% CI Lower": "CI_Lower",
    "95% CI Upper": "CI_Upper"
})

# Create a label that combines Trait and Model for plotting
df["Trait_Model"] = df["Trait"] + " - " + df["Model"]

# Get unique prompt levels
prompt_levels = sorted(df["Prompt Level"].unique())

for prompt in prompt_levels:
    prompt_df = df[df["Prompt Level"] == prompt].copy()

    # Sort for consistent visual order
    prompt_df = prompt_df.sort_values(by="QWK", ascending=True)

    # Calculate error bars
    yerr = [
        prompt_df["QWK"] - prompt_df["CI_Lower"],
        prompt_df["CI_Upper"] - prompt_df["QWK"]
    ]

    # Plot
    plt.figure(figsize=(12, 10))
    plt.barh(
        y=prompt_df["Trait_Model"],
        width=prompt_df["QWK"],
        xerr=yerr,
        align="center",
        color="lightgreen",
        edgecolor="black",
        capsize=5
    )
    plt.xlabel("QWK Score")
    plt.title(f"QWK with 95% CI — Prompt Level {prompt}")
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Save plot
    filename = f"{prompt}_qwk_ci_plot.png".replace(" ", "_")
    plt.savefig(filename)
    plt.close()
