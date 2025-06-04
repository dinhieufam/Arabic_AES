import pandas as pd
import matplotlib.pyplot as plt

# Sample data loading (replace with your actual file reading logic)
evaluation_path = [
    ["evaluation_results/model_1/prompt_level_1.csv", "evaluation_results/model_1/prompt_level_2.csv", "evaluation_results/model_1/prompt_level_3.csv"],
    ["evaluation_results/model_2/prompt_level_1.csv", "evaluation_results/model_2/prompt_level_2.csv", "evaluation_results/model_2/prompt_level_3.csv"],
    ["evaluation_results/model_3/prompt_level_1.csv", "evaluation_results/model_3/prompt_level_2.csv", "evaluation_results/model_3/prompt_level_3.csv"],
    ["evaluation_results/model_4/prompt_level_1.csv", "evaluation_results/model_4/prompt_level_2.csv", "evaluation_results/model_4/prompt_level_3.csv"]
]

MODELS = ["Qwen1.5-1.8B-Chat", "Qwen2.5-7B-Instruct", "ALLaM-7B-Instruct-preview", "jais-family-13b-chat"]

# Load data into a dictionary
data = {}
for i, model_paths in enumerate(evaluation_path, 1):
    model_name = MODELS[i-1]
    data[model_name] = []
    for path in model_paths:
        df = pd.read_csv(path)  # Read each CSV
        data[model_name].append(df.iloc[0])  # Assume one row per file

# Aspects to plot
aspects = ['organization', 'vocabulary', 'style', 'development', 'mechanics', 'structure', 'relevance', 'final_score', 'total_score']

# Create a 4x2 subplot figure
fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharex=True)
axes = axes.flatten()  # Flatten for easy iteration

# Plot each aspect
for idx, aspect in enumerate(aspects):
    ax = axes[idx]
    for model_name in data.keys():
        scores = [data[model_name][p][aspect] for p in range(3)]  # Scores for prompt levels 1, 2, 3
        ax.plot([1, 2, 3], scores, label=model_name, marker='o')
    ax.set_title(aspect.capitalize())
    ax.set_xlabel('Prompt Level')
    ax.set_ylabel('Score')
    ax.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
plt.show()