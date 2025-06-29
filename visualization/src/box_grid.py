import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define input paths and model names
evaluation_path = [
    ["evaluation_results/model_1/prompt_level_1.csv", "evaluation_results/model_1/prompt_level_2.csv", "evaluation_results/model_1/prompt_level_3.csv"],
    ["evaluation_results/model_2/prompt_level_1.csv", "evaluation_results/model_2/prompt_level_2.csv", "evaluation_results/model_2/prompt_level_3.csv"],
    ["evaluation_results/model_3/prompt_level_1.csv", "evaluation_results/model_3/prompt_level_2.csv", "evaluation_results/model_3/prompt_level_3.csv"],
    ["evaluation_results/model_4/prompt_level_1.csv", "evaluation_results/model_4/prompt_level_2.csv", "evaluation_results/model_4/prompt_level_3.csv"],
    ["evaluation_results/model_5/prompt_level_1.csv", "evaluation_results/model_5/prompt_level_2.csv", "evaluation_results/model_5/prompt_level_3.csv"],
    ["evaluation_results/model_6/prompt_level_1.csv", "evaluation_results/model_6/prompt_level_2.csv", "evaluation_results/model_6/prompt_level_3.csv"],
    ["evaluation_results/gpt4/prompt_level_1.csv", "evaluation_results/gpt4/prompt_level_2.csv"]
]

MODELS = ["Qwen1.5-1.8B-Chat", "Qwen2.5-7B-Instruct", "ALLaM-7B-Instruct-preview",
          "jais-family-13b-chat", "Fanar-1-9B-Instruct", "Llama-2-7b-chat-hf", "ChatGPT-4"]

# Define aspects
aspects = ['organization', 'vocabulary', 'style', 'development',
           'mechanics', 'structure', 'relevance', 'total_score']

# Prepare data with manual positions by prompt level
rows = []
for model_idx, model_paths in enumerate(evaluation_path):
    model_name = MODELS[model_idx]
    for level in range(1, 4):
        if model_idx == 6 and level == 3:
            continue
        df = pd.read_csv(model_paths[level - 1])
        values = df.iloc[0][aspects].astype(float)
        for aspect_score in values:
            # Create x positions with staggered layout per level
            base = (level - 1) * 30  # base: 0 for L1, 30 for L2, 60 for L3
            pos = base + model_idx * 2  # spread models within level
            rows.append({
                'Model': model_name,
                'Prompt Level': f'Level {level}',
                'Score (%)': aspect_score,
                'x': pos
            })

# Create DataFrame
plot_df = pd.DataFrame(rows)

# Plot
plt.figure(figsize=(20, 15))
sns.set(style="whitegrid")
ax = sns.boxplot(
    data=plot_df,
    x="x",
    y="Score (%)",
    hue="Model",
    palette="Set2",
    linewidth=1.5
)

# Custom x-ticks: center of each prompt level group
xtick_pos = [3.25, 10, 16.5]  # approximate center of each cluster
xtick_label = ['Prompt Level 1', 'Prompt Level 2', 'Prompt Level 3']
plt.xticks(xtick_pos, xtick_label, fontsize=15)
plt.xlabel("", fontsize=12)

# Add vertical lines to separate clusters
plt.axvline(x=6.5, color='gray', linestyle='--')
plt.axvline(x=13.5, color='gray', linestyle='--')

# Final formatting
plt.title('Model Comparison Across Prompt Levels', fontsize=20)
plt.ylabel('QWK Score (%)', fontsize=20)
plt.legend(title="Model", loc='upper right', bbox_to_anchor=(0.21, 0.98), frameon=True, fontsize=15)
plt.tight_layout()
plt.savefig('visualization/fig/box_grid.png', dpi=300)
plt.show()
