import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm   

# Config
ground_truth_path = '/Users/salimalmandhari/Documents/Project/ArabicNLP_2025-extended/dataset.csv'
predictions_root = '/Users/salimalmandhari/Downloads/Arabic_AES-master/predictions'  # this directory contains 8 model subfolders
output_path = 'qwk_ci_summary.csv'
traits = ['organization', 'vocabulary', 'style', 'development',
          'mechanics', 'structure', 'relevance', 'final_score']

# Load ground truth
df_true = pd.read_csv(ground_truth_path)

# Function to compute QWK with bootstrapped CI
def bootstrap_qwk_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(cohen_kappa_score(np.array(y_true)[idx], np.array(y_pred)[idx], weights='quadratic'))
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lower, upper

# Collect all results
summary = []

# Loop over each model directory
for model_dir in os.listdir(predictions_root):
    model_path = os.path.join(predictions_root, model_dir)
    if os.path.isdir(model_path):
        for csv_file in glob.glob(os.path.join(model_path, '*.csv')):
            prompt_level = os.path.splitext(os.path.basename(csv_file))[0]  # e.g., "prompt_level_1"
            df_pred = pd.read_csv(csv_file)

            for trait in traits:
                if trait in df_pred.columns and trait in df_true.columns:
                    y_true = df_true[trait].dropna().astype(int)
                    y_pred = df_pred[trait].dropna().astype(int)

                    if len(y_true) == len(y_pred):
                        qwk, lower, upper = bootstrap_qwk_ci(y_true, y_pred)
                        summary.append({
                            'Model': model_dir,
                            'Prompt Level': prompt_level,
                            'Trait': trait.replace('_fn', ''),
                            'QWK': round(qwk, 3),
                            '95% CI Lower': round(lower, 3),
                            '95% CI Upper': round(upper, 3)
                        })

# Save results
pd.DataFrame(summary).to_csv(output_path, index=False)
print(f"\n Results saved to: {output_path}")
