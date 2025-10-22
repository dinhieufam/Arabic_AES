import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Load predictions and references
preds = pd.read_csv("confidance_intervals/prompt_level_3.csv")
refs = pd.read_csv("dataset.csv")

# List of traits to evaluate
traits = ['organization', 'vocabulary', 'style', 'development', 'mechanics', 'structure', 'relevance', 'final_score']

# Function to compute bootstrapped confidence intervals
def compute_qwk_ci(y_true, y_pred, n_iterations=1000, ci=95):
    scores = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        score = cohen_kappa_score(y_true[indices], y_pred[indices], weights='quadratic')
        scores.append(score)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), (lower, upper)

# Calculate QWK + confidence intervals per trait
results = []
for trait in traits:
    y_true = refs[trait].values
    y_pred = preds[trait].values
    mean_qwk, (ci_low, ci_high) = compute_qwk_ci(y_true, y_pred)
    results.append({
        "Trait": trait,
        "QWK": round(mean_qwk, 3),
        "95% CI Lower": round(ci_low, 3),
        "95% CI Upper": round(ci_high, 3)
    })

# Save results to CSV
qwk_df = pd.DataFrame(results)
qwk_df.to_csv("qwk_confidence_intervals.csv", index=False)
print(qwk_df)
