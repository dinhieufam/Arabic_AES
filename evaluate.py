import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import statsmodels.stats.inter_rater
import json
import os

# Load configuration into global constants
with open("evaluate.json", 'r') as f:
    config = json.load(f)

# Paths
GROUND_TRUTH_PATH = config['paths']['ground_truth']
MODEL_PREDICTIONS_PATH = config['paths']['model_predictions']
OUTPUT_PATH = config['paths']['output']

def calculate_qwk(ground_truth, predictions, min_rating=0, max_rating=5):
    """
    Calculate Quadratic Weighted Kappa between two raters
    
    Args:
        ground_truth: Ground truth ratings
        predictions: Predicted ratings
        min_rating: Minimum possible rating
        max_rating: Maximum possible rating
        
    Returns:
        float: Quadratic Weighted Kappa score
    """
    ground_truth = np.array(ground_truth, dtype=int)
    predictions = np.array(predictions, dtype=int)
    
    return cohen_kappa_score(
        ground_truth,
        predictions,
        weights='quadratic'
    )

def evaluate_model_predictions():
    # Read the model predictions
    model_preds = pd.read_csv(MODEL_PREDICTIONS_PATH, index_col="essay_id")
    
    # Read the QAES ground truth data
    qaes_data = pd.read_excel(GROUND_TRUTH_PATH, index_col="essay_id")

    # Remove 'A' from model predictions index
    model_preds.index = model_preds.index.str.replace('A', '').astype(int)
    
    # Initialize dictionary to store QWK scores for each trait
    qwk_scores = {}
    
    # Calculate QWK for each scoring trait
    traits = ['organization', 'vocabulary', 'style', 'development', 
              'mechanics', 'structure', 'relevance']
    
    # Process each essay index
    for index in model_preds.index:
        print(f"\nProcessing essay {index}:")
        
        # Skip if essay not in model predictions
        if index not in model_preds.index:
            print(f"Essay {index} not found in model predictions, skipping...")
            continue
        
        print(qaes_data.index)
        print(model_preds.index)

        # Get scores for this essay
        ground_truth_scores = qaes_data.loc[index]
        model_scores = model_preds.loc[index]
        
        # Compare scores for each trait
        for trait in traits:
            gt_score = ground_truth_scores[f"{trait}_fn"]
            pred_score = model_scores[trait]
            print(f"{trait}: Ground Truth = {gt_score}, Predicted = {pred_score}")
            
            # Initialize trait in qwk_scores if not present
            if trait not in qwk_scores:
                qwk_scores[trait] = {'ground_truth': [], 'predictions': []}
                
            # Append scores for QWK calculation
            qwk_scores[trait]['ground_truth'].append(gt_score)
            qwk_scores[trait]['predictions'].append(pred_score)
    
    # Calculate final QWK scores for each trait
    results = {}
    print("\nFinal QWK Scores:")
    for trait in traits:
        if trait in qwk_scores:
            qwk = calculate_qwk(
                qwk_scores[trait]['ground_truth'],
                qwk_scores[trait]['predictions']
            )
            print(f"{trait}: {qwk:.3f}")
            results[trait] = qwk
            
    # Calculate and print average QWK
    trait_qwks = [calculate_qwk(qwk_scores[t]['ground_truth'], 
                               qwk_scores[t]['predictions']) 
                  for t in traits if t in qwk_scores]
    avg_qwk = np.mean(trait_qwks)
    print(f"\nAverage QWK across all traits: {avg_qwk:.3f}")
    results['average'] = avg_qwk
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    evaluate_model_predictions()
