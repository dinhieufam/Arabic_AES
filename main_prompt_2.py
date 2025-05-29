import os
import csv
import json

from prediction_prompt_2 import evaluate_essay, RATER_SPECIALIZATIONS, RUBRIC_MAPPING
from utils import load_essays

with open("main_config.json", "r") as f:
    config = json.load(f)

# Configuration 
ESSAY_FOLDER = config["essay_folder"]
OUTPUT_CSV = "predictions/model_3/prompt_level_2.csv"
MAX_ESSAYS = config["max_essays"]

def save_to_csv(results, filename):
    # Define the fieldnames for the CSV file
    fieldnames = ["essay_id"]
    fieldnames.extend([f"rater_{r}" for r in RATER_SPECIALIZATIONS.keys()])
    fieldnames.extend(RUBRIC_MAPPING.keys())
    fieldnames.append("final_score")
    
    # Save the results to the CSV file
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    print("🔍 Loading essays...")
    essays = load_essays(ESSAY_FOLDER, limit=MAX_ESSAYS)
    
    print(f"✅ Starting evaluation of {len(essays)} essays...")
    results = []                                                                          
    for idx, (eid, text) in enumerate(essays, 1):
        print(f"  ⏳ Processing essay {idx}/{len(essays)}: {eid}")
        results.append(evaluate_essay(eid, text))
    
    print(f"💾 Saving results to {OUTPUT_CSV}...")
    save_to_csv(results, OUTPUT_CSV)
    print("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()