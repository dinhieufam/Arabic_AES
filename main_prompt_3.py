import csv
import json

from prediction_prompt_3 import run_model_and_parse_response
from utils import load_essays

with open("main_config.json", "r") as f:
    config = json.load(f)

OUTPUT_CSV = "predictions/prompt_level_3.csv"
MAX_ESSAYS = config["max_essays"]
MODEL_NAME = config["model_name"]

RUBRICS = ["organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance"]

def save_to_csv(results, filename):
    # Define the fieldnames for the CSV file
    fieldnames = ["essay_id"] + RUBRICS + ["final_score"]

    # Write the fieldnames to the CSV file
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    print(f"🔢 Limiting evaluation to {MAX_ESSAYS} essays...")

    essays = load_essays("essays", limit=MAX_ESSAYS)
    results = []

    for i, (essay_id, text) in enumerate(essays):
        print(f"✍️ Evaluating essay {i+1}/{len(essays)} - ID: {essay_id}")
        row = {"essay_id": essay_id}
        total = 0
        for rubric in RUBRICS:
            result = run_model_and_parse_response(MODEL_NAME, rubric, text)
            score = float(result.get("score", 0))
            row[rubric] = score
            total += score
            print(f"  📌 {rubric}: {score}")
        row["final_score"] = round(total, 2)
        results.append(row)

    save_to_csv(results, OUTPUT_CSV)
    print(f"✅ Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
