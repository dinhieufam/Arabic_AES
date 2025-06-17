import csv
import json
import torch
import pandas as pd

from prediction_prompt_3 import run_model_and_parse_response
from transformers import AutoTokenizer, AutoModelForCausalLM

with open("main_config.json", "r") as f:
    config = json.load(f)

OUTPUT_CSV = "predictions/model_6/prompt_level_3.csv"
MAX_ESSAYS = config["max_essays"]
DATASET_NAME = config["dataset_name"]
MODEL_NAME = config["model_name"]

RUBRICS = ["organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance"]

def load_essays(limit=None):
    # Read the Excel file
    df = pd.read_excel(DATASET_NAME)
    
    # Get essay_id and text columns
    essays = list(zip(df['essay_id'], df['text']))
    
    # Apply limit if specified
    if limit:
        essays = essays[:limit]
        
    return essays

def save_to_csv(results, filename):
    # Define the fieldnames for the CSV file
    fieldnames = ["essay_id"] + RUBRICS + ["final_score", "total_score"]

    # Write the fieldnames to the CSV file
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    print(f"🧠 Using model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=False,
        device_map="auto"
    )

    # Set pad_token to eos_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        trust_remote_code=False,
        device_map="auto"
    )

    print(f"🔢 Limiting evaluation to {MAX_ESSAYS} essays...")

    essays = load_essays(limit=MAX_ESSAYS)
    results = []

    for i, (essay_id, text) in enumerate(essays):
        print(f"✍️ Evaluating essay {i+1}/{len(essays)} - ID: {essay_id}")
        row = {"essay_id": essay_id}
        total = 0
        for rubric in RUBRICS:
            # Run the model and parse the response
            result = run_model_and_parse_response(rubric, text, model, tokenizer)
            score = round(result.get("score", 0))

            # Add the score to the row
            row[rubric] = score

            # Add the score to the total
            total += score

            # print(f"  📌 {rubric}: {score}")

        row["final_score"] = round(total)
        row["total_score"] = round(total)

        print(row)

        results.append(row)

    save_to_csv(results, OUTPUT_CSV)
    print(f"✅ Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
