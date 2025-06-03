import csv
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from prediction_prompt_1 import run_model_and_parse_response
from utils import load_essays

with open("main_config.json", "r") as f:
    config = json.load(f)

OUTPUT_CSV = "predictions/model_3/prompt_level_1.csv"
MAX_ESSAYS = config["max_essays"]
ESSAY_FOLDER = config["essay_folder"]
MODEL_NAME = config["model_name"]

def save_to_csv(results, filename):
    # Define the fieldnames for the CSV file
    fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance", "final_score", "total_score"]

    # Save the results to the CSV file
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():

    print(f"🔢 Limiting evaluation to {MAX_ESSAYS} essays...")
    print(f"🧠 Using model: {MODEL_NAME}")

    print("🔍 Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=False,
        device_map="auto"
    )

    print("🔍 Loading model...")

    # ✅ Set pad_token to eos_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype="auto", 
        trust_remote_code=False,
        device_map="auto",
    )

    print("🔍 Setting model to evaluation mode...")

    model.eval()

    essays = load_essays(ESSAY_FOLDER, limit=MAX_ESSAYS)
    results = []

    for i, (essay_id, text) in enumerate(essays, start=1):
        print(f"  ⏳ Processing essay {i}/{len(essays)}: {essay_id}")
        scores = run_model_and_parse_response(text, model, tokenizer)

        # Add essay_id to the scores dictionary
        scores["essay_id"] = essay_id

        # Append the scores to the results list
        results.append(scores)

        print(scores)

    save_to_csv(results, OUTPUT_CSV)
    print(f"💾 Saved results to {OUTPUT_CSV}")
    print("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()


