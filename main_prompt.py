import argparse
import csv
from prompt_evaluator import run_model_and_parse_response
from utils import load_essays

OUTPUT_CSV = "unified_prompt_scores.csv"

def save_to_csv(results, filename):
    fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance", "final_score"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="Number of essays to process")
    parser.add_argument("--model", type=str, required=True, help="Model name on Hugging Face")
    args = parser.parse_args()

    print(f"🔢 Limiting evaluation to {args.limit} essays...")
    print(f"🧠 Using model: {args.model}")

    essays = load_essays("essays", limit=args.limit)
    results = []

    for i, (essay_id, text) in enumerate(essays, start=1):
        print(f"  ⏳ Processing essay {i}/{len(essays)}: {essay_id}")
        scores = run_model_and_parse_response(text, args.model)
        scores["essay_id"] = essay_id
        results.append(scores)

    save_to_csv(results, OUTPUT_CSV)
    print(f"💾 Saved results to {OUTPUT_CSV}")
    print("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()


# or Qwen/Qwen2.5-7B)"/"ALLaM-AI/ALLaM-7B-Instruct-preview"/"Qwen/Qwen1.5-1.8B" / "mistralai/Mistral-7B-Instruct-v0.2" /"NousResearch/Hermes-2-Pro-Llama-3-8B"