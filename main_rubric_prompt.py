
import argparse
import csv
from prompt_rubric_evaluator import run_model_and_parse_response
from utils import load_essays

OUTPUT_CSV = "rubric_scores_qwen.csv"
RUBRICS = ["organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance"]

def save_to_csv(results, filename):
    fieldnames = ["essay_id"] + RUBRICS + ["final_score"]
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

    for i, (essay_id, text) in enumerate(essays):
        print(f"✍️ Evaluating essay {i+1}/{len(essays)} - ID: {essay_id}")
        row = {"essay_id": essay_id}
        total = 0
        for rubric in RUBRICS:
            result = run_model_and_parse_response(args.model, rubric, text)
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
