import os
import csv
from hybrid_llm_evaluation import evaluate_essay
from hybrid_llm_evaluation import RATER_SPECIALIZATIONS, RUBRIC_MAPPING 

# Configuration
ESSAY_FOLDER = "QCAW_Arabic"
OUTPUT_CSV = "essay_scores_full.csv"
MAX_ESSAYS = 195

def load_essays(folder, limit=40):
    """Load essays from text files"""
    essays = []
    for fname in sorted(os.listdir(folder))[:limit]:
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                essays.append((fname.replace(".txt", ""), f.read()))
    return essays

def save_to_csv(results, filename):
    """Save results to CSV with proper encoding"""
    if not results:
        return
    
    # Generate comprehensive fieldnames
    fieldnames = ["essay_id"]
    fieldnames.extend([f"rater_{r}" for r in RATER_SPECIALIZATIONS.keys()])
    fieldnames.extend(RUBRIC_MAPPING.keys())
    fieldnames.append("total")
    
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    print("🔍 Loading essays...")
    essays = load_essays(ESSAY_FOLDER, MAX_ESSAYS)
    
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