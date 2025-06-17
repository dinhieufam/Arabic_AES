import json
import re

def extract_scores(output):
    traits_order = [
        "organization", "vocabulary", "style", "development",
        "mechanics", "structure", "relevance", "final_score"
    ]
    
    try:
        # Try parsing as JSON
        json_start = output.rfind('{')
        json_end = output.find('}', json_start) + 1
        json_str = output[json_start:json_end]
        parsed = json.loads(json_str)

        scores = {
            "organization": int(min(parsed.get("organization", 0), 5)),
            "vocabulary": int(min(parsed.get("vocabulary", 0), 5)),
            "style": int(min(parsed.get("style", 0), 5)),
            "development": int(min(parsed.get("development", 0), 5)),
            "mechanics": int(min(parsed.get("mechanics", 0), 5)),
            "structure": int(min(parsed.get("structure", 0), 5)),
            "relevance": int(min(parsed.get("relevance", 0), 2)),
            "final_score": int(min(parsed.get("final_score", 0), 32))
        }

    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        print("🔎 Falling back to text-based parsing.")
        
        scores = {}
        lines = output.splitlines()
        trait_idx = 0

        for line in lines:
            if trait_idx >= len(traits_order):
                break

            # Match lines like 'trait: number' or 'trait: number/max'
            match = re.search(r':\s*([0-9]+)(?:\s*/\s*[0-9]+)?', line)
            if match:
                score = int(match.group(1))
                trait = traits_order[trait_idx]
                max_score = 5 if trait != "relevance" else 2
                max_score = 32 if trait == "final_score" else max_score
                scores[trait] = min(score, max_score)
                trait_idx += 1

        # Fill in missing scores
        for trait in traits_order:
            if trait not in scores:
                scores[trait] = 0

    # Calculate total score (excluding final_score)
    scores["total_score"] = (
        scores["organization"] +
        scores["vocabulary"] +
        scores["style"] +
        scores["development"] +
        scores["mechanics"] +
        scores["structure"] +
        scores["relevance"]
    )

    return scores
