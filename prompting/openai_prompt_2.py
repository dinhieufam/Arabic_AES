import openai
import time
import pandas as pd
import json
import csv
from secret_key import openai_key

openai.api_key = openai_key

output_file = "predictions/gpt4/prompt_level_2.csv"

RATER_SPECIALIZATIONS = {
    "A": "Evaluate the essay's organization and how well ideas are connected. "
            "Consider logical flow, paragraph structure, and transitions. "
            "Provide a score between 0-5",
    
    "B": "Assess the vocabulary quality and lexical variety in the essay. "
            "Consider word choice, sophistication, and avoidance of repetition. "
            "Provide a score between 0-5",
    
    "C": "Evaluate grammar, spelling, punctuation, and mechanical accuracy. "
            "Identify errors and assess their impact on readability. "
            "Provide a score between 0-5",
    
    "D": "Analyze content development and reasoning quality. "
            "Consider depth of analysis, argument strength, and evidence use. "
            "Provide a score between 0-5",
    
    "E": "Assess style, tone, and contextual appropriateness. "
            "Consider voice, audience awareness, and stylistic effectiveness. "
            "Provide a score between 0-5"
}

RUBRIC_MAPPING = {
    "organization": ["A", "D", "C"],
    "vocabulary": ["B", "E", "C"],
    "style": ["B", "E", "C"],
    "development": ["D", "A", "B"],
    "mechanics": ["C"],
    "structure": ["A", "B", "C"],
    "relevance": ["D", "B", "E"]
}

RUBRIC_MAX_SCORES = {
    "organization": 5,
    "vocabulary": 5,
    "style": 5,
    "development": 5,
    "mechanics": 5,
    "structure": 5,
    "relevance": 5  # Will be normalized to 0-2 later
}

INSTRUCTION_PROMPT = f"""
You are an Arabic essay scoring assistant. You will read a student's Arabic essay and assign scores from 0 to 5 for the following five linguistic dimensions:

A: {RATER_SPECIALIZATIONS['A']}
B: {RATER_SPECIALIZATIONS['B']}
C: {RATER_SPECIALIZATIONS['C']}
D: {RATER_SPECIALIZATIONS['D']}
E: {RATER_SPECIALIZATIONS['E']}

Return ONLY this JSON object with your scores (replace X with actual numbers):
{{
    "A": X,
    "B": X,
    "C": X,
    "D": X,
    "E": X
}}
"""

def get_response_from_gpt4(essay_text):
    while True:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful and precise Arabic essay scoring assistant."},
                {"role": "user", "content": f"{INSTRUCTION_PROMPT}\n\nEssay:\n{essay_text.strip()}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error: {e}")
            return ""

df = pd.read_excel("dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print("📝 Scoring Essay ID:", essay_id)
    raw_response = get_response_from_gpt4(essay_text)
    print(raw_response)

    try:
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        json_str = raw_response[json_start:json_end]
        parsed = json.loads(json_str)

        # Clamp all values to [0,5]
        scores = {
            "A": int(min(parsed.get("A", 0), 5)),
            "B": int(min(parsed.get("B", 0), 5)),
            "C": int(min(parsed.get("C", 0), 5)),
            "D": int(min(parsed.get("D", 0), 5)),
            "E": int(min(parsed.get("E", 0), 5)),
        }

        rubric_scores = {"essay_id": essay_id}
        for rubric, keys in RUBRIC_MAPPING.items():
            avg_score = sum(scores.get(k, 0) for k in keys) // len(keys)
            rubric_scores[rubric] = min(avg_score, RUBRIC_MAX_SCORES[rubric])

        # Normalize relevance (0-5) → (0-2)
        rubric_scores["relevance"] = round(rubric_scores["relevance"] / 5 * 2)

        rubric_scores["final_score"] = sum(
            rubric_scores[r] for r in [
                "organization", "vocabulary", "style",
                "development", "mechanics", "structure", "relevance"
            ]
        )
        rubric_scores["total_score"] = rubric_scores["final_score"]

        rubric_scores.update({f"rater_{k}": scores[k] for k in scores})
        results.append(rubric_scores)

    except Exception as e:
        print(f"❌ Failed to parse response for essay {essay_id}: {e}")
        fallback = {
            "essay_id": essay_id,
            "organization": 0, "vocabulary": 0, "style": 0, "development": 0,
            "mechanics": 0, "structure": 0, "relevance": 0,
            "final_score": 0, "total_score": 0,
            "rater_A": 0, "rater_B": 0, "rater_C": 0, "rater_D": 0, "rater_E": 0
        }
        results.append(fallback)

    time.sleep(15)

fieldnames = [
    "essay_id", "rater_A", "rater_B", "rater_C", "rater_D", "rater_E",
    "organization", "vocabulary", "style", "development",
    "mechanics", "structure", "relevance", "final_score", "total_score"
]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved GPT-4 results to {output_file}")
