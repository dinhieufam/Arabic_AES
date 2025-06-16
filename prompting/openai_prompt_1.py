import openai 
import time
import pandas as pd
import json
import csv

from secret_key import openai_key

openai.api_key = openai_key  # 🔑 Set your API key here

SCORING_PROMPT = """
You are an expert Arabic language evaluator. Your task is to assess the proficiency of an Arabic essay based on seven traits:
1. Organization (0-5): How well-structured and coherent is the essay?
2. Vocabulary (0-5): Does the writer use a rich and appropriate vocabulary?
3. Style (0-5): Is the writing engaging, fluent, and stylistically appropriate?
4. Development (0-5): Are ideas elaborated with sufficient details and examples?
5. Mechanics (0-5): Are grammar, spelling, and punctuation correct?
6. Structure (0-5): Does the essay follow proper syntactic structures?
7. Relevance (0-2): Does the essay address the given topic appropriately?
8. Final Score (0-32): The sum of all the scores.

Each trait should be scored on a scale from 0 (poor) to 5 (excellent), except for relevance which is scored on a scale from 0 (poor) to 2 (excellent).
The final score should be the sum of all the scores on a scale from 0 to 32.

Return ONLY this JSON object with your scores (replace X with actual numbers):
{
    "organization": X,
    "vocabulary": X,
    "style": X,
    "development": X,
    "mechanics": X,
    "structure": X,
    "relevance": X, 
    "final_score": X
}
"""

def get_response_from_gpt4(essay_text):
    while True:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful and strict Arabic language evaluator."},
                {"role": "user", "content": f"{SCORING_PROMPT}\n\nEssay:\n{essay_text.strip()}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
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
        scores = json.loads(json_str)

        scores["essay_id"] = essay_id
        scores["total_score"] = sum(scores[k] for k in [
            "organization", "vocabulary", "style",
            "development", "mechanics", "structure", "relevance"
        ])
        results.append(scores)

    except Exception as e:
        print(f"❌ Failed to parse response for essay {essay_id}: {e}")
        results.append({
            "essay_id": essay_id,
            "organization": 0,
            "vocabulary": 0,
            "style": 0,
            "development": 0,
            "mechanics": 0,
            "structure": 0,
            "relevance": 0,
            "final_score": 0,
            "total_score": 0
        })

    # Wait for 10 seconds to prevent rate limit
    time.sleep(15)

# Save to CSV
output_file = "predictions/gpt4/prompt_level_1.csv"
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development",
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved GPT-4 results to {output_file}")
