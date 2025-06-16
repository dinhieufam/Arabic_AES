# -*- coding: utf-8 -*-
import pandas as pd
import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Output file
output_file = "predictions/model_6/prompt_1.csv"

# Model path for Aya 101
checkpoint = "CohereLabs/aya-101"

print(f"🧠 Using model: {checkpoint}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: " + device)

print("🔍 Loading tokenizer...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    device_map="auto"
)

print("🔍 Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto"
).to(device)

print("Start evaluating")

# Arabic Scoring Prompt 
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

def generate_with_aya(text, model=model, tokenizer=tokenizer):
    input_text = SCORING_PROMPT + "\n\nEssay:\n" + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=512)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

# Load essay dataset
df = pd.read_excel("dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print(f"📝 Evaluating Essay ID: {essay_id}")
    
    try:
        response = generate_with_aya(essay_text)
        print("📥 Response:", response)
        
        # Extract JSON from model response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        scores = json.loads(json_str)

        scores["essay_id"] = essay_id
        scores["total_score"] = sum(scores[k] for k in ["organization", "vocabulary", "style", 
                                                        "development", "mechanics", "structure", "relevance"])

        results.append(scores)

    except Exception as e:
        print(f"❌ Failed to parse Essay ID {essay_id}: {e}")
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

# Save results
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Results saved to {output_file}")
