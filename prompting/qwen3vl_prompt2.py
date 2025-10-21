# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"

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
    "relevance": 5
}

prompt = f"""
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

prompt_eng = "### Instruction: You are a helpful assistant. Complete the conversation between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n[|AI|]\n### Response :"

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

def get_response(text, processor=processor, model=model):
    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # Apply chat template and prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
    # Process generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

df = pd.read_excel("../dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print("😍 Output of Essay ID: ", essay_id)
    prompt = prompt_eng.format_map({'Question': prompt + "\n\nEssay:\n" + essay_text.strip()})
    
    response = get_response(prompt)
    print(response)
    
    try:
        # Find JSON in response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        # Parse scores
        scores = json.loads(json_str)
        
        parsed = json.loads(json_str)
        scores = {
            "A": int(min(parsed.get("A", 0), 5)),
            "B": int(min(parsed.get("B", 0), 5)), 
            "C": int(min(parsed.get("C", 0), 5)),
            "D": int(min(parsed.get("D", 0), 5)),
            "E": int(min(parsed.get("E", 0), 5))
        }

        print("🔍 Parsed scores:", scores)
        
    except Exception as e:
        print(f"Failed to parse response for essay {essay_id}: {e}")

        scores = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
            "E": 0,
        }
    
    # Map back to full rubric categories
    rubric_scores = {"essay_id": essay_id}
    for rubric, keys in RUBRIC_MAPPING.items():
        rubric_scores[rubric] = min(sum(scores.get(k, 0) for k in keys) // len(keys), RUBRIC_MAX_SCORES[rubric])

    # Normalize relevance score to 0-2
    rubric_scores["relevance"] = round(rubric_scores["relevance"] / 5 * 2)

    # Calculate final score
    rubric_scores["final_score"] = rubric_scores["organization"] + rubric_scores["vocabulary"] + rubric_scores["style"] + rubric_scores["development"] + rubric_scores["mechanics"] + rubric_scores["structure"] + rubric_scores["relevance"]
    rubric_scores["total_score"] = rubric_scores["final_score"]

    rubric_scores.update({f"rater_{k}": scores.get(k, 0) for k in RATER_SPECIALIZATIONS})
    
    print("🔍 Rubric scores:", rubric_scores)

    results.append(rubric_scores)

# Save results to CSV
output_file = "../predictions/qwen3vl/prompt_level_2.csv"
fieldnames = ["essay_id", "rater_A", "rater_B", "rater_C", "rater_D", "rater_E", 
              "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved results to {output_file}")