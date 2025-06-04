# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "inceptionai/jais-family-13b-chat"

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

prompt_eng = "### Instruction: You are a helpful assistant. Complete the conversation between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n[|AI|]\n### Response :"
# prompt_ar = "### Instruction:اسمك \"جيس\" وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception في الإمارات. أنت مساعد مفيد ومحترم وصادق. أجب دائمًا بأكبر قدر ممكن من المساعدة، مع الحفاظ على البقاء أمناً. أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {Question}\n[|AI|]\n### Response :"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)


def get_response(text, tokenizer=tokenizer, model=model):
    # Tokenize with padding and truncation, and move to device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Generate model output with attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, outputs)
    ]

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = decoded_output[0]
    response = output.split("### Response :")[-1]
    return response


df = pd.read_excel("../dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print("😍 Output of Essay ID: ", essay_id)
    prompt = prompt_eng.format_map({'Question': SCORING_PROMPT + "\n\nEssay:\n" + essay_text.strip()})
    
    response = get_response(prompt)
    print(response)
    
    try:
        # Find JSON in response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        # Parse scores
        scores = json.loads(json_str)
        
        # Add essay_id and calculate total score
        scores["essay_id"] = essay_id
        scores["total_score"] = sum(scores[k] for k in ["organization", "vocabulary", "style", 
                                                       "development", "mechanics", "structure", "relevance"])
        
        results.append(scores)
        
    except Exception as e:
        print(f"Failed to parse response for essay {essay_id}: {e}")
        # Add default scores on failure
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

# Save results to CSV
output_file = "../predictions/model_4/prompt_1.csv"
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved results to {output_file}")