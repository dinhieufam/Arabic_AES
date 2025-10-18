# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"

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
output_file = "../predictions/qwen3vl/prompt_1.csv"
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved results to {output_file}")