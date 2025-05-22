import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

SCORING_PROMPT = """
You are an expert Arabic language evaluator. Your task is to assess the proficiency of an Arabic essay based on seven traits:
1. Organization (0-5): How well-structured and coherent is the essay?
2. Vocabulary (0-5): Does the writer use a rich and appropriate vocabulary?
3. Style (0-5): Is the writing engaging, fluent, and stylistically appropriate?
4. Development (0-5): Are ideas elaborated with sufficient details and examples?
5. Mechanics (0-5): Are grammar, spelling, and punctuation correct?
6. Structure (0-5): Does the essay follow proper syntactic structures?
7. Relevance (0-2): Does the essay address the given topic appropriately?

Each trait should be scored on a scale from 0 (poor) to 5 (excellent).
Finally, calculate the total score by summing all traits, with a maximum possible score of 32.

Respond in JSON format:
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

def run_model_and_parse_response(essay_text, model_name):
    prompt = SCORING_PROMPT + "\n\nEssay:\n" + essay_text.strip()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("📤 Model output:\n", decoded_output)

    try:
        json_start = decoded_output.find('{')
        json_str = decoded_output[json_start:]
        parsed = json.loads(json_str)
        return {
            "organization": parsed.get("organization", 0),
            "vocabulary": parsed.get("vocabulary", 0),
            "style": parsed.get("style", 0),
            "development": parsed.get("development", 0),
            "mechanics": parsed.get("mechanics", 0),
            "structure": parsed.get("structure", 0),
            "relevance": parsed.get("relevance", 0),
            "final_score": parsed.get("final_score", 0),
        }
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        print("🔎 Raw output was:\n", decoded_output)
        return {
            "organization": 0,
            "vocabulary": 0,
            "style": 0,
            "development": 0,
            "mechanics": 0,
            "structure": 0,
            "relevance": 0,
            "final_score": 0,
        }
