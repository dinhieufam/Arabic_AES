import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCORING_PROMPT = """
You are an expert Arabic language evaluator. Your task is to assess the proficiency of an Arabic essay based on seven traits:
1. Organization (0-5): How well-structured and coherent is the essay?
2. Vocabulary (0-5): Does the writer use a rich and appropriate vocabulary?
3. Style (0-5): Is the writing engaging, fluent, and stylistically appropriate?
4. Development (0-5): Are ideas elaborated with sufficient details and examples?
5. Mechanics (0-5): Are grammar, spelling, and punctuation correct?
6. Structure (0-5): Does the essay follow proper syntactic structures?
7. Relevance (0-2): Does the essay address the given topic appropriately?
8. Final Score (0-32): The sum of all traits.

Each trait should be scored on a scale from 0 (poor) to 5 (excellent).
Finally, calculate the total score by summing all traits, with a maximum possible score of 32.

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

def run_model_and_parse_response(essay_text, model_name):
    prompt = SCORING_PROMPT + "\n\nEssay:\n" + essay_text.strip()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=False,
        device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        trust_remote_code=False,
        device_map="auto"
    )

    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = decoded_output[0]

    # print("📤 Model output:\n", decoded_output[0])

    try:
        json_start = output.rfind('{')
        print("Found JSON at index:", json_start)
        json_end = output.find('}', json_start) + 1
        json_str = output[json_start:json_end]
        print("JSON string:", repr(json_str))  # Using repr() to see hidden characters
        print("JSON string length:", len(json_str))
        print("First few characters:", [ord(c) for c in json_str[:10]])  # Print ASCII values of first few chars
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
        print("🔎 Raw output was:\n", output)
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
