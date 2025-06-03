import json
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("main_config.json", "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]

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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    use_fast=False,
    trust_remote_code=False
)

# Set pad_token to eos_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    trust_remote_code=False, 
    device_map="auto"
)

model.eval()

def evaluate_essay(essay_id: str, text: str):
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

    Essay:
    {text}
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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

    # print(f"Raw Output:\n {output}")

    try:
        json_start = output.find('{')
        # print("Found JSON at index:", json_start)
        # Find the first closing brace after the opening brace
        json_end = json_start
        for i in range(json_start, len(output)):
            if output[i] == '}':
                json_end = i + 1
                break
        json_str = output[json_start:json_end]
        # print("JSON string:", repr(json_str))  # Using repr() to see hidden characters
        # print("JSON string length:", len(json_str))
        # print("First few characters:", [ord(c) for c in json_str[:10]])  # Print ASCII values of first few chars
        parsed = json.loads(json_str)
        scores = {
            "A": int(min(parsed.get("A", 0), 5)),
            "B": int(min(parsed.get("B", 0), 5)), 
            "C": int(min(parsed.get("C", 0), 5)),
            "D": int(min(parsed.get("D", 0), 5)),
            "E": int(min(parsed.get("E", 0), 5))
        }
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        print("🔎 Raw output was:\n", output)
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
    return rubric_scores
