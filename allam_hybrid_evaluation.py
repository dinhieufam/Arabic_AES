
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

RATER_SPECIALIZATIONS = {
            "A": "Evaluate the essay's organization and how well ideas are connected. "
                 "Consider logical flow, paragraph structure, and transitions. "
                 "Provide a score between 0-5 with 0.1 increments.",
            
            "B": "Assess the vocabulary quality and lexical variety in the essay. "
                 "Consider word choice, sophistication, and avoidance of repetition. "
                 "Provide a score between 0-5 with 0.1 increments.",
            
            "C": "Evaluate grammar, spelling, punctuation, and mechanical accuracy. "
                 "Identify errors and assess their impact on readability. "
                 "Provide a score between 0-5 with 0.1 increments.",
            
            "D": "Analyze content development and reasoning quality. "
                 "Consider depth of analysis, argument strength, and evidence use. "
                 "Provide a score between 0-5 with 0.1 increments.",
            
            "E": "Assess style, tone, and contextual appropriateness. "
                 "Consider voice, audience awareness, and stylistic effectiveness. "
                 "Provide a score between 0-5 with 0.1 increments."
        }

RUBRIC_MAPPING = {
    "Organization": ["A", "D", "C"],
    "Vocabulary": ["B", "E", "C"],
    "Style": ["B", "E", "C"],
    "Development": ["D", "A", "B"],
    "Mechanics": ["C"],
    "Structure": ["A", "B", "C"],
    "Relevance": ["D", "B", "E"]
}

RUBRIC_MAX_SCORES = {
    "Organization": 5,
    "Vocabulary": 5,
    "Style": 5,
    "Development": 5,
    "Mechanics": 5,
    "Structure": 5,
    "Relevance": 2
}

# Load ALLaM-7B model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model.eval()

def evaluate_essay(text: str):
    prompt = f"""You are an Arabic essay scoring assistant. You will read a student's Arabic essay and assign scores from 0 to 5 for the following five linguistic dimensions:

A: {RATER_SPECIALIZATIONS['A']}
B: {RATER_SPECIALIZATIONS['B']}
C: {RATER_SPECIALIZATIONS['C']}
D: {RATER_SPECIALIZATIONS['D']}
E: {RATER_SPECIALIZATIONS['E']}

Return only a valid JSON object.

Essay:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the last valid JSON structure
    matches = re.findall(r'{\s*"A":\s*\d,\s*"B":\s*\d,\s*"C":\s*\d,\s*"D":\s*\d,\s*"E":\s*\d\s*}', result)
    if matches:
        scores = json.loads(matches[-1])
    else:
        print("❌ Failed to parse JSON from:", result)
        scores = {key: 0 for key in RATER_SPECIALIZATIONS.keys()}

    # Map back to full rubric categories
    rubric_scores = {}
    for rubric, keys in RUBRIC_MAPPING.items():
        rubric_scores[rubric] = min(sum(scores.get(k, 0) for k in keys) // len(keys), RUBRIC_MAX_SCORES[rubric])

    rubric_scores.update({f"rater_{k}": scores.get(k, 0) for k in RATER_SPECIALIZATIONS})
    return rubric_scores
