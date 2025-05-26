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
    "relevance": 2
}

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    device_map="auto"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    trust_remote_code=True, device_map="auto"
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

    Return only a valid JSON object.

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

    # print(f"Raw Output:\n {output}")

    try:
        json_start = output.find('{')
        print("Found JSON at index:", json_start)
        # Find the 6th closing brace after the opening brace
        json_end = json_start
        brace_count = 0
        for i in range(json_start, len(output)):
            if output[i] == '}':
                brace_count += 1
                if brace_count == 6:
                    json_end = i + 1
                    break
        json_str = output[json_start:json_end]
        print("JSON string:", repr(json_str))  # Using repr() to see hidden characters
        print("JSON string length:", len(json_str))
        print("First few characters:", [ord(c) for c in json_str[:10]])  # Print ASCII values of first few chars
        parsed = json.loads(json_str)
        scores = {
            "A": parsed.get("A", {}).get("score", 0),
            "B": parsed.get("B", {}).get("score", 0), 
            "C": parsed.get("C", {}).get("score", 0),
            "D": parsed.get("D", {}).get("score", 0),
            "E": parsed.get("E", {}).get("score", 0)
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

    # # Extract the last valid JSON structure
    # matches = re.findall(r'{\s*"A":\s*\d,\s*"B":\s*\d,\s*"C":\s*\d,\s*"D":\s*\d,\s*"E":\s*\d\s*}', output)
    # if matches:
    #     scores = json.loads(matches[-1])
    # else:
    #     print("❌ Failed to parse JSON from:", output)
    #     scores = {key: 0 for key in RATER_SPECIALIZATIONS.keys()}

    # Map back to full rubric categories
    rubric_scores = {"essay_id": essay_id}
    for rubric, keys in RUBRIC_MAPPING.items():
        rubric_scores[rubric] = min(sum(scores.get(k, 0) for k in keys) // len(keys), RUBRIC_MAX_SCORES[rubric])

    rubric_scores.update({f"rater_{k}": scores.get(k, 0) for k in RATER_SPECIALIZATIONS})
    return rubric_scores
