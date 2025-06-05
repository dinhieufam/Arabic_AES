import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define SCORING_PROMPT (shortened to save tokens)
SCORING_PROMPT = """You are an Arabic essay evaluator. Score these traits (0-5, except relevance 0-2):
1. Organization: Structure and coherence
2. Vocabulary: Word choice richness  
3. Style: Writing fluency
4. Development: Idea elaboration
5. Mechanics: Grammar/spelling
6. Structure: Syntax correctness
7. Relevance: Topic adherence (0-2)
8. Final Score: Sum of all scores (0-32)

Return JSON only:
{"organization": X, "vocabulary": X, "style": X, "development": X, "mechanics": X, "structure": X, "relevance": X, "final_score": X}"""

# Load dataset from a single CSV file and split it
try:
    dataset = load_dataset("csv", data_files="dataset.csv")
    print(f"Original dataset size: {len(dataset['train'])}")
    
    # Check the first few samples to understand the structure
    sample = dataset["train"][0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample text length: {len(str(sample.get('text', '')))}")
    
    # Split into train (80%), validation (10%), and test (10%)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_val = dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset = {
        "train": dataset["train"],
        "validation": test_val["train"],
        "test": test_val["test"]
    }
    print(f"Dataset split: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load model and tokenizer
model_name = "Qwen/Qwen1.5-1.8B"
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print(f"LoRA applied. Trainable parameters: {model.num_parameters()}")

def truncate_essay(text, max_essay_tokens=400):
    """Truncate essay to fit within token limits while preserving meaning"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_essay_tokens:
        return text
    
    # Truncate and decode back to text
    truncated_tokens = tokens[:max_essay_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Try to end at a sentence boundary
    sentences = truncated_text.split('.')
    if len(sentences) > 1:
        truncated_text = '.'.join(sentences[:-1]) + '.'
    
    return truncated_text

def formatting_func(example):
    """Format example for training with proper truncation"""
    try:
        # Get and clean the essay text
        essay = str(example["text"]).strip()
        if not essay:
            return None
            
        # Truncate essay to manageable length
        essay = truncate_essay(essay, max_essay_tokens=350)
        
        # Extract and validate scores
        scores = {
            "organization": int(float(example["organization"])),
            "vocabulary": int(float(example["vocabulary"])),
            "style": int(float(example["style"])),
            "development": int(float(example["development"])),
            "mechanics": int(float(example["mechanics"])),
            "structure": int(float(example["structure"])),
            "relevance": int(float(example["relevance"])),
            "final_score": int(float(example["final_score"]))
        }
        
        # Validate score ranges
        for key, value in scores.items():
            if key == "relevance" and not (0 <= value <= 2):
                print(f"Invalid relevance score: {value}, setting to 1")
                scores[key] = 1
            elif key == "final_score" and not (0 <= value <= 32):
                print(f"Invalid final score: {value}, recalculating")
                scores[key] = sum(v for k, v in scores.items() if k != "final_score")
            elif key not in ["relevance", "final_score"] and not (0 <= value <= 5):
                print(f"Invalid {key} score: {value}, setting to 3")
                scores[key] = 3
        
        # Ensure final score is correct
        expected_final = sum(v for k, v in scores.items() if k != "final_score")
        scores["final_score"] = expected_final
        
        json_scores = json.dumps(scores, ensure_ascii=False)
        formatted_text = f"{SCORING_PROMPT}\n\nEssay:\n{essay}\n\n### Response:\n{json_scores}"
        
        # Check final token count
        total_tokens = len(tokenizer.encode(formatted_text))
        if total_tokens > 1000:  # Conservative limit
            print(f"Still too long ({total_tokens} tokens), further truncating...")
            essay = truncate_essay(essay, max_essay_tokens=250)
            formatted_text = f"{SCORING_PROMPT}\n\nEssay:\n{essay}\n\n### Response:\n{json_scores}"
        
        return formatted_text
        
    except Exception as e:
        print(f"Error in formatting_func: {e}")
        print(f"Example keys: {list(example.keys()) if hasattr(example, 'keys') else 'No keys'}")
        return None

def clean_dataset(dataset_split):
    """Clean dataset and remove problematic samples"""
    cleaned_data = []
    
    for i, example in enumerate(dataset_split):
        formatted = formatting_func(example)
        if formatted is not None:
            # Create a clean example with only the text field
            cleaned_data.append({"text": formatted})
        else:
            print(f"Skipping sample {i} due to formatting issues")
    
    print(f"Cleaned dataset: {len(cleaned_data)} samples")
    return Dataset.from_list(cleaned_data)

# Clean datasets
print("Cleaning datasets...")
clean_train = clean_dataset(dataset["train"])
clean_val = clean_dataset(dataset["validation"])

if len(clean_train) == 0 or len(clean_val) == 0:
    print("Error: No valid samples after cleaning!")
    exit(1)

print(f"Clean datasets: Train={len(clean_train)}, Val={len(clean_val)}")

# Check a sample
sample_formatted = clean_train[0]["text"]
sample_tokens = len(tokenizer.encode(sample_formatted))
print(f"Sample token count: {sample_tokens}")
print(f"Sample preview: {sample_formatted[:300]}...")

# Set up SFTConfig with conservative settings
sft_config = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=1,  # Very small batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Simulate larger batch
    learning_rate=1e-5,  # Very low learning rate
    num_train_epochs=2,  # Fewer epochs
    max_seq_length=1024,  # Increased max length
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=5,
    packing=False,
    dataloader_drop_last=True,
    remove_unused_columns=True,  # Remove extra columns
    fp16=True,
    warmup_steps=20,
    weight_decay=0.01,
    report_to=[],  # Disable wandb
)

# Simple formatting function for SFT
def simple_formatting_func(example):
    return example["text"]

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=clean_train,
    eval_dataset=clean_val,
    formatting_func=simple_formatting_func,
    args=sft_config,
)

print("\n=== Starting Training ===")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    print("Attempting to continue with inference testing...")

# Improved inference function
def generate_scores(input_text, max_new_tokens=100):
    """Generate scores for an input essay"""
    # Truncate input essay
    input_text = truncate_essay(input_text, max_essay_tokens=350)
    
    prompt = f"{SCORING_PROMPT}\n\nEssay:\n{input_text}\n\n### Response:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enable sampling
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    response_start = generated_text.find("### Response:")
    if response_start == -1:
        return None
    
    json_str = generated_text[response_start + len("### Response:"):].strip()
    
    # Extract JSON using regex
    json_match = re.search(r'\{[^}]*\}', json_str)
    if json_match:
        json_str = json_match.group(0)
    
    return json_str

def parse_scores(json_str):
    """Parse and validate JSON scores"""
    if not json_str:
        return None
        
    try:
        scores = json.loads(json_str)
        
        required_keys = ["organization", "vocabulary", "style", "development", 
                        "mechanics", "structure", "relevance", "final_score"]
        
        if not all(key in scores for key in required_keys):
            print(f"Missing keys. Found: {list(scores.keys())}")
            return None
        
        # Validate and fix scores
        for key, value in scores.items():
            try:
                value = int(float(value))
                scores[key] = value
                
                if key == "relevance" and not (0 <= value <= 2):
                    print(f"Invalid relevance score: {value}")
                    return None
                elif key == "final_score" and not (0 <= value <= 32):
                    print(f"Invalid final score: {value}")
                    return None
                elif key not in ["relevance", "final_score"] and not (0 <= value <= 5):
                    print(f"Invalid score for {key}: {value}")
                    return None
            except (ValueError, TypeError):
                print(f"Invalid score type for {key}: {value}")
                return None
        
        return scores
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

# Test inference
print("\n=== Testing Inference ===")
test_essay = "هذا مقال تجريبي للاختبار. يحتوي على نص عربي بسيط لتقييم النموذج. يجب أن يكون النص واضحاً ومفهوماً. هذا النص يحتوي على عدة جمل لاختبار قدرة النموذج على التقييم."

predicted_json = generate_scores(test_essay)
print(f"Generated JSON: {predicted_json}")

if predicted_json:
    parsed_scores = parse_scores(predicted_json)
    if parsed_scores:
        print("Predicted Scores:", parsed_scores)
    else:
        print("Failed to parse scores")
else:
    print("No JSON generated")

# Save model
print("\n=== Saving Model ===")
try:
    model.save_pretrained("model_checkpoints/instr_tune_1")
    tokenizer.save_pretrained("model_checkpoints/instr_tune_1")
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")