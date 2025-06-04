import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from utils import load_essays

# Load config
with open("main_config.json", "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
ESSAY_FOLDER = config["essay_folder"]
MAX_ESSAYS = config["max_essays"]

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

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

def build_prompt():
    training_data = []

    for essay_text in essays:
        training_data.append SCORING_PROMPT + "\n\nEssay:\n" + essay_text.strip()

def prepare_training_data(essays):
    """Convert essays and their scores into instruction format for training"""
    training_data = []
    
    for essay_id, text in essays:
        # Create instruction format
        instruction = """You are an expert Arabic language evaluator. Your task is to assess the proficiency of an Arabic essay based on seven traits:
1. Organization (0-5): How well-structured and coherent is the essay?
2. Vocabulary (0-5): Does the writer use a rich and appropriate vocabulary?
3. Style (0-5): Is the writing engaging, fluent, and stylistically appropriate?
4. Development (0-5): Are ideas elaborated with sufficient details and examples?
5. Mechanics (0-5): Are grammar, spelling, and punctuation correct?
6. Structure (0-5): Does the essay follow proper syntactic structures?
7. Relevance (0-2): Does the essay address the given topic appropriately?
8. Final Score (0-32): The sum of all the scores.

Essay:
{text}

Please evaluate the essay and provide scores in JSON format."""

        # Create expected output format
        output = """{
    "organization": X,
    "vocabulary": X,
    "style": X,
    "development": X,
    "mechanics": X,
    "structure": X,
    "relevance": X,
    "final_score": X
}"""

        training_data.append({
            "instruction": instruction.format(text=text),
            "output": output
        })
    
    return training_data

def prepare_model_and_tokenizer():
    """Initialize model and tokenizer with LoRA configuration"""
    print("🔍 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔍 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        device_map="auto"
    )

    # Prepare model for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def main():
    print(f"🧠 Using model: {MODEL_NAME}")
    
    # Load essays
    essays = load_essays(ESSAY_FOLDER, limit=MAX_ESSAYS)
    
    # Prepare training data
    training_data = prepare_training_data(essays)
    dataset = Dataset.from_list(training_data)
    
    # Initialize model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model()
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main() 