import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np

# Define your rubric keys
rubric_keys = ['organization', 'vocabulary', 'style', 'development', 
               'mechanics', 'structure', 'relevance', 'final_score']

# Model definition
class MultiRubricClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        dropout_prob = getattr(self.config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        self.heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, 6),  # organization
            nn.Linear(self.config.hidden_size, 6),  # vocabulary
            nn.Linear(self.config.hidden_size, 6),  # style
            nn.Linear(self.config.hidden_size, 6),  # development
            nn.Linear(self.config.hidden_size, 6),  # mechanics
            nn.Linear(self.config.hidden_size, 6),  # structure
            nn.Linear(self.config.hidden_size, 3),  # relevance
            nn.Linear(self.config.hidden_size, 33),  # final_score
        ])

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token (first token) for Qwen models
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        all_logits = [head(pooled_output) for head in self.heads]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            
            # Debug: Check label ranges and adjust if necessary
            for i, (logits, head) in enumerate(zip(all_logits, self.heads)):
                current_labels = labels[:, i]
                num_classes = logits.size(-1)
                
                # Clamp labels to valid range [0, num_classes-1]
                current_labels = torch.clamp(current_labels, 0, num_classes - 1)
                
                # Debug print (remove after testing)
                # print(f"Head {i} ({rubric_keys[i]}): labels range [{current_labels.min()}, {current_labels.max()}], num_classes: {num_classes}")
                
                loss += loss_fct(logits, current_labels)
            
            return {"loss": loss, "logits": all_logits}
        else:
            return {"logits": all_logits}

# Fixed collate function that properly handles the data
def collate_fn(batch):
    # Debug print to see what we're getting
    # print("First batch item keys:", list(batch[0].keys()))
    
    # Handle input_ids and attention_mask - ensure they're properly padded
    max_length = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        # Pad sequences to max_length
        ids = item['input_ids']
        mask = item['attention_mask']
        
        # Pad if necessary
        if len(ids) < max_length:
            pad_length = max_length - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad_length
            mask = mask + [0] * pad_length
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    # Convert to tensors
    batch_dict = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }
    
    # Handle labels if they exist
    if f"label_{rubric_keys[0]}" in batch[0]:
        labels_list = []
        for key in rubric_keys:
            label_values = [item[f"label_{key}"] for item in batch]
            labels_list.append(label_values)
        
        # Transpose to get [batch_size, num_rubrics] instead of [num_rubrics, batch_size]
        labels_array = np.array(labels_list).T  # Transpose here
        
        # Convert labels to 0-based indexing if they aren't already
        # Most scoring rubrics start from 1, but neural networks expect 0-based
        labels_array = labels_array - 1  # Convert from 1-based to 0-based
        
        # Ensure labels are non-negative (in case some were already 0-based)
        labels_array = np.maximum(labels_array, 0)
        
        batch_dict["labels"] = torch.tensor(labels_array, dtype=torch.long)
        
        # Debug: Print label statistics
        print(f"Labels shape: {labels_array.shape}")
        print(f"Labels range: [{labels_array.min()}, {labels_array.max()}]")
        print(f"Sample labels: {labels_array[0] if len(labels_array) > 0 else 'None'}")
    
    return batch_dict

# Fixed compute metrics
def compute_metrics(pred):
    # The model returns a dictionary with 'logits' key
    if isinstance(pred.predictions, dict):
        all_logits = pred.predictions["logits"]
    else:
        all_logits = pred.predictions
    
    all_labels = pred.label_ids
    
    metrics = {}
    for i, key in enumerate(rubric_keys):
        # all_logits is a list of arrays, one per head
        if isinstance(all_logits, (list, tuple)):
            preds = np.argmax(all_logits[i], axis=1)
        else:
            # Fallback if it's not a list
            preds = np.argmax(all_logits, axis=1)
        
        labels = all_labels[:, i]
        acc = np.mean(preds == labels)
        metrics[f"{key}_accuracy"] = acc
    
    return metrics

# Main function
def main():
    global tokenizer
    model_name = 'Qwen/Qwen1.5-1.8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    df = pd.read_csv('dataset.csv')
    dataset = Dataset.from_pandas(df)

    # Preprocessing function
    def preprocess_function(examples):
        # Don't use padding='max_length' here, let collate_fn handle it
        tokenized = tokenizer(examples['text'], truncation=True, max_length=512)
        # Add labels to the tokenized data
        for key in rubric_keys:
            tokenized[f"label_{key}"] = examples[key]
        return tokenized

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print("Dataset fields after tokenization:", tokenized_dataset.column_names)
    
    # Debug: Check label value ranges in your dataset
    print("\nLabel value ranges in dataset:")
    for key in rubric_keys:
        if key in df.columns:
            values = df[key].values
            print(f"{key}: min={values.min()}, max={values.max()}, unique={sorted(set(values))}")
    print()

    # Better dataset split
    dataset_size = len(tokenized_dataset)
    if dataset_size < 10:
        print(f"Warning: Very small dataset size ({dataset_size}). Consider using more data.")
    
    train_size = max(1, int(0.8 * dataset_size))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, dataset_size))

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Initialize model
    model = MultiRubricClassifier(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='model_checkpoints/finetuning',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy='epoch',
        logging_dir='./logs',
        save_strategy='no', # Set to epoch if wanna save checkpoints after each epoch
        load_best_model_at_end=False, # Set to True if wanna save
        logging_steps=1,
        remove_unused_columns=False,  # Important: keep all columns
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        data_collator=collate_fn,
    )

    # Train
    trainer.train()

    trainer.save_model("model_checkpoints/final_model")

if __name__ == '__main__':
    main()