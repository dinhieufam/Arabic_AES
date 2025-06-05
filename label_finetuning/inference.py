import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Define your rubric keys (same as training)
rubric_keys = ['organization', 'vocabulary', 'style', 'development', 
               'mechanics', 'structure', 'relevance', 'final_score']

# Model definition (must match training exactly)
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
            nn.Linear(self.config.hidden_size, 6),  # final_score
        ])

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        all_logits = [head(pooled_output) for head in self.heads]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            for i, (logits, head) in enumerate(zip(all_logits, self.heads)):
                current_labels = labels[:, i]
                num_classes = logits.size(-1)
                current_labels = torch.clamp(current_labels, 0, num_classes - 1)
                loss += loss_fct(logits, current_labels)
            return {"loss": loss, "logits": all_logits}
        else:
            return {"logits": all_logits}

class EssayScorePredictor:
    def __init__(self, model_checkpoint_path, model_name='Qwen/Qwen1.5-1.8B'):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Args:
            model_checkpoint_path: Path to the saved model checkpoint
            model_name: Name of the base model used during training
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = MultiRubricClassifier(model_name)
        
        # Load the trained weights
        if os.path.isdir(model_checkpoint_path):
            # If it's a directory, look for pytorch_model.bin or model.safetensors
            if os.path.exists(os.path.join(model_checkpoint_path, 'pytorch_model.bin')):
                checkpoint_file = os.path.join(model_checkpoint_path, 'pytorch_model.bin')
                state_dict = torch.load(checkpoint_file, map_location=self.device)
            elif os.path.exists(os.path.join(model_checkpoint_path, 'model.safetensors')):
                from safetensors.torch import load_file
                checkpoint_file = os.path.join(model_checkpoint_path, 'model.safetensors')
                state_dict = load_file(checkpoint_file)
            else:
                raise FileNotFoundError(f"No model file found in {model_checkpoint_path}")
        else:
            # Direct file path
            state_dict = torch.load(model_checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict_single(self, text):
        """
        Predict scores for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with rubric scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            max_length=512, 
            padding=True, 
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
        
        # Convert logits to predictions
        predictions = {}
        for i, key in enumerate(rubric_keys):
            pred_class = torch.argmax(logits[i], dim=1).cpu().item()
            # Convert back to 1-based scoring (since we converted to 0-based during training)
            pred_score = pred_class + 1
            predictions[key] = pred_score
        
        return predictions
    
    def predict_batch(self, texts, batch_size=8):
        """
        Predict scores for multiple texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with rubric scores
        """
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits']
            
            # Convert logits to predictions for this batch
            batch_size_actual = inputs['input_ids'].size(0)
            for j in range(batch_size_actual):
                predictions = {}
                for k, key in enumerate(rubric_keys):
                    pred_class = torch.argmax(logits[k][j]).cpu().item()
                    # Convert back to 1-based scoring
                    pred_score = pred_class + 1
                    predictions[key] = pred_score
                all_predictions.append(predictions)
        
        return all_predictions
    
    def predict_from_csv(self, input_csv_path, text_column='text', output_csv_path=None):
        """
        Predict scores for texts in a CSV file.
        
        Args:
            input_csv_path: Path to input CSV file
            text_column: Name of column containing text
            output_csv_path: Path to save output CSV (optional)
            
        Returns:
            DataFrame with original data and predictions
        """
        # Load data
        df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(df)} texts from {input_csv_path}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts)
        
        # Add predictions to dataframe
        for key in rubric_keys:
            df[f'predicted_{key}'] = [pred[key] for pred in predictions]
        
        # Save if output path provided
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")
        
        return df

def main():
    """
    Example usage of the predictor
    """
    # Path to your trained model checkpoint
    model_checkpoint_path = "model_checkpoints/final_model"  # Adjust this path
    
    # Initialize predictor
    predictor = EssayScorePredictor(model_checkpoint_path)
    
    # Example 1: Single text prediction
    sample_text = """
    The importance of education cannot be overstated in today's society. Education serves as the foundation 
    for personal growth, economic development, and social progress. Through education, individuals acquire 
    the knowledge and skills necessary to navigate an increasingly complex world. Furthermore, education 
    promotes critical thinking, creativity, and innovation, which are essential for solving the challenges 
    we face as a global community. In conclusion, investing in education is investing in our future.
    """
    
    print("Single text prediction:")
    single_prediction = predictor.predict_single(sample_text)
    for rubric, score in single_prediction.items():
        print(f"{rubric}: {score}")
    print()
    
    # Example 2: CSV file prediction
    # Create a sample CSV for demonstration
    sample_data = {
        'text': [
            sample_text,
            "This is a short essay about cats. Cats are good pets. They are fluffy and nice.",
            "Climate change is one of the most pressing issues of our time. We must take action now to reduce greenhouse gas emissions and transition to renewable energy sources. The future of our planet depends on the decisions we make today."
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_input.csv', index=False)
    
    print("CSV prediction:")
    result_df = predictor.predict_from_csv(
        input_csv_path='sample_input.csv',
        text_column='text',
        output_csv_path='predictions_output.csv'
    )
    
    print(result_df[['text', 'predicted_organization', 'predicted_vocabulary', 'predicted_final_score']].head())

if __name__ == "__main__":
    main()