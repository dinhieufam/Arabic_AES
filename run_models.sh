#!/bin/bash

# Create necessary directories
mkdir -p predictions
mkdir -p evaluation_results

# Function to update config files
update_configs() {
    local model_name=$1
    local model_num=$2
    
    # Update main_config.json
    jq --arg model "$model_name" '.model_name = $model' main_config.json > temp.json && mv temp.json main_config.json
    
    # Update evaluate.json for each prompt level
    for level in 1 2 3; do
        jq --arg pred "predictions/model_${model_num}/prompt_level_${level}.csv" \
           --arg out "evaluation_results/model_${model_num}/prompt_level_${level}.csv" \
           '.paths.model_predictions = $pred | .paths.output = $out' evaluate.json > temp.json && mv temp.json evaluate.json
    done
}

# Function to run evaluation pipeline
run_evaluation() {
    local model_name=$1
    local model_num=$2
    
    echo "🚀 Running evaluation for model: $model_name"
    
    # Create model-specific directories
    mkdir -p "predictions/model_${model_num}"
    mkdir -p "evaluation_results/model_${model_num}"
    
    # Update configurations
    update_configs "$model_name" "$model_num"
    
    # Run prompt level 1
    echo "📝 Running prompt level 1..."
    python main_prompt_1.py
    python evaluate.py
    
    # Run prompt level 2
    echo "📝 Running prompt level 2..."
    python main_prompt_2.py
    python evaluate.py
    
    # Run prompt level 3
    echo "📝 Running prompt level 3..."
    python main_prompt_3.py
    python evaluate.py
    
    echo "✅ Completed evaluation for model: $model_name"
    echo "----------------------------------------"
}

# List of models to evaluate
models=(
    "Qwen/Qwen1.5-1.8B-Chat"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "ALLaM-AI/ALLaM-7B-Instruct-preview"
    # "AceGPT/AceGPT-1.5B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"
    # "NousResearch/Hermes-2-Pro-Llama-3-8B"
)

# Run evaluation for each model
for i in "${!models[@]}"; do
    model_num=$((i + 1))  # Start from model 1
    run_evaluation "${models[$i]}" "$model_num"
done

echo "🎉 All evaluations completed!" 