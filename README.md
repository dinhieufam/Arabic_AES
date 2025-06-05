# Automatic Arabic Essay Scoring

## Model
- model_1: Qwen/Qwen1.5-1.8B-Chat
- model_2: Qwen/Qwen2.5-7B-Instruct
- model_3: ALLaM-AI/ALLaM-7B-Instruct-preview
- model_4: inceptionai/jais-family-13b-chat
- model_5: FreedomIntelligence/AceGPT-7B-chat
<!-- - model_6: mistralai/Mistral-7B-Instruct-v0.2
- model_7: NousResearch/Hermes-2-Pro-Llama-3-8B -->

## Finished evaluation
- Prompt Engineer
    - model_1, model_2, model_3
    - All prompt_3 have to be run again because forgot to remove normalize of relevance
        - model_1: DONE
        - model_2: DONE
        - model_3: .....

    - model_4 prompt_1: good
    - model_4 prompt_2: output is unstable, do not follow the JSON format. Maybe it is because the prompt is in English. Because Jais in an Arabic LLM, maybe we could try Arabic prompt
    - model_4 prompt_3: all 0, even worse than using prompt_2

- Instruction-Tuning:
    - Code available
    - To inference just load the model checkpoint and use the code in prompting folder

- Label-Supervised Learning:
    - Finetuning code available
    - Inference code available