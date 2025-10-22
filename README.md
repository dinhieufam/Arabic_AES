# Arabic Automatic Essay Scoring with Prompt Engineering

This repository contains the code and resources for research on **trait-specific Automatic Essay Scoring (AES) in Arabic using Large Language Models (LLMs)** through novel prompt engineering techniques.

## 📋 Abstract

This paper presents a novel prompt engineering framework for trait-specific Automatic Essay Scoring (AES) in Arabic, leveraging large language models (LLMs) under zero-shot and few-shot configurations. Addressing the scarcity of scalable, linguistically informed AES tools for Arabic, we introduce a three-tier prompting strategy: standard, hybrid, and rubric-guided that guides LLMs in evaluating distinct language proficiency traits such as organization, vocabulary, development, and style. The hybrid approach simulates multi-agent evaluation with trait specialist raters, while the rubric-guided method incorporates scored exemplars to enhance model alignment.

These findings confirm that structured prompting, not model scale alone, enables effective AES in Arabic. This study presents the first comprehensive framework for proficiency-oriented Arabic AES and sets the foundation for scalable assessment in low-resource educational contexts.

## 🚀 Key Features

- **Three-tier prompting strategy**: Standard, Hybrid, and Rubric-guided approaches
- **Trait-specific evaluation**: Organization, Vocabulary, Development, Style, Structure, Mechanics, and Relevance
- **Zero-shot and few-shot configurations**: No model fine-tuning required
- **Comprehensive evaluation**: Testing on 8 different LLMs
- **First Arabic AES framework**: Specialized for Arabic language proficiency assessment

## 📊 Dataset

This project uses the **QAES dataset** - the first publicly available Arabic AES resource with trait-level annotations. The dataset includes essays with detailed scoring across multiple linguistic traits.

## 🏗️ Project Structure

```
Arabic_AES/
├── README.md                    # Project documentation
├── dataset.csv                  # Main dataset file
├── dataset.xlsx                 # Dataset in Excel format
├── main_config.json            # Main configuration file
├── evaluate.json               # Evaluation configuration
├── evaluate.py                 # Main evaluation script
├── essays/                     # Essay corpus
├── evaluation_results/         # Model evaluation outputs
│   ├── ...                   
├── predictions/                # Model predictions
│   ├── ...                   
├── prompting/                  # Prompt engineering scripts
│   ├── main_prompt_*.py        # Main prompting strategies
│   ├── *_prompt_*.py          # Model-specific prompts
│   └── util.py                # Utility functions
├── rubric_examples/            # Trait-specific rubric examples
│   ├── development.txt
│   ├── mechanics.txt
│   ├── organization.txt
│   ├── relevance.txt
│   ├── structure.txt
│   ├── style.txt
│   └── vocabulary.txt
└── visualization/             # Results visualization
    └── ...
```

## 🏃‍♂️ Usage

### Prompting Strategies

The project implements three main prompting approaches:

#### 1. Standard Prompting
```bash
python prompting/main_prompt_1.py
```

#### 2. Hybrid Prompting (Multi-agent simulation)
```bash
python prompting/main_prompt_2.py
```

#### 3. Rubric-guided Prompting
```bash
python prompting/main_prompt_3.py
```

### Model-specific Scripts

For different LLMs, use the corresponding scripts:
- **OpenAI models**: `openai_prompt_*.py`
- **JAIS models**: `jais_prompt_*.py`
- **Llama models**: `llama_prompt_*.py`
- **Aya models**: `aya_prompt_*.py`
- **Qwen3-VL models**: `qwen3vl_prompt_*.py`

### Visualization

Generate result visualizations:
```bash
python visualization/src/box_grid.py
python visualization/src/line_graph.py
```

## 📝 Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)**: Primary metric for inter-rater agreement
- **Trait-level analysis**: Separate evaluation for each linguistic trait
- **Cross-model comparison**: Performance across 7 different LLMs

## 🎯 Evaluated Traits

1. **Organization**: Essay structure and logical flow
2. **Vocabulary**: Word choice and lexical diversity
3. **Development**: Idea elaboration and argumentation
4. **Style**: Writing tone and register
5. **Structure**: Sentence and paragraph construction
6. **Mechanics**: Grammar, spelling, and punctuation
7. **Relevance**: Topic adherence and content appropriateness