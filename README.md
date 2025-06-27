# Arabic Automatic Essay Scoring with Prompt Engineering

This repository contains the code and resources for our research on **trait-specific Automatic Essay Scoring (AES) in Arabic using Large Language Models (LLMs)** through novel prompt engineering techniques.

## 📋 Abstract

This paper presents a novel prompt engineering framework for trait-specific Automatic Essay Scoring (AES) in Arabic, leveraging large language models (LLMs) under zero-shot and few-shot configurations. Addressing the scarcity of scalable, linguistically informed AES tools for Arabic, we introduce a three-tier prompting strategy: standard, hybrid, and rubric-guided that guides LLMs in evaluating distinct language proficiency traits such as organization, vocabulary, development, and style. 

The hybrid approach simulates multi-agent evaluation with trait specialist raters, while the rubric-guided method incorporates scored exemplars to enhance model alignment. Without any model fine-tuning, we evaluate seven LLMs on the QAES dataset, the first publicly available Arabic AES resource with trait-level annotations. Experimental results using Quadratic Weighted Kappa (QWK) show that Fanar-1-9B-Instruct achieves the highest trait-level agreement in both zero and few-shot prompting (QWK = 0.28), followed by ALLaM-7B-Instruct-preview (QWK = 0.26), with rubric-guided prompting yielding consistent gains across all traits and models.

## 🚀 Key Features

- **Three-tier prompting strategy**: Standard, Hybrid, and Rubric-guided approaches
- **Trait-specific evaluation**: Organization, Vocabulary, Development, Style, Structure, Mechanics, and Relevance
- **Zero-shot and few-shot configurations**: No model fine-tuning required
- **Comprehensive evaluation**: Testing on 7 different LLMs
- **First Arabic AES framework**: Specialized for Arabic language proficiency assessment

## 📊 Dataset

This project uses the **QAES dataset** - the first publicly available Arabic AES resource with trait-level annotations. The dataset includes essays with detailed scoring across multiple linguistic traits.

## 🏗️ Project Structure

```
ArabicNLP_2025/
├── README.md                    # Project documentation
├── dataset.csv                  # Main dataset file
├── dataset.xlsx                 # Dataset in Excel format
├── main_config.json            # Main configuration file
├── evaluate.json               # Evaluation configuration
├── evaluate.py                 # Main evaluation script
├── essays/                     # Essay corpus
├── evaluation_results/         # Model evaluation outputs
│   ├── gpt4/                   # GPT-4 results
│   ├── model_1/ to model_6/    # Other LLM results
├── predictions/                # Model predictions
│   ├── gpt4/                   # GPT-4 predictions
│   ├── model_1/ to model_6/    # Other LLM predictions
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
    └── src/
        ├── box_grid.py
        └── line_graph.py
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/dinhieufam/ArabicNLP_2025.git
cd ArabicNLP_2025
```

## 🏃‍♂️ Usage

### Quick Start

1. **Configure your settings** in `evaluate.json`
2. **Run evaluation**:
```bash
python evaluate.py
```

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

<!-- ## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Arabic Automatic Essay Scoring with Prompt Engineering: A Trait-Specific Approach},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2025}
}
``` -->

<!-- ## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- QAES dataset contributors
- The Arabic NLP research community
- All LLM providers for API access -->

<!-- ## 📞 Contact

For questions or collaborations, please contact:
- Email: your.email@university.edu
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Note**: This is the first comprehensive framework for proficiency-oriented Arabic AES, setting the foundation for scalable assessment in low-resource educational contexts. -->