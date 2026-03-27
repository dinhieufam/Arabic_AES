# Structured Prompting for Arabic Essay Proficiency: A Trait-Centric Evaluation Approach

[![arXiv](https://img.shields.io/badge/arXiv-2603.19668-b31b1b.svg)](https://arxiv.org/abs/2603.19668)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> **Official repository** for the paper *"Structured Prompting for Arabic Essay Proficiency: A Trait-Centric Evaluation Approach"*

---

## Abstract

This paper presents a novel prompt engineering framework for trait-specific Automatic Essay Scoring (AES) in Arabic, leveraging large language models (LLMs) under zero-shot and few-shot configurations. Addressing the scarcity of scalable, linguistically informed AES tools for Arabic, we introduce a three-tier prompting strategy — **standard**, **hybrid**, and **rubric-guided** — that guides LLMs in evaluating distinct language proficiency traits such as organization, vocabulary, development, and style. The hybrid approach simulates multi-agent evaluation with trait-specialist raters, while the rubric-guided method incorporates scored exemplars to enhance model alignment. These findings confirm that structured prompting, not model scale alone, enables effective AES in Arabic. This study presents the first comprehensive framework for proficiency-oriented Arabic AES and sets the foundation for scalable assessment in low-resource educational contexts.

---

## Key Contributions

- **Three-tier prompting strategy** — standard, hybrid, and rubric-guided approaches that systematically improve scoring alignment without model fine-tuning.
- **Multi-agent simulation** — the hybrid approach coordinates trait-specialist rater personas within a single inference pass.
- **First Arabic AES framework** — proficiency-oriented, trait-level scoring for Arabic as a low-resource language.
- **Comprehensive benchmarking** — evaluation across eight LLMs under both zero-shot and few-shot configurations.
- **Open dataset support** — built on QAES, the first publicly available Arabic AES resource with trait-level annotations.

---

## Dataset

This work uses the **QAES dataset** — the first publicly available Arabic AES resource featuring trait-level annotations across seven distinct linguistic dimensions. Essays are annotated along the following traits:

| Trait | Description |
|---|---|
| **Organization** | Logical flow and overall structural coherence |
| **Vocabulary** | Lexical richness, precision, and diversity |
| **Development** | Depth of argumentation and idea elaboration |
| **Style** | Register, tone, and stylistic appropriateness |
| **Structure** | Sentence and paragraph construction |
| **Mechanics** | Grammar, spelling, and punctuation |
| **Relevance** | Topic adherence and content alignment |

---

## Repository Structure

```
Arabic_AES/
├── README.md                    # Project documentation
├── dataset.csv                  # Main dataset (CSV)
├── dataset.xlsx                 # Main dataset (Excel)
├── main_config.json             # Global configuration
├── evaluate.json                # Evaluation configuration
├── evaluate.py                  # Evaluation pipeline
│
├── essays/                      # Raw essay corpus
│
├── prompting/                   # Prompt engineering scripts
│   ├── main_prompt_1.py         # Standard prompting
│   ├── main_prompt_2.py         # Hybrid (multi-agent) prompting
│   ├── main_prompt_3.py         # Rubric-guided prompting
│   ├── openai_prompt_*.py       # OpenAI model scripts
│   ├── jais_prompt_*.py         # JAIS model scripts
│   ├── llama_prompt_*.py        # Llama model scripts
│   ├── aya_prompt_*.py          # Aya model scripts
│   ├── qwen3vl_prompt_*.py      # Qwen3-VL model scripts
│   └── util.py                  # Shared utilities
│
├── rubric_examples/             # Trait-specific scored exemplars
│   ├── development.txt
│   ├── mechanics.txt
│   ├── organization.txt
│   ├── relevance.txt
│   ├── structure.txt
│   ├── style.txt
│   └── vocabulary.txt
│
├── predictions/                 # Model prediction outputs
├── evaluation_results/          # Scoring and metric outputs
└── visualization/               # Result plots and figures
    ├── src/box_grid.py
    └── src/line_graph.py
```

---

## Prompting Strategies

### 1. Standard Prompting

A direct, single-pass prompt that instructs the model to score each trait independently.

```bash
python prompting/main_prompt_1.py
```

### 2. Hybrid Prompting

Simulates a multi-agent panel by assigning the model distinct trait-specialist rater personas within one inference call. Each rater focuses on a single trait before a final aggregation step.

```bash
python prompting/main_prompt_2.py
```

### 3. Rubric-Guided Prompting

Augments the prompt with scored exemplars from `rubric_examples/` to align model outputs with human annotation standards via in-context learning.

```bash
python prompting/main_prompt_3.py
```

---

## Model-Specific Scripts

The repository includes dedicated scripts for each evaluated model family:

| Model Family | Script Pattern |
|---|---|
| OpenAI (GPT series) | `prompting/openai_prompt_*.py` |
| JAIS | `prompting/jais_prompt_*.py` |
| Llama | `prompting/llama_prompt_*.py` |
| Aya | `prompting/aya_prompt_*.py` |
| Qwen3-VL | `prompting/qwen3vl_prompt_*.py` |

---

## Evaluation

The primary evaluation metric is **Quadratic Weighted Kappa (QWK)**, which captures inter-rater agreement while penalising large scoring discrepancies more heavily than small ones. Evaluation is conducted per trait and aggregated across model configurations.

```bash
python evaluate.py
```

---

## Visualization

Generate box plots and trend graphs across models and traits:

```bash
python visualization/src/box_grid.py
python visualization/src/line_graph.py
```

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{arabic_aes_2026,
  title   = {Prompt Engineering for Trait-Specific Automatic Essay Scoring in Arabic},
  journal = {arXiv preprint arXiv:2603.19668},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.19668}
}
```

---

## License

This project is released under the [MIT License](LICENSE).