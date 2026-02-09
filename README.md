# LLM Dialect Robustness

Official implementation for the paper: **Enhancing Dialect Robustness in Large Language Models via Parameter-Efficient Fine-Tuning**

## Overview

Despite the success of large language models (LLMs) in a wide range of applications, it has been shown that their performance varies across English dialects. Differences among English dialects are reflected in vocabulary, syntax, and writing style, and can adversely affect model performance. Several studies evaluate the dialect robustness of LLMs, yet research on enhancing their robustness to dialectal variation remains limited.

In this work, we propose two parameter-efficient frameworks for improving dialectal robustness in LLMs:

1. **DialectFusion**: We train separate LoRA layers for each dialect and apply different LoRA merging methods (TIES, CAT) to create a unified model that handles multiple dialects effectively.

2. **DialectMoE**: Built on top of Mixture of Experts LoRA, this approach introduces multiple LoRA-based experts to the feed-forward layer to internally model dialectal dependencies with learned routing mechanisms.

Our comprehensive analysis on five open-source LLMs for sentiment and sarcasm tasks in zero- and few-shot settings shows that our proposed approaches enhance the dialect robustness of LLMs and outperform instruct and LoRA fine-tuning based approaches.

### Key Features

- **Two Parameter-Efficient Methods**: DialectFusion (adapter merging) and DialectMoE (mixture of experts)
- **Multi-Dialect Evaluation**: Comprehensive testing across en-AU (Australian), en-IN (Indian), en-UK (British) varieties using BESSTIE benchmark
- **Multiple Model Families**: Llama, Mistral, Gemma, Qwen, Phi (5 open-source LLMs)
- **Efficient Training Support**: DeepSpeed ZeRO-3 and FSDP integration with QLoRA for memory-efficient training

## Installation

We require two separate environments for running DialectMoE and DialectFusion, this is due to conflict in dependencies of MoE-PEFT on which DialectMoE is based upon.

### Requirements

For DialectFusion, please use the following requirements :

```bash
# Python 3.11+
pip install -r requirements.txt
```

For DialectMoE, please use the following requirements :

```bash
# Python 3.11+
pip install -r requirements_moe.txt
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Sanjh-Maheshwari/LLM-Dialect-Robustness.git
cd LLM-Dialect-Robustness
pip install -r requirements.txt # or the MoE-PEFT requirements
```

### 2. Prepare Data

The BESSTIE dataset is downloaded and processed to store in friendly format in `notebooks/analysis.ipynb` which stores the data in JSON format which is then futher use to obtain training / test datasets for DialectFusion and DialectMoE.

```bash
# Preprocess and validate BESSTIE dataset
python synthesizer/preprocess_besstie_dataset.py

# Generate DialectMoE-specific training data
python synthesizer/preprocess_besstie_moepeft_dataset.py
```

This creates:

- `data/axolotl` - Dialect-specific splits in instruction format for DialectFusion training
- `data/moe_peft` - Dialect-specific splits in instruction format for DialectMoE training

### 3. Train Models

There are two different libraries used for training Axolotl and MoE-PEFT. The checkpoints and adapteres are saved to `output` directory as default.

#### Train Dialect-Specific Adapters (using Axolotl)

```bash
# Train individual adapters for all variants of BESSTIE dataset
bash scripts/train_all_dialects.sh llama

# Or run each variant and dialect individually (ex : llama, sarcasm)
axolotl train trainer/axolotl_configs/llama/llama_sarcasm_au.yaml --debug --debug-num-examples 5
axolotl train trainer/axolotl_configs/llama/llama_sarcasm_in.yaml --debug --debug-num-examples 5
axolotl train trainer/axolotl_configs/llama/llama_sarcasm_uk.yaml --debug --debug-num-examples 5
```

#### Train MixLoRA Adapters (using MoE-PEFT)

```bash
# Train DialectMoE on mixed dialect data
python moe_peft.py --base_model google/gemma-2-9b-it --config trainer/moe_configs/moe_peft_sarcasm_gemma.json --bf16 --load_4bit --dir ./output
```

#### Train Baseline LoRA (using Axolotl)

```bash
# Train baselines on mixed dialect data
bash scripts/train_all_baselines.sh llama

# Or run each variant and dialect individually (ex : llama, sarcasm)
axolotl train trainer/axolotl_configs_baseline/llama/llama_sarcasm.yaml --debug --debug-num-examples 5
```

### 4. Evaluate Models & Analyse Results

#### Evaluate DialectFusion

##### Zero-Shot Evaluation

```bash
python evaluator_unified/evaluate_merge_methods_zeroshot.py --model phi --method lora_grouping

# All available models: phi, mistral7b, mistral2409, qwen, llama, gemma2
# All available methods: lora_grouping, cat, ties, base_instruct, individual_dialect
```

##### Few-Shot Evaluation

```bash
python evaluator_unified/evaluate_merge_methods_fewshot.py --model phi --method cat

# All available models: phi, mistral7b, mistral2409, qwen, llama, gemma2
# All available methods: lora_grouping, cat, ties, base_instruct, individual_dialect
```

or optionally run following to evaluate combination of models and methods :

```bash
python evaluator_unified/run_all_evaluations.py
```

#### Evaluate DialectMoE (Mixture of Experts)

##### Zero-Shot Evaluation

```bash
python evaluator_moe/evaluate_moe_zeroshot.py --model phi

# All available models: phi, mistral, gemma, qwen, llama
```

##### Few-Shot Evaluation

```bash
# Evaluate individual MOE models (few-shot, with 2 examples per instance)
python evaluator_moe/evaluate_moe_fewshot.py --model phi

# All available models: phi, mistral, gemma, qwen, llama
```

Results are saved to `results` directory by default but can be changed by using `--output-dir` argument. Each result file contains following structure :

```json
{
  "overall": {
    "accuracy": 0.XX,
    "f1_macro": 0.XX,
    "f1_weighted": 0.XX
  },
  "per_dialect": {
    "en-AU": {"accuracy": 0.XX, "f1": 0.XX},
    "en-IN": {"accuracy": 0.XX, "f1": 0.XX},
    "en-UK": {"accuracy": 0.XX, "f1": 0.XX}
  },
  "per_task": {
    "Sentiment": {"accuracy": 0.XX, "f1": 0.XX},
    "Sarcasm": {"accuracy": 0.XX, "f1": 0.XX}
  },
  "confusion_matrix": [...],
  "predictions": [...]
}
```

## Experimental Settings

### GPU Requirements

It is recommended to use an NVIDIA GPU with minimum of 40 GB RAM. For the experiments we USED two NVIDIA A100 40GB GPUs and two NVIDIA A100 80GB GPUs from CREATE HPC depending on availability.

### Supported Models

| Model Family | Model ID                                              | Parameters |
| ------------ | ----------------------------------------------------- | ---------- |
| Llama        | `meta-llama/Llama-2-7b-hf`                            | 7B         |
| Mistral      | `mistralai/Mistral-7B-v0.1`                           | 7B         |
| Gemma        | `google/gemma-2b`, `google/gemma-7b`                  | 2B, 7B     |
| Qwen         | `Qwen/Qwen-7B`                                        | 7B         |
| Phi          | `microsoft/phi-2`, `microsoft/Phi-3-mini-4k-instruct` | 2.7B, 3.8B |

### Evaluation Settings

- **Zero-shot**: Base model without fine-tuning
- **Few-shot**: k-shot in-context learning (n=2, 3, 5)
- **Fine-tuned**: Dialect-specific adapter (single dialect) and combined adapter
- **DialectMoE**: Mixture of dialect experts with learned routing mechanism
- **DialectFusion** (Merged Adapters):
  - **TIES**: Task-wise interference elimination and scaling
  - **CAT**: Concatenation-based expert selection

### Datasets

- **BESSTIE**: British English Sentiment and Sarcasm in Text across dialects
  - Varieties: en-AU, en-IN, en-UK
  - Tasks: Sentiment (positive/negative), Sarcasm (sarcastic/non-sarcastic)
  - Domains: Reddit, Google Reviews
- **mlabonne/FineTome-100k**: Ultra-high quality instruction-following dataset
  - Size: 100K samples
  - Content: Conversations, reasoning problems, function calling, and more
  - Source: Filtered subset of arcee-ai/The-Tome using HuggingFaceFW/fineweb-edu-classifier
  - Language: English
  - Use Case: General-purpose fine-tuning for large language models

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourpaper2024,
  title={[Your Paper Title]},
  author={[Your Names]},
  booktitle={[Conference Name]},
  year={2024}
}
``` -->

<!-- ## License

This project is licensed under the MIT License. -->

## Acknowledgments

- BESSTIE dataset: [unswnlporg/BESSTIE](https://huggingface.co/datasets/unswnlporg/BESSTIE)
- FineTome-100k dataset: [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) by Maxime Labonne
  - Source dataset: [arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)
  - Filtering classifier: [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)
- Hugging Face [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) libraries
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) and Microsoft Research
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) training framework
- [MixLoRA (MoE-PEFT)](https://github.com/TUDB-Labs/MixLoRA) for Mixture of Expert Parameter-Efficient Fine-Tuning
