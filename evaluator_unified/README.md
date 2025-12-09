# Unified LoRA Merging Evaluation Pipeline

A unified evaluation pipeline for testing different LoRA adapter merging strategies across multiple LLMs on the BESSTIE dialect dataset.

## Overview

This pipeline evaluates **6 models** with **3 merging methods**, producing comprehensive results for dialect-robust classification tasks.

### Supported Models
1. **Phi-3** (microsoft/Phi-3-medium-4k-instruct)
2. **Mistral 7B** (mistralai/Mistral-7B-Instruct-v0.1)
3. **Mistral 2409** (mistralai/Mistral-Small-Instruct-2409)
4. **Qwen** (Qwen/Qwen2.5-7B-Instruct)
5. **LLaMA** (meta-llama/Llama-3.1-8B-Instruct)
6. **Gemma 2** (google/gemma-2-9b-it)

### Merging Methods
1. **LoRA Grouping**: Single adapter trained on all dialects combined (baseline)
2. **CAT**: Concatenation-based merging of 3 dialect-specific adapters
3. **TIES**: Task Arithmetic with Ties-Merging of 3 dialect-specific adapters

## Directory Structure

```
evaluator_unified/
├── llm_services/
│   ├── phi_unified.py
│   ├── mistral7b_unified.py
│   ├── mistral2409_unified.py
│   ├── qwen_unified.py
│   ├── llama_unified.py
│   └── gemma2_unified.py
├── evaluate_merge_methods.py    # Single evaluation script
├── run_all_evaluations.py       # Batch runner
├── run_all.sh                   # Shell script for convenience
└── README.md                    # This file
```

## Usage

### 1. Run a Single Evaluation

Evaluate one model with one method:

```bash
python evaluator_unified/evaluate_merge_methods.py --model phi --method lora_grouping
python evaluator_unified/evaluate_merge_methods.py --model qwen --method cat
python evaluator_unified/evaluate_merge_methods.py --model llama --method ties
```

**Arguments:**
- `--model`: Choose from `phi`, `mistral7b`, `mistral2409`, `qwen`, `llama`, `gemma2`
- `--method`: Choose from `lora_grouping`, `cat`, `ties`
- `--output-dir`: (Optional) Output directory, default: `results_besstie_unified`

### 2. Run All Evaluations (18 total)

Run all 6 models × 3 methods automatically:

```bash
python evaluator_unified/run_all_evaluations.py
```

Or use the shell script:

```bash
bash evaluator_unified/run_all.sh
```

**Options:**
- Select specific models:
  ```bash
  python evaluator_unified/run_all_evaluations.py --models phi qwen
  ```

- Select specific methods:
  ```bash
  python evaluator_unified/run_all_evaluations.py --methods cat ties
  ```

- Combine both:
  ```bash
  python evaluator_unified/run_all_evaluations.py --models phi qwen --methods cat ties
  ```

### 3. Check Results

Results are saved in JSON format:

```
results_besstie_unified/
├── phi/
│   ├── lora_grouping/results.json
│   ├── cat/results.json
│   └── ties/results.json
├── mistral7b/
│   ├── lora_grouping/results.json
│   ├── cat/results.json
│   └── ties/results.json
├── ...
└── all_results_summary.json  # Aggregated summary
```

## Results Format

Each `results.json` file contains:

```json
{
  "model": "phi",
  "method": "lora_grouping",
  "adapter_type": "baseline_single",
  "timestamp": "2025-12-09T...",
  "config": {
    "varieties": ["en-AU", "en-IN", "en-UK"],
    "tasks": ["Sarcasm", "Sentiment"],
    "domains": ["Reddit"]
  },
  "results": {
    "Sarcasm": {
      "Reddit": {
        "en-AU": {"acc": 0.85, "f1": 0.84},
        "en-IN": {"acc": 0.82, "f1": 0.81},
        "en-UK": {"acc": 0.87, "f1": 0.86}
      }
    },
    "Sentiment": { ... }
  },
  "summary": {
    "average_accuracy": 0.8467,
    "average_f1": 0.8433
  }
}
```

## Implementation Details

### LoRA Grouping
- Uses single adapter from `{BASELINE_ADAPTER_DIR}/en_{task}_{domain}_adapter`
- Adapter trained on combined data from all dialects
- No merging required

### CAT (Concatenation)
- Loads 3 dialect-specific adapters: `en-AU`, `en-IN`, `en-UK`
- Merges using PEFT's `add_weighted_adapter()` with `combination_type="cat"`
- Equal weights: `[0.33, 0.33, 0.34]`

### TIES (Task Arithmetic with Ties)
- Loads 3 dialect-specific adapters: `en-AU`, `en-IN`, `en-UK`
- Merges using PEFT's `add_weighted_adapter()` with `combination_type="ties"`
- Equal weights: `[0.33, 0.33, 0.34]`
- Density parameter: `0.2` (sparsity)

## Adapter Paths

Each model expects adapters in specific directories:

```python
DIALECT_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/{model}/"
BASELINE_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/{model}/baseline/"
```

Where `{model}` is:
- `phi_3` for Phi-3
- `mistral7b` for Mistral 7B
- `mistral` for Mistral 2409
- `qwen` for Qwen
- `llama` for LLaMA
- `gemma` for Gemma 2

## Evaluation Details

- **Dataset**: BESSTIE (Dialect-robust classification)
- **Tasks**: Sarcasm detection, Sentiment analysis
- **Domains**: Reddit
- **Dialects**: Australian (en-AU), Indian (en-IN), British (en-UK) English
- **Metrics**: Accuracy, F1-score (weighted)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- scikit-learn
- pandas
- tqdm
- loguru

## Notes

- GPU memory is managed by clearing adapters between evaluations
- Each evaluation is independent and can be run separately
- Failed evaluations are logged and can be re-run individually
- Batch runner provides progress tracking and summary statistics
