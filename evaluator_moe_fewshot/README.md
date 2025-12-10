# MixLoRA (MOE) Few-Shot Evaluation

This directory contains the few-shot evaluation framework for MixLoRA (Mixture of Experts) models, separate from the unified LoRA evaluation in `evaluator_unified/`.

## Directory Structure

```
evaluator_moe_fewshot/
├── moe_services/           # MOE model service implementations
│   ├── __init__.py
│   ├── mistral_moe.py     # Mistral-Small with MixLoRA
│   ├── gemma_moe.py       # Gemma-2-9b with MixLoRA
│   ├── qwen_moe.py        # Qwen-2.5-7B with MixLoRA
│   ├── phi_moe.py         # Phi-3.5 with MixLoRA
│   └── llama_moe.py       # Llama-3.2 with MixLoRA
├── evaluate_moe_fewshot.py # Main evaluation script
├── __init__.py
└── README.md              # This file
```

## Features

- **Few-shot evaluation** with 2 examples per test instance
- **Deterministic sampling** using `seed = 42 + current_idx` for reproducibility
- **5 MOE models** supported: Mistral, Gemma, Qwen, Phi, Llama
- **MixLoRA adapters** for dialect-specific fine-tuning
- **Memory efficient** with adapter load/unload pattern

## Key Differences from `evaluator_unified/`

| Feature | `evaluator_unified/` | `evaluator_moe_fewshot/` |
|---------|---------------------|-------------------------|
| Framework | transformers + PEFT | MOE-PEFT |
| Adapter Type | Standard LoRA | MixLoRA (Expert-based) |
| Methods | lora_grouping, cat, ties, base_instruct, individual_dialect | MixLoRA only |
| Routing | N/A | Top-k, Dynamic, Switch |
| Few-shot | ✓ | ✓ |

## Usage

### Evaluate a single model

```bash
# Mistral MOE
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model mistral

# Gemma MOE
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model gemma

# Qwen MOE
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model qwen

# Phi MOE
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model phi

# Llama MOE
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model llama
```

### Custom output directory

```bash
python evaluator_moe_fewshot/evaluate_moe_fewshot.py --model mistral --output-dir custom_results
```

## Model Services

Each MOE service file implements:

### Key Methods

- `__init__()`: Initialize MOE model with MoE-PEFT
- `format_fewshot_prompt()`: Format prompt with few-shot examples
- `evaluate()`: Generate response using MOE model
- `predict()`: Zero-shot prediction (for comparison)
- `predict_fewshot()`: Few-shot prediction with examples

### Adapter Naming Convention

MixLoRA adapters follow this pattern:
```
mixlora_{model}_{task}_{domain}_0
```

Examples:
- `mixlora_mistral_sentiment_reddit_0`
- `mixlora_gemma_sarcasm_reddit_0`
- `mixlora_qwen_sentiment_reddit_0`

## Few-Shot Sampling

The evaluation uses **deterministic seeding** to ensure reproducibility:

```python
random.seed(42 + current_idx)
sampled_indices = random.sample(available_indices, num_shots)
```

This ensures:
- ✅ Same 2 few-shot examples for the same test instance across ALL models
- ✅ Same examples across multiple runs
- ✅ Different few-shot examples for different test instances

## Output Format

Results are saved to `results_moe_fewshot/{model}/results.json`:

```json
{
  "model": "mistral",
  "model_type": "mixlora_moe",
  "evaluation_type": "few_shot",
  "num_shots": 2,
  "sampling_strategy": "random_per_instance_with_seed",
  "seed_formula": "42 + current_idx",
  "results": {
    "Sentiment": {
      "Reddit": {
        "en-AU": {"acc": 0.85, "f1": 0.84},
        "en-IN": {"acc": 0.83, "f1": 0.82},
        "en-UK": {"acc": 0.87, "f1": 0.86}
      }
    },
    "Sarcasm": {
      "Reddit": {
        "en-AU": {"acc": 0.72, "f1": 0.71},
        "en-IN": {"acc": 0.70, "f1": 0.69},
        "en-UK": {"acc": 0.74, "f1": 0.73}
      }
    }
  },
  "summary": {
    "average_accuracy": 0.785,
    "average_f1": 0.775
  }
}
```

## Evaluated Dialects & Tasks

- **Dialects**: en-AU (Australian), en-IN (Indian), en-UK (British)
- **Tasks**: Sentiment Analysis, Sarcasm Detection
- **Domains**: Reddit

## Requirements

- `moe_peft` library (custom MOE-PEFT framework)
- PyTorch with CUDA support
- transformers, pandas, scikit-learn
- loguru for logging
- tqdm for progress bars

## Notes

- Each model loads/unloads adapters for memory efficiency
- Uses `torch.bfloat16` for reduced memory footprint
- Adapter path defaults to `/scratch/users/k24053411/`
- Test data path: `data/instruction/besstie/test.json`
