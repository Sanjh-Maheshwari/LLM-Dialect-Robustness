# Usage Examples

## Quick Start Examples

### Example 1: Evaluate Phi-3 with LoRA Grouping

```bash
python evaluator_unified/evaluate_merge_methods.py --model phi --method lora_grouping
```

**What this does:**
- Loads Phi-3 base model
- Loads single baseline adapter trained on all dialects
- Evaluates on en-AU, en-IN, en-UK test sets
- Saves results to `results_besstie_unified/phi/lora_grouping/results.json`

### Example 2: Evaluate Qwen with CAT Merging

```bash
python evaluator_unified/evaluate_merge_methods.py --model qwen --method cat
```

**What this does:**
- Loads Qwen base model
- Loads 3 dialect-specific adapters (en-AU, en-IN, en-UK)
- Merges them using CAT (concatenation) method
- Evaluates merged adapter on all test sets
- Saves results to `results_besstie_unified/qwen/cat/results.json`

### Example 3: Evaluate LLaMA with TIES Merging

```bash
python evaluator_unified/evaluate_merge_methods.py --model llama --method ties
```

**What this does:**
- Loads LLaMA base model
- Loads 3 dialect-specific adapters
- Merges them using TIES (Task Arithmetic) method
- Evaluates merged adapter on all test sets
- Saves results to `results_besstie_unified/llama/ties/results.json`

## Batch Evaluation Examples

### Example 4: Run All Evaluations (18 total)

```bash
python evaluator_unified/run_all_evaluations.py
```

**What this does:**
- Runs all 6 models × 3 methods = 18 evaluations
- Shows progress bar for each evaluation
- Saves individual results for each model/method combination
- Generates aggregated summary in `results_besstie_unified/all_results_summary.json`
- Prints final summary table

### Example 5: Evaluate Specific Models Only

```bash
python evaluator_unified/run_all_evaluations.py --models phi qwen llama
```

**What this does:**
- Runs only Phi-3, Qwen, and LLaMA
- Tests all 3 methods for each: 3 models × 3 methods = 9 evaluations

### Example 6: Test Specific Methods Only

```bash
python evaluator_unified/run_all_evaluations.py --methods cat ties
```

**What this does:**
- Runs CAT and TIES methods only (skips LoRA Grouping)
- Tests all 6 models: 6 models × 2 methods = 12 evaluations

### Example 7: Targeted Evaluation

```bash
python evaluator_unified/run_all_evaluations.py --models phi mistral7b --methods lora_grouping cat
```

**What this does:**
- Evaluates only Phi-3 and Mistral-7B
- Tests only LoRA Grouping and CAT methods
- Total: 2 models × 2 methods = 4 evaluations

## Results Inspection Examples

### Example 8: View Single Model Results

```bash
# Pretty print JSON results
cat results_besstie_unified/phi/lora_grouping/results.json | python -m json.tool
```

### Example 9: Extract Average Metrics

```bash
# Extract average accuracy for all Phi-3 evaluations
for method in lora_grouping cat ties; do
    echo -n "$method: "
    jq '.summary.average_accuracy' results_besstie_unified/phi/$method/results.json
done
```

### Example 10: Compare Methods for One Model

```python
import json

model = "qwen"
methods = ["lora_grouping", "cat", "ties"]

print(f"Comparison for {model}:")
print("-" * 50)

for method in methods:
    with open(f"results_besstie_unified/{model}/{method}/results.json") as f:
        data = json.load(f)
        acc = data["summary"]["average_accuracy"]
        f1 = data["summary"]["average_f1"]
        print(f"{method:15} | Acc: {acc:.4f} | F1: {f1:.4f}")
```

## Advanced Examples

### Example 11: Custom Output Directory

```bash
python evaluator_unified/evaluate_merge_methods.py \
    --model phi \
    --method ties \
    --output-dir my_custom_results
```

### Example 12: Run Evaluations in Parallel (Advanced)

```bash
# Run 3 different evaluations in parallel (requires sufficient GPU memory)
python evaluator_unified/evaluate_merge_methods.py --model phi --method lora_grouping &
python evaluator_unified/evaluate_merge_methods.py --model qwen --method lora_grouping &
python evaluator_unified/evaluate_merge_methods.py --model llama --method lora_grouping &
wait
```

### Example 13: Skip Aggregation (for debugging)

```bash
python evaluator_unified/run_all_evaluations.py --skip-aggregation
```

## Expected Output Format

When you run an evaluation, you'll see output like:

```
=== Unified LoRA Merging Evaluation ===
Model: phi
Method: lora_grouping
Output directory: results_besstie_unified

Initializing phi model...
Loading microsoft/Phi-3-medium-4k-instruct

============================================================
Evaluating Task: Sarcasm
============================================================

Domain: Reddit
Evaluating Sarcasm/Reddit for dialect: en-AU using method: lora_grouping
en-AU: 100%|████████████████████| 500/500 [02:15<00:00,  3.70it/s]
Results - Accuracy: 0.8520, F1: 0.8498

Evaluating Sarcasm/Reddit for dialect: en-IN using method: lora_grouping
en-IN: 100%|████████████████████| 500/500 [02:15<00:00,  3.70it/s]
Results - Accuracy: 0.8340, F1: 0.8312

...

============================================================
Results Summary
============================================================
Average Accuracy: 0.8467
Average F1: 0.8433

Results saved to: results_besstie_unified/phi/lora_grouping/results.json
============================================================
```

## Troubleshooting

### Issue: Out of GPU Memory

**Solution:** Run evaluations sequentially instead of in parallel, or use a smaller batch of models:

```bash
python evaluator_unified/run_all_evaluations.py --models phi
python evaluator_unified/run_all_evaluations.py --models qwen
# etc.
```

### Issue: Missing Adapter Files

**Error:** `FileNotFoundError: Adapter not found`

**Solution:** Check that adapter directories exist:
```bash
ls /scratch/users/k24053411/axolotl/phi_3/
ls /scratch/users/k24053411/axolotl/phi_3/baseline/
```

### Issue: Import Errors

**Error:** `ModuleNotFoundError: No module named 'evaluator_unified'`

**Solution:** Run from the project root directory:
```bash
cd /home/aniket/personal/LLM-Dialect-Robustness
python evaluator_unified/evaluate_merge_methods.py --model phi --method lora_grouping
```
