#!/bin/bash

# Run few-shot evaluation for all models
for model in mistral gemma qwen phi llama; do
    echo "Running evaluation for $model..."
    python -m evaluator_moe_fewshot.evaluate_moe_fewshot --model $model
done

echo "All evaluations complete!"