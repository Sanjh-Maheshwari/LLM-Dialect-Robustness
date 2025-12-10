#!/bin/bash

# Convenience script to run all evaluations
# Usage: bash evaluator_unified/run_all.sh

echo "========================================"
echo "Running All LoRA Merging Evaluations"
echo "========================================"
echo ""
echo "This will run 18 evaluations:"
echo "  - 6 models: phi, mistral7b, mistral2409, qwen, llama, gemma2"
echo "  - 3 methods: lora_grouping, cat, ties"
echo ""
echo "Press Ctrl+C to cancel..."
sleep 3

python evaluator_unified/run_all_evaluations.py

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "Check results in: results_besstie_unified/"
echo "========================================"
