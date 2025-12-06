#!/bin/bash

# Sequential training script for multiple dialect configs
# Usage: ./train_all_dialects.sh [model_name]

MODEL_NAME=${1:-qwen}  # Default to qwen if no argument provided
CONFIG_DIR="trainer/axolotl_configs/${MODEL_NAME}"
DIALECTS=("au" "in" "uk")

echo "Starting sequential training for model: ${MODEL_NAME}"
echo "================================================"

for dialect in "${DIALECTS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/${MODEL_NAME}_sarcasm_${dialect}_google.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        continue
    fi
    
    echo ""
    echo "Training dialect: ${dialect}"
    echo "Config: ${CONFIG_FILE}"
    echo "Started at: $(date)"
    echo "------------------------------------------------"
    
    # Run axolotl training
    axolotl train "$CONFIG_FILE" --debug --debug-num-examples 5
    
    TRAIN_STATUS=$?
    
    if [ $TRAIN_STATUS -eq 0 ]; then
        echo "✓ Successfully completed training for: ${dialect}"
        echo "Finished at: $(date)"
    else
        echo "✗ Training failed for: ${dialect} (exit code: $TRAIN_STATUS)"
        echo "Do you want to continue with remaining dialects? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping training pipeline."
            exit $TRAIN_STATUS
        fi
    fi
done

echo ""
echo "================================================"
echo "All training jobs completed!"
echo "Finished at: $(date)"