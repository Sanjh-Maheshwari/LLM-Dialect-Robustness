#!/bin/bash

# Sequential training script for baseline configs
# Usage: ./train_all_baselines.sh [model_name]

MODEL_NAME=${1:-qwen}  # Default to qwen if no argument provided
CONFIG_DIR="trainer/axolotl_configs_baseline/${MODEL_NAME}"
VARIANTS=("sarcasm_google" "sarcasm" "sentiment")

echo "Starting sequential baseline training for model: ${MODEL_NAME}"
echo "================================================"

for variant in "${VARIANTS[@]}"; do
    # Handle naming convention for different models
    if [ "$MODEL_NAME" = "phi_3" ]; then
        CONFIG_FILE="${CONFIG_DIR}/phi_${variant}.yaml"
    elif [ "$MODEL_NAME" = "mistral_7b" ]; then
        CONFIG_FILE="${CONFIG_DIR}/mistral_${variant}.yaml"
    else
        CONFIG_FILE="${CONFIG_DIR}/${MODEL_NAME}_${variant}.yaml"
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        continue
    fi

    echo ""
    echo "Training variant: ${variant}"
    echo "Config: ${CONFIG_FILE}"
    echo "Started at: $(date)"
    echo "------------------------------------------------"

    # Run axolotl training
    axolotl train "$CONFIG_FILE" --debug --debug-num-examples 5

    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -eq 0 ]; then
        echo "✓ Successfully completed training for: ${variant}"
        echo "Finished at: $(date)"
    else
        echo "✗ Training failed for: ${variant} (exit code: $TRAIN_STATUS)"
        echo "Do you want to continue with remaining variants? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping training pipeline."
            exit $TRAIN_STATUS
        fi
    fi
done

echo ""
echo "================================================"
echo "All baseline training jobs completed!"
echo "Finished at: $(date)"
