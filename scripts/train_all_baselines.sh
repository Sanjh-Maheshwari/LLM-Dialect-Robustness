#!/bin/bash

MODEL_NAME=${1:-qwen}
CONFIG_DIR="trainer/axolotl_configs_baseline/${MODEL_NAME}"
VARIANTS=("sentiment_google" "sarcasm" "sentiment")

echo "Starting sequential baseline training for model: ${MODEL_NAME}"

for variant in "${VARIANTS[@]}"; do
    
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

    echo "Training variant: ${variant}"
    echo "Config: ${CONFIG_FILE}"
    echo "Started at: $(date)"

    # Run axolotl training
    axolotl train "$CONFIG_FILE" --debug --debug-num-examples 5

    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -eq 0 ]; then
        echo "Successfully completed training for: ${variant}"
    else
        echo "Training failed for: ${variant} (exit code: $TRAIN_STATUS)"
        echo "Do you want to continue with remaining variants? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping training pipeline."
            exit $TRAIN_STATUS
        fi
    fi
done

echo "All baseline training jobs completed!"
echo "Finished at: $(date)"
