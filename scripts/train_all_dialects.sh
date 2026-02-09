#!/bin/bash

MODEL_NAME=${1:-qwen} 
CONFIG_DIR="trainer/axolotl_configs/${MODEL_NAME}"
VARIANTS=("sentiment_google" "sarcasm" "sentiment")
DIALECTS=("au" "in" "uk")

echo "Starting sequential training for model: ${MODEL_NAME}"

for variant in "${VARIANTS[@]}"; do
    for dialect in "${DIALECTS[@]}"; do
        if [ "$MODEL_NAME" = "phi_3" ]; then
            CONFIG_FILE="${CONFIG_DIR}/phi_${variant}_${dialect}.yaml"
        elif [ "$MODEL_NAME" = "mistral_7b" ]; then
            CONFIG_FILE="${CONFIG_DIR}/mistral_${variant}_${dialect}.yaml"
        else
            CONFIG_FILE="${CONFIG_DIR}/${MODEL_NAME}_${variant}_${dialect}.yaml"
        fi
        
        echo "Training variant: ${variant}, dialect : ${dialect}"
        echo "Config: ${CONFIG_FILE}"
        echo "Started at: $(date)"
        
        # Run axolotl training
        axolotl train "$CONFIG_FILE" --debug --debug-num-examples 5
        
        TRAIN_STATUS=$?
        
        if [ $TRAIN_STATUS -eq 0 ]; then
            echo "Successfully completed training for: ${variant}, ${dialect}"
        else
            echo "Training failed for: ${variant}, ${dialect} (exit code: $TRAIN_STATUS)"
            echo "Do you want to continue with remaining dialects? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Stopping training pipeline."
                exit $TRAIN_STATUS
            fi
        fi
    done
done

echo "All training jobs completed!"
echo "Finished at: $(date)"