#!/bin/bash -l
#SBATCH --output=/users/k24053411/individual_project/fever-final/logs/llama_train.out
#SBATCH --job-name=train_lora_llama
#SBATCH --gres=gpu
#SBATCH --constraint=[a100_80g]
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --signal=USR2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00

# Load required modules
module load python/3.11.6-gcc-13.2.0
module load anaconda3/2022.10-gcc-13.2.0 

# Source bashrc and activate conda environment
source /users/k24053411/.bashrc
source activate fever

# Change to the project directory
cd ~/individual_project/fever-final/

export HF_HOME="/scratch/users/k24053411"

# Run the Python script with specified arguments
CUDA_VISIBLE_DEVICES=0 python trainer/peft_tuning_besstie.py