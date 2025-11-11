#!/bin/bash -l
#SBATCH --output=/users/k24053411/individual_project/logs/axolotl/mistral_train_sarcasm.out
#SBATCH --job-name=train_moe_lora_mistral_sarcasm
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_80g
#SBATCH --gpus=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --signal=USR2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00

# Load required modules
module load python/3.11.6-gcc-13.2.0
module load anaconda3/2022.10-gcc-13.2.0 

# Source bashrc and activate conda environment
source /users/k24053411/.bashrc
source activate axolotl

# Change to the project directory
cd /users/k24053411/individual_project/LLM-Dialect-Robustness

export HF_HOME="/scratch/users/k24053411"

axolotl train trainer/axolotl_configs/mistral.yaml