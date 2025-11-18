#!/bin/bash -l
#SBATCH --output=/users/k24053411/individual_project/logs/axolotl/gemma_sentiment_uk_ds.out
#SBATCH --job-name=gemma_sentiment_uk_ds
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
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

axolotl train trainer/axolotl_configs/gemma/gemma_sentiment_uk_ds.yaml