#!/bin/bash -l
#SBATCH --output=/users/k24053411/individual_project/fever-final/logs/llama_eval.out
#SBATCH --job-name=evaluate_testset
#SBATCH --gres=gpu:1
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
source activate fever

# Change to the project directory
cd ~/individual-project/fever-final

export HF_HOME="/scratch/users/k24053411"

# Run the Python script with specified arguments
CUDA_VISIBLE_DEVICES=0 python -m evaluator.evaluate_llama_v1