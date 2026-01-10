#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 01:00:00
#SBATCH -o logs/eval_%j.out
#SBATCH -A {your_project_id}                
#SBATCH -J tutorial-grpo-eval            

module load Mamba
source activate /path/to/your/conda/env

# Configuration
MODEL_PATH=/path/to/your/model
DATASET_DISK_PATH=/path/to/your/gms8k/dataset

python eval.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_DISK_PATH \
    --num_samples 500