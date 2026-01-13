#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4                     
#SBATCH -t 01:00:00
#SBATCH -A {your_project_id}                
#SBATCH -J tutorial-grpo-eval            
#SBATCH -o logs/eval_%j.out

module load Mamba
source activate /path/to/your/conda/env

python eval.py \
    --model_path /path/to/your/model \
    --lora_path /path/to/your/lora \
    --dataset_path /path/to/your/gms8k/dataset \
    --num_samples 500