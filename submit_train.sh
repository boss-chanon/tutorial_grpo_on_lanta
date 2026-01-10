#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1                            
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4                     
#SBATCH -t 01:00:00
#SBATCH -A {your_project_id}                
#SBATCH -J tutorial-grpo-train            
#SBATCH -o logs/train_%j.out

module restore
module load Mamba
module load cuda
module load gcc/10.3.0

conda deactivate
conda activate /path/to/your/conda/env

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_TIMEOUT=360000
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=./.cache
export WANDB_MODE="offline"

export HF_HOME=./.cache
export HF_HUB_CACHE=./.cache
export HF_DATASETS_CACHE=./.cache

echo myuser=`whoami`
echo "Total Nodes: ${#NODELIST[@]}"

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --multi_gpu \
    --mixed_precision fp16 \
    train_grpo.py \
        --model_name /path/to/your/model \
        --dataset_path /path/to/your/gms8k/dataset \
        --use_lora True \
        --deepspeed /path/to/your/deepspeed/config \
        --output_dir /path/to/your/output \
        --per_device_train_batch_size 4 \
        --num_generations 8 \
        --bf16 True \
        --save_strategy "steps" \
        --save_steps 10 \