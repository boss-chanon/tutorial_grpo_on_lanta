# Tutorial GRPO on Lanta

This project implements Group Relative Policy Optimization (GRPO) for Reinforcement learning with Verifiable Rewards (RLVR), specifically optimized for mathematical reasoning tasks like GSM8K.

## Setup Instructions
1. Clone the Repository

    First, clone the GRPO tutorial repository to your project directory on Lanta:
    ```bash
    git clone https://github.com/boss-chanon/tutorial_grpo_on_lanta.git
    ```

2. Environment Preparation

    Create a Conda environment on Lanta and install the required stack:
    ```bash
    module load Mamba
    conda create -p ./env python=3.11 -y
    conda activate ./env

    # Install TRL with vLLM support
    pip install -r requirements.txt
    ```

3. Download Data & Model to local

    Supercomputer compute nodes often lack internet access. Use the Hugging Face CLI to download resources to a shared project directory:
    ```bash
    #!/bin/bash

    module load Mamba
    conda activate ./env

    huggingface-cli download Qwen/Qwen3-1.7B --cache-dir ./hf_cache --local-dir /path/to/your/model

    python3 -c "from datasets import load_dataset; \
    data = load_dataset('openai/gsm8k', 'main'); \
    data.save_to_disk('/path/to/your/gms8k/dataset')" 
    ```

## Configuration
### Reward Functions
The trainer utilizes four specific reward functions to guide the model:

- Correctness Reward: Validates the extracted XML answer against the ground truth (Weight: 2.0).
- Format Reward: Ensures the model uses <think>...</think> and <answer>...</answer> tags (Weight: 0.5).
- Integer Reward: Incentivizes providing numeric answers for math problems (Weight: 0.5).
- XML Count Reward: Penalizes repetitive or malformed XML structures.

### LoRA vs. Full Fine-Tuning
You can toggle LoRA by passing `--use_lora` `True` or `False`. The default configuration uses `r=16` and `α=64` targeting all linear modules.

## Running the Job
### Training
Submit the job to Slurm:
```bash
sbatch submit_train.sh
```
#### Key Training Arguments
Since the script inherits from GRPOConfig, you can modify these directly in the .sh file:

- `--num_generations`: Number of completions to sample per prompt (Default: 8).
- `--per_device_train_batch_size`: Batch size per GPU (Default: 4).

### Evaluation
Submit the job to Slurm:
```bash
sbatch submit_eval.sh
```

## Inference
After training, the final LoRA adapter is saved to your output directory. You can load it for inference using `PeftModel`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("/path/to/your/original/model")
model = PeftModel.from_pretrained(base_model, "/path/to/your/final/model")
```