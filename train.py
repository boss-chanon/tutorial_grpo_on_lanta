import os
import re
import torch
from dataclasses import dataclass, field
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

import os

REASONING_START = "<think>"
REASONING_END = "</thnk>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

SYSTEM_PROMPT = f"""
Respond in the following format:

{REASONING_START}
...
{REASONING_END}
{ANSWER_START}
...
{ANSWER_END}
"""
COT_FORMAT = f"""
{REASONING_START}
{{reasoning}}
{REASONING_END}
{ANSWER_START}
{{answer}}
{ANSWER_END}
"""

# One-shot examples
ONE_SHOT_EXAMPLES = {
    "question": "What is the largest single-digit prime number?",
    "reasoning": "9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
    "answer": "7",
}


@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    dataset_path: str = field(default=None, metadata={"help": "Local path to dataset"})
    use_lora: bool = field(default=True)


def extract_xml_answer(text: str) -> str:
    try:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    except:
        return ""


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K format (#### answer)"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from MATH format (\boxed{answer})"""
    pattern = r"\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def load_and_prepare_dataset(
    dataset_path: str, use_one_shot: bool = True, max_samples: int = None
):
    """Load and prepare dataset with chat formatting."""

    # Load dataset
    data = load_from_disk(dataset_path)["train"]
    question_field = "question"
    answer_fn = lambda x: extract_hash_answer(x["answer"])

    def format_example(x):
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add one-shot example
        if use_one_shot:
            ex = ONE_SHOT_EXAMPLES
            prompt.extend(
                [
                    {"role": "user", "content": ex["question"]},
                    {
                        "role": "assistant",
                        "content": COT_FORMAT.format(
                            reasoning=ex["reasoning"], answer=ex["answer"]
                        ),
                    },
                ]
            )

        prompt.append({"role": "user", "content": x[question_field]})
        return {"prompt": prompt, "answer": answer_fn(x)}

    # Format dataset
    formatted = data.map(format_example)

    # Limit samples if specified
    if max_samples and len(formatted) > max_samples:
        formatted = formatted.shuffle(seed=42).select(range(max_samples))

    return formatted


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


def format_reward_func(completions, **kwargs):
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load local data
    dataset = load_and_prepare_dataset(script_args.dataset_path)
    if isinstance(dataset, dict):
        dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Logic
    peft_config = None
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=16, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"
        )

    print(dataset)

    trainer = GRPOTrainer(
        model=script_args.model_name,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func, format_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
