import argparse
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Constants
REASONING_START, REASONING_END = "<think>", "</think>"  # Fixed typo in </think>
ANSWER_START, ANSWER_END = "<answer>", "</answer>"
SYSTEM_PROMPT = f"""
Respond in the following format:

{REASONING_START}
...
{REASONING_END}
{ANSWER_START}
...
{ANSWER_END}
"""


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags."""
    try:
        if ANSWER_START in text and ANSWER_END in text:
            return text.split(ANSWER_START)[-1].split(ANSWER_END)[0].strip()
        elif ANSWER_START in text:
            return text.split(ANSWER_START)[-1].strip()
        return ""
    except IndexError:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged model or base model",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset folder"
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    args = parser.parse_args()

    # 1. Load Dataset from Disk
    print(f"📂 Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    if hasattr(dataset, "keys") and "test" in dataset:
        dataset = dataset["test"]

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    # 2. Setup Transformers Pipeline
    print(f"🚀 Initializing Transformers with model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Ensure pad_token is set (crucial for batching in Transformers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # 3. Prepare Data for Pipeline
    prompts = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": example.get("question", example.get("problem", "")),
            },
        ]
        # Transformers pipeline handles the chat template and generation prompt
        prompts.append(messages)

    # 4. Generate Answers
    print(f"📝 Generating answers for {len(prompts)} samples...")
    # Note: Use batch_size here to speed up inference if your GPU supports it
    results = generator(
        prompts,
        max_new_tokens=512,
        temperature=0.001,  # Transformers doesn't support 0.0 directly in all configs
        do_sample=False,  # Equivalent to temperature 0 (greedy)
        batch_size=args.batch_size,
        return_full_text=False,
        use_cache=True,
    )

    # 5. Calculate Accuracy
    correct = 0
    for i, output in enumerate(results):
        # results is a list of lists because of batching/multiple returns
        generated_text = output[0]["generated_text"]
        extracted = extract_xml_answer(generated_text)

        gold_raw = dataset[i].get("answer", "")
        gold = (
            gold_raw.split("####")[-1].strip()
            if "####" in gold_raw
            else gold_raw.strip()
        )

        if extracted == gold:
            correct += 1

    print(f"\n📊 Evaluation Results:")
    print(f"   Accuracy: {correct/len(dataset):.2%} ({correct}/{len(dataset)})")


if __name__ == "__main__":
    main()
