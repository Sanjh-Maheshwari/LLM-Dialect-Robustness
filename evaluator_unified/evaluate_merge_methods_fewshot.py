"""
Few-shot evaluation script for LoRA merging methods
Supports 6 models: Phi-3, Mistral 7B, Mistral 2409, Qwen, LLaMA, Gemma 2

Methods:
- lora_grouping: Single adapter trained on all dialects
- cat: Merge 3 dialect adapters using concatenation
- ties: Merge 3 dialect adapters using TIES
- base_instruct: No adapter, just the base instruct model
- individual_dialect: Evaluate each dialect adapter separately

For each test instance, samples 2 random examples from the same dialect/task/domain
from the test set to use as few-shot examples.

Usage:
    python evaluate_merge_methods_fewshot.py --model phi --method lora_grouping
    python evaluate_merge_methods_fewshot.py --model mistral7b --method cat
    python evaluate_merge_methods_fewshot.py --model qwen --method ties
    python evaluate_merge_methods_fewshot.py --model llama --method base_instruct
    python evaluate_merge_methods_fewshot.py --model gemma2 --method individual_dialect
"""

import json
import os
import argparse
from datetime import datetime
from pprint import pprint
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from loguru import logger
import warnings
import random

warnings.filterwarnings('ignore')

# Import model classifiers
from evaluator_unified.llm_services.phi_unified import Phi3UnifiedClassifier
from evaluator_unified.llm_services.mistral7b_unified import Mistral7BUnifiedClassifier
from evaluator_unified.llm_services.mistral2409_unified import Mistral2409UnifiedClassifier
from evaluator_unified.llm_services.qwen_unified import QwenUnifiedClassifier
from evaluator_unified.llm_services.llama_unified import LlamaUnifiedClassifier
from evaluator_unified.llm_services.gemma2_unified import Gemma2UnifiedClassifier

# Constants
VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit"]
NUM_SHOTS = 2  # Number of few-shot examples

TEST_DATA_PATH = "data/instruction/besstie/test.json"

# Model mapping
MODEL_CLASSES = {
    "phi": Phi3UnifiedClassifier,
    "mistral7b": Mistral7BUnifiedClassifier,
    "mistral2409": Mistral2409UnifiedClassifier,
    "qwen": QwenUnifiedClassifier,
    "llama": LlamaUnifiedClassifier,
    "gemma2": Gemma2UnifiedClassifier,
}

VALID_METHODS = ["lora_grouping", "cat", "ties", "base_instruct", "individual_dialect"]


def convert_besstie_to_instruction_format(text, label, task, variety, example_id):
    """Convert BESSTIE classification data to instruction-following format"""

    if task == "Sentiment":
        instruction = (
            "Generate the sentiment of the given text. "
            "1 for positive sentiment, and 0 for negative sentiment. "
            "Do not give an explanation."
        )
    else:  # Sarcasm
        instruction = (
            "Predict if the given text is sarcastic. "
            "1 if the text is sarcastic, and 0 if the text is not sarcastic. "
            "Do not give an explanation."
        )

    return {
        "example_id": example_id,
        "variety": variety,
        "task": task,
        "instruction": instruction,
        "context": text,
        "response": str(label)
    }


def load_besstie_data(json_path, variety, task, domain):
    """Load data from BESSTIE JSON structure"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    try:
        variety_data = data[task][domain][variety]
    except KeyError:
        logger.warning(f"No data found for {task}/{domain}/{variety}")
        return []

    samples = []
    for idx, item in enumerate(variety_data):
        sample = convert_besstie_to_instruction_format(
            text=item['text'] if isinstance(item, dict) else item,
            label=item['label'] if isinstance(item, dict) else 0,
            task=task,
            variety=variety,
            example_id=idx
        )
        samples.append(sample)

    df = pd.DataFrame.from_records(samples)

    return df


def sample_fewshot_examples(dialect_df, current_idx, num_shots=NUM_SHOTS):
    """
    Sample random few-shot examples from the test set, excluding the current test instance

    Args:
        dialect_df: DataFrame containing all test examples for the dialect/task/domain
        current_idx: Index of the current test instance (to exclude from sampling)
        num_shots: Number of examples to sample (default: 2)

    Returns:
        List of dicts with 'context' and 'response' keys
    """
    # Get all indices except the current one
    available_indices = [i for i in range(len(dialect_df)) if i != current_idx]

    # Sample random indices
    if len(available_indices) < num_shots:
        # If not enough examples, use what we have
        sampled_indices = available_indices
    else:
        sampled_indices = random.sample(available_indices, num_shots)

    # Extract examples
    few_shot_examples = []
    for idx in sampled_indices:
        row = dialect_df.iloc[idx]
        few_shot_examples.append({
            'context': row['context'],
            'response': row['response']
        })

    return few_shot_examples


def evaluate_dialect_fewshot(model, variety, task, domain, method, json_path):
    """Evaluate model on a specific dialect using few-shot prompting"""

    logger.info(f"Evaluating {task}/{domain} for dialect: {variety} using method: {method} (Few-shot: {NUM_SHOTS} examples)")

    dialect_df = load_besstie_data(json_path, variety, task, domain)

    if len(dialect_df) == 0:
        logger.warning(f"No data found for {variety}/{task}/{domain}")
        return {"acc": 0.0, "f1": 0.0}

    predictions = []
    for i, row in tqdm(dialect_df.iterrows(), total=dialect_df.shape[0], desc=f"{variety}"):
        instruction = row['instruction']
        context = row['context']

        # Sample few-shot examples (excluding current instance)
        few_shot_examples = sample_fewshot_examples(dialect_df, i, num_shots=NUM_SHOTS)

        prediction = model.predict_fewshot(
            instruction=instruction,
            context=context,
            few_shot_examples=few_shot_examples,
            dialect=variety,
            task=task,
            domain=domain,
            method=method
        )
        predictions.append(prediction)

    true_labels = dialect_df['response'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    logger.info(f"Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "acc": round(accuracy, 4),
        "f1": round(f1, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Few-shot evaluation of LoRA merging methods on BESSTIE dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CLASSES.keys()),
        help="Model to evaluate"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=VALID_METHODS,
        help="Merging method to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_besstie_unified_fewshot",
        help="Output directory for results"
    )

    args = parser.parse_args()

    logger.info(f"=== Unified LoRA Merging Evaluation (Few-shot) ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Few-shot examples: {NUM_SHOTS}")
    logger.info(f"Output directory: {args.output_dir}\n")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.model, args.method)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass()

    if model.model is None:
        logger.error(f"Failed to load {args.model} model. Please check your setup.")
        return

    # Evaluation loop
    all_results = {}

    for task in TASKS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Task: {task}")
        logger.info(f"{'='*60}")

        domains = DOMAINS if task == "Sentiment" else ["Reddit"]
        domain_results = {}

        for domain in domains:
            logger.info(f"\nDomain: {domain}")
            dialect_results = {}

            for variety in VARIETIES:
                result = evaluate_dialect_fewshot(
                    model=model,
                    variety=variety,
                    task=task,
                    domain=domain,
                    method=args.method,
                    json_path=TEST_DATA_PATH,
                )
                dialect_results[variety] = result

            domain_results[domain] = dialect_results

        all_results[task] = domain_results

    # Compute average metrics
    logger.info(f"\n{'='*60}")
    logger.info("Computing average metrics...")
    logger.info(f"{'='*60}")

    total_acc = []
    total_f1 = []

    for task in all_results:
        for domain in all_results[task]:
            for variety in all_results[task][domain]:
                total_acc.append(all_results[task][domain][variety]["acc"])
                total_f1.append(all_results[task][domain][variety]["f1"])

    avg_acc = sum(total_acc) / len(total_acc) if total_acc else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    # Determine adapter type based on method
    if args.method == "lora_grouping":
        adapter_type = "baseline_single"
    elif args.method in ["cat", "ties"]:
        adapter_type = "merged_3dialects"
    elif args.method == "base_instruct":
        adapter_type = "none_base_model_only"
    elif args.method == "individual_dialect":
        adapter_type = "individual_dialect_adapter"
    else:
        adapter_type = "unknown"

    # Prepare output structure
    output_data = {
        "model": args.model,
        "method": args.method,
        "evaluation_type": "few_shot",
        "num_shots": NUM_SHOTS,
        "sampling_strategy": "random_per_instance",
        "adapter_type": adapter_type,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "varieties": VARIETIES,
            "tasks": TASKS,
            "domains": DOMAINS
        },
        "results": all_results,
        "summary": {
            "average_accuracy": round(avg_acc, 4),
            "average_f1": round(avg_f1, 4)
        }
    }

    # Save results
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results Summary (Few-shot: {NUM_SHOTS} examples)")
    logger.info(f"{'='*60}")
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Average F1: {avg_f1:.4f}")
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"{'='*60}\n")

    # Print full results
    pprint(output_data)


if __name__ == "__main__":
    main()
