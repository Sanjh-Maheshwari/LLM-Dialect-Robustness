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

from evaluator_moe.moe_services.mistral_moe import MistralMOEClassifier
from evaluator_moe.moe_services.gemma_moe import GemmaMOEClassifier
from evaluator_moe.moe_services.qwen_moe import QwenMOEClassifier
from evaluator_moe.moe_services.phi_moe import PhiMOEClassifier
from evaluator_moe.moe_services.llama_moe import LlamaMOEClassifier

VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sentiment"]
DOMAINS = ["Google"]
NUM_SHOTS = 2  # Number of few-shot examples
TEST_DATA_PATH = "data/instruction/besstie/test.json"
MODEL_CLASSES = {
    "mistral": MistralMOEClassifier,
    "gemma": GemmaMOEClassifier,
    "qwen": QwenMOEClassifier,
    "phi": PhiMOEClassifier,
    "llama": LlamaMOEClassifier,
}

def convert_besstie_to_instruction_format(text, label, task, variety, example_id):
 
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

    available_indices = [i for i in range(len(dialect_df)) if i != current_idx]

    if len(available_indices) < num_shots:
        sampled_indices = available_indices
    else:
        # Set seed based on current_idx to ensure reproducibility across models
        random.seed(42 + current_idx)
        sampled_indices = random.sample(available_indices, num_shots)

    few_shot_examples = []
    for idx in sampled_indices:
        row = dialect_df.iloc[idx]
        few_shot_examples.append({
            'context': row['context'],
            'response': row['response']
        })

    return few_shot_examples


def evaluate_dialect_fewshot(model, variety, task, domain, json_path):
   
    logger.info(f"Evaluating {task}/{domain} for dialect: {variety} (Few-shot: {NUM_SHOTS} examples)")

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
            domain=domain
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
    parser = argparse.ArgumentParser(description="Few-shot evaluation of MixLoRA (MOE) models on BESSTIE dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CLASSES.keys()),
        help="Model to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    logger.info(f"=== MixLoRA (MOE) Few-shot Evaluation ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Few-shot examples: {NUM_SHOTS}")
    logger.info(f"Output directory: {args.output_dir}\n")

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Initializing {args.model} MOE model...")
    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass()

    if model.model is None:
        logger.error(f"Failed to load {args.model} MOE model. Please check your setup.")
        return

    all_results = {}

    for task in TASKS:
        logger.info(f"Evaluating Task: {task}")

        domains = DOMAINS if task == "Sentiment" else ["Reddit"]
        domain_results = {}

        for domain in domains:
            logger.info(f"Domain: {domain}")
            dialect_results = {}

            for variety in VARIETIES:
                result = evaluate_dialect_fewshot(
                    model=model,
                    variety=variety,
                    task=task,
                    domain=domain,
                    json_path=TEST_DATA_PATH,
                )
                dialect_results[variety] = result

            domain_results[domain] = dialect_results

        all_results[task] = domain_results

    total_acc = []
    total_f1 = []

    for task in all_results:
        for domain in all_results[task]:
            for variety in all_results[task][domain]:
                total_acc.append(all_results[task][domain][variety]["acc"])
                total_f1.append(all_results[task][domain][variety]["f1"])

    avg_acc = sum(total_acc) / len(total_acc) if total_acc else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    output_data = {
        "model": args.model,
        "model_type": "mixlora_moe",
        "evaluation_type": "few_shot",
        "num_shots": NUM_SHOTS,
        "sampling_strategy": "random_per_instance_with_seed",
        "seed_formula": "42 + current_idx",
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

    logger.info(f"Summary (Few-shot: {NUM_SHOTS} examples)")
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Average F1: {avg_f1:.4f}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()