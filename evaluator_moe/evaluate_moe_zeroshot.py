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

warnings.filterwarnings('ignore')

# Import MOE model classifiers
from moe_services.mistral_moe import MistralMOEClassifier
from moe_services.gemma_moe import GemmaMOEClassifier
from moe_services.qwen_moe import QwenMOEClassifier
from moe_services.phi_moe import PhiMOEClassifier
from moe_services.llama_moe import LlamaMOEClassifier

VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit", "Google"]

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


def evaluate_dialect(model, variety, task, domain, json_path):
   
    logger.info(f"Evaluating {task}/{domain} for dialect: {variety} (Zero-shot)")

    dialect_df = load_besstie_data(json_path, variety, task, domain)

    if len(dialect_df) == 0:
        logger.warning(f"No data found for {variety}/{task}/{domain}")
        return {"acc": 0.0, "f1": 0.0}

    predictions = []
    for i, row in tqdm(dialect_df.iterrows(), total=dialect_df.shape[0], desc=f"{variety}"):
        instruction = row['instruction']
        context = row['context']

        prediction = model.predict(
            instruction=instruction,
            context=context,
            dialect=variety,
            task=task,
            domain=domain
        )
        predictions.append(prediction)

    true_labels = dialect_df['response'].tolist()

    average = "weighted"
    accuracy = accuracy_score(true_labels, predictions)

    logger.warning(f"Using following average for f1 : {average}")
    f1 = f1_score(true_labels, predictions, average=average, zero_division=0)

    logger.info(f"Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "acc": round(accuracy, 4),
        "f1": round(f1, 4)
    }


def main():
    
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of MixLoRA (MOE) models on BESSTIE dataset")
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

    logger.info(f"=== MixLoRA (MOE) Zero-shot Evaluation ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Evaluation type: Zero-shot")
    logger.info(f"Output directory: {args.output_dir}")

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass()

    if model.model is None:
        logger.error(f"Failed to load {args.model} MOE model. Please check your setup.")
        return

    # Evaluation loop
    all_results = {}

    for task in TASKS:
        logger.info(f"Evaluating Task: {task}")

        domains = DOMAINS if task == "Sentiment" else ["Reddit"]
        domain_results = {}

        for domain in domains:
            logger.info(f"\nDomain: {domain}")
            dialect_results = {}

            for variety in VARIETIES:
                result = evaluate_dialect(
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
        "evaluation_type": "zero_shot",
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

    logger.info(f"Results Summary (Zero-shot)")
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Average F1: {avg_f1:.4f}")
    logger.info(f"\Results saved to: {output_file}")

if __name__ == "__main__":
    main()
