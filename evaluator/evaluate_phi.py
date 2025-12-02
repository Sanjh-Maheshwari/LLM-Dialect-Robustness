import json
from datasets import load_dataset, Dataset
from loguru import logger
import os
import json
import random
import warnings
from pprint import pprint
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from evaluator.llm_services.phi_w_adapters import Phi3Classifier

warnings.filterwarnings('ignore')
    
VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit"]

TEST_DATA_PATH = "data/instruction/besstie/test.json"
RESULTS_DIR = "results_besstie/v1/phi"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
            label=item['label'] if isinstance(item, dict) else 0,  # Adjust accordingly
            task=task,
            variety=variety,
            example_id=idx
        )
        samples.append(sample)

    df = pd.DataFrame.from_records(samples)
    
    return df

def evaluate_dialect(model, variety, task, domain, json_path): 

    logger.info(f"Evaluating {task}, {domain} for dialect : {variety}")

    dialect_df = load_besstie_data(json_path, variety, task, domain)

    predictions = []
    for i, row in tqdm(dialect_df.iterrows(), total=dialect_df.shape[0]):
        instruction = row['instruction']
        context = row['context']

        prediction = model.predict(
            instruction = instruction, 
            context = context, 
            dialect = variety,
            task = task,
            domain = domain 
        )
        predictions.append(prediction) 

    true_labels = dialect_df['response'].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "acc" : accuracy,
        "f1" : f1
    }

def main():
    logger.info("=== Besstie Dialect Evaluation with Phi 3 ===\n")

    all_results = {}    

    logger.info("Initializing Phi 3 model...")
    model = Phi3Classifier()

    if model.model is None:
        print("Failed to load Phi 3 model. Please check your setup.")
        return
    
    for task in TASKS:
        domains = DOMAINS if task == "Sentiment" else ["Reddit"]
        
        domain_results = {}
        
        for domain in domains:

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

            # TODO : can also store the average results ? 
            domain_results[domain] = dialect_results
        
        all_results[task] = domain_results 
    
    pprint(all_results)

if __name__ == "__main__": 
    main()