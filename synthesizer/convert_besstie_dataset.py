import pandas as pd
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
import warnings
from datasets import load_dataset
import random
from loguru import logger

warnings.filterwarnings('ignore')

OUTPUT_JSON_PATH = "data/instruction/besstie_instruction_format.json"


def load_besstie_dataset(sample_size = None): 

    try: 
        print("Loading Besstie dataset from Hugging Face...")
        dataset = load_dataset("unswnlporg/BESSTIE", split =  "train")

        print(f"Loaded Dolly dataset with {len(dataset)} total samples")
        print(f"Dataset columns: {list(dataset.features.keys())}")

        # Sample if requested
        if sample_size and sample_size < len(dataset):
            
            # Convert to pandas for easier sampling
            df = dataset.to_pandas()
            sampled_df = df.sample(n=sample_size, random_state=RANDOM_SEED)
            print(f"Sampled {sample_size} entries from {len(dataset)} total entries")
            return sampled_df
        else:
            print(f"Using all {len(dataset)} entries")
            return dataset.to_pandas()
            
    except Exception as e:
        print(f"Error loading Dolly dataset: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
        return None

def split_and_convert(train_df): 

    tasks = train_df['task'].unique()

    dialect_mapping = {
        "en-AU" : "Australian_English",
        "en-IN" : "Indian_English",
        "en-UK" : "British_English"
    }

    for task in tasks:

        task_df = df[df['task'] == task].copy()
        task_name = task.lower()

        examples = []

        for idx, row in df.iterrows():

            dialect = row["varaiety"]

            example = {
                "example_id" : idx, 
                "category" : task_name,
                
            }
