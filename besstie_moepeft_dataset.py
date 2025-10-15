from datasets import load_dataset, Dataset
from loguru import logger
import os
import json
import random

# Define varieties and tasks
VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit", "Google"]

class BESTTIEDataset:
    def __init__(self):
        pass

    def apply_template(self, example):
        """Apply ChatML-style template to examples"""
        
        # Combine instruction and context
        if example.get('context', '').strip():
            user_input = f"{example['instruction']}\n{example['context']}"
        else:
            user_input = example['instruction']
        
        # Format using ChatML template
        formatted_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        return tokenized

    def load_finetome_samples(self, num_samples=1000, seed=42):
        """Load samples from FineTome-100k dataset"""
        
        try:
            finetome_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
            random.seed(seed)
            
            if num_samples >= len(finetome_dataset):
                sampled_dataset = finetome_dataset
            else:
                indices = random.sample(range(len(finetome_dataset)), num_samples)
                sampled_dataset = finetome_dataset.select(indices)
            
            return sampled_dataset
        
        except Exception as e:
            logger.error(f"Error loading FineTome dataset: {e}")
            return None

    def convert_finetome_to_format(self, finetome_sample, variety, task, example_id_offset=0):
        """Convert FineTome sample to instruction format"""
        
        conversations = finetome_sample['conversations']
        
        user_input = ""
        assistant_response = ""
        
        for conv in conversations:
            if conv['from'] in ['human', 'user']:
                user_input = conv['value']
            elif conv['from'] in ['gpt', 'assistant']:
                assistant_response = conv['value']
                break
        
        # Create sample in instruction format
        sample = {
            "example_id": example_id_offset,
            "variety": variety,
            "task": task,
            "instruction": user_input,
            "input": "",
            "output": assistant_response
        }
        
        return sample

    def convert_besstie_to_instruction_format(self, text, label, task, variety, example_id):
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
            "input": text,
            "output": str(label)
        }

    def load_besstie_data(self, json_path, variety, task, domain):
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
            # Adjust based on your actual JSON structure
            # Assuming item is a dict with 'text' and 'label' keys
            sample = self.convert_besstie_to_instruction_format(
                text=item['text'] if isinstance(item, dict) else item,
                label=item['label'] if isinstance(item, dict) else 0,  # Adjust accordingly
                task=task,
                variety=variety,
                example_id=idx
            )
            samples.append(sample)
        
        return samples

    def augment_with_finetome(self, original_samples, variety, task, num_samples=1000, seed=42):
        """Augment BESSTIE samples with FineTome for better instruction tuning"""
        
        logger.info(f"Loading {num_samples} samples from FineTome-100k for augmentation...")
        finetome_samples = self.load_finetome_samples(num_samples, seed)
        
        if finetome_samples is None:
            logger.warning("Failed to load FineTome. Using only BESSTIE data.")
            return original_samples
        
        augmented_samples = original_samples.copy()
        
        # Find max ID to avoid conflicts
        max_id = max([s['example_id'] for s in original_samples]) if original_samples else 0
        
        # Add FineTome samples
        logger.info(f"Adding {len(finetome_samples)} FineTome samples for {variety}/{task}")
        for i, sample in enumerate(finetome_samples):
            converted_sample = self.convert_finetome_to_format(
                sample, variety, task, max_id + i + 1
            )
            augmented_samples.append(converted_sample)
        
        logger.info(f"Original: {len(original_samples)}, Augmented: {len(augmented_samples)}")
        
        return augmented_samples

    def prepare_dataset(
        self, 
        json_path, 
        task, 
        domain,
        variety = "all", 
        augment_with_finetome=True,
        finetome_samples=3000,
        finetome_seed=42
    ):
        """Prepare dataset for specific variety-task-domain combination"""
        
        # Load BESSTIE data
        if variety == "all": 
            samples = []
            for variety in VARIETIES:
                samples.extend(self.load_besstie_data(json_path, variety, task, domain))
        else:
            samples = self.load_besstie_data(json_path, variety, task, domain)
        
        if not samples:
            logger.warning(f"No BESSTIE samples found for {variety}/{task}/{domain}")
            return None
        
        # Augment with FineTome
        if augment_with_finetome:
            samples = self.augment_with_finetome(
                samples, 
                variety, 
                task,
                num_samples=finetome_samples,
                seed=finetome_seed
            )
        
        # Convert to HF Dataset
        dataset = Dataset.from_list(samples)
        
        # No need to apply template for MoE-PEFT
        # dataset = dataset.map(
        #     self.apply_template,
        #     batched=False,
        #     remove_columns=dataset.column_names
        # )

        json_data = []
        for i, _ in enumerate(dataset):
            json_data.append(dataset[i])
        
        return json_data

    def generate_all_data(
        self, 
        json_path, 
        output_dir,
        augment_finetome=True,
        finetome_samples_per_variety=3000
    ):
        """Generate dataset for all dialects in single dataset but separate task and domains """
        
        # Generate dataset for each combination 
        for task in TASKS:
            # Google only has sentiment
            domains = DOMAINS if task == "Sentiment" else ["Reddit"]
            
            for domain in domains:
                dataset = self.prepare_dataset(
                    json_path=json_path,
                    task=task,
                    domain=domain,
                    augment_with_finetome=augment_finetome,
                    finetome_samples=finetome_samples_per_variety
                )
                
                # Save the dataset
                ds_path = os.path.join(output_dir, task, domain, "merged.json")
                with open(ds_path, "w") as f:
                    json.dump(dataset, f, indent=2)

        logger.info("="*60)
        logger.info("Training data generation completed!")
        logger.info("="*60)


def main():

    dataset_generator = BESTTIEDataset()

    dataset_generator.generate_all_data(
        json_path="data/instruction/besstie.json",
        output_dir="MOE_peft/dialect_data/besstie",
        augment_finetome=True,
        finetome_samples_per_variety=2000
    )

if __name__ == "__main__":
    main()