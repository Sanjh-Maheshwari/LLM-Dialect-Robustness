import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
from loguru import logger
import os
import json
import random
from accelerate import PartialState

# Define varieties and tasks
VARIETIES = ["en-AU", "en-IN", "en-UK"]
# VARIETIES = ["en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit", "Google"]

class BESTTIEPeftQLoraTrainer:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B"):
        self.model_id = model_id
        self.base_model = None
        self.tokenizer = None
        self.variety_adapters = {}
        
    def get_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bnb_config = BitsAndBytesConfig(  
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= False,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=PartialState().process_index,
            # device_map="auto",
            quantization_config=self.bnb_config,
            trust_remote_code=True,
        )
        
        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        return self.base_model, self.tokenizer

    def apply_template(self, example):
        """Apply ChatML-style template to examples"""
        
        # Combine instruction and context
        if example.get('context', '').strip():
            user_input = f"{example['instruction']}\n{example['context']}"
        else:
            user_input = example['instruction']
        
        
        # Check if we are dealing with instruct model
        if "instruct" in self.model_id.lower():
            
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": example['response']}
            ]

            formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Default : format using ChatML template
        else:    
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
            "context": "",
            "response": assistant_response
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
            "context": text,
            "response": str(label)
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
        variety, 
        task, 
        domain,
        augment_with_finetome=True,
        finetome_samples=3000,
        finetome_seed=42
    ):
        """Prepare dataset for specific variety-task-domain combination"""
        
        # Load BESSTIE data
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
        
        # Apply ChatML template
        dataset = dataset.map(
            self.apply_template,
            batched=False,
            remove_columns=dataset.column_names
        )

        print(dataset[0])
        
        return dataset

    def setup_peft_model(self):
        """Setup PEFT model with LoRA configuration"""

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        peft_model = get_peft_model(self.base_model, peft_config)
        peft_model.enable_input_require_grads()
        peft_model.print_trainable_parameters()
        
        return peft_model, peft_config

    def train_variety_adapter(
        self, 
        variety, 
        task, 
        domain, 
        json_path, 
        output_dir,
        augment_finetome=True,
        finetome_samples=3000
    ):
        """Train LoRA adapter for specific variety-task-domain"""
        
        adapter_name = f"{variety}_{task}_{domain}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Training adapter: {adapter_name}")
        logger.info(f"{'='*60}\n")
        
        # Setup PEFT model
        peft_model, peft_config = self.setup_peft_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset(
            json_path, 
            variety, 
            task, 
            domain,
            augment_with_finetome=augment_finetome,
            finetome_samples=finetome_samples
        )
        
        if dataset is None or len(dataset) == 0:
            logger.warning(f"Skipping {adapter_name} - no data available")
            return
        
        logger.info(f"Training on {len(dataset)} samples")
        
        # Training arguments (matching paper specs)
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{adapter_name}_adapter",
            num_train_epochs=5,  # As per paper
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,  # Effective batch size = 8
            learning_rate=3e-4,  # From paper's grid search
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="no", 
            optim="paged_adamw_32bit",
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            weight_decay=0.01,
            fp16=False,
            bf16=False,
            report_to=None,
            remove_unused_columns=False,
            group_by_length=True,
            dataloader_pin_memory=False, 
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=peft_model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save adapter
        adapter_output_dir = f"{output_dir}/{adapter_name}_adapter"
        os.makedirs(adapter_output_dir, exist_ok=True)
        peft_model.save_pretrained(adapter_output_dir)
        
        self.variety_adapters[adapter_name] = adapter_output_dir
        logger.info(f"Adapter saved to {adapter_output_dir}")
        
        # Clean up
        del peft_model
        del trainer
        torch.cuda.empty_cache()

    def train_all_adapters(
        self, 
        json_path, 
        output_dir="./output/besstie_adapters",
        augment_finetome=True,
        finetome_samples_per_variety=3000
    ):
        """Train all variety-task-domain adapter combinations"""
        
        # Initialize base model
        self.get_model_and_tokenizer()
        
        total_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f'Total parameters: {total_params:,}')
        
        # Train each combination
        for variety in VARIETIES:
            for task in TASKS:
                # Google only has sentiment
                domains = DOMAINS if task == "Sentiment" else ["Reddit"]
                
                for domain in domains:
                    self.train_variety_adapter(
                        variety=variety,
                        task=task,
                        domain=domain,
                        json_path=json_path,
                        output_dir=output_dir,
                        augment_finetome=augment_finetome,
                        finetome_samples=finetome_samples_per_variety
                    )
        
        logger.info("\n" + "="*60)
        logger.info("All adapters training completed!")
        logger.info("="*60)
        
        # Save registry
        registry_path = os.path.join(output_dir, "adapter_registry.json")
        with open(registry_path, 'w') as f:
            json.dump(self.variety_adapters, f, indent=2)
        logger.info(f"Adapter registry saved to {registry_path}")

def main():
    
    # Initialize trainer
    trainer = BESTTIEPeftQLoraTrainer("mistralai/Mistral-Small-Instruct-2409")
    
    # Train all adapters with FineTome augmentation
    trainer.train_all_adapters(
        json_path="data/instruction/besstie.json",
        output_dir="output/besstie_adapters_multi",
        augment_finetome=True,
        finetome_samples_per_variety=2000
    )

if __name__ == "__main__":
    main()