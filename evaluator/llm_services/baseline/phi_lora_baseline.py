import pandas as pd
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import warnings
from loguru import logger
from peft import PeftModel, PeftConfig
from transformers import StoppingCriteria, StoppingCriteriaList

ADAPTER_DIR = "/scratch/users/k24053411/axolotl/phi_3/baseline"

warnings.filterwarnings('ignore')

class Phi3StoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<|end|>", "<|user|>", "<|assistant|>"]
        
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        
        self.stop_token_ids = []
        for stop_string in stop_strings:
            # Encode the stop string
            tokens = tokenizer.encode(stop_string, add_special_tokens=False)
            if tokens: 
                self.stop_token_ids.append(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_tokens in self.stop_token_ids:
            stop_length = len(stop_tokens)
            if input_ids.shape[1] >= stop_length:
                last_tokens = input_ids[0, -stop_length:].tolist()
                if last_tokens == stop_tokens:
                    return True
        return False

class Phi3Classifier:
    """Phi-3 model for fact checking"""
    
    def __init__(self, model_id="microsoft/Phi-3-medium-4k-instruct"):
        logger.info(f"Loading {model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token            
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # trust_remote_code=True,
                low_cpu_mem_usage=False
            )
            
            # Initialize with base model (no adapter loaded)
            self.model = self.base_model
            self.current_dialect = "original"  # Track current adapter
            self.peft_models = {}  # Cache loaded PEFT models
            self.dialects = ["en-AU", "en-IN", "en-UK"]

            logger.info(f"Phi-3-Medium-4K-Instruct model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Phi-3 model: {e}")
            self.model = None

    def load_merged_dialect_adapter(self, task: str, domain: str):

        adapters = []
        weights = [0.33, 0.33, 0.33]
        density = 0.2
        combination_type = "ties"
        adapter_name = f"merge_{task}_{domain}"
        self.current_task = f"{task}_{domain}"

        if adapter_name not in self.peft_models: 
            
            # Add the first adapter
            adapter_path = os.path.join(ADAPTER_DIR, f"en_{task}_{domain}_adapter")
            temp_model = PeftModel.from_pretrained(
                self.base_model, 
                adapter_path,
                torch_dtype=torch.bfloat16,
                adapter_name=adapter_name
            )
                
            self.peft_models[adapter_name] = temp_model
            logger.info(f"Initialzed merged adapter for task: {self.current_task}")

        self.model = self.peft_models[adapter_name]
        self.model.set_adapter(adapter_name)

        logger.info(f"Switched to merged adapter for task: {self.current_task}")
    
    def unload_current_adapter(self):
        """Switch back to base model (no adapter)"""
        
        self.model = self.base_model
        self.current_dialect = "original"
        logger.info("Switched back to base model (no adapter)")

    def clear_all_adapters(self):
        """Clear all cached PEFT models to free memory"""
        
        self.peft_models.clear()
        self.model = self.base_model
        self.current_dialect = "original"
        logger.info("Cleared all cached adapters")

    def format_prompt(self, instruction: str, context: str = "") -> str:
        """Format prompt using Phi-3 chat template"""
        
        if context.strip():
            user_input = f"{instruction}\n{context}"
        else:
            user_input = instruction
        
        messages = [
            {"role": "user", "content": user_input},
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt
    
    def predict(self, instruction: str, context: str, dialect: str, task: str, domain: str, max_new_tokens: int = 50) -> str:
        """Predict fact verification label"""
        
        try:
            self.load_merged_dialect_adapter(task, domain)
            prompt = self.format_prompt(instruction, context=context)

            logger.warning(prompt)

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=4096,  # Phi-3-Medium supports 4K context
                add_special_tokens=False
            )
            inputs = inputs.to("cuda")

            stopping_criteria = StoppingCriteriaList([
                Phi3StoppingCriteria(self.tokenizer, ["<|end|>"])
            ])

            self.model.eval()
            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs, 
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False).strip()
            
            # Clean up response - remove Phi-3 special tokens
            for stop_string in ["<|end|>", "<|user|>", "<|assistant|>"]:
                response_text = response_text.replace(stop_string, "").strip()
    
            logger.info(response_text)

            # Extract numeric label
            for char in response_text:
                if char in ['0', '1']:
                    return char

            # Fallback interpretation
            response_lower = response_text.lower()
            if any(word in response_lower for word in ['support', 'confirm', 'true', 'correct']):
                return '0'
            elif any(word in response_lower for word in ['refute', 'contradict', 'false', 'incorrect']):
                return '1'
            else:
                return '0'
        
        except Exception as e:
            logger.error(f"Error in Phi-3 prediction for dialect {dialect}: {e}")
            return '0'