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

ADAPTER_DIR = "/scratch/users/k24053411/axolotl/gemma"

warnings.filterwarnings('ignore')

class ChatMLStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<start_of_turn>", "<end_of_turn>"]
        
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

class GemmaClassifier:
    """Gemma 2 9B IT model for fact checking"""
    
    def __init__(self, model_id="google/gemma-2-9b-it"):
        logger.info(f"Loading {model_id}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Gemma 2 has its own pad token, but we can set it if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Initialize with base model (no adapter loaded)
            self.model = self.base_model
            self.current_dialect = "original"  # Track current adapter
            self.peft_models = {}  # Cache loaded PEFT models
            
            print(f"Gemma-2-9b-it model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Gemma 2 model: {e}")
            self.model = None

    def load_dialect_adapter(self, dialect: str, task: str, domain: str):
        """Load adapter for specific dialect"""
        
        if dialect == "original":
            # Use base model without adapter
            self.model = self.base_model
            self.current_dialect = "original"
            logger.info("Using original model (no adapter)")
            return
            
        if dialect == self.current_dialect:
            return
            
        try:
            # Check if we've already loaded this PEFT model
            if dialect not in self.peft_models:
                adapter_path = os.path.join(ADAPTER_DIR, f"{dialect}_{task}_{domain}_adapter")
                
                # Load PEFT model
                self.peft_models[dialect] = PeftModel.from_pretrained(
                    self.base_model,  # Always use base model as foundation
                    adapter_path,
                    torch_dtype=torch.bfloat16
                )
                logger.info(f"Loaded PEFT model for dialect: {dialect}")
            
            # Switch to the requested dialect
            self.model = self.peft_models[dialect]
            self.current_dialect = dialect
            logger.info(f"Switched to dialect: {dialect}")
            
        except Exception as e:
            logger.error(f"Error loading adapter for {dialect}: {e}")
            # Fallback to base model
            self.model = self.base_model
            self.current_dialect = "original"
    
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

    def format_prompt(self, instruction: str, context: str = "") -> List[Dict[str, str]]:
        """Format prompt using Gemma 2 chat template"""
        
        if context.strip():
            user_input = f"{instruction}\n{context}"
        else:
            user_input = instruction
        
        messages = [
            {"role": "user", "content": user_input}
        ]

        return messages
    
    def predict(self, instruction: str, context: str, dialect: str, task: str, domain: str, max_new_tokens: int = 50) -> str:
        """Predict fact verification label"""
        
        self.load_dialect_adapter(dialect, task, domain)
        messages = self.format_prompt(instruction, context=context)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        stopping_criteria = StoppingCriteriaList([
            ChatMLStoppingCriteria(self.tokenizer, ["<start_of_turn>", "<end_of_turn>"])
        ])

        self.model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs, 
                do_sample=True,
                temperature = 0.1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # stopping_criteria=stopping_criteria,
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[-1]
        response_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
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