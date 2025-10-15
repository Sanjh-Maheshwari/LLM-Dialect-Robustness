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
    DataCollatorForLanguageModeling
)
import warnings
from loguru import logger
from peft import PeftModel, PeftConfig
from transformers import StoppingCriteria, StoppingCriteriaList

ADAPTER_DIR = "output/besstie_adapters_completed"

warnings.filterwarnings('ignore')

class ChatMLStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<|im_start|>", "<|im_end|>"]
        
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        
        self.stop_token_ids = []
        for stop_string in stop_strings:
            # Encode the stop string
            tokens = tokenizer.encode(stop_string, add_special_tokens=False)
            if tokens: 
                self.stop_token_ids.append(tokens)
        
        # print(f"Stop strings: {stop_strings}")
        # print(f"Stop token IDs: {self.stop_token_ids}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for stop_tokens in self.stop_token_ids:
            stop_length = len(stop_tokens)
            if input_ids.shape[1] >= stop_length:
                last_tokens = input_ids[0, -stop_length:].tolist()
                if last_tokens == stop_tokens:
                    return True
        return False

class LlamaClassifier:
    """LLama model for fact checking"""
    
    def __init__(self, model_id="meta-llama/Llama-3.1-8B"):
        logger.info(f"Loading {model_id}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token            
            
            # Load base model first
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for consistency
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Initialize with base model (no adapter loaded)
            self.model = self.base_model
            self.current_dialect = "original"  # Track current adapter
            self.peft_models = {}  # Cache loaded PEFT models
            
            print(f"Llama base model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            self.model = None

    def load_dialect_adapter(self, dialect: str, task:str, domain:str):
        """Load adapter for specific dialect"""
        
        if dialect == "original":
            # Use base model without adapter
            self.model = self.base_model
            self.current_dialect = "original"
            logger.info("Using original model (no adapter)")
            return
            
        if dialect == self.current_dialect:
            # logger.info(f"Adapter {dialect} already loaded")
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
                # logger.info(f"Loaded PEFT model for dialect: {dialect}")
            
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

    def format_chatml_prompt(self, instruction: str, context: str = "") -> str:
        """Format prompt using ChatML template"""
        
        if context.strip():
            user_input = f"{instruction}\n{context}"
        else:
            user_input = instruction
        
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def predict(self, instruction:str, context:str, dialect: str, task:str, domain:str, max_new_tokens: int = 50) -> str:
        """Predict fact verification label"""
        
        try:
            self.load_dialect_adapter(dialect, task, domain)
            prompt = self.format_chatml_prompt(instruction, context=context)

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048, 
                add_special_tokens=False
            )
            inputs = inputs.to("cuda")

            stopping_criteria = StoppingCriteriaList([
                ChatMLStoppingCriteria(self.tokenizer, ["<|im_start|>", "<|im_end|>"])
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
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up response
            for stop_string in ["<|im_start|>", "<|im_end|>"]:
                response_text = response_text.replace(stop_string, "").strip()
    
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
            logger.error(f"Error in Llama prediction for dialect {dialect}: {e}")
            return '0'