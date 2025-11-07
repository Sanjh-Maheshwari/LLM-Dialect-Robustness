import pandas as pd
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from vllm import LLM, SamplingParams
import warnings
from loguru import logger

class LlamaBaseClassifier:
    """LLama model for fact checking"""
    
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        
        logger.info(f"Loading {model_id}")
        try:

            # Initialize vLLM with optimized settings
            self.model = LLM(
                model=model_id,
                tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                dtype="bfloat16",
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                hf_token="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"
            )
            
            # Sampling parameters for consistent, deterministic outputs
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=10,
                stop=["\n", "Explanation:", "Because:"],
            )
            
            print(f"Llama model loaded successfully with vLLM")
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            self.model = None

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
            # system_prompt = "You are a helpful assistant"
            user_content = f"{instruction}\n{context}"
            conversation = [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            # Generate response using vLLM chat interface
            outputs = self.model.chat(
                messages=[conversation],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            response_text = outputs[0].outputs[0].text.strip()

            logger.debug(response_text)

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