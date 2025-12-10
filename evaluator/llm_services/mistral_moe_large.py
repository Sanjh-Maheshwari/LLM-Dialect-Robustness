import pandas as pd
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import warnings
from loguru import logger
import moe_peft

warnings.filterwarnings('ignore')

class MistralClassifier:
    
    def __init__(self, model_id="mistralai/Mistral-Small-Instruct-2409", lora_path = "/scratch/users/k24053411/"):
        
        logger.info(f"Loading {model_id}")
        try:
            # Initialize mode model with optimized settings
            self.model = moe_peft.LLMModel.from_pretrained(
                    model_id,
                    device=moe_peft.executor.default_device_name(),
                    attn_impl="eager",
                    load_dtype=torch.bfloat16,
            )

            self.tokenizer = moe_peft.Tokenizer(model_id)
            self.lora_path = lora_path
            self.generation_config = moe_peft.GenerateConfig(
                adapter_name="default",
                prompt_template="mistral",
            )

            print(f"Mistral model loaded successfully with MoE-PEFT")
        except Exception as e:
            print(f"Error loading Mistral model: {e}")
            self.model = None

    def evaluate(
        self,
        instruction,
        input="",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=50,
        stream_output=False,
    ):
        instruction = instruction.strip()
        if len(instruction) == 0:
            instruction = placeholder_text

        input = input.strip()
        if len(input) == 0:
            input = None

        # logger.debug(f"Prompt : {instruction}, Context : {input}")

        self.generation_config.prompts = [(instruction, input)]
        self.generation_config.temperature = temperature
        self.generation_config.top_p = top_p
        self.generation_config.top_k = top_k
        self.generation_config.repetition_penalty = repetition_penalty

        generate_params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "configs": [self.generation_config],
            "max_gen_len": max_new_tokens,
        }

        # Without streaming
        output = moe_peft.generate(**generate_params)

        return output["default"][0]
    
    def predict(self, instruction:str, context:str, dialect: str, task:str, domain:str, max_new_tokens: int = 50) -> str:
        """Predict fact verification label"""
        
  
        # lora_weights = os.path.join(self.lora_path, f"mixlora_mistral_{task.lower()}_{domain.lower()}_0")
        lora_weights = os.path.join(self.lora_path, f"mixlora_mistral_sentiment_reddit_0/mixlora_mistral_sentiment_reddit_0_3000")
        

        self.model.load_adapter(lora_weights, "default")
        
        response_text = self.evaluate(
                                    instruction=instruction, 
                                    input=context,
                                    stream_output=False
                            )
        
        self.model.unload_adapter("default")

        logger.debug(response_text)

        for char in response_text:
            if char in ['0', '1']:
                return char

        response_lower = response_text.lower()
        
        if any(word in response_lower for word in ['support', 'confirm', 'true', 'correct']):
            return '0'
        elif any(word in response_lower for word in ['refute', 'contradict', 'false', 'incorrect']):
            return '1'
        else:
            return '0'

