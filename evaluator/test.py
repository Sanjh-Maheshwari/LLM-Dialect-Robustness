import pandas as pd
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report
from evaluator.llm_services.prompts import system_prompt_general, evaluation_prompt_zeroshot, evaluation_prompt_fewshot
from evaluator.llm_services.utils import format_fewshot_prompt
import warnings
from loguru import logger
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    StoppingCriteria, 
    StoppingCriteriaList
)
from peft import PeftModel, PeftConfig
import torch

warnings.filterwarnings('ignore')

ADAPTER_DIR = "output/peft_lora"
model_id = "meta-llama/Llama-3.1-8B"
dialect = "Indian_English"

# Load PEFT config
peft_config_path = os.path.join(ADAPTER_DIR, f"{dialect}_adapter")
config = PeftConfig.from_pretrained(peft_config_path)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token            

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,  # Use base model from config
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Load the PEFT model with trained adapter
model = PeftModel.from_pretrained(
    model, 
    peft_config_path,
    torch_dtype=torch.bfloat16
)

print(f"Loaded PEFT model with adapter: {dialect}")
print(f"Base model: {config.base_model_name_or_path}")
print(f"PEFT type: {config.peft_type}")
print(f"Task type: {config.task_type}")


class ChatMLStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<|im_start|>", "<|im_end|>"]
        
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        
        # Pre-encode stop strings to token IDs for efficient comparison
        self.stop_token_ids = []
        for stop_string in stop_strings:
            # Encode the stop string
            tokens = tokenizer.encode(stop_string, add_special_tokens=False)
            if tokens:  # Only add non-empty token lists
                self.stop_token_ids.append(tokens)
        
        print(f"Stop strings: {stop_strings}")
        print(f"Stop token IDs: {self.stop_token_ids}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check the last generated tokens for any stop sequences
        for stop_tokens in self.stop_token_ids:
            stop_length = len(stop_tokens)
            if input_ids.shape[1] >= stop_length:
                # Check if the last tokens match any stop sequence
                last_tokens = input_ids[0, -stop_length:].tolist()
                if last_tokens == stop_tokens:
                    return True
        return False

def format_chatml_prompt(instruction: str, context: str = "") -> str:
    """Format prompt using ChatML template (your custom format)"""
    
    # Combine instruction and context
    if context.strip():
        user_input = f"{instruction}\n{context}"
    else:
        user_input = instruction
    
    # Use your ChatML format
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt

def generate_response(model, tokenizer, instruction: str, context: str = "", max_new_tokens: int = 256) -> str:
    """Generate response using ChatML template with custom stopping criteria"""
    
    # Format prompt using ChatML template
    prompt = format_chatml_prompt(instruction, context)

    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048,
        add_special_tokens=False
    )
    inputs = inputs.to("cuda")
    
    # Create stopping criteria
    stopping_criteria = StoppingCriteriaList([
        ChatMLStoppingCriteria(tokenizer, ["<|im_start|>", "<|im_end|>"])
    ])
    
    # Generate
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs, 
            do_sample=True,
            temperature=0.1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    
    # Decode only the new tokens (response part)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    # Clean up response (remove any remaining special tokens and stop strings)
    for stop_string in ["<|im_start|>", "<|im_end|>"]:
        response_text = response_text.replace(stop_string, "").strip()
    
    return response_text

# Test the model
test_instruction = "Based on the provided evidence, classify the claim using one of these labels:\n- 0: SUPPORTS (evidence confirms the claim)\n- 1: REFUTES (evidence contradicts the claim)  \n- 2: NOT ENOUGH INFO (evidence is insufficient to determine truth)\n- 3: DISPUTED (evidence shows conflicting information about the claim)\n\n**CLAIM**: Increased atmospheric carbon dioxide has helped in raising global food production and reducing poverty.\n**RESPONSE**: Provide only the numeric label (0, 1, 2, or 3).\n"
test_context = "1. In the 1980s and 1990s low world market prices for cereals and livestock resulted in decreased agricultural growth and increased rural poverty.\n2. While increased CO 2 levels help crop growth at lower temperature increases, those crops do become less nutritious.\n3. Global warming is the result of increasing atmospheric carbon dioxide concentrations which is caused primarily by the combustion of fossil energy sources such as petroleum, coal, and natural gas, and to an unknown extent by destruction of forests, increased methane, volcanic activity and cement production.\n4. Potential negative environmental impacts caused by increasing atmospheric carbon dioxide concentrations are rising global air temperatures, altered hydrogeological cycles resulting in more frequent and severe droughts, storms, and floods, as well as sea level rise and ecosystem disruption.\n5. A major hurdle to achieve sustainability is the alleviation of poverty."

print(f"Testing model with dialect: {dialect}")
print(f"Instruction: {test_instruction}")
print(f"Context: {test_context}")
print("="*50)

# Method 1: Simple generation
response = generate_response(
    model=model,
    tokenizer=tokenizer, 
    instruction=test_instruction,
    context=test_context,
    max_new_tokens=50
)

print("Generated Response (Method 1):")
print(response)
print("="*50)

# Optional: If you want to merge adapter weights with base model for faster inference
# model = model.merge_and_unload()
# print("Adapter weights merged with base model")

# Optional: If you want to disable adapter temporarily
# model.disable_adapter()
# print("Adapter disabled")

# Optional: If you want to re-enable adapter
# model.enable_adapter()
# print("Adapter enabled")