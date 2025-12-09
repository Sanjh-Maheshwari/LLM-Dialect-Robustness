import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
import warnings
from loguru import logger
from peft import PeftModel

warnings.filterwarnings('ignore')

# Adapter directories
DIALECT_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/gemma"
BASELINE_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/gemma/baseline"


class GemmaStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<start_of_turn>", "<end_of_turn>"]

        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

        self.stop_token_ids = []
        for stop_string in stop_strings:
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


class Gemma2UnifiedClassifier:
    """Unified Gemma-2 classifier supporting LoRA Grouping, CAT, and TIES methods"""

    def __init__(self, model_id="google/gemma-2-9b-it"):
        logger.info(f"Loading {model_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            # Initialize with base model
            self.model = self.base_model
            self.peft_models = {}
            self.dialects = ["en-AU", "en-IN", "en-UK"]
            self.current_method = None
            self.current_task = None

            logger.info(f"Gemma-2-9b-it model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Gemma 2 model: {e}")
            self.model = None

    def load_adapter(self, task: str, domain: str, method: str):
        """Load adapter based on method"""
        adapter_key = f"{method}_{task}_{domain}"
        self.current_task = f"{task}_{domain}"
        self.current_method = method

        if adapter_key in self.peft_models:
            self.model = self.peft_models[adapter_key]
            if method != "lora_grouping":
                self.model.set_adapter(adapter_key)
            logger.info(f"Using cached adapter: {adapter_key}")
            return

        if method == "lora_grouping":
            adapter_path = os.path.join(BASELINE_ADAPTER_DIR, f"en_{task}_{domain}_adapter")
            logger.info(f"Loading LoRA Grouping adapter from: {adapter_path}")

            temp_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                torch_dtype=torch.bfloat16,
                adapter_name=adapter_key
            )

            self.peft_models[adapter_key] = temp_model
            self.model = temp_model
            self.model.set_adapter(adapter_key)
            logger.info(f"Loaded LoRA Grouping adapter for {self.current_task}")

        elif method in ["cat", "ties"]:
            logger.info(f"Loading {method.upper()} merged adapter")

            adapter_path = os.path.join(DIALECT_ADAPTER_DIR, f"{self.dialects[0]}_{task}_{domain}_adapter")
            logger.info(f"Loading first adapter: {adapter_path}")

            temp_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                torch_dtype=torch.bfloat16,
                adapter_name=self.dialects[0]
            )

            adapters = [self.dialects[0]]

            for dialect in self.dialects[1:]:
                adapter_path = os.path.join(DIALECT_ADAPTER_DIR, f"{dialect}_{task}_{domain}_adapter")
                logger.info(f"Loading adapter: {adapter_path}")
                temp_model.load_adapter(adapter_path, adapter_name=dialect)
                adapters.append(dialect)

            weights = [0.33, 0.33, 0.34]
            density = 0.2

            logger.info(f"Merging adapters using {method.upper()} method")
            temp_model.add_weighted_adapter(
                adapters,
                weights,
                adapter_key,
                combination_type=method,
                density=density if method == "ties" else None
            )

            self.peft_models[adapter_key] = temp_model
            self.model = temp_model
            self.model.set_adapter(adapter_key)
            logger.info(f"Merged adapter created and activated: {adapter_key}")

        else:
            raise ValueError(f"Unknown method: {method}. Must be one of ['lora_grouping', 'cat', 'ties']")

    def format_prompt(self, instruction: str, context: str = "") -> str:
        """Format prompt using tokenizer's chat template"""
        if context.strip():
            user_input = f"{instruction}\n{context}"
        else:
            user_input = instruction

        messages = [
            {"role": "user", "content": user_input}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def predict(self, instruction: str, context: str, dialect: str,
                task: str, domain: str, method: str, max_new_tokens: int = 50) -> str:
        """Predict classification label"""

        try:
            self.load_adapter(task, domain, method)
            prompt = self.format_prompt(instruction, context=context)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                add_special_tokens=False
            )
            inputs = inputs.to("cuda")

            stopping_criteria = StoppingCriteriaList([
                GemmaStoppingCriteria(self.tokenizer, ["<end_of_turn>"])
            ])

            self.model.eval()
            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )

            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Extract numeric label
            for char in response_text:
                if char in ['0', '1']:
                    return char

            # Fallback
            response_lower = response_text.lower()
            if any(word in response_lower for word in ['support', 'confirm', 'true', 'correct']):
                return '0'
            elif any(word in response_lower for word in ['refute', 'contradict', 'false', 'incorrect']):
                return '1'
            else:
                return '0'

        except Exception as e:
            logger.error(f"Error in Gemma 2 prediction for dialect {dialect}: {e}")
            return '0'

    def clear_all_adapters(self):
        """Clear all cached PEFT models to free memory"""
        self.peft_models.clear()
        self.model = self.base_model
        logger.info("Cleared all cached adapters")
