import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import List, Dict
import warnings
from loguru import logger
from peft import PeftModel

warnings.filterwarnings('ignore')

# Adapter directories
DIALECT_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/qwen"
BASELINE_ADAPTER_DIR = "/scratch/users/k24053411/axolotl/qwen/baseline"


class QwenStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=None):
        if stop_strings is None:
            stop_strings = ["<|im_end|>", "<|endoftext|>"]

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


class QwenUnifiedClassifier:
    """Unified Qwen classifier supporting LoRA Grouping, CAT, and TIES methods"""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
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
                low_cpu_mem_usage=False
            )

            # Initialize with base model
            self.model = self.base_model
            self.peft_models = {}
            self.dialects = ["en-AU", "en-IN", "en-UK"]
            self.current_method = None
            self.current_task = None

            logger.info(f"Qwen2.5-7B-Instruct model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Qwen 2.5 model: {e}")
            self.model = None

    def load_adapter(self, task: str, domain: str, method: str, dialect: str = None):
        """Load adapter based on method"""
        # For base_instruct, just use the base model without any adapter
        if method == "base_instruct":
            logger.info("Using base instruct model without any adapter")
            self.model = self.base_model
            self.current_task = f"{task}_{domain}"
            self.current_method = method
            return

        # For individual_dialect, include dialect in the key
        if method == "individual_dialect":
            if dialect is None:
                raise ValueError("dialect parameter is required for individual_dialect method")
            adapter_key = f"{method}_{dialect}_{task}_{domain}"
        else:
            adapter_key = f"{method}_{task}_{domain}"

        self.current_task = f"{task}_{domain}"
        self.current_method = method

        if adapter_key in self.peft_models:
            self.model = self.peft_models[adapter_key]
            if method not in ["lora_grouping", "individual_dialect"]:
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

        elif method == "individual_dialect":
            # Load only the specific dialect adapter
            adapter_path = os.path.join(DIALECT_ADAPTER_DIR, f"{dialect}_{task}_{domain}_adapter")
            logger.info(f"Loading individual dialect adapter from: {adapter_path}")

            temp_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                torch_dtype=torch.bfloat16,
                adapter_name=adapter_key
            )

            self.peft_models[adapter_key] = temp_model
            self.model = temp_model
            self.model.set_adapter(adapter_key)
            logger.info(f"Loaded individual dialect adapter for {dialect}/{self.current_task}")

        else:
            raise ValueError(f"Unknown method: {method}. Must be one of ['lora_grouping', 'cat', 'ties', 'base_instruct', 'individual_dialect']")

    def format_prompt(self, instruction: str, context: str = "") -> str:
        """Format prompt using Qwen2.5 chat template"""

        if context.strip():
            user_input = f"{instruction}\n{context}"
        else:
            user_input = instruction

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def format_fewshot_prompt(self, instruction: str, few_shot_examples: list, context: str = "") -> str:
        """Format few-shot prompt using Qwen2.5 chat template

        Args:
            instruction: Task instruction
            few_shot_examples: List of dicts with 'context' and 'response' keys
            context: Current test context
        """
        # Build few-shot examples
        examples_text = ""
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {example['context']}\n"
            examples_text += f"Answer: {example['response']}\n"

        # Construct full prompt
        user_input = f"{instruction}\n{examples_text}\nNow predict:\nText: {context}\nAnswer:"

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
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
            self.load_adapter(task, domain, method, dialect=dialect)
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
                QwenStoppingCriteria(self.tokenizer, ["<|im_end|>", "<|endoftext|>"])
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
                )

            input_length = inputs['input_ids'].shape[-1]
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
            logger.error(f"Error in Qwen prediction for dialect {dialect}: {e}")
            return '0'

    def predict_fewshot(self, instruction: str, context: str, few_shot_examples: list,
                        dialect: str, task: str, domain: str, method: str,
                        max_new_tokens: int = 50) -> str:
        """Predict classification label with few-shot examples

        Args:
            instruction: Task instruction
            context: Test context to classify
            few_shot_examples: List of dicts with 'context' and 'response' keys
            dialect: Dialect variety
            task: Task name
            domain: Domain name
            method: Merging method
            max_new_tokens: Max tokens to generate
        """
        try:
            self.load_adapter(task, domain, method, dialect=dialect)
            prompt = self.format_fewshot_prompt(instruction, few_shot_examples, context=context)

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
                QwenStoppingCriteria(self.tokenizer, ["<|im_end|>", "<|endoftext|>"])
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
                )

            input_length = inputs['input_ids'].shape[-1]
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
            logger.error(f"Error in Qwen few-shot prediction for dialect {dialect}: {e}")
            return '0'

    def clear_all_adapters(self):
        """Clear all cached PEFT models to free memory"""
        self.peft_models.clear()
        self.model = self.base_model
        logger.info("Cleared all cached adapters")
