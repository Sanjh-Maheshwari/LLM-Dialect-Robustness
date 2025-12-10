import os
import torch
import warnings
from loguru import logger
import moe_peft

warnings.filterwarnings('ignore')


class GemmaMOEClassifier:
    """Gemma MOE classifier with few-shot support"""

    def __init__(self, model_id="google/gemma-2-9b-it", lora_path="/scratch/users/k24053411/"):
        logger.info(f"Loading {model_id} with MoE-PEFT")

        try:
            # Initialize MOE model with optimized settings
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
                prompt_template="gemma",
                stop_token="<end_of_turn>"
            )

            logger.info(f"Gemma MOE model loaded successfully with MoE-PEFT")
        except Exception as e:
            logger.error(f"Error loading Gemma MOE model: {e}")
            self.model = None

    def format_fewshot_prompt(self, instruction: str, few_shot_examples: list, context: str = "") -> str:
        """Format few-shot prompt with examples

        Args:
            instruction: Task instruction
            few_shot_examples: List of dicts with 'context' and 'response' keys
            context: Current test context

        Returns:
            Formatted prompt with few-shot examples
        """
        # Build few-shot examples
        examples_text = ""
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {example['context']}\n"
            examples_text += f"Answer: {example['response']}\n"

        # Construct full prompt
        full_instruction = f"{instruction}\n{examples_text}\nNow predict:\nText: {context}\nAnswer:"

        return full_instruction

    def evaluate(
        self,
        instruction,
        input="",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=50,
    ):
        """Generate response using MOE model"""
        instruction = instruction.strip()
        input = input.strip() if input else None

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

        output = moe_peft.generate(**generate_params)
        return output["default"][0]

    def predict(self, instruction: str, context: str, dialect: str, task: str, domain: str, max_new_tokens: int = 50) -> str:
        """Predict classification label (zero-shot)"""

        lora_weights = os.path.join(self.lora_path, f"mixlora_gemma_{task.lower()}_{domain.lower()}_0")

        self.model.load_adapter(lora_weights, "default")

        response_text = self.evaluate(
            instruction=instruction,
            input=context,
        ).strip("<end_of_turn>")

        self.model.unload_adapter("default")

        logger.debug(f"Response: {response_text}")

        # Extract numeric label
        for char in response_text:
            if char in ['0', '1']:
                return char

        # Fallback
        response_lower = response_text.lower()
        if any(word in response_lower for word in ['positive', 'sarcastic', 'yes', 'true']):
            return '1'
        elif any(word in response_lower for word in ['negative', 'not sarcastic', 'no', 'false']):
            return '0'
        else:
            return '0'

    def predict_fewshot(self, instruction: str, context: str, few_shot_examples: list,
                        dialect: str, task: str, domain: str, max_new_tokens: int = 50) -> str:
        """Predict classification label with few-shot examples

        Args:
            instruction: Task instruction
            context: Test context to classify
            few_shot_examples: List of dicts with 'context' and 'response' keys
            dialect: Dialect variety
            task: Task name
            domain: Domain name
            max_new_tokens: Max tokens to generate

        Returns:
            Predicted label ('0' or '1')
        """
        try:
            # Load MixLoRA adapter
            lora_weights = os.path.join(self.lora_path, f"mixlora_gemma_{task.lower()}_{domain.lower()}_0")
            self.model.load_adapter(lora_weights, "default")

            # Format prompt with few-shot examples
            fewshot_instruction = self.format_fewshot_prompt(instruction, few_shot_examples, context)

            # Generate response
            response_text = self.evaluate(
                instruction=fewshot_instruction,
                input="",
                max_new_tokens=max_new_tokens
            ).strip("<end_of_turn>")

            # Unload adapter to free memory
            self.model.unload_adapter("default")

            logger.debug(f"Few-shot response: {response_text}")

            # Extract numeric label
            for char in response_text:
                if char in ['0', '1']:
                    return char

            # Fallback
            response_lower = response_text.lower()
            if any(word in response_lower for word in ['positive', 'sarcastic', 'yes', 'true']):
                return '1'
            elif any(word in response_lower for word in ['negative', 'not sarcastic', 'no', 'false']):
                return '0'
            else:
                return '0'

        except Exception as e:
            logger.error(f"Error in Gemma MOE few-shot prediction for dialect {dialect}: {e}")
            return '0'
