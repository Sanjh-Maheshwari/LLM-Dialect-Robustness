import json
import logging
import os.path as osp
from typing import Dict, Optional, Union
from transformers import AutoTokenizer

prompt_templates = {
    "moe_peft": {
        "description": "Default Prompt Template Provided by MoE-PEFT",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n",
        "prompt_no_input": "### Instruction:\n{instruction}\n\n### Output:\n",
        "response_split": "### Output:",
    },
    "alpaca": {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, "
        + "paired with an input that provides further context. "
        + "Write a response that appropriately completes the request.\n\n"
        + "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. "
        + "Write a response that appropriately completes the request.\n\n"
        + "### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
    },
}


# manage templates and prompt building.
class Prompter:
    def __init__(self, template: Optional[Union[Dict, str]] = None):
        if template is None:
            self.template = prompt_templates["moe_peft"]
        elif isinstance(template, str):
            if osp.exists(template):
                with open(template) as fp:
                    self.template = json.load(fp)
            else:
                if template in prompt_templates: 
                    self.template = prompt_templates[template]
                else:
                    self.template = template
        else:
            self.template = template

        logging.info(f"Using prompt template: {self.template}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
        tokenizer: AutoTokenizer = None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        
        if isinstance(self.template, str) and self.template == "gemma": 

            if input:
                user_input = f"{instruction}\n{input}"
            else:
                user_input = instruction

            if not tokenizer: 
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
                
            if label:
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": label},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False,
                )
            else:
                messages = [
                    {"role": "user", "content": user_input},
                ]
                
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )

        elif isinstance(self.template, str) and self.template == "mistral": 

            if input:
                user_input = f"{instruction}\n{input}"
            else:
                user_input = instruction

            if not tokenizer: 
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

            if label:
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": label},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False,
                )
            else:
                messages = [
                    {"role": "user", "content": user_input},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )

        elif isinstance(self.template, str) and self.template == "phi": 

            if input:
                user_input = f"{instruction}\n{input}"
            else:
                user_input = instruction

            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct")

            if label:
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": label},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False,
                )
            else:
                messages = [
                    {"role": "user", "content": user_input},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )

        elif isinstance(self.template, str) and self.template == "qwen": 

            if input:
                user_input = f"{instruction}\n{input}"
            else:
                user_input = instruction

            if not tokenizer: 
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

            if label:
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": label},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False,
                )
            else:
                messages = [
                    {"role": "user", "content": user_input},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )

        elif isinstance(self.template, str) and self.template == "llama": 

            if input:
                user_input = f"{instruction}\n{input}"
            else:
                user_input = instruction
            
            if not tokenizer: 
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

            if label:
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": label},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False,
                )
            else:
                messages = [
                    {"role": "user", "content": user_input},
                ]
                res = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )

        else:
            if input:
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_no_input"].format(instruction=instruction)

            if label:
                res = f"{res}{label}\n"

        return res

    def get_response(self, output: str) -> str:
        
        if isinstance(self.template, str):
            if self.template == "gemma":
                if "<start_of_turn>model\n" in output:
                    return output.split("<start_of_turn>model\n")[-1].strip()
                return output.strip()
            
            elif self.template == "mistral":
                if "[/INST]" in output:
                    return output.split("[/INST]")[-1].strip()
                return output.strip()
            
            elif self.template == "phi":
                if "<|assistant|>" in output:
                    return output.split("<|assistant|>")[-1].strip()
                return output.strip()
            
            elif self.template == "qwen":
                if "<|im_start|>assistant\n" in output:
                    return output.split("<|im_start|>assistant\n")[-1].strip()
                return output.strip()
            
            elif self.template == "llama":
                if "<|start_header_id|>assistant<|end_header_id|>\n\n" in output:
                    return output.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].strip()
                return output.strip()

            else:
                logging.warning(f"Unknown template type: {self.template}, returning output as-is")
                return output.strip()
        
        elif isinstance(self.template, dict):
            if "response_split" in self.template:
                return output.split(self.template["response_split"])[-1].strip()
            else:
                logging.warning("Template dict missing 'response_split' key, returning output as-is")
                return output.strip()
        
        else:
            logging.warning(f"Unexpected template type: {type(self.template)}, returning output as-is")
            return output.strip()