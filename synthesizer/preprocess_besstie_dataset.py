import json
import os
from datasets import load_dataset
from pathlib import Path
from loguru import logger

# Configuration
VARIETIES = ["en-AU", "en-IN", "en-UK"]
TASKS = ["Sarcasm", "Sentiment"]
DOMAINS = ["Reddit", "Google"]

def validate_message_alternation(messages):
    """
    Validate that messages alternate properly between user and assistant roles.
    Returns (is_valid, error_message)
    """
    
    if not messages or len(messages) == 0:
        return False, "Empty messages list"
    
    # Filter out system messages and tool-related messages for validation
    conversation_messages = [
        msg for msg in messages 
        if msg.get('role') not in ['system', 'tool', 'tool_results']
    ]
    
    if len(conversation_messages) == 0:
        return False, "No user/assistant messages found"
    
    # First message should be from user
    if conversation_messages[0]['role'] != 'user':
        return False, f"First message must be 'user', got '{conversation_messages[0]['role']}'"
    
    # Check alternation
    for i in range(1, len(conversation_messages)):
        current_role = conversation_messages[i]['role']
        previous_role = conversation_messages[i-1]['role']
        
        if current_role == previous_role:
            return False, f"Consecutive {current_role} messages at position {i}"
        
        # Only user and assistant are allowed (after filtering)
        if current_role not in ['user', 'assistant']:
            return False, f"Invalid role '{current_role}' at position {i}"
    
    # Should end with assistant message for training
    if conversation_messages[-1]['role'] != 'assistant':
        return False, f"Last message should be 'assistant', got '{conversation_messages[-1]['role']}'"
    
    return True, None

def convert_besstie_to_chat_format(text, label, task, variety, example_id):
    """Convert BESSTIE data to OpenAI chat format"""
    
    if task == "Sentiment":
        instruction = (
            "Generate the sentiment of the given text. "
            "1 for positive sentiment, and 0 for negative sentiment. "
            "Do not give an explanation."
        )
    else:  # Sarcasm
        instruction = (
            "Predict if the given text is sarcastic. "
            "1 if the text is sarcastic, and 0 if the text is not sarcastic. "
            "Do not give an explanation."
        )
    
    # Create chat messages in OpenAI format
    messages = [
        {
            "role": "user",
            "content": f"{instruction}\n{text}"
        },
        {
            "role": "assistant",
            "content": str(label)
        }
    ]
    
    return {
        "messages": messages,
        "example_id": example_id,
        "variety": variety,
        "task": task
    }

def load_besstie_samples(json_path, variety, task, domain):
    """Load samples from BESSTIE JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        variety_data = data[task][domain][variety]
    except KeyError:
        logger.warning(f"No data found for {task}/{domain}/{variety}")
        return []
    
    samples = []
    for idx, item in enumerate(variety_data):
        sample = convert_besstie_to_chat_format(
            text=item['text'] if isinstance(item, dict) else item,
            label=item['label'] if isinstance(item, dict) else 0,
            task=task,
            variety=variety,
            example_id=idx
        )
        samples.append(sample)
    
    return samples

def convert_finetome_to_chat_format(finetome_sample, variety, task, example_id_offset=0):
    """Convert FineTome sample to OpenAI chat format"""
    
    conversations = finetome_sample['conversations']
    
    messages = []
    for conv in conversations:
        role = "user" if conv['from'] in ['human', 'user'] else "assistant"
        messages.append({
            "role": role,
            "content": conv['value']
        })
    
    return {
        "messages": messages,
        "example_id": example_id_offset,
        "variety": variety,
        "task": task
    }

def augment_with_finetome(original_samples, variety, task, num_samples=2000, seed=42):
    """Augment with FineTome samples"""
    
    logger.info(f"Loading {num_samples} samples from FineTome-100k...")
    
    try:
        # Load FineTome dataset
        finetome = load_dataset("mlabonne/FineTome-100k", split="train", streaming=True)
        
        # Sample from streaming dataset
        augmented_samples = original_samples.copy()
        max_id = max([s['example_id'] for s in original_samples]) if original_samples else 0
        
        count = 0
        for sample in finetome:
            if count >= num_samples:
                break
            
            converted = convert_finetome_to_chat_format(
                sample, variety, task, max_id + count + 1
            )
            augmented_samples.append(converted)
            count += 1
        
        logger.info(f"Augmented: {len(original_samples)} -> {len(augmented_samples)} samples")
        return augmented_samples
        
    except Exception as e:
        logger.error(f"Error loading FineTome: {e}")
        logger.warning("Continuing with only BESSTIE data")
        return original_samples

def prepare_dataset(
    besstie_json_path,
    output_dir="data/instruction",
    augment=True,
    finetome_samples=2000,
    seed=42
):
    """Prepare all dataset variants"""
    
    os.makedirs(output_dir, exist_ok=True)

    total_samples = 0
    total_skipped = 0
    
    for variety in VARIETIES:
        for task in TASKS:
            domains = DOMAINS if task == "Sarcasm" else ["Reddit"]
            
            for domain in domains:
                logger.info(f"{'='*60}")
                logger.info(f"Preparing: {variety}/{task}/{domain}")
                logger.info(f"{'='*60}")
                
                # Load BESSTIE samples
                samples = load_besstie_samples(
                    besstie_json_path, variety, task, domain
                )
                
                if not samples:
                    logger.warning(f"No samples found, skipping...")
                    continue
                
                logger.info(f"Loaded {len(samples)} BESSTIE samples")
                
                # Augment with FineTome
                if augment:
                    samples = augment_with_finetome(
                        samples, variety, task, finetome_samples, seed
                    )

                valid_samples = []
                final_skipped = 0
                for sample in samples:
                    is_valid, error = validate_message_alternation(sample['messages'])
                    if is_valid:
                        valid_samples.append(sample)
                    else:
                        final_skipped += 1
                        logger.debug(f"Skipped in final pass: {error}")
                
                if final_skipped > 0:
                    logger.warning(f"Skipped {final_skipped} samples in final validation")
                
                # Save as JSONL
                output_file = f"{output_dir}/{variety}_{task}_{domain}.jsonl"
                with open(output_file, 'w') as f:
                    for sample in valid_samples:
                        f.write(json.dumps(sample) + '\n')
                
                logger.info(f"Saved {len(valid_samples)} samples to {output_file}")
                total_samples += len(valid_samples)
                total_skipped += final_skipped
    
    logger.info("="*60)
    logger.info("Dataset preparation complete!")
    logger.info(f"Total valid samples: {total_samples}")
    logger.info(f"Total skipped samples: {total_skipped}")
    logger.info("="*60)

def main():
    
    prepare_dataset(
        besstie_json_path="data/instruction/besstie.json",
        output_dir="data/axolotl",
        augment=True,
        finetome_samples=2000,
        seed=42
    )

if __name__ == "__main__":
    main()