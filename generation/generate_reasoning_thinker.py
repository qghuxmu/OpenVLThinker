import os
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import argparse
from mathruler.grader import grade_answer
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.clean_answer import clean_answer

def load_image_dataset(dataset_name: str) -> List[Dict]:
    """
    Load dataset from Hugging Face and extract image URLs and metadata
    """
    data = load_dataset(dataset_name, split="testmini")
    
    return [
        {
            'image_url': item['images'][0],
            'image_url_old': item['images_url'],
            'instruction': item.get('problem', ''),
            'response': item.get('answer', '')
        }
        for item in data
    ]

def download_image(url: str) -> Optional[Image.Image]:
    """
    Download image from URL and return PIL Image object
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def generate_answer(image_url: str, instruction: str, processor, model, device, k=4) -> List[str]:
    """
    Generate k different reasoning attempts with temperature 0.6
    Returns a list of k reasoning attempts
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {
                        "type": "text",
                        "text": instruction
                    },
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate k different reasoning attempts with temperature 0.6
        reasoning_attempts = []
        
        # Use a single batch generation with multiple samples instead of a loop
        generated_ids = model.generate(
            **inputs, 
            do_sample=True, 
            max_new_tokens=2048, 
            top_p=0.001, #0.95, #0.001, 
            top_k=1, #20, #1, 
            temperature=0.01, #0.6, 0.01,
            repetition_penalty=1,
            num_return_sequences=1,  # Generate k sequences in one call
        )
        
        # Process all generated sequences at once
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        reasoning_attempts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return reasoning_attempts
    
    except Exception as e:
        print(f"Error generating reasoning: {str(e)}")
        return [None] * k

def extract_answer_from_reasoning(reasoning: str) -> Optional[str]:
    """
    Extract the answer from the reasoning text
    """
    if reasoning is None or "</answer>" not in reasoning:
        return None
    
    answer = reasoning.split("<answer>")[-1].split("</answer>")[0].strip()
    return clean_answer(answer)

def save_descriptions(descriptions: List[Dict], output_file: str):
    """
    Save generated descriptions to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(descriptions, f, indent=2)

def get_shard_indices(total_size: int, shard_id: int, num_shards: int) -> Tuple[int, int]:
    """
    Calculate the start and end indices for a specific shard
    
    Args:
        total_size: Total number of items in the dataset
        shard_id: ID of the shard to process (0-based)
        num_shards: Total number of shards
        
    Returns:
        Tuple of (start_index, end_index) for the shard
    """
    shard_size = total_size // num_shards
    remainder = total_size % num_shards
    
    start_idx = shard_id * shard_size
    if shard_id < remainder:
        start_idx += shard_id
    else:
        start_idx += remainder
        
    end_idx = start_idx + shard_size
    if shard_id < remainder:
        end_idx += 1
        
    return start_idx, end_idx

def select_best_reasoning(all_correct_paths: List[Tuple[str, str, int]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Select the best reasoning from a list of correct paths
    
    Logic:
    - If there is only one correct path, use it
    - Else if there are two, select the longest one
    - Else, pop the two shortest ones, use the next
    
    Args:
        all_correct_paths: List of tuples (reasoning, answer, word_count)
        
    Returns:
        Tuple of (chosen_reasoning, chosen_answer)
    """
    if not all_correct_paths:
        return None, None
        
    if len(all_correct_paths) == 1:
        return all_correct_paths[0][0], all_correct_paths[0][1]
    
    # Sort all paths by length (ascending)
    all_correct_paths.sort(key=lambda x: x[2])

    # remove the longest one
    all_correct_paths = all_correct_paths[:-1]
    
    return all_correct_paths[-1][0], all_correct_paths[-1][1]

def process_item(item, processor, model, device, k):
    """
    Process a single item from the dataset
    
    Args:
        item: Dataset item
        processor: Model processor
        model: Model
        device: Device to run on
        k: Number of reasoning attempts
        
    Returns:
        Tuple of (description, is_correct)
    """
    # Generate k reasoning attempts
    reasoning_attempts = generate_answer(item['decoded_image'], item['problem'], processor, model, device, k=k)
    response = item['answer']
    
    # Find correct reasoning paths
    all_correct_paths = []
    
    for reasoning in reasoning_attempts:
        if reasoning is None:
            continue
            
        answer = extract_answer_from_reasoning(reasoning)
        if answer is None:
            continue
            
        # Check if answer is correct
        if "-" in response and len(response) >= 5:
            if response.lower() == answer.lower():
                # Count words in reasoning
                word_count = len(reasoning.split())
                # Add all correct paths to the list
                all_correct_paths.append((reasoning, answer, word_count))
        elif response.lower() == answer.lower() or grade_answer(response, answer):
            # Count words in reasoning
            word_count = len(reasoning.split())
            # Add all correct paths to the list
            all_correct_paths.append((reasoning, answer, word_count))
    
    # Calculate model accuracy for this question
    # model_acc = len(all_correct_paths) / k if k > 0 else 0.0
    
    # Select the best reasoning
    chosen_correct_reasoning, chosen_correct_answer = select_best_reasoning(all_correct_paths)
    
    # If we found a correct reasoning, create description
    if chosen_correct_reasoning is not None:
        description = {
            'image_url': item['image'],
            'instruction': item['problem'],
            'response': item['answer'],
            'reasoning': chosen_correct_reasoning,
            'answer': chosen_correct_answer,
            # 'model_acc': model_acc  # Add model accuracy to the description
        }
        return description, True
    
    return None, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=4, help='Number of reasoning attempts to generate')
    parser.add_argument('--shard_id', type=int, default=0, help='ID of the shard to process (0-based)')
    parser.add_argument('--num_shards', type=int, default=2, help='Total number of shards to split the data into')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing items')
    parser.add_argument('--save_frequency', type=int, default=10, help='Save frequency in number of items')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = args.k
    shard_id = args.shard_id
    num_shards = args.num_shards
    batch_size = args.batch_size
    save_frequency = args.save_frequency

    # Dataset name on Hugging Face
    model_path = "checkpoints/openvlthinker_grpo_iter2_hard_step30" 
    # Output file for descriptions
    output_file = f"./data/sft_iter3_shard{shard_id}.json"
    
    # Initialize LLaVA model and processor
    print(f"Loading {model_path.split('/')[-1]}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load dataset from hub
    data = load_dataset("ydeng9/sft_seed", split="train")
    sources_to_remove = {"scienceqa", "ai2d"} #, "chartqa", "tabmwp", "dvqa", "vizwiz", "iconqa"}
    data = data.filter(lambda example: example['source'] not in sources_to_remove)

    # Calculate shard indices
    start_idx, end_idx = get_shard_indices(len(data), shard_id, num_shards)
    print(f"Processing shard {shard_id+1}/{num_shards} (indices {start_idx} to {end_idx-1})")
    
    # Create list to store descriptions
    descriptions = []
    correct = 0

    # Process each image in the shard
    data = data.select(range(start_idx, end_idx))
    
    # Process items in batches
    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing shard {shard_id+1}/{num_shards}"):
        batch = data.select(range(i, min(i + batch_size, len(data))))
        
        # Process each item in the batch
        for item in batch:
            description, is_correct = process_item(item, processor, model, device, k)
            
            if is_correct:
                correct += 1
                descriptions.append(description)
        
        # Save periodically
        if (i + batch_size) % save_frequency == 0 or i + batch_size >= len(data):
            save_descriptions(descriptions, output_file)
            print(f"\nSaved {len(descriptions)} descriptions to {output_file}")
    
    # Final save
    save_descriptions(descriptions, output_file)
    print(f"\nCompleted! Generated descriptions saved to {output_file}")

    print(f"Accuracy for shard {shard_id+1}: {correct/(end_idx-start_idx)}")

if __name__ == "__main__":
    main() 
