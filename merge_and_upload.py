#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model and upload to HuggingFace Hub.
Reads configuration from train_config.yml and creates a merged model.
"""

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys
from datetime import datetime

def load_config(config_path="train_config.yml"):
    """Load training configuration to get model details."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

def merge_and_upload_model(config):
    """
    Merge LoRA adapter with base model and upload to HuggingFace Hub.
    
    Args:
        config (dict): Training configuration dictionary
    """
    # Extract configuration
    base_model_name = config.get('base_model')
    adapter_model_id = config.get('hub_model_id')
    
    if not base_model_name or not adapter_model_id:
        print("Error: Missing base_model or hub_model_id in configuration")
        sys.exit(1)
    
    print(f"🚀 Starting model merge process")
    print(f"Base model: {base_model_name}")
    print(f"Adapter model: {adapter_model_id}")
    print("-" * 50)
    
    try:
        # Load tokenizer from the adapter (which has the correct vocab size)
        print("📝 Loading tokenizer from adapter...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
        
        # Load base model
        print("🔧 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize base model embeddings to match adapter's vocabulary size
        print("🔧 Resizing base model embeddings to match adapter vocabulary...")
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load and merge LoRA adapter
        print("🔗 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_model_id)
        
        print("⚡ Merging adapter with base model...")
        merged_model = model.merge_and_unload()
        
        # Create merged model repository name
        merged_model_id = f"{adapter_model_id}-merged"
        
        print(f"📤 Uploading merged model to: {merged_model_id}")
        
        # Upload merged model
        merged_model.push_to_hub(
            merged_model_id,
            private=False,
            commit_message=f"Merged LoRA adapter with {base_model_name} - {datetime.now().isoformat()}"
        )
        
        # Upload tokenizer
        tokenizer.push_to_hub(
            merged_model_id,
            private=False,
            commit_message=f"Tokenizer for merged model - {datetime.now().isoformat()}"
        )
        
        print("✅ Model merge and upload completed successfully!")
        print(f"🎯 Merged model available at: https://huggingface.co/{merged_model_id}")
        
        # Update evaluation script suggestion
        print("\n📋 To evaluate the merged model, use:")
        print(f"euroeval --model {merged_model_id} --language da --split validation --save-results --verbose")
        
        return merged_model_id
        
    except Exception as e:
        print(f"❌ Error during merge and upload: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're logged into HuggingFace: huggingface-cli login")
        print("2. Check that the adapter model exists and is accessible")
        print("3. Verify you have write permissions to the repository")
        sys.exit(1)

def update_evaluation_script(merged_model_id):
    """
    Optionally update the evaluation script to use the merged model.
    
    Args:
        merged_model_id (str): ID of the merged model
    """
    try:
        # Read current evaluation script
        with open("evaluate_model.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add a note about the merged model
        note = f"""
# NOTE: Merged model available at: {merged_model_id}
# You can evaluate it directly with:
# euroeval --model {merged_model_id} --language da --split validation --save-results --verbose
"""
        
        # Prepend the note to the file
        updated_content = note + content
        
        with open("evaluate_model.py", 'w', encoding='utf-8') as f:
            f.write(updated_content)
            
        print(f"📝 Updated evaluate_model.py with merged model information")
        
    except Exception as e:
        print(f"Warning: Could not update evaluation script: {e}")

def main():
    """Main function to orchestrate the merge and upload process."""
    print("🔄 LoRA Adapter Merge and Upload Tool")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Perform merge and upload
    merged_model_id = merge_and_upload_model(config)
    
    # Update evaluation script with merged model info
    update_evaluation_script(merged_model_id)
    
    print("\n🎉 Process completed!")
    print(f"Your merged model '{merged_model_id}' is ready for evaluation.")

if __name__ == "__main__":
    main() 