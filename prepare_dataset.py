#!/usr/bin/env python3
"""
Script to create a proper train/validation split for the Danish dialogue dataset.
Splits by conversation_id to avoid data leakage between train/val sets.
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def create_conversation_split():
    """Split dataset by conversation_id to avoid leakage."""
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("mikeriess/LM_dialogues_3k_gemma_2_27b_prompt1", split="train")
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)
    
    # Get unique conversation IDs
    unique_conversations = df['conversation_id'].unique()
    print(f"Total conversations: {len(unique_conversations)}")
    print(f"Total samples: {len(df)}")
    
    # Split conversation IDs (not individual samples)
    train_conv_ids, val_conv_ids = train_test_split(
        unique_conversations, 
        test_size=0.15, 
        random_state=42
    )
    
    # Split dataframe based on conversation IDs
    train_df = df[df['conversation_id'].isin(train_conv_ids)]
    val_df = df[df['conversation_id'].isin(val_conv_ids)]
    
    print(f"Train conversations: {len(train_conv_ids)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val conversations: {len(val_conv_ids)}")
    print(f"Val samples: {len(val_df)}")
    
    # Convert back to datasets
    from datasets import Dataset, DatasetDict
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # Save locally
    dataset_dict.save_to_disk("./data/danish_dialogues_split")
    print("Dataset saved to ./data/danish_dialogues_split")
    
    # Optionally push to HuggingFace Hub
    # dataset_dict.push_to_hub("mikeriess/LM_dialogues_3k_gemma_2_27b_prompt1_split")
    
    return dataset_dict

if __name__ == "__main__":
    create_conversation_split() 