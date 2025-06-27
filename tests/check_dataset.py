from datasets import load_from_disk
import sys
import json

# The field name from your train_config.yml
MESSAGES_FIELD = "synthetic_messages"
DATASET_PATH = './data/danish_dialogues_split'

def check_dataset_format():
    """
    Loads the dataset and checks if the message structure is what Axolotl expects for chat templates.
    """
    try:
        print(f"--- Attempting to load dataset from: {DATASET_PATH} ---")
        dataset = load_from_disk(DATASET_PATH)
        print("✅ Dataset loaded successfully!")
        print("\nDataset structure:")
        print(dataset)

        if 'train' not in dataset or len(dataset['train']) == 0:
            print("\n❌ 'train' split not found or is empty. Cannot check format.")
            return

        print("\n--- Inspecting 'train' split ---")
        train_split = dataset['train']
        first_example = train_split[0]

        print("\nFull content of the first example row:")
        print(json.dumps(first_example, indent=2, ensure_ascii=False))

        # --- Deeper check for the message format ---
        if MESSAGES_FIELD not in first_example or not first_example[MESSAGES_FIELD]:
            print(f"\n❌ ERROR: Field '{MESSAGES_FIELD}' not found or is empty in the first example.")
            print("Available fields are:", list(first_example.keys()))
            return
        
        print(f"\n✅ Field '{MESSAGES_FIELD}' found. Inspecting its content...")
        
        messages_list = first_example[MESSAGES_FIELD]
        if not isinstance(messages_list, list) or len(messages_list) == 0:
             print(f"\n❌ ERROR: '{MESSAGES_FIELD}' is not a list or is empty.")
             return
        
        first_message = messages_list[0]
        print("\nContent of the first message in the list:")
        print(json.dumps(first_message, indent=2, ensure_ascii=False))

        # Check for required keys
        required_keys = {"role", "content"}
        actual_keys = set(first_message.keys())

        if required_keys.issubset(actual_keys):
            print(f"\n✅ The first message contains the required keys: {required_keys}")
        else:
            print(f"\n❌ ERROR: The first message is MISSING required keys.")
            print(f"   - Expected keys: {required_keys}")
            print(f"   - Actual keys found: {actual_keys}")
            print(f"\nSuggestion: Modify 'prepare_dataset.py' to format each message dictionary to have 'role' and 'content' keys.")
            print("For example, rename the 'text' key to 'content' and add a 'role' key based on 'poster_id'.")


    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    check_dataset_format() 