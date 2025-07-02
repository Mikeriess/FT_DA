#!/usr/bin/env python3
"""
Script to automatically evaluate the fine-tuned model using EuroEval.
Reads the model ID from train_config.yml and benchmarks on Danish tasks.
Results are saved locally in eval/<model_name>/ directory.
"""

import yaml
import subprocess
import sys
import os
from datetime import datetime

def load_model_id_from_config(config_path="train_config.yml"):
    """Extract the hub_model_id from the training configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        model_id = config.get('hub_model_id')
        if not model_id:
            raise ValueError("hub_model_id not found in configuration file")
        
        print(f"Found model ID: {model_id}")
        return model_id
    
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

def create_eval_directory(model_id):
    """Create evaluation directory based on model ID."""
    # Convert model ID to safe directory name (replace / with _)
    safe_model_name = model_id.replace("/", "_")
    
    # Create eval directory structure
    eval_dir = os.path.join("eval", safe_model_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Evaluation results will be saved to: {eval_dir}")
    return eval_dir

def check_euroeval_installed():
    """Check if EuroEval is installed, and install if not."""
    try:
        import euroeval
        print("EuroEval is already installed.")
        return True
    except ImportError:
        print("EuroEval not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "euroeval[all]"])
            print("EuroEval installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install EuroEval: {e}")
            return False

def run_euroeval_benchmark(model_id, eval_dir, language="da", split="validation"):
    """
    Run EuroEval benchmark on the specified model for Danish language tasks.
    
    Args:
        model_id (str): HuggingFace model ID
        eval_dir (str): Directory to save evaluation results
        language (str): Language code (default: "da" for Danish)
        split (str): Dataset split to use (default: "validation")
    """
    print(f"Starting EuroEval benchmark for model: {model_id}")
    print(f"Language: {language}")
    print(f"Split: {split}")
    print(f"Results directory: {eval_dir}")
    print("-" * 50)
    
    # Generate timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output files
    log_file = os.path.join(eval_dir, f"euroeval_log_{timestamp}.txt")
    
    # EuroEval command with specific parameters
    # Force use of transformers backend instead of vLLM to support LoRA adapters
    cmd = [
        "euroeval",
        "--model", model_id,
        "--language", language,
        "--split", split,
        "--save-results",
        "--verbose",
        "--batch-size", "1"  # Smaller batch size for transformers backend
    ]
    
    try:
        # Run the benchmark with output logging
        print(f"Running command: {' '.join(cmd)}")
        
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # Write command and timestamp to log
            log_f.write(f"EuroEval Benchmark Log\n")
            log_f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            log_f.write(f"Model: {model_id}\n")
            log_f.write(f"Language: {language}\n")
            log_f.write(f"Split: {split}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("-" * 50 + "\n\n")
            log_f.flush()
            
            # Run the benchmark
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=log_f, 
                stderr=subprocess.STDOUT, 
                text=True
            )
        
        print("-" * 50)
        print("EuroEval benchmark completed successfully!")
        print(f"📝 Full log saved to: {log_file}")
        
        # Create a summary file
        create_evaluation_summary(eval_dir, model_id, timestamp, None, log_file)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"EuroEval benchmark failed with exit code {e.returncode}")
        print("This might be because:")
        print("1. The model is not yet available on HuggingFace Hub")
        print("2. The model is private and requires authentication")
        print("3. Network connectivity issues")
        print("\nTo manually run the benchmark later:")
        print(f"euroeval --model {model_id} --language {language} --split {split}")
        return False
        
    except FileNotFoundError:
        print("Error: 'euroeval' command not found.")
        print("Please ensure EuroEval is properly installed:")
        print("pip install euroeval[all]")
        return False

def create_evaluation_summary(eval_dir, model_id, timestamp, results_file, log_file):
    """Create a summary file for the evaluation."""
    summary_file = os.path.join(eval_dir, f"evaluation_summary_{timestamp}.md")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# EuroEval Benchmark Summary\n\n")
        f.write(f"**Model:** {model_id}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Language:** Danish (da)\n")
        f.write(f"**Split:** validation\n\n")
        f.write(f"## Files Generated\n\n")
        if results_file and os.path.exists(results_file):
            f.write(f"- **Results:** `{os.path.basename(results_file)}`\n")
        f.write(f"- **Log:** `{os.path.basename(log_file)}`\n")
        f.write(f"- **Summary:** `{os.path.basename(summary_file)}`\n\n")
        f.write(f"## Manual Command\n\n")
        f.write(f"```bash\n")
        f.write(f"euroeval --model {model_id} --language da --split validation\n")
        f.write(f"```\n")
    
    print(f"📋 Summary saved to: {summary_file}")

def main():
    """Main function to orchestrate the evaluation process."""
    print("🚀 Automated Model Evaluation with EuroEval")
    print("=" * 50)
    
    # Step 1: Load model ID from config
    model_id = load_model_id_from_config()
    
    # Step 2: Create evaluation directory
    eval_dir = create_eval_directory(model_id)
    
    # Step 3: Check/install EuroEval
    if not check_euroeval_installed():
        print("Failed to install EuroEval. Exiting.")
        sys.exit(1)
    
    # Step 4: Run the benchmark
    success = run_euroeval_benchmark(model_id, eval_dir, language="da", split="validation")
    
    if success:
        print("\n✅ Evaluation completed!")
        print(f"Your Danish model '{model_id}' has been benchmarked on EuroEval.")
        print(f"📁 All results saved in: {eval_dir}")
        print("\nResults will show how your fine-tuned model performs on standardized Danish NLP tasks.")
    else:
        print("\n❌ Evaluation failed.")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 