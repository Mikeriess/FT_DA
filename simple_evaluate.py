#!/usr/bin/env python3
"""
Simple evaluation script for the merged Danish model.
Tests basic functionality and Danish language capabilities.
"""

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def load_model_id_from_config(config_path="train_config.yml"):
    """Extract the hub_model_id from the training configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        model_id = config.get('hub_model_id')
        if model_id:
            # Use merged model if available
            merged_model_id = f"{model_id}-merged"
            return merged_model_id
        else:
            raise ValueError("hub_model_id not found in configuration file")
    
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def test_danish_dialogue(model, tokenizer, test_prompts):
    """Test the model on Danish dialogue examples."""
    print("🇩🇰 Testing Danish Dialogue Capabilities")
    print("=" * 50)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt[:50]}...")
        
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(input_text):].strip()
        
        print(f"👤 User: {prompt}")
        print(f"🤖 Assistant: {response}")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": generation_time,
            "input_length": len(inputs.input_ids[0]),
            "output_length": len(outputs[0]) - len(inputs.input_ids[0])
        })
    
    return results

def calculate_basic_metrics(results):
    """Calculate basic performance metrics."""
    total_time = sum(r["generation_time"] for r in results)
    avg_time = total_time / len(results)
    avg_input_length = sum(r["input_length"] for r in results) / len(results)
    avg_output_length = sum(r["output_length"] for r in results) / len(results)
    
    print("\n📊 Performance Metrics")
    print("=" * 30)
    print(f"Total tests: {len(results)}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average input length: {avg_input_length:.1f} tokens")
    print(f"Average output length: {avg_output_length:.1f} tokens")
    print(f"Tokens per second: {avg_output_length/avg_time:.1f}")

def main():
    """Main evaluation function."""
    print("🚀 Simple Danish Model Evaluation")
    print("=" * 50)
    
    # Load model ID
    model_id = load_model_id_from_config()
    if not model_id:
        print("❌ Could not load model ID from config")
        return
    
    print(f"📋 Testing model: {model_id}")
    
    try:
        # Load model and tokenizer
        print("🔧 Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Parameters: {model.num_parameters():,}")
        print(f"📚 Vocabulary size: {len(tokenizer)}")
        
        # Danish test prompts
        test_prompts = [
            "Hej! Hvordan har du det i dag?",
            "Kan du forklare mig, hvad kunstig intelligens er?",
            "Hvad er forskellen mellem kaffe og te?",
            "Fortæl mig en kort historie om en kat.",
            "Hvordan laver man pandekager?",
            "Hvad er vejret som i dag?",
            "Kan du hjælpe mig med at lære dansk?",
            "Hvad synes du om dansk kultur?"
        ]
        
        # Run tests
        results = test_danish_dialogue(model, tokenizer, test_prompts)
        
        # Calculate metrics
        calculate_basic_metrics(results)
        
        # Save results
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simple_eval_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_id": model_id,
                "timestamp": timestamp,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 