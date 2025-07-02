# Danish Dialogue Fine-tuning with Axolotl

Fine-tune Llama-3.2-1B on Danish dialogues using LoRA and the official Axolotl Docker image.

## Quick Start (OpenShift Pod)

### 1. Deploy the Pod

This project uses the official Axolotl Docker image, which includes all necessary dependencies. Deploy the pod to your OpenShift cluster using the provided configuration:

```bash
oc delete pod axolotl-gpu
```
```bash
oc apply -f axolotl-gpu-pod.yaml
```
```bash
oc get pods
```

### 2. Copy Project Files

Copy your local project files into the running pod:

```bash
oc cp ./ axolotl-gpu:/workspace/
```

### 3. Access the Pod

Open a shell inside the pod to execute the next steps:

```bash
oc rsh axolotl-gpu
```

### 4. Authenticate Services

Inside the pod, authenticate with Weights & Biases and Hugging Face:

```bash
# Login to Weights & Biases
wandb login

# Login to HuggingFace Hub
huggingface-cli login
```

### 5. Prepare the Dataset

Run the script to prepare the conversation-aware dataset split:

```bash
python prepare_dataset.py
```

### 6. Run Training

The `train_config.yml` is pre-configured. Start the training process:

```bash
cd ../
accelerate launch -m axolotl.cli.train train_config.yml
```

### 7. Monitor Training
- **Weights & Biases:** [wandb.ai/mikeriess/Llama-3.2-1B_LM](https://wandb.ai/mikeriess/Llama-3.2-1B_LM)
- **Hugging Face:** Models are automatically uploaded to [mikeriess/Llama-3.2-1B_LM](httpse://huggingface.co/mikeriess/Llama-3.2-1B_LM)

### 8. Create Merged Model (Post-Training)

After training completes, create a merged model for easier evaluation and deployment:

```bash
python merge_and_upload.py
```

This script will:
1. Load your LoRA adapter from HuggingFace
2. Merge it with the base model (Llama-3.2-1B)
3. Upload the merged model to `mikeriess/Llama-3.2-1B_LM-merged`
4. Update evaluation scripts with the merged model ID

## Evaluation

After training completes, evaluate your fine-tuned model on Danish language benchmarks using EuroEval.

### Automated Evaluation

The project includes an automated evaluation script that reads your model ID from the training configuration and runs comprehensive Danish benchmarks:

```bash
python evaluate_model.py
```

This script will:
1. **Read model ID** from `train_config.yml` (uses `hub_model_id` field)
2. **Install EuroEval** if not already available
3. **Run Danish benchmarks** on the validation split
4. **Save results** in organized directory structure: `eval/<model_name>/`
5. **Generate logs and summaries** with timestamps

### Manual Evaluation

You can also run EuroEval manually with custom parameters:

```bash
# Install EuroEval (if needed)
pip install euroeval[all]

# Evaluate on Danish tasks (validation split)
euroeval --model mikeriess/Llama-3.2-1B_LM --language da --split validation --save-results --verbose

# Evaluate on test split (final evaluation only)
euroeval --model mikeriess/Llama-3.2-1B_LM --language da --split test --save-results --verbose
```

### Evaluation Results

Results are saved in the following structure:
```
eval/
└── mikeriess_Llama-3.2-1B_LM/
    ├── euroeval_log_20241215_143022.txt       # Detailed benchmark log
    ├── evaluation_summary_20241215_143022.md  # Summary report
    └── euroeval_benchmark_results.jsonl       # Structured results data
```

### Understanding Results

EuroEval benchmarks your model on standardized Danish NLP tasks including:
- **Text Classification**: Sentiment analysis, topic classification
- **Natural Language Inference**: Logical reasoning in Danish
- **Question Answering**: Reading comprehension tasks
- **Language Modeling**: Perplexity on Danish text

Higher scores indicate better performance on Danish language understanding tasks.

### Dataset Info
- **Source**: 30,000 Danish dialogue samples with synthetic augmentation
- **Split**: Conversation-aware 85/15 train/validation split
- **Target**: Fine-tune assistant responses only

