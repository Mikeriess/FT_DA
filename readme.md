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

### Dataset Info
- **Source**: 3,000 Danish dialogue samples with synthetic augmentation
- **Split**: Conversation-aware 85/15 train/validation split
- **Target**: Fine-tune assistant responses only

