# Danish Dialogue Fine-tuning with Axolotl

Fine-tune Llama-3.2-1B on Danish dialogues using LoRA and conversation-aware dataset splitting.

## Quick Start (OpenShift Pod)

### Prerequisites
- OpenShift pod with NVIDIA drivers and Python installed
- CLI access to the pod

### Step-by-Step Procedure

1. **Install Dependencies**
```bash
pip install datasets pandas scikit-learn
pip install axolotl[flash-attn,deepspeed] --extra-index-url https://download.pytorch.org/whl/cu121
```

2. **Authenticate Services**
```bash
# Login to Weights & Biases
wandb login
# Enter your WandB API key when prompted
# Get from wandb.ai/settings → API Keys

# Login to HuggingFace Hub
huggingface-cli login
# Enter your HF token when prompted
#  Get from huggingface.co/settings/tokens → New Token (with write permissions)
```

3. **Prepare Dataset with Conversation-Aware Split**
```bash
python prepare_dataset.py
```

4. **Update Training Config**
Edit `train_config.yml` to use local split dataset:
```yaml
datasets:
  - path: ./data/danish_dialogues_split
    val_split: validation
# Remove: val_set_size: 0.15
```

5. **Run Training**
```bash
accelerate launch -m axolotl.cli.train train_config.yml
```

6. **Monitor Training**
- Check WandB: `wandb.ai/mikeriess/Llama-3.2-1B_LM`
- Model auto-uploads to: `mikeriess/Llama-3.2-1B_LM`

### Dataset Info
- **Source**: 3,000 Danish dialogue samples with synthetic augmentation
- **Split**: Conversation-aware 85/15 train/validation split
- **Target**: Fine-tune assistant responses only

