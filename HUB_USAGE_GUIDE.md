# Hugging Face Hub Integration Usage Guide

This guide shows how to use the new Hugging Face Hub integration features in AlbertMoE.

## Quick Start

### 1. Set up your Hugging Face token

```bash
export HF_TOKEN="your_hugging_face_token"
```

### 2. Train and upload a model

```bash
# Train CLM model and upload to Hub
python scripts/train_clm.py \
    --task_type clm \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_epochs 3 \
    --batch_size 8 \
    --push_to_hub username/my-albert-clm \
    --hub_token $HF_TOKEN

# Train MLM model and upload to Hub
python scripts/train_mlm.py \
    --task_type mlm \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_epochs 3 \
    --batch_size 8 \
    --push_to_hub username/my-albert-mlm \
    --hub_private  # Create private repo
```

## Programmatic Usage

```python
from albertmoe import AlbertMoEConfig, AlbertForCausalLM, CLMTrainer, push_to_hub
from transformers import AutoTokenizer

# 1. Create and train your model
config = AlbertMoEConfig(vocab_size=30000, hidden_size=768, num_experts=8)
model = AlbertForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

# ... training code ...

# 2. Save locally and push to Hub
trainer.save_model(
    save_path="./my_model",
    push_to_hub_repo="username/my-albert-model",
    hub_token="your_token",
    hub_private=False,
    training_args={
        "dataset": "wikitext",
        "epochs": 3,
        "batch_size": 8
    }
)

# 3. Or use the direct function
push_to_hub(
    local_path="./my_model",
    repo_id="username/my-albert-model",
    model_config=config.__dict__,
    task_type="clm",
    token="your_token"
)
```

## Features

### Automatic Repository Management
- Creates new repositories if they don't exist
- Updates existing repositories with new model versions
- Handles both public and private repositories

### Generated Files
- **README.md**: Comprehensive model card with usage examples
- **config.json**: Model configuration for Hub compatibility
- **pytorch_model.bin**: Model weights
- **tokenizer files**: For proper tokenization

### Error Handling
- Graceful handling of network issues
- Clear error messages for authentication problems
- Fallback to local-only saving if Hub upload fails

## Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--push_to_hub` | Hub repository ID (e.g., 'username/model-name') | No |
| `--hub_token` | Hugging Face authentication token | No* |
| `--hub_private` | Create private repository | No |
| `--hub_commit_message` | Custom commit message | No |

*Required if not set via environment variables

## Environment Variables

```bash
# Option 1
export HF_TOKEN="your_token"

# Option 2  
export HUGGINGFACE_HUB_TOKEN="your_token"
```

## Examples

### Create a public model
```bash
python scripts/train_clm.py --push_to_hub myusername/albert-moe-demo
```

### Create a private model
```bash
python scripts/train_clm.py --push_to_hub myusername/albert-moe-private --hub_private
```

### Update an existing model
```bash
python scripts/train_clm.py --push_to_hub myusername/albert-moe-demo --hub_commit_message "Updated with better hyperparameters"
```

## Troubleshooting

### Authentication Issues
- Ensure your token has write permissions
- Check that the token is correctly set in environment variables
- Verify the repository name format: `username/repository-name`

### Network Issues
- Models save locally even if Hub upload fails
- Check internet connectivity
- Retry upload manually using the `push_to_hub()` function

### Repository Conflicts
- If repository exists but you don't have access, use a different name
- For organizations, ensure you have the correct permissions

## Model Card Generation

The system automatically generates comprehensive model cards including:

- Model architecture and configuration
- Training details and hyperparameters  
- Usage examples in Python
- Citation information
- Performance metrics (if available)

This ensures your models are well-documented and easy for others to use!