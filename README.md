# AlbertMoE

A modular and production-ready implementation of ALBERT (A Lite BERT) with Mixture of Experts (MoE) architecture, supporting both Causal Language Modeling (CLM) and Masked Language Modeling (MLM) objectives.

## 🚀 Features

- **Modular Architecture**: Clean separation of components for easy extension and maintenance
- **Dual Training Objectives**: Support for both CLM (GPT-style) and MLM (BERT-style) training
- **Mixture of Experts**: Efficient MoE implementation for better model capacity
- **Production Ready**: Proper Python package structure with comprehensive configuration
- **Advanced Optimizer**: Custom ChillAdam optimizer combining Adam with adaptive learning rates
- **Rotary Embeddings**: Modern positional encoding for better sequence understanding
- **MTEB Evaluation**: Built-in support for evaluating sentence embeddings
- **Weights & Biases Integration**: Comprehensive logging and experiment tracking
- **🤗 Hugging Face Hub Integration**: Seamless model sharing and collaboration

## 📦 Installation

### From Source

```bash
git clone https://github.com/mlnomadpy/albertmoe.git
cd albertmoe
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- See `requirements.txt` for full dependency list

## 🎯 Quick Start

### Training a Causal Language Model (CLM)

```bash
python scripts/train_clm.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_epochs 3 \
    --batch_size 8 \
    --hidden_size 768 \
    --num_experts 8 \
    --max_position_embeddings 512 \
    --use_wandb
```

### Training with Hugging Face Hub Integration

```bash
python scripts/train_clm.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_epochs 3 \
    --batch_size 8 \
    --hidden_size 768 \
    --num_experts 8 \
    --max_position_embeddings 512 \
    --push_to_hub username/my-albert-moe \
    --hub_token your_hf_token \
    --use_wandb
```

### Training a Masked Language Model (MLM)

```bash
python scripts/train_mlm.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_epochs 3 \
    --batch_size 8 \
    --hidden_size 768 \
    --num_experts 8 \
    --max_position_embeddings 512 \
    --use_wandb
```

### Training with Streaming Datasets 🚀

For large datasets that don't fit in memory, AlbertMoE supports streaming datasets:

```bash
# Stream a large dataset with memory efficiency
python scripts/train_clm.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --streaming \
    --max_samples 50000 \
    --batch_size 8 \
    --hidden_size 768 \
    --num_experts 8 \
    --max_position_embeddings 512 \
    --use_wandb
```

```bash
# Stream with custom text column
python scripts/train_clm.py \
    --mode pretrain \
    --dataset squad \
    --dataset_config plain_text \
    --text_column context \
    --streaming \
    --max_samples 10000 \
    --batch_size 4 \
    --max_position_embeddings 512
```

**Benefits of Streaming:**
- 🔥 Memory efficient - doesn't load entire dataset into RAM
- ⚡ Faster startup time for large datasets
- 🎯 Controllable dataset size with `--max_samples`
- 🌐 Works with any Hugging Face dataset

### Using as a Python Library

```python
from albertmoe import AlbertMoEConfig, AlbertForCausalLM, ChillAdam, push_to_hub

# Create configuration
config = AlbertMoEConfig(
    vocab_size=30000,
    hidden_size=768,
    num_hidden_layers=12,
    num_experts=8,
    top_k_experts=2
)

# Create model
model = AlbertForCausalLM(config)

# Create optimizer
optimizer = ChillAdam(model.parameters())

# Your training loop here...

# Save locally and push to Hub
push_to_hub(
    local_path="./my_model",
    repo_id="username/my-albert-moe",
    model_config=config.__dict__,
    task_type="clm",
    token="your_hf_token"
)
```

## 🏗️ Architecture

### Package Structure

```
albertmoe/
├── __init__.py          # Main package exports
├── config.py            # Configuration classes
├── models.py            # Model implementations (CLM, MLM)
├── components.py        # Core components (MoE, Attention, etc.)
├── optimizers.py        # Custom optimizers
├── training.py          # Training utilities and trainers
└── evaluation.py        # Evaluation utilities
scripts/
├── train_clm.py         # CLM training script
└── train_mlm.py         # MLM training script
```

### Key Components

- **AlbertMoEConfig**: Comprehensive configuration class
- **ALBERT**: Base model with shared layers
- **AlbertForCausalLM**: Model for next-token prediction (GPT-style)
- **AlbertForMaskedLM**: Model for masked token prediction (BERT-style)
- **MoE**: Mixture of Experts implementation with routing
- **ChillAdam**: Advanced optimizer with adaptive learning rates
- **RotaryEmbedding**: Modern positional encoding

## ⚙️ Configuration

### Model Configuration

```python
config = AlbertMoEConfig(
    # Model architecture
    vocab_size=30000,              # Vocabulary size
    embedding_size=128,            # Embedding dimension
    hidden_size=768,               # Hidden state dimension
    num_hidden_layers=12,          # Number of transformer layers
    num_attention_heads=12,        # Number of attention heads
    intermediate_size=3072,        # FFN intermediate size
    
    # MoE configuration
    num_experts=8,                 # Number of experts
    top_k_experts=2,               # Active experts per token
    aux_loss_coefficient=0.01,     # Auxiliary loss weight
    
    # Training configuration
    max_position_embeddings=512,   # Maximum sequence length
    use_rotary=True,              # Use rotary embeddings
)
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: pretrain, evaluate, serve, all | all |
| `--task_type` | Task type: clm, mlm | Required |
| `--dataset` | Dataset name | wikitext |
| `--batch_size` | Training batch size | 8 |
| `--num_epochs` | Number of training epochs | 3 |
| `--hidden_size` | Model hidden size | 768 |
| `--num_experts` | Number of MoE experts | 8 |
| `--max_position_embeddings` | Maximum sequence length | 512 |
| `--streaming` | Enable dataset streaming for large datasets | False |
| `--max_samples` | Maximum number of samples to use from dataset | None |
| `--text_column` | Name of the text column in the dataset | text |
| `--use_wandb` | Enable Weights & Biases logging | False |
| `--push_to_hub` | Hub repository ID (e.g., 'username/model-name') | None |
| `--hub_token` | Hugging Face authentication token | None |
| `--hub_private` | Create private repository on Hub | False |

## 🚀 Serving Mode for Inference

AlbertMoE now supports a dedicated serving mode for using pretrained models for inference without requiring the full training setup.

### CLM (Causal Language Model) Serving

Use your trained CLM models for text generation and completion:

```bash
# Interactive text generation
python scripts/train_clm.py \
    --mode serve \
    --model_path ./my_clm_model \
    --interactive

# Single text completion
python scripts/train_clm.py \
    --mode serve \
    --model_path ./my_clm_model \
    --input_text "Once upon a time" \
    --gen_max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95
```

### MLM (Masked Language Model) Serving

Use your trained MLM models for masked token prediction:

```bash
# Interactive masked token prediction
python scripts/train_mlm.py \
    --mode serve \
    --model_path ./my_mlm_model \
    --interactive

# Single masked token prediction
python scripts/train_mlm.py \
    --mode serve \
    --model_path ./my_mlm_model \
    --input_text "The weather today is [MASK] and [MASK]."
```

### Serving Configuration Options

| Argument | Description | Default | Task |
|----------|-------------|---------|------|
| `--interactive` | Enable interactive serving mode | False | Both |
| `--input_text` | Input text for non-interactive mode | None | Both |
| `--gen_max_length` | Maximum generation length | 100 | CLM |
| `--temperature` | Sampling temperature | 1.0 | CLM |
| `--top_k` | Top-k sampling | 50 | CLM |
| `--top_p` | Nucleus sampling threshold | 0.95 | CLM |

### Serving Mode Features

**CLM Serving:**
- 🎯 **Text Generation**: Complete text given a prompt
- ⚙️ **Configurable Sampling**: Temperature, top-k, top-p parameters
- 🔄 **Interactive Mode**: Continuous prompt-response interface
- 📝 **Single Prediction**: One-time generation with `--input_text`

**MLM Serving:**
- 🎭 **Masked Token Prediction**: Predict `[MASK]` tokens in sentences
- 📊 **Top-K Predictions**: Multiple candidates with probability scores
- 🔄 **Interactive Mode**: Continuous masked text prediction
- 📝 **Single Prediction**: One-time prediction with `--input_text`

### Example Workflow

```bash
# 1. Train a model
python scripts/train_clm.py --mode pretrain --num_epochs 3

# 2. Serve the trained model interactively
python scripts/train_clm.py --mode serve \
    --model_path ./albert_from_scratch_output \
    --interactive

# 3. Or use for single predictions
python scripts/train_clm.py --mode serve \
    --model_path ./albert_from_scratch_output \
    --input_text "The future of AI is" \
    --gen_max_length 50
```

## 🤗 Hugging Face Hub Integration

AlbertMoE now supports seamless integration with Hugging Face Hub for model sharing and collaboration.

### Automatic Model Upload

Train and automatically upload to Hub:

```bash
# CLM training with Hub upload
python scripts/train_clm.py \
    --push_to_hub username/my-albert-clm \
    --hub_token your_hf_token \
    --max_position_embeddings 512 \
    --hub_private  # Optional: create private repo

# MLM training with Hub upload  
python scripts/train_mlm.py \
    --push_to_hub username/my-albert-mlm \
    --hub_token your_hf_token \
    --max_position_embeddings 512
```

### Programmatic Hub Integration

```python
from albertmoe import push_to_hub, AlbertMoEConfig

# After training your model locally
success = push_to_hub(
    local_path="./my_trained_model",
    repo_id="username/my-albert-moe", 
    model_config=config.__dict__,
    task_type="clm",  # or "mlm"
    token="your_hf_token",
    private=False,
    commit_message="Upload trained AlbertMoE model"
)
```

### Repository Management

The Hub integration automatically:

- **Creates repositories** if they don't exist
- **Updates existing repositories** with new model versions
- **Generates model cards** with training details and usage examples
- **Creates config.json** for model compatibility
- **Handles authentication** via tokens or environment variables

### Environment Variables

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
# or
export HUGGINGFACE_HUB_TOKEN="your_hugging_face_token"
```

### Generated Model Card

The integration automatically creates comprehensive model cards including:

- Model architecture details
- Training configuration
- Usage examples
- Performance metrics
- Citation information

## 🔬 Evaluation

The package includes comprehensive evaluation capabilities using MTEB (Massive Text Embedding Benchmark):

```bash
# Evaluate on basic tasks
python scripts/train_clm.py --mode evaluate --eval_tasks basic --max_position_embeddings 512

# Evaluate on comprehensive benchmark
python scripts/train_clm.py --mode evaluate --eval_tasks comprehensive --max_position_embeddings 512

# Custom tasks
python scripts/train_clm.py --mode evaluate --custom_tasks STSBenchmark SummEval --max_position_embeddings 512
```

## 📊 Monitoring

### Weights & Biases Integration

Enable comprehensive experiment tracking:

```bash
export WANDB_PROJECT="your-project"
python scripts/train_clm.py --use_wandb --wandb_project albert-moe-experiments
```

Logged metrics include:
- Training losses (total, task-specific, auxiliary)
- Learning rates per parameter group
- MTEB evaluation scores
- Model configuration and hyperparameters

## 🧪 Advanced Usage

### Custom Training Loop

```python
from albertmoe import CLMTrainer, AlbertMoEConfig
from albertmoe.training import get_common_parser

# Parse arguments
parser = get_common_parser()
args = parser.parse_args()

# Create trainer
trainer = CLMTrainer.create_trainer(args)

# Prepare dataset
dataset = trainer.prepare_dataset("wikitext", "wikitext-2-raw-v1", "train")

# Your custom training logic...
```

### Model Inference

```python
import torch
from transformers import AutoTokenizer
from albertmoe import AlbertForCausalLM, AlbertMoEConfig

# Load trained model
config = AlbertMoEConfig()
model = AlbertForCausalLM(config)
model.load_state_dict(torch.load("path/to/model.bin"))

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer("Hello world", return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs["logits"]
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ALBERT paper: [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
- Mixture of Experts: [Switch Transformer: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- RoPE: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## 📞 Support

For questions, issues, or suggestions:

- Open an issue on GitHub
- Check existing documentation
- Review example usage in the scripts

---

**Happy Training! 🎉**