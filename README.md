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
    --use_wandb
```

### Using as a Python Library

```python
from albertmoe import AlbertMoEConfig, AlbertForCausalLM, ChillAdam

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
| `--mode` | Training mode: pretrain, evaluate, all | all |
| `--task_type` | Task type: clm, mlm | Required |
| `--dataset` | Dataset name | wikitext |
| `--batch_size` | Training batch size | 8 |
| `--num_epochs` | Number of training epochs | 3 |
| `--hidden_size` | Model hidden size | 768 |
| `--num_experts` | Number of MoE experts | 8 |
| `--use_wandb` | Enable Weights & Biases logging | False |

## 🔬 Evaluation

The package includes comprehensive evaluation capabilities using MTEB (Massive Text Embedding Benchmark):

```bash
# Evaluate on basic tasks
python scripts/train_clm.py --mode evaluate --eval_tasks basic

# Evaluate on comprehensive benchmark
python scripts/train_clm.py --mode evaluate --eval_tasks comprehensive

# Custom tasks
python scripts/train_clm.py --mode evaluate --custom_tasks STSBenchmark SummEval
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