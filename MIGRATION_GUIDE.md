# Migration Guide: From Monolithic Scripts to Modular AlbertMoE

This guide shows how to migrate from the original monolithic scripts (`clm_moealbert.py` and `mlm_moealbert.py`) to the new modular AlbertMoE package.

## What Changed

### Before (Monolithic)
- Two large, nearly identical files (~1074 total lines)
- 95%+ code duplication between CLM and MLM scripts
- All components mixed together in single files
- No proper package structure
- Hard to maintain and extend

### After (Modular)
- Clean package structure with separated concerns
- Zero code duplication
- Proper Python package with setup.py
- Importable modules for custom usage
- Production-ready structure

## Package Structure

```
albertmoe/
├── __init__.py          # Main package exports
├── config.py            # Configuration classes
├── models.py            # Model implementations
├── components.py        # Core components (MoE, Attention, etc.)
├── optimizers.py        # Custom optimizers
├── training.py          # Training utilities
└── evaluation.py        # Evaluation utilities
scripts/
├── train_clm.py         # CLM training script
└── train_mlm.py         # MLM training script
```

## Migration Examples

### 1. Basic Model Usage

**Before:**
```python
# Had to copy/paste classes from the scripts
# No clean imports possible
```

**After:**
```python
from albertmoe import AlbertMoEConfig, AlbertForCausalLM, ChillAdam

config = AlbertMoEConfig(
    vocab_size=30000,
    hidden_size=768,
    num_experts=8
)
model = AlbertForCausalLM(config)
optimizer = ChillAdam(model.parameters())
```

### 2. Training Scripts

**Before:**
```bash
python clm_moealbert.py --mode pretrain --dataset wikitext
python mlm_moealbert.py --mode pretrain --dataset wikitext
```

**After:**
```bash
python scripts/train_clm.py --mode pretrain --dataset wikitext
python scripts/train_mlm.py --mode pretrain --dataset wikitext
```

### 3. Custom Training Loop

**Before:**
```python
# Had to copy entire classes and modify
# No clean separation of concerns
```

**After:**
```python
from albertmoe.training import CLMTrainer, get_common_parser

# Use built-in trainer
parser = get_common_parser()
args = parser.parse_args()
trainer = CLMTrainer.create_trainer(args)

# Or build custom training
from albertmoe import AlbertForCausalLM, AlbertMoEConfig

config = AlbertMoEConfig()
model = AlbertForCausalLM(config)
# Your custom training logic
```

## Component Mapping

| Original Location | New Module | New Class/Function |
|------------------|------------|-------------------|
| ChillAdam optimizer | `albertmoe.optimizers` | `ChillAdam` |
| RotaryEmbedding | `albertmoe.components` | `RotaryEmbedding` |
| MoE layer | `albertmoe.components` | `MoE` |
| Expert | `albertmoe.components` | `Expert` |
| AlbertEmbeddings | `albertmoe.components` | `AlbertEmbeddings` |
| MultiHeadAttention | `albertmoe.components` | `MultiHeadAttention` |
| AlbertLayer | `albertmoe.components` | `AlbertLayer` |
| ALBERT base | `albertmoe.models` | `ALBERT` |
| AlbertForCausalLM | `albertmoe.models` | `AlbertForCausalLM` |
| AlbertForMaskedLM | `albertmoe.models` | `AlbertForMaskedLM` |
| SentenceAlbert | `albertmoe.models` | `SentenceAlbert` |
| Configuration | `albertmoe.config` | `AlbertMoEConfig` |

## Parameter Mapping

All parameters from the original scripts are preserved. The argument parser in `training.py` maintains compatibility:

```python
# Old script arguments work the same way
--vocab_size 30000
--hidden_size 768
--num_experts 8
--top_k_experts 2
--aux_loss_coefficient 0.01
```

## Breaking Changes

### None! 
The migration maintains full backward compatibility for:
- All model parameters and hyperparameters
- Training arguments and configurations  
- Model architectures and forward passes
- Loss computation and optimization

## Benefits of Migration

1. **Code Reuse**: Import and use components in other projects
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new model variants
4. **Testing**: Components can be tested independently
5. **Documentation**: Each module is self-documented
6. **Packaging**: Can be installed as a proper Python package

## Installation

```bash
# From the repository root
pip install -e .

# Or install requirements manually
pip install -r requirements.txt
```

## Quick Start

```python
# Import the package
import albertmoe

# Create and use models
config = albertmoe.AlbertMoEConfig()
model = albertmoe.AlbertForCausalLM(config)

# Use in your training loop
optimizer = albertmoe.ChillAdam(model.parameters())
```

This modular structure provides the same functionality as the original scripts while offering much better maintainability, reusability, and extensibility.