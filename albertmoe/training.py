"""
Training utilities for AlbertMoE models.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import transformers

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import AlbertMoEConfig
from .models import AlbertForCausalLM, AlbertForMaskedLM
from .optimizers import ChillAdam

# Set the transformers logger to warning level to suppress step-by-step loss prints
transformers.logging.set_verbosity_warning()


class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(self, config, model, tokenizer, optimizer, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        
    def prepare_dataset(self, dataset_name, dataset_config, dataset_split, max_length=512):
        """Prepare dataset for training."""
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        return tokenized_dataset
    
    def train_epoch(self, dataloader, use_wandb=False):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_clm_loss = 0.0
        total_mlm_loss = 0.0
        total_aux_loss = 0.0
        
        for step, batch in enumerate(tqdm(dataloader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_aux_loss += outputs["aux_loss"].item()
            
            if "clm_loss" in outputs and outputs["clm_loss"] is not None:
                total_clm_loss += outputs["clm_loss"].item()
            if "mlm_loss" in outputs and outputs["mlm_loss"] is not None:
                total_mlm_loss += outputs["mlm_loss"].item()
            
            # Log to wandb
            if use_wandb and step % 100 == 0 and WANDB_AVAILABLE:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/aux_loss": outputs["aux_loss"].item(),
                    "train/step": step
                }
                if "clm_loss" in outputs and outputs["clm_loss"] is not None:
                    log_dict["train/clm_loss"] = outputs["clm_loss"].item()
                if "mlm_loss" in outputs and outputs["mlm_loss"] is not None:
                    log_dict["train/mlm_loss"] = outputs["mlm_loss"].item()
                wandb.log(log_dict)
        
        # Return average losses
        num_steps = len(dataloader)
        return {
            "total_loss": total_loss / num_steps,
            "clm_loss": total_clm_loss / num_steps if total_clm_loss > 0 else None,
            "mlm_loss": total_mlm_loss / num_steps if total_mlm_loss > 0 else None,
            "aux_loss": total_aux_loss / num_steps
        }


class CLMTrainer(BaseTrainer):
    """Trainer for Causal Language Modeling."""
    
    @staticmethod
    def create_trainer(args):
        """Factory method to create CLM trainer."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config
        albert_config = AlbertMoEConfig(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            max_position_embeddings=args.max_position_embeddings,
            aux_loss_coefficient=args.aux_loss_coefficient,
            top_k_experts=args.top_k_experts,
        )
        
        # Create model
        model = AlbertForCausalLM(config=albert_config).to(device)
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create optimizer
        optimizer = ChillAdam(
            model.parameters(),
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        
        return CLMTrainer(albert_config, model, tokenizer, optimizer, device)
    
    def get_data_collator(self):
        """Get data collator for CLM."""
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)


class MLMTrainer(BaseTrainer):
    """Trainer for Masked Language Modeling."""
    
    @staticmethod
    def create_trainer(args):
        """Factory method to create MLM trainer."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config
        albert_config = AlbertMoEConfig(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            max_position_embeddings=args.max_position_embeddings,
            aux_loss_coefficient=args.aux_loss_coefficient,
            top_k_experts=args.top_k_experts,
        )
        
        # Create model
        model = AlbertForMaskedLM(config=albert_config).to(device)
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create optimizer
        optimizer = ChillAdam(
            model.parameters(),
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        
        return MLMTrainer(albert_config, model, tokenizer, optimizer, device)
    
    def get_data_collator(self):
        """Get data collator for MLM."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )


def get_common_parser():
    """Get argument parser with common training arguments."""
    parser = argparse.ArgumentParser(description="Pre-train or evaluate a custom ALBERT model.")
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument("--mode", type=str, default="all", choices=["pretrain", "evaluate", "all"])
    mode_group.add_argument("--model_path", type=str, default="./albert_from_scratch_output")
    mode_group.add_argument("--task_type", type=str, choices=["clm", "mlm"], required=True)

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--vocab_size", type=int, default=30000)
    model_group.add_argument("--embedding_size", type=int, default=128)
    model_group.add_argument("--hidden_size", type=int, default=768)
    model_group.add_argument("--num_hidden_layers", type=int, default=12)
    model_group.add_argument("--num_attention_heads", type=int, default=12)
    model_group.add_argument("--intermediate_size", type=int, default=3072)
    model_group.add_argument("--num_experts", type=int, default=8)
    model_group.add_argument("--top_k_experts", type=int, default=2)
    model_group.add_argument("--max_position_embeddings", type=int, default=512)
    model_group.add_argument("--aux_loss_coefficient", type=float, default=0.01)

    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument("--dataset", type=str, default="wikitext")
    dataset_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    dataset_group.add_argument("--dataset_split", type=str, default="train")
    dataset_group.add_argument("--tokenizer_name", type=str, default="albert-base-v2")

    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument("--num_epochs", type=int, default=3)
    training_group.add_argument("--batch_size", type=int, default=8)
    training_group.add_argument("--max_length", type=int, default=512)
    training_group.add_argument("--save_every", type=int, default=1000)

    # Optimizer configuration
    optimizer_group = parser.add_argument_group('Optimizer Configuration')
    optimizer_group.add_argument("--min_lr", type=float, default=1e-4)
    optimizer_group.add_argument("--max_lr", type=float, default=1.0)
    optimizer_group.add_argument("--eps", type=float, default=1e-3)
    optimizer_group.add_argument("--beta1", type=float, default=0.9)
    optimizer_group.add_argument("--beta2", type=float, default=0.999)
    optimizer_group.add_argument("--weight_decay", type=float, default=0.0)

    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument("--eval_tasks", type=str, default="basic", choices=["basic", "comprehensive"])
    eval_group.add_argument("--custom_tasks", nargs="+", default=None)

    # Logging configuration
    logging_group = parser.add_argument_group('Logging Configuration')
    logging_group.add_argument("--use_wandb", action="store_true")
    logging_group.add_argument("--wandb_project", type=str, default="albert-moe")
    logging_group.add_argument("--wandb_name", type=str, default=None)

    return parser