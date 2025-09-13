#!/usr/bin/env python3
"""
Main script for Causal Language Modeling training and evaluation.
"""

import os
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from albertmoe.training import CLMTrainer, get_common_parser
from albertmoe.evaluation import evaluate_model


def train_clm_model(trainer, args):
    """Train a CLM model."""
    print("🚀 Starting CLM training...")
    
    # Prepare dataset
    tokenized_dataset = trainer.prepare_dataset(
        args.dataset, 
        args.dataset_config, 
        args.dataset_split,
        args.max_length
    )
    
    # Create data loader
    data_collator = trainer.get_data_collator()
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"clm-training-{args.model_path.split('/')[-1]}",
            config=vars(args)
        )
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{args.num_epochs}")
        
        losses = trainer.train_epoch(dataloader, args.use_wandb)
        
        print(f"Average losses - Total: {losses['total_loss']:.4f}, "
              f"CLM: {losses['clm_loss']:.4f}, "
              f"Aux: {losses['aux_loss']:.4f}")
        
        # Save model
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_path = f"{args.model_path}/epoch_{epoch + 1}"
            os.makedirs(save_path, exist_ok=True)
            torch.save(trainer.model.state_dict(), f"{save_path}/pytorch_model.bin")
            trainer.tokenizer.save_pretrained(save_path)
            print(f"💾 Model saved to {save_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    print("✅ CLM training completed!")


def main():
    parser = get_common_parser()
    args = parser.parse_args()
    
    # Ensure task type is CLM
    args.task_type = "clm"
    
    if args.mode in ["pretrain", "all"]:
        # Create trainer
        trainer = CLMTrainer.create_trainer(args)
        
        # Train model
        train_clm_model(trainer, args)
    
    if args.mode in ["evaluate", "all"]:
        # Evaluate model
        print("🔍 Starting evaluation...")
        results = evaluate_model(
            model_path=args.model_path,
            task_type="clm",
            eval_tasks=args.eval_tasks,
            custom_tasks=args.custom_tasks,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
        print("✅ Evaluation completed!")
    
    print("\n🎉 SCRIPT EXECUTION COMPLETE")


if __name__ == "__main__":
    main()