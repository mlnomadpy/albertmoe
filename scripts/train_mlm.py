#!/usr/bin/env python3
"""
Main script for Masked Language Modeling training and evaluation.
"""

import os
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from albertmoe.training import MLMTrainer, get_common_parser
from albertmoe.evaluation import evaluate_model


def train_mlm_model(trainer, args):
    """Train an MLM model."""
    print("🚀 Starting MLM training...")
    
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
            name=args.wandb_name or f"mlm-training-{args.model_path.split('/')[-1]}",
            config=vars(args)
        )
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{args.num_epochs}")
        
        losses = trainer.train_epoch(dataloader, args.use_wandb)
        
        print(f"Average losses - Total: {losses['total_loss']:.4f}, "
              f"MLM: {losses['mlm_loss']:.4f}, "
              f"Aux: {losses['aux_loss']:.4f}")
        
        # Save model
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_path = f"{args.model_path}/epoch_{epoch + 1}"
            
            # Prepare training args for model card
            training_args_dict = {
                "task_type": "mlm",
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "hidden_size": args.hidden_size,
                "num_experts": args.num_experts,
                "top_k_experts": args.top_k_experts,
                "max_position_embeddings": args.max_position_embeddings,
            } if args.push_to_hub else None
            
            trainer.save_model(
                save_path=save_path,
                push_to_hub_repo=args.push_to_hub,
                hub_token=args.hub_token,
                hub_private=args.hub_private,
                training_args=training_args_dict
            )
    
    if args.use_wandb:
        wandb.finish()
    
    print("✅ MLM training completed!")


def main():
    parser = get_common_parser()
    args = parser.parse_args()
    
    # Ensure task type is MLM
    args.task_type = "mlm"
    
    if args.mode in ["pretrain", "all"]:
        # Create trainer
        trainer = MLMTrainer.create_trainer(args)
        
        # Train model
        train_mlm_model(trainer, args)
    
    if args.mode in ["evaluate", "all"]:
        # Evaluate model
        print("🔍 Starting evaluation...")
        results = evaluate_model(
            model_path=args.model_path,
            task_type="mlm",
            eval_tasks=args.eval_tasks,
            custom_tasks=args.custom_tasks,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
        print("✅ Evaluation completed!")
    
    print("\n🎉 SCRIPT EXECUTION COMPLETE")


if __name__ == "__main__":
    main()