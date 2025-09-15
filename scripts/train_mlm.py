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
        args.max_length,
        args.streaming,
        args.max_samples,
        args.text_column
    )
    
    # Create data loader
    data_collator = trainer.get_data_collator()
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,  # Don't shuffle streaming datasets
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
        
        losses = trainer.train_epoch(
            dataloader, 
            args.use_wandb, 
            args.streaming, 
            args.max_samples, 
            args.batch_size
        )
        
        # Format losses, handling None values gracefully
        total_loss = losses['total_loss']
        mlm_loss = losses['mlm_loss']  
        aux_loss = losses['aux_loss']
        
        total_str = f"{total_loss:.4f}" if total_loss is not None else "N/A"
        mlm_str = f"{mlm_loss:.4f}" if mlm_loss is not None else "N/A"
        aux_str = f"{aux_loss:.4f}" if aux_loss is not None else "N/A"
        
        print(f"Average losses - Total: {total_str}, "
              f"MLM: {mlm_str}, "
              f"Aux: {aux_str}")
        
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
    
    if args.mode == "serve":
        # Serving mode
        print("🚀 Starting MLM serving mode...")
        
        # Create trainer for serving
        trainer = MLMTrainer.create_trainer(args)
        
        # Load the trained model
        if not trainer.load_for_serving(args.model_path):
            print("❌ Failed to load model for serving. Please check the model path.")
            return
        
        if args.interactive:
            # Interactive serving
            trainer.serve_interactive(args)
        else:
            # Single prediction mode
            if args.input_text:
                print(f"🎯 Input: {args.input_text}")
                print("🔄 Predicting masked tokens...")
                
                results = trainer.predict_masked_tokens(args.input_text, top_k=5)
                
                if isinstance(results, str):
                    print(results)
                else:
                    print(f"📝 Original text: {args.input_text}")
                    print(f"🎯 Predictions:")
                    
                    for i, result in enumerate(results):
                        print(f"\n   Mask #{i+1} (position {result['position']}):")
                        for pred in result['predictions']:
                            print(f"      {pred['rank']}. '{pred['token']}' "
                                f"(probability: {pred['probability']:.3f})")
            else:
                print("⚠️  Please provide --input_text for non-interactive serving mode.")
                print("💡 Or use --interactive flag for interactive mode.")
                print("💡 Example: --input_text 'The weather is [MASK] today.'")
    
    print("\n🎉 SCRIPT EXECUTION COMPLETE")


if __name__ == "__main__":
    main()