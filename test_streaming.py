#!/usr/bin/env python3
"""
Test script to verify streaming dataset functionality.
"""

import os
import sys
import torch
from albertmoe.training import CLMTrainer, get_common_parser

def test_streaming():
    """Test streaming dataset functionality."""
    print("🧪 Testing Streaming Dataset Functionality")
    print("=" * 50)
    
    # Create test arguments
    test_args = [
        "--task_type", "clm",
        "--dataset", "wikitext",
        "--dataset_config", "wikitext-2-raw-v1",
        "--streaming",
        "--max_samples", "100",
        "--batch_size", "2",
        "--num_epochs", "1",
        "--hidden_size", "256",
        "--num_hidden_layers", "2",
        "--num_attention_heads", "8",  # 256 / 8 = 32, which is valid
        "--num_experts", "4",
        "--max_length", "128",
    ]
    
    # Parse arguments
    parser = get_common_parser()
    args = parser.parse_args(test_args)
    
    print(f"✅ Arguments parsed with streaming={args.streaming}, max_samples={args.max_samples}")
    
    try:
        # Create trainer
        trainer = CLMTrainer.create_trainer(args)
        print("✅ CLM Trainer created successfully")
        
        # Test dataset preparation with streaming
        print("\n📊 Testing dataset preparation...")
        tokenized_dataset = trainer.prepare_dataset(
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            args.max_length,
            args.streaming,
            args.max_samples,
            args.text_column
        )
        print("✅ Streaming dataset prepared successfully")
        
        # Create data loader
        from torch.utils.data import DataLoader
        data_collator = trainer.get_data_collator()
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            shuffle=not args.streaming,
            collate_fn=data_collator
        )
        print("✅ DataLoader created for streaming dataset")
        
        # Test a few batches
        print("\n🔄 Testing training loop with streaming...")
        trainer.model.train()
        total_batches = 0
        
        for i, batch in enumerate(dataloader):
            total_batches += 1
            # Test forward pass
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(**batch)
            loss = outputs["loss"]
            
            print(f"   Batch {i+1}: Loss = {loss.item():.4f}")
            
            # Stop after a few batches to verify functionality
            if i >= 2:
                break
        
        print(f"✅ Processed {total_batches} batches successfully")
        
        print("\n🎉 All streaming tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming()
    sys.exit(0 if success else 1)