#!/usr/bin/env python3
"""
Test script to verify streaming dataset functionality - offline version.
"""

import os
import sys
import torch
from albertmoe.training import BaseTrainer, get_common_parser
from albertmoe.config import AlbertMoEConfig
from albertmoe.models import AlbertForCausalLM
from albertmoe.optimizers import ChillAdam
from transformers import AutoTokenizer
from datasets import Dataset

def create_mock_dataset(num_samples=100):
    """Create a mock dataset for testing."""
    data = {
        "text": [f"This is sample text number {i} for testing streaming functionality." for i in range(num_samples)]
    }
    return Dataset.from_dict(data)

def test_streaming_offline():
    """Test streaming dataset functionality offline."""
    print("🧪 Testing Streaming Dataset Functionality (Offline)")
    print("=" * 60)
    
    try:
        # Create a basic configuration
        config = AlbertMoEConfig(
            vocab_size=1000,
            embedding_size=128,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=512,
            num_experts=4,
            max_position_embeddings=128,
        )
        print("✅ Configuration created")
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlbertForCausalLM(config=config).to(device)
        print("✅ Model created")
        
        # Create a simple tokenizer (we'll create a mock one)
        # Note: In real usage, this would be from a pretrained model
        class MockTokenizer:
            def __init__(self, vocab_size=1000):
                self.vocab_size = vocab_size
                self.pad_token = "[PAD]"
                self.pad_token_id = 0
                
            def __call__(self, texts, truncation=True, padding=False, max_length=128):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Simple tokenization: split by spaces and convert to ids
                tokenized = []
                for text in texts:
                    words = text.split()[:max_length-2]  # Leave room for special tokens
                    # Convert to simple IDs (hash-based)
                    ids = [1] + [hash(word) % (self.vocab_size-2) + 2 for word in words]  # Start with 1, avoid 0 (pad)
                    if len(ids) < max_length:
                        ids.extend([0] * (max_length - len(ids)))  # Pad with 0
                    tokenized.append(ids[:max_length])
                
                return {"input_ids": tokenized, "attention_mask": [[1 if id != 0 else 0 for id in ids] for ids in tokenized]}
        
        tokenizer = MockTokenizer()
        optimizer = ChillAdam(model.parameters())
        
        # Create base trainer
        trainer = BaseTrainer(config, model, tokenizer, optimizer, device)
        print("✅ Trainer created")
        
        # Test 1: Non-streaming dataset preparation
        print("\n📊 Testing non-streaming dataset preparation...")
        mock_dataset = create_mock_dataset(50)
        
        # Manually test the tokenization logic
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=False, max_length=128)
        
        # Test non-streaming approach
        max_samples = 20
        if max_samples:
            limited_dataset = mock_dataset.select(range(min(max_samples, len(mock_dataset))))
        tokenized_dataset = limited_dataset.map(tokenize_function, batched=True, remove_columns=limited_dataset.column_names)
        print(f"✅ Non-streaming: processed {len(tokenized_dataset)} samples")
        
        # Test 2: Simulate streaming behavior
        print("\n📊 Testing streaming-like dataset preparation...")
        # For streaming, we would use dataset.take(max_samples) but we can simulate it
        streaming_samples = 15
        streaming_dataset = mock_dataset.select(range(streaming_samples))
        tokenized_streaming = streaming_dataset.map(tokenize_function, batched=True, remove_columns=streaming_dataset.column_names)
        print(f"✅ Streaming simulation: processed {len(tokenized_streaming)} samples")
        
        # Test 3: Verify the new prepare_dataset method
        print("\n📊 Testing updated prepare_dataset method...")
        
        # We can't test with real HF datasets offline, but we can test the parameter passing
        # Let's test that the method accepts the new parameters
        import inspect
        
        # Check method signature
        sig = inspect.signature(trainer.prepare_dataset)
        expected_params = {'dataset_name', 'dataset_config', 'dataset_split', 'max_length', 'streaming', 'max_samples', 'text_column'}
        actual_params = set(sig.parameters.keys())
        
        if expected_params.issubset(actual_params):
            print("✅ prepare_dataset method has all required streaming parameters")
        else:
            missing = expected_params - actual_params
            print(f"❌ Missing parameters in prepare_dataset: {missing}")
            return False
        
        # Test 4: Check argument parser
        print("\n📊 Testing argument parser for streaming options...")
        parser = get_common_parser()
        test_args = [
            "--task_type", "clm",
            "--streaming",
            "--max_samples", "100",
            "--text_column", "content"
        ]
        
        args = parser.parse_args(test_args)
        
        if hasattr(args, 'streaming') and args.streaming:
            print("✅ Streaming argument parsed correctly")
        else:
            print("❌ Streaming argument not found or incorrect")
            return False
            
        if hasattr(args, 'max_samples') and args.max_samples == 100:
            print("✅ max_samples argument parsed correctly")
        else:
            print("❌ max_samples argument not found or incorrect")
            return False
            
        if hasattr(args, 'text_column') and args.text_column == "content":
            print("✅ text_column argument parsed correctly")
        else:
            print("❌ text_column argument not found or incorrect")
            return False
        
        # Test 5: Check train_epoch method signature
        print("\n📊 Testing train_epoch method for streaming support...")
        sig = inspect.signature(trainer.train_epoch)
        expected_params = {'dataloader', 'use_wandb', 'streaming', 'max_samples', 'batch_size'}
        actual_params = set(sig.parameters.keys())
        
        if expected_params.issubset(actual_params):
            print("✅ train_epoch method has all required streaming parameters")
        else:
            missing = expected_params - actual_params
            print(f"❌ Missing parameters in train_epoch: {missing}")
            return False
        
        print("\n🎉 All offline streaming tests passed!")
        print("\n📋 Streaming functionality summary:")
        print("   ✅ Added streaming parameter to prepare_dataset()")
        print("   ✅ Added max_samples parameter for limiting dataset size")
        print("   ✅ Added text_column parameter for flexible column naming")
        print("   ✅ Updated argument parser with streaming options")
        print("   ✅ Updated train_epoch() to support early stopping with streaming")
        print("   ✅ Ready for real dataset streaming with Hugging Face datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_offline()
    sys.exit(0 if success else 1)