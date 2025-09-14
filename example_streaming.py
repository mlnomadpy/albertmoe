#!/usr/bin/env python3
"""
Example script demonstrating streaming dataset functionality.

This script shows how to use the new streaming dataset feature to train
AlbertMoE models on large datasets without loading them entirely into memory.
"""

import argparse
import sys

def main():
    """Demonstrate streaming dataset usage."""
    print("🚀 AlbertMoE Streaming Dataset Example")
    print("=" * 50)
    
    print("\n📖 Streaming Dataset Training Examples:")
    
    print("\n1️⃣ Basic streaming training with CLM:")
    print("python scripts/train_clm.py \\")
    print("    --mode pretrain \\")
    print("    --dataset wikitext \\")
    print("    --dataset_config wikitext-103-raw-v1 \\")
    print("    --streaming \\")
    print("    --max_samples 10000 \\")
    print("    --batch_size 8 \\")
    print("    --num_epochs 1 \\")
    print("    --hidden_size 768 \\")
    print("    --num_experts 8")
    
    print("\n2️⃣ Streaming training with custom dataset:")
    print("python scripts/train_clm.py \\")
    print("    --mode pretrain \\")
    print("    --dataset squad \\")
    print("    --dataset_config plain_text \\")
    print("    --text_column context \\")
    print("    --streaming \\")
    print("    --max_samples 5000 \\")
    print("    --batch_size 4 \\")
    print("    --num_epochs 2")
    
    print("\n3️⃣ Streaming MLM training:")
    print("python scripts/train_mlm.py \\")
    print("    --mode pretrain \\")
    print("    --dataset c4 \\")
    print("    --dataset_config en \\")
    print("    --streaming \\")
    print("    --max_samples 20000 \\")
    print("    --batch_size 16 \\")
    print("    --use_wandb")
    
    print("\n4️⃣ Streaming with Hub integration:")
    print("python scripts/train_clm.py \\")
    print("    --dataset oscar \\")
    print("    --dataset_config unshuffled_deduplicated_en \\")
    print("    --streaming \\")
    print("    --max_samples 50000 \\")
    print("    --batch_size 8 \\")
    print("    --push_to_hub username/albert-moe-oscar \\")
    print("    --hub_token your_hf_token")
    
    print("\n📋 Key Benefits of Streaming:")
    print("   ✅ Memory efficient - doesn't load entire dataset")
    print("   ✅ Works with large datasets (100GB+ datasets)")
    print("   ✅ Faster startup time")
    print("   ✅ Controllable dataset size with --max_samples")
    print("   ✅ Supports any Hugging Face dataset")
    
    print("\n⚠️  Important Notes:")
    print("   • Streaming datasets cannot be shuffled traditionally")
    print("   • Use --max_samples to limit dataset size")
    print("   • Some datasets may require specific --text_column")
    print("   • Progress bars show processed batches, not total dataset size")
    
    print("\n💡 Tips for Large Dataset Training:")
    print("   • Start with small --max_samples to test")
    print("   • Use appropriate --batch_size for your GPU memory")
    print("   • Enable --use_wandb for monitoring")
    print("   • Consider using gradient accumulation for large effective batch sizes")

if __name__ == "__main__":
    main()