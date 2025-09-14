#!/usr/bin/env python3
"""
Demonstration script for AlbertMoE serving mode functionality.

This script shows examples of how to use the new serving mode for both
CLM (Causal Language Model) and MLM (Masked Language Model) tasks.
"""

def main():
    """Demonstrate serving mode usage."""
    print("🚀 AlbertMoE Serving Mode Examples")
    print("=" * 50)
    
    print("\n📖 Serving Mode Documentation:")
    print("\nThe new serving mode allows you to use pretrained AlbertMoE models for inference")
    print("without requiring the full training setup. It supports both interactive and")
    print("single-prediction modes for both CLM and MLM tasks.")
    
    print("\n🎯 CLM (Causal Language Model) Serving Examples:")
    print("\n1️⃣ Interactive text generation:")
    print("python scripts/train_clm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_clm_model \\")
    print("    --interactive")
    
    print("\n2️⃣ Single text completion:")
    print("python scripts/train_clm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_clm_model \\")
    print("    --input_text 'Once upon a time' \\")
    print("    --gen_max_length 100 \\")
    print("    --temperature 0.8 \\")
    print("    --top_k 50 \\")
    print("    --top_p 0.95")
    
    print("\n3️⃣ Creative text generation with higher temperature:")
    print("python scripts/train_clm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_clm_model \\")
    print("    --input_text 'The future of AI is' \\")
    print("    --temperature 1.2 \\")
    print("    --gen_max_length 150")
    
    print("\n🎯 MLM (Masked Language Model) Serving Examples:")
    print("\n1️⃣ Interactive masked token prediction:")
    print("python scripts/train_mlm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_mlm_model \\")
    print("    --interactive")
    
    print("\n2️⃣ Single masked token prediction:")
    print("python scripts/train_mlm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_mlm_model \\")
    print("    --input_text 'The weather today is [MASK] and [MASK].'")
    
    print("\n3️⃣ Fill-in-the-blank tasks:")
    print("python scripts/train_mlm.py \\")
    print("    --mode serve \\")
    print("    --model_path ./my_mlm_model \\")
    print("    --input_text 'Paris is the [MASK] of France.'")
    
    print("\n⚙️  Serving Configuration Options:")
    print("\n📝 Common Options:")
    print("   --mode serve              # Enable serving mode")
    print("   --model_path PATH         # Path to pretrained model directory")
    print("   --interactive             # Enable interactive mode (continuous prompts)")
    print("   --input_text TEXT         # Single input for non-interactive mode")
    
    print("\n🔄 CLM-Specific Options:")
    print("   --gen_max_length INT      # Maximum generation length (default: 100)")
    print("   --temperature FLOAT       # Sampling temperature (default: 1.0)")
    print("   --top_k INT              # Top-k sampling (default: 50)")
    print("   --top_p FLOAT            # Nucleus sampling threshold (default: 0.95)")
    
    print("\n💡 MLM-Specific Notes:")
    print("   • Use [MASK] tokens in your input text")
    print("   • The model will predict top-5 candidates for each [MASK]")
    print("   • Predictions include probability scores")
    
    print("\n🔧 Example Workflow:")
    print("\n1. Train a model:")
    print("   python scripts/train_clm.py --mode pretrain --num_epochs 3")
    print("\n2. Serve the trained model:")
    print("   python scripts/train_clm.py --mode serve --model_path ./albert_from_scratch_output --interactive")
    
    print("\n📚 Interactive Mode Features:")
    print("   ✅ Continuous input prompts")
    print("   ✅ Type 'quit' or 'exit' to stop")
    print("   ✅ Real-time text generation/prediction")
    print("   ✅ Error handling and user feedback")
    
    print("\n🎁 Benefits of Serving Mode:")
    print("   🚀 Quick inference without training setup")
    print("   🎯 Both batch and interactive processing")
    print("   ⚡ Optimized for CPU and GPU inference")
    print("   🔧 Configurable generation parameters")
    print("   📊 Detailed prediction scores (MLM)")
    
    print("\n💻 Try it out:")
    print("   1. Train a small model for testing")
    print("   2. Use --interactive for real-time exploration")
    print("   3. Use --input_text for scripted workflows")
    print("   4. Experiment with different generation parameters")

if __name__ == "__main__":
    main()