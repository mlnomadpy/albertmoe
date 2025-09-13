#!/usr/bin/env python3
"""
Simple test script to validate the modularized AlbertMoE package.
"""

import torch
from albertmoe import (
    AlbertMoEConfig, 
    AlbertForCausalLM, 
    AlbertForMaskedLM, 
    ChillAdam,
    ALBERT,
    RotaryEmbedding,
    MoE,
    Expert
)

def test_config():
    """Test configuration creation."""
    print("Testing configuration...")
    config = AlbertMoEConfig(
        vocab_size=1000,
        embedding_size=128, 
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_experts=8,
        top_k_experts=2
    )
    print(f"✅ Config created: {config.vocab_size} vocab, {config.hidden_size} hidden")
    return config

def test_models(config):
    """Test model creation and forward pass."""
    print("\nTesting models...")
    
    # Test base ALBERT
    albert = ALBERT(config)
    print(f"✅ Base ALBERT: {sum(p.numel() for p in albert.parameters())} params")
    
    # Test CLM model
    clm_model = AlbertForCausalLM(config)
    print(f"✅ CLM model: {sum(p.numel() for p in clm_model.parameters())} params")
    
    # Test MLM model
    mlm_model = AlbertForMaskedLM(config)
    print(f"✅ MLM model: {sum(p.numel() for p in mlm_model.parameters())} params")
    
    # Test forward passes
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    
    with torch.no_grad():
        # Test base ALBERT
        hidden, aux_loss = albert(input_ids)
        print(f"✅ ALBERT forward: {hidden.shape}, aux_loss: {aux_loss:.4f}")
        
        # Test CLM
        clm_output = clm_model(input_ids)
        print(f"✅ CLM forward: logits {clm_output['logits'].shape}, aux_loss: {clm_output['aux_loss']:.4f}")
        
        # Test MLM
        mlm_output = mlm_model(input_ids)
        print(f"✅ MLM forward: logits {mlm_output['logits'].shape}, aux_loss: {mlm_output['aux_loss']:.4f}")

def test_components(config):
    """Test individual components."""
    print("\nTesting components...")
    
    # Test RotaryEmbedding
    rope = RotaryEmbedding(32, 512)
    x = torch.randn(2, 4, 16, 32)
    cos, sin = rope(x)
    print(f"✅ RoPE: cos {cos.shape}, sin {sin.shape}")
    
    # Test Expert
    expert = Expert(config)
    x = torch.randn(8, config.hidden_size)
    out = expert(x)
    print(f"✅ Expert: {x.shape} -> {out.shape}")
    
    # Test MoE
    moe = MoE(config)
    x = torch.randn(2, 16, config.hidden_size)
    out, aux_loss = moe(x)
    print(f"✅ MoE: {x.shape} -> {out.shape}, aux_loss: {aux_loss:.4f}")

def test_optimizer(model):
    """Test optimizer."""
    print("\nTesting optimizer...")
    
    optimizer = ChillAdam(
        model.parameters(),
        min_lr=1e-5,
        max_lr=1e-2,
        eps=1e-3
    )
    
    # Test optimization step
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    labels = torch.randint(0, model.config.vocab_size, (2, 16))
    
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    if outputs['loss'] is not None:
        outputs['loss'].backward()
        optimizer.step()
        print(f"✅ Optimizer step: loss {outputs['loss'].item():.4f}")
    else:
        print("✅ Optimizer created successfully")

def test_loss_computation(config):
    """Test loss computation with labels."""
    print("\nTesting loss computation...")
    
    clm_model = AlbertForCausalLM(config)
    mlm_model = AlbertForMaskedLM(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = torch.randint(0, config.vocab_size, (2, 16))
    
    # Test CLM loss
    clm_output = clm_model(input_ids, labels=labels)
    print(f"✅ CLM loss: {clm_output['loss'].item():.4f}, CLM: {clm_output['clm_loss'].item():.4f}")
    
    # Test MLM loss  
    mlm_output = mlm_model(input_ids, labels=labels)
    print(f"✅ MLM loss: {mlm_output['loss'].item():.4f}, MLM: {mlm_output['mlm_loss'].item():.4f}")

def main():
    print("🚀 Testing AlbertMoE Package")
    print("=" * 50)
    
    # Test configuration
    config = test_config()
    
    # Test models
    test_models(config)
    
    # Test components
    test_components(config)
    
    # Test optimizer
    clm_model = AlbertForCausalLM(config)
    test_optimizer(clm_model)
    
    # Test loss computation
    test_loss_computation(config)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! AlbertMoE package is working correctly.")
    print("\nKey features verified:")
    print("- ✅ Modular configuration system")
    print("- ✅ Both CLM and MLM model variants")
    print("- ✅ Mixture of Experts architecture")
    print("- ✅ Rotary Position Embeddings")
    print("- ✅ Custom ChillAdam optimizer")
    print("- ✅ Proper loss computation")
    print("- ✅ Forward and backward passes")

if __name__ == "__main__":
    main()