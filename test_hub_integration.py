#!/usr/bin/env python3
"""
Test script for Hugging Face Hub integration functionality.
"""

import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from albertmoe import AlbertMoEConfig, AlbertForCausalLM, AlbertForMaskedLM
from albertmoe.hub_utils import HubManager, push_to_hub
from transformers import AutoTokenizer


def test_hub_manager_creation():
    """Test HubManager creation and basic functionality."""
    print("Testing HubManager creation...")
    
    # Test with mock token
    with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
        try:
            hub_manager = HubManager()
            print("✅ HubManager created successfully")
            return True
        except ImportError:
            print("⚠️  HubManager requires huggingface_hub (expected in test environment)")
            return False
        except Exception as e:
            print(f"❌ Error creating HubManager: {e}")
            return False


def test_model_card_generation():
    """Test model card generation."""
    print("\nTesting model card generation...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock HubManager
            with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
                hub_manager = HubManager()
                
                # Test config
                config_dict = {
                    "vocab_size": 30000,
                    "hidden_size": 768,
                    "num_experts": 8
                }
                
                training_args = {
                    "num_epochs": 3,
                    "batch_size": 8,
                    "dataset": "wikitext"
                }
                
                # Generate model card
                card_path = hub_manager.create_model_card(
                    temp_dir, config_dict, "clm", training_args
                )
                
                # Check if file was created
                if os.path.exists(card_path):
                    with open(card_path, 'r') as f:
                        content = f.read()
                        if "AlbertMoE" in content and "Causal Language Modeling" in content:
                            print("✅ Model card generated successfully")
                            return True
                        else:
                            print("❌ Model card content is incorrect")
                            return False
                else:
                    print("❌ Model card file was not created")
                    return False
                    
    except Exception as e:
        print(f"❌ Error testing model card generation: {e}")
        return False


def test_config_json_generation():
    """Test config.json generation."""
    print("\nTesting config.json generation...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
                hub_manager = HubManager()
                
                config_dict = {
                    "vocab_size": 30000,
                    "hidden_size": 768,
                    "num_experts": 8,
                    "model_type": "albert_moe"
                }
                
                # Save config
                config_path = hub_manager.save_config_json(temp_dir, config_dict)
                
                # Check if file was created and has correct content
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        saved_config = json.load(f)
                        if saved_config == config_dict:
                            print("✅ Config.json generated successfully")
                            return True
                        else:
                            print("❌ Config.json content is incorrect")
                            return False
                else:
                    print("❌ Config.json file was not created")
                    return False
                    
    except Exception as e:
        print(f"❌ Error testing config.json generation: {e}")
        return False


def test_push_to_hub_function():
    """Test the high-level push_to_hub function."""
    print("\nTesting push_to_hub function...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal model directory structure
            model_path = os.path.join(temp_dir, "pytorch_model.bin")
            with open(model_path, 'w') as f:
                f.write("dummy model file")
            
            config_dict = {
                "vocab_size": 1000,
                "hidden_size": 256,
                "num_experts": 4,
                "model_type": "albert_moe"
            }
            
            # Mock the HubManager methods to avoid actual Hub calls
            with patch('albertmoe.hub_utils.HubManager') as mock_hub_manager:
                mock_instance = MagicMock()
                mock_instance.create_model_card.return_value = "README.md"
                mock_instance.save_config_json.return_value = "config.json"
                mock_instance.upload_model.return_value = True
                mock_hub_manager.return_value = mock_instance
                
                # Test push_to_hub
                result = push_to_hub(
                    local_path=temp_dir,
                    repo_id="test/albert-moe",
                    model_config=config_dict,
                    task_type="clm",
                    token="test_token"
                )
                
                if result:
                    print("✅ push_to_hub function works correctly")
                    return True
                else:
                    print("❌ push_to_hub function failed")
                    return False
                    
    except Exception as e:
        print(f"❌ Error testing push_to_hub function: {e}")
        return False


def test_integration_with_model():
    """Test integration with actual model saving."""
    print("\nTesting integration with model saving...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a small model for testing
            config = AlbertMoEConfig(
                vocab_size=1000,
                embedding_size=64,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_experts=4,
                top_k_experts=2
            )
            
            model = AlbertForCausalLM(config)
            
            # Save model locally
            model_path = os.path.join(temp_dir, "test_model")
            os.makedirs(model_path, exist_ok=True)
            
            import torch
            torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
            
            # Create mock tokenizer files
            tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
            for file_name in tokenizer_files:
                with open(os.path.join(model_path, file_name), 'w') as f:
                    f.write('{"mock": "tokenizer"}')
            
            # Test that model files were created
            required_files = ["pytorch_model.bin", "tokenizer_config.json", "tokenizer.json"]
            missing_files = []
            
            for file_name in required_files:
                if not os.path.exists(os.path.join(model_path, file_name)):
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"❌ Missing model files: {missing_files}")
                return False
            
            # Mock all Hub operations to test integration without network calls
            with patch('albertmoe.hub_utils.HubManager') as mock_hub_manager_class, \
                 patch('albertmoe.hub_utils.upload_folder') as mock_upload, \
                 patch('albertmoe.hub_utils.create_repo') as mock_create_repo:
                
                # Setup mocks
                mock_instance = MagicMock()
                mock_instance.repo_exists.return_value = False
                mock_instance.create_repository.return_value = True
                mock_instance.upload_model.return_value = True
                mock_instance.create_model_card.return_value = os.path.join(model_path, "README.md") 
                mock_instance.save_config_json.return_value = os.path.join(model_path, "config.json")
                mock_hub_manager_class.return_value = mock_instance
                
                mock_upload.return_value = True
                mock_create_repo.return_value = True
                
                config_dict = {
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "num_experts": config.num_experts,
                    "model_type": "albert_moe"
                }
                
                result = push_to_hub(
                    local_path=model_path,
                    repo_id="test/albert-moe-integration",
                    model_config=config_dict,
                    task_type="clm",
                    token="test_token"
                )
                
                if result:
                    print("✅ Model integration test passed")
                    
                    # Verify that our mocked methods were called to create files
                    # The actual files should be created by the mocked methods
                    mock_instance.create_model_card.assert_called_once()
                    mock_instance.save_config_json.assert_called_once()
                    mock_instance.upload_model.assert_called_once()
                    
                    print("✅ All Hub operations were called correctly")
                    return True
                else:
                    print("❌ Model integration test failed")
                    return False
                    
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        return False


def main():
    """Run all Hub functionality tests."""
    print("🚀 Testing Hugging Face Hub Integration")
    print("=" * 50)
    
    tests = [
        ("Hub Manager Creation", test_hub_manager_creation),
        ("Model Card Generation", test_model_card_generation),
        ("Config JSON Generation", test_config_json_generation),
        ("Push to Hub Function", test_push_to_hub_function),
        ("Model Integration", test_integration_with_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Hub integration tests passed!")
        print("\nKey Hub features verified:")
        print("- ✅ HubManager creation and configuration")
        print("- ✅ Model card generation with training details")
        print("- ✅ Config.json generation for Hub compatibility")
        print("- ✅ High-level push_to_hub function")
        print("- ✅ Integration with model saving workflow")
        print("\nUsage example:")
        print("python scripts/train_clm.py --push_to_hub username/my-albert-model --hub_token your_token")
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    main()