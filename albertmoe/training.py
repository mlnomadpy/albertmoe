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
from .hub_utils import push_to_hub

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
        
    def prepare_dataset(self, dataset_name, dataset_config, dataset_split, max_length=512, streaming=False, max_samples=None, text_column="text"):
        """Prepare dataset for training."""
        try:
            # Handle None config (convert to None for datasets library)
            config_to_use = None if dataset_config and dataset_config.lower() in ['none', 'null', ''] else dataset_config
            dataset = load_dataset(dataset_name, config_to_use, split=dataset_split, streaming=streaming)
        except Exception as e:
            print(f"✗ Error loading dataset {dataset_name}: {e}. Falling back to wikitext.")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=streaming)
            text_column = "text"  # Update text column for fallback dataset
        
        def tokenize_function(examples):
            return self.tokenizer(examples[text_column], truncation=True, padding=False, max_length=max_length)
        
        if streaming:
            # For streaming datasets, use map without num_proc and remove_columns from features
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=list(dataset.features.keys()))
            if max_samples:
                tokenized_dataset = tokenized_dataset.take(max_samples)
        else:
            # For non-streaming datasets, limit samples before tokenization if needed
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        return tokenized_dataset
    
    def save_model(self, save_path, push_to_hub_repo=None, hub_token=None, hub_private=False, training_args=None):
        """
        Save model locally and optionally push to Hugging Face Hub.
        
        Args:
            save_path: Local path to save the model
            push_to_hub_repo: Hub repository ID (e.g., "username/model-name"). If None, only saves locally.
            hub_token: Hugging Face token for authentication
            hub_private: Whether to create a private repository on the Hub
            training_args: Training arguments to include in model card
        """
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict and tokenizer locally
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_path)
        
        print(f"💾 Model saved locally to {save_path}")
        
        # Push to Hub if requested
        if push_to_hub_repo:
            # Create config dictionary for Hub
            config_dict = {
                "architectures": [self.model.__class__.__name__],
                "model_type": "albert_moe",
                "vocab_size": self.config.vocab_size,
                "embedding_size": self.config.embedding_size,
                "hidden_size": self.config.hidden_size,
                "num_hidden_layers": self.config.num_hidden_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "intermediate_size": self.config.intermediate_size,
                "num_experts": self.config.num_experts,
                "top_k_experts": self.config.top_k_experts,
                "max_position_embeddings": self.config.max_position_embeddings,
                "aux_loss_coefficient": getattr(self.config, 'aux_loss_coefficient', 0.01),
                "use_rotary": getattr(self.config, 'use_rotary', True),
            }
            
            # Determine task type based on model class
            task_type = "clm" if "CausalLM" in self.model.__class__.__name__ else "mlm"
            
            # Push to Hub
            success = push_to_hub(
                local_path=save_path,
                repo_id=push_to_hub_repo,
                model_config=config_dict,
                task_type=task_type,
                token=hub_token,
                private=hub_private,
                training_args=training_args
            )
            
            if success:
                print(f"🚀 Model successfully pushed to Hub: https://huggingface.co/{push_to_hub_repo}")
            else:
                print(f"❌ Failed to push model to Hub: {push_to_hub_repo}")
    
    def train_epoch(self, dataloader, use_wandb=False, streaming=False, max_samples=None, batch_size=None):
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
            
            # Early stopping for streaming datasets with max_samples
            if streaming and max_samples and batch_size and (step + 1) * batch_size >= max_samples:
                print(f"Reached max_samples ({max_samples}) for streaming dataset, stopping training.")
                break
            
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
        
        # Return average losses (use step+1 for actual number of steps processed)
        num_steps = step + 1
        return {
            "total_loss": total_loss / num_steps,
            "clm_loss": total_clm_loss / num_steps if total_clm_loss > 0 else None,
            "mlm_loss": total_mlm_loss / num_steps if total_mlm_loss > 0 else None,
            "aux_loss": total_aux_loss / num_steps
        }

    def load_for_serving(self, model_path):
        """Load a trained model for serving/inference."""
        try:
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ Model loaded from {model_path}")
                return True
            else:
                print(f"✗ Model file not found at {model_file}")
                return False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False


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

    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        """Generate text given a prompt using the CLM model."""
        self.model.eval()
        with torch.no_grad():
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            # Generate tokens one by one
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                # Get logits for the last token
                last_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)
                    filtered_logits = torch.full_like(last_token_logits, float('-inf'))
                    filtered_logits[top_k_indices] = top_k_logits
                    last_token_logits = filtered_logits
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    last_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append token to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return generated_text

    def serve_interactive(self, args):
        """Interactive serving mode for CLM."""
        print("🤖 AlbertMoE CLM Interactive Serving Mode")
        print("=" * 50)
        print("Enter text prompts to generate completions.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                prompt = input("🎯 Enter your prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    print("⚠️  Please enter a non-empty prompt.")
                    continue
                
                print("🔄 Generating...")
                generated = self.generate_text(
                    prompt, 
                    max_length=args.gen_max_length, 
                    temperature=args.temperature,
                    top_k=args.top_k, 
                    top_p=args.top_p
                )
                
                print(f"📝 Generated text:\n{generated}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during generation: {e}")


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

    def predict_masked_tokens(self, text, top_k=5):
        """Predict masked tokens in the given text."""
        self.model.eval()
        with torch.no_grad():
            # Check if text contains [MASK] tokens
            if "[MASK]" not in text:
                return f"⚠️  No [MASK] tokens found in text. Please include [MASK] tokens to predict."
            
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            # Find mask token positions
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
            
            if len(mask_positions[1]) == 0:
                return f"⚠️  No [MASK] tokens found after tokenization."
            
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs["logits"]
            
            results = []
            for pos in mask_positions[1]:
                # Get predictions for this mask position
                mask_logits = logits[0, pos, :]
                top_predictions = torch.topk(mask_logits, top_k)
                
                # Decode top predictions
                predictions = []
                for i, (score, token_id) in enumerate(zip(top_predictions.values, top_predictions.indices)):
                    token = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    probability = F.softmax(mask_logits, dim=-1)[token_id].item()
                    predictions.append({
                        'rank': i + 1,
                        'token': token,
                        'probability': probability,
                        'score': score.item()
                    })
                
                results.append({
                    'position': pos.item(),
                    'predictions': predictions
                })
            
            return results

    def serve_interactive(self, args):
        """Interactive serving mode for MLM."""
        print("🤖 AlbertMoE MLM Interactive Serving Mode")
        print("=" * 50)
        print("Enter text with [MASK] tokens to predict masked words.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        print("💡 Example: 'The weather today is [MASK] and [MASK].'")
        print("💡 Note: Use [MASK] (not <mask>) as the mask token.\n")
        
        while True:
            try:
                text = input("🎯 Enter text with [MASK] tokens: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("⚠️  Please enter non-empty text.")
                    continue
                
                print("🔄 Predicting masked tokens...")
                results = self.predict_masked_tokens(text, top_k=5)
                
                if isinstance(results, str):
                    print(results)
                else:
                    print(f"📝 Original text: {text}")
                    print(f"🎯 Predictions:")
                    
                    for i, result in enumerate(results):
                        print(f"\n   Mask #{i+1} (position {result['position']}):")
                        for pred in result['predictions']:
                            print(f"      {pred['rank']}. '{pred['token']}' "
                                f"(probability: {pred['probability']:.3f})")
                
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during prediction: {e}")


def get_common_parser():
    """Get argument parser with common training arguments."""
    parser = argparse.ArgumentParser(description="Pre-train or evaluate a custom ALBERT model.")
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument("--mode", type=str, default="all", choices=["pretrain", "evaluate", "serve", "all"])
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
    dataset_group.add_argument("--streaming", action="store_true", default=False,
                              help="Enable dataset streaming for large datasets")
    dataset_group.add_argument("--max_samples", type=int, default=None,
                              help="Maximum number of samples to use from dataset")
    dataset_group.add_argument("--text_column", type=str, default="text",
                              help="Name of the text column in the dataset")

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

    # Hugging Face Hub configuration
    hub_group = parser.add_argument_group('Hugging Face Hub Configuration')
    hub_group.add_argument("--push_to_hub", type=str, default=None, 
                          help="Hub repository ID (e.g., 'username/model-name'). If provided, model will be pushed to Hub.")
    hub_group.add_argument("--hub_token", type=str, default=None,
                          help="Hugging Face token. If not provided, will try to get from environment variables.")
    hub_group.add_argument("--hub_private", action="store_true",
                          help="Create a private repository on the Hub.")
    hub_group.add_argument("--hub_commit_message", type=str, default=None,
                          help="Custom commit message for Hub upload.")

    # Serving configuration
    serving_group = parser.add_argument_group('Serving Configuration')
    serving_group.add_argument("--interactive", action="store_true", default=False,
                              help="Enable interactive serving mode with continuous input prompts.")
    serving_group.add_argument("--input_text", type=str, default=None,
                              help="Single input text for serving mode (non-interactive).")
    serving_group.add_argument("--gen_max_length", type=int, default=100,
                              help="Maximum length for text generation (CLM only).")
    serving_group.add_argument("--temperature", type=float, default=1.0,
                              help="Temperature for text generation (CLM only).")
    serving_group.add_argument("--top_k", type=int, default=50,
                              help="Top-k sampling for text generation (CLM only).")
    serving_group.add_argument("--top_p", type=float, default=0.95,
                              help="Top-p (nucleus) sampling for text generation (CLM only).")

    return parser