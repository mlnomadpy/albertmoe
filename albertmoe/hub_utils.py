"""
Utilities for Hugging Face Hub integration.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, Repository
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Hub functionality disabled.")


class HubManager:
    """Manager for Hugging Face Hub operations."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize Hub manager.
        
        Args:
            token: Hugging Face token. If None, will try to get from environment.
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hub functionality")
        
        self.token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)
    
    def repo_exists(self, repo_id: str, repo_type: str = "model") -> bool:
        """
        Check if a repository exists on the Hub.
        
        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            repo_type: Type of repository ("model", "dataset", "space")
            
        Returns:
            True if repository exists, False otherwise
        """
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return True
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            print(f"Error checking repository existence: {e}")
            return False
    
    def create_repository(
        self, 
        repo_id: str, 
        private: bool = False,
        exist_ok: bool = True,
        repo_type: str = "model"
    ) -> bool:
        """
        Create a new repository on the Hub.
        
        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            private: Whether to create a private repository
            exist_ok: If True, don't raise error if repo already exists
            repo_type: Type of repository ("model", "dataset", "space")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.repo_exists(repo_id, repo_type) and exist_ok:
                print(f"Repository {repo_id} already exists. Skipping creation.")
                return True
            
            create_repo(
                repo_id=repo_id,
                private=private,
                repo_type=repo_type,
                token=self.token,
                exist_ok=exist_ok
            )
            print(f"Successfully created repository: {repo_id}")
            return True
        except Exception as e:
            print(f"Error creating repository {repo_id}: {e}")
            return False
    
    def upload_model(
        self,
        local_path: str,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        create_pr: bool = False
    ) -> bool:
        """
        Upload a model to the Hub.
        
        Args:
            local_path: Path to local model directory
            repo_id: Repository ID (e.g., "username/model-name")
            commit_message: Commit message for the upload
            private: Whether to create a private repository if it doesn't exist
            create_pr: Whether to create a pull request instead of committing directly
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure repository exists
            if not self.repo_exists(repo_id):
                if not self.create_repository(repo_id, private=private):
                    return False
            
            # Upload the folder
            upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                commit_message=commit_message,
                token=self.token,
                create_pr=create_pr,
                repo_type="model"
            )
            
            print(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")
            return True
        except Exception as e:
            print(f"Error uploading model to {repo_id}: {e}")
            return False
    
    def create_model_card(
        self,
        local_path: str,
        model_config: Dict[str, Any],
        task_type: str,
        training_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a model card for the model.
        
        Args:
            local_path: Path to local model directory
            model_config: Model configuration dictionary
            task_type: Type of task ("clm" or "mlm")
            training_args: Training arguments used
            
        Returns:
            Path to the created model card
        """
        card_content = self._generate_model_card_content(
            model_config, task_type, training_args
        )
        
        card_path = os.path.join(local_path, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        
        return card_path
    
    def _generate_model_card_content(
        self,
        model_config: Dict[str, Any],
        task_type: str,
        training_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate model card content."""
        task_name = "Causal Language Modeling" if task_type == "clm" else "Masked Language Modeling"
        
        content = f"""---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- albert
- mixture-of-experts
- {task_type}
- pytorch
---

# AlbertMoE: {task_name} Model

This model is an implementation of ALBERT (A Lite BERT) with Mixture of Experts (MoE) architecture, trained for {task_name.lower()}.

## Model Details

- **Model Type**: AlbertMoE
- **Task**: {task_name}
- **Architecture**: ALBERT with Mixture of Experts
- **Framework**: PyTorch
- **License**: MIT

## Model Configuration

"""
        
        # Add configuration details
        if model_config:
            content += "```json\n"
            content += json.dumps(model_config, indent=2)
            content += "\n```\n\n"
        
        # Add training details if available
        if training_args:
            content += "## Training Details\n\n"
            content += "| Parameter | Value |\n"
            content += "|-----------|-------|\n"
            for key, value in training_args.items():
                if key not in ['token', 'wandb_key']:  # Skip sensitive info
                    content += f"| {key} | {value} |\n"
            content += "\n"
        
        content += """## Usage

```python
import torch
from transformers import AutoTokenizer
from albertmoe import AlbertForCausalLM, AlbertMoEConfig  # or AlbertForMaskedLM

# Load the model
config = AlbertMoEConfig()
model = AlbertForCausalLM(config)  # or AlbertForMaskedLM(config)
model.load_state_dict(torch.load("pytorch_model.bin"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

# Example usage
inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs["logits"]
```

## Citation

```bibtex
@misc{albertmoe,
  title={AlbertMoE: ALBERT with Mixture of Experts},
  author={MLNomad},
  year={2024},
  url={https://github.com/mlnomadpy/albertmoe}
}
```

## Acknowledgments

- ALBERT paper: [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
- Mixture of Experts: [Switch Transformer: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
"""
        
        return content
    
    def save_config_json(self, local_path: str, config_dict: Dict[str, Any]) -> str:
        """
        Save model configuration as config.json.
        
        Args:
            local_path: Path to local model directory
            config_dict: Configuration dictionary
            
        Returns:
            Path to the saved config file
        """
        config_path = os.path.join(local_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path


def push_to_hub(
    local_path: str,
    repo_id: str,
    model_config: Dict[str, Any],
    task_type: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    training_args: Optional[Dict[str, Any]] = None
) -> bool:
    """
    High-level function to push a model to Hugging Face Hub.
    
    Args:
        local_path: Path to local model directory
        repo_id: Repository ID (e.g., "username/model-name")
        model_config: Model configuration dictionary
        task_type: Type of task ("clm" or "mlm")
        token: Hugging Face token
        private: Whether to create a private repository
        commit_message: Custom commit message
        training_args: Training arguments used
        
    Returns:
        True if successful, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available. Cannot push to Hub.")
        return False
    
    try:
        hub_manager = HubManager(token=token)
        
        # Create model card and config.json
        hub_manager.create_model_card(local_path, model_config, task_type, training_args)
        hub_manager.save_config_json(local_path, model_config)
        
        # Set default commit message
        if commit_message is None:
            task_name = "CLM" if task_type == "clm" else "MLM"
            commit_message = f"Upload AlbertMoE {task_name} model"
        
        # Upload to Hub
        return hub_manager.upload_model(
            local_path=local_path,
            repo_id=repo_id,
            commit_message=commit_message,
            private=private
        )
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        return False