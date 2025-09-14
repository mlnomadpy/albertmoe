"""
AlbertMoE model implementations for different tasks.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from .components import AlbertEmbeddings, AlbertLayer


class ALBERT(nn.Module):
    """Base ALBERT model with Mixture of Experts."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.shared_layer = AlbertLayer(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        x = self.embeddings(input_ids, token_type_ids, position_ids)
        total_aux_loss = 0.0
        for _ in range(self.config.num_hidden_layers):
            x, aux_loss = self.shared_layer(x, attention_mask, position_ids)
            total_aux_loss += aux_loss
        return x, total_aux_loss / self.config.num_hidden_layers


class AlbertForCausalLM(nn.Module):
    """ALBERT model for Causal Language Modeling (like GPT)."""
    
    def __init__(self, config):
        super().__init__()
        self.albert = ALBERT(config)
        self.lm_head_transform = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.RMSNorm(config.embedding_size)
        self.lm_head_decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, position_ids=None):
        hidden_states, aux_loss = self.albert(input_ids, attention_mask, token_type_ids, position_ids)
        transformed_states = self.norm(self.gelu(self.lm_head_transform(hidden_states)))
        logits = self.lm_head_decoder(transformed_states)
        
        loss, clm_loss = None, None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            clm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            aux_loss_coefficient = getattr(self.config, 'aux_loss_coefficient', 0.01)
            total_loss = clm_loss + aux_loss_coefficient * aux_loss
            loss = total_loss

        return {"loss": loss, "clm_loss": clm_loss, "logits": logits, "aux_loss": aux_loss}


class AlbertForMaskedLM(nn.Module):
    """ALBERT model for Masked Language Modeling (like BERT)."""
    
    def __init__(self, config):
        super().__init__()
        self.albert = ALBERT(config)
        self.mlm_head_transform = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.RMSNorm(config.embedding_size)
        self.mlm_head_decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, position_ids=None):
        hidden_states, aux_loss = self.albert(input_ids, attention_mask, token_type_ids, position_ids)
        transformed_states = self.norm(self.gelu(self.mlm_head_transform(hidden_states)))
        logits = self.mlm_head_decoder(transformed_states)

        loss, mlm_loss = None, None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            aux_loss_coefficient = getattr(self.config, 'aux_loss_coefficient', 0.01)
            total_loss = mlm_loss + aux_loss_coefficient * aux_loss
            loss = total_loss

        return {"loss": loss, "mlm_loss": mlm_loss, "logits": logits, "aux_loss": aux_loss}


class SentenceAlbert(nn.Module):
    """Wrapper for MTEB evaluation."""
    
    def __init__(self, config, model_path):
        super().__init__()
        
        # Try to load tokenizer with proper error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except (ValueError, TypeError, OSError) as e:
            # If tokenizer loading fails, provide helpful error message and fallback
            if "not a string" in str(e) or "vocab_file" in str(e):
                raise ValueError(
                    f"Failed to load tokenizer from '{model_path}'. "
                    f"The model directory appears to be missing required tokenizer files "
                    f"(e.g., vocab file, tokenizer config). "
                    f"Please ensure the model was saved correctly with all tokenizer files. "
                    f"Original error: {e}"
                ) from e
            elif "Unrecognized model" in str(e):
                raise ValueError(
                    f"Model path '{model_path}' does not contain a valid model configuration. "
                    f"Please ensure the directory contains config.json and other required model files. "
                    f"Original error: {e}"
                ) from e
            else:
                # Re-raise other tokenizer errors with context
                raise ValueError(
                    f"Failed to load tokenizer from '{model_path}': {e}"
                ) from e
        
        self.albert = ALBERT(config)

        # Load model state with proper error handling
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file '{model_file}' not found. "
                f"Please ensure the model was saved correctly and the path '{model_path}' "
                f"contains the required pytorch_model.bin file."
            )
        
        try:
            state_dict = torch.load(model_file, map_location="cpu")
            
            # Handle the case where the state dict was saved from a full model (e.g., AlbertForCausalLM)
            # and contains keys with "albert." prefix, but we need to load into just the ALBERT component
            albert_state_dict = {}
            albert_prefix = "albert."
            
            # Check if state dict has "albert." prefixed keys
            has_albert_prefix = any(key.startswith(albert_prefix) for key in state_dict.keys())
            
            if has_albert_prefix:
                # Extract only the albert component weights and remove the prefix
                for key, value in state_dict.items():
                    if key.startswith(albert_prefix):
                        # Remove the "albert." prefix
                        new_key = key[len(albert_prefix):]
                        albert_state_dict[new_key] = value
            else:
                # State dict is already in the expected format (no prefix)
                albert_state_dict = state_dict
            
            self.albert.load_state_dict(albert_state_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to load model weights from '{model_file}'. "
                f"The file may be corrupted or incompatible with the current model architecture. "
                f"Original error: {e}"
            ) from e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.albert.to(self.device)
        self.albert.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # ALBERT now returns a tuple
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=32, **kwargs):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.device)
            model_output = self.albert(**inputs)
            sentence_embedding = self.mean_pooling(model_output, inputs["attention_mask"])
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
            all_embeddings.append(sentence_embedding.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()