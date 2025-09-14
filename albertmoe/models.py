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
        self.norm = nn.LayerNorm(config.embedding_size)
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
        self.norm = nn.LayerNorm(config.embedding_size)
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.albert = ALBERT(config)

        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        self.albert.load_state_dict(state_dict)

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