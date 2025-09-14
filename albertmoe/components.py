"""
Core model components for AlbertMoE.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Union
from abc import ABC, abstractmethod

try:
    from torchtune.modules import RotaryPositionalEmbeddings as TorchTuneRoPE
    TORCHTUNE_AVAILABLE = True
except ImportError:
    TORCHTUNE_AVAILABLE = False


class AbstractPositionalEmbedding(ABC, nn.Module):
    """Abstract base class for positional embeddings."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply positional embeddings to input tensor.
        
        Args:
            x: Input tensor
            input_pos: Optional position indices
            
        Returns:
            Tensor with positional embeddings applied
        """
        pass


class TorchTuneRotaryEmbedding(AbstractPositionalEmbedding):
    """Wrapper around torchtune.modules.RotaryPositionalEmbeddings."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        if not TORCHTUNE_AVAILABLE:
            raise ImportError("torchtune is required for TorchTuneRotaryEmbedding. Install with: pip install torchtune torchao")
        
        self.rope = TorchTuneRoPE(dim=dim, max_seq_len=max_seq_len, base=base)
        
    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor following torchtune API.
        
        Args:
            x (torch.Tensor): input tensor with shape [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. Shape [b, s]. If None, assume sequential positions.
                
        Returns:
            torch.Tensor: output tensor with shape [b, s, n_h, h_d]
        """
        return self.rope(x, input_pos=input_pos)


class RotaryEmbedding(AbstractPositionalEmbedding):
    """Legacy RotaryEmbedding implementation for backward compatibility."""
    
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply RoPE using legacy implementation."""
        seq_len = x.shape[2] if x.dim() == 4 else x.shape[1]
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        return apply_rotary_pos_emb(x, x, cos, sin)[0]  # Return rotated query (x)


class AbsolutePositionalEmbedding(AbstractPositionalEmbedding):
    """Traditional absolute positional embeddings."""
    
    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add absolute positional embeddings to input.
        
        Args:
            x: Input embeddings with shape [batch_size, seq_len, hidden_size]
            input_pos: Optional position indices [batch_size, seq_len]
            
        Returns:
            Input embeddings with positional embeddings added
        """
        batch_size, seq_len = x.shape[:2]
        
        if input_pos is None:
            # Create default sequential position indices
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = input_pos
            
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings


def create_positional_embedding(config) -> AbstractPositionalEmbedding:
    """Factory function to create positional embeddings based on config."""
    if config.position_embedding_type == "rope":
        # Use TorchTune implementation when available, fallback to legacy
        if TORCHTUNE_AVAILABLE:
            return TorchTuneRotaryEmbedding(
                dim=config.rotary_dim,
                max_seq_len=config.rope_max_seq_len,
                base=config.rope_base
            )
        else:
            # Fallback to legacy implementation
            return RotaryEmbedding(
                dim=config.rotary_dim,
                max_seq_len=config.max_position_embeddings
            )
    elif config.position_embedding_type == "absolute":
        return AbsolutePositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size
        )
    else:
        raise ValueError(f"Unsupported position_embedding_type: {config.position_embedding_type}")


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.linear2(self.gelu(self.linear1(x))))


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k_experts = getattr(config, 'top_k_experts', 2)  # Default to 2 if not specified
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.view(-1, hidden_size)
        
        router_logits = self.gate(x)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=1, dtype=torch.float), 
            self.top_k_experts, 
            dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        expert_counts = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1).sum(dim=0)
        f_i = expert_counts / num_tokens
        P_i = router_logits.softmax(dim=-1).mean(dim=0)
        aux_loss = self.num_experts * (f_i * P_i).sum()
        
        final_hidden_states = torch.zeros_like(x)
        
        flat_selected_experts = selected_experts.view(-1)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_selected_experts == i)
            flat_expert_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            if flat_expert_indices.numel() == 0:
                continue
            
            expert_indices = flat_expert_indices // self.top_k_experts
            tokens_for_expert = x[expert_indices]
            expert_output = expert(tokens_for_expert)
            weights_for_expert = routing_weights.view(-1)[expert_mask]
            
            final_hidden_states.index_add_(
                0, expert_indices, 
                expert_output * weights_for_expert.unsqueeze(1).to(x.dtype)
            )

        return final_hidden_states.view(batch_size, seq_len, hidden_size), aux_loss


class AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        
        # Project from embedding_size to hidden_size
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Positional embeddings - only for absolute positioning
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.register_buffer("token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(input_ids.shape[0], -1)
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        
        # Project to hidden size first
        projected_embeddings = self.embedding_hidden_mapping_in(embeddings)
        
        # Add positional embeddings if using absolute positioning
        if self.position_embedding_type == "absolute":
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
            position_embeddings = self.position_embeddings(position_ids)
            projected_embeddings = projected_embeddings + position_embeddings
        
        embeddings = self.norm(projected_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        assert config.hidden_size % self.num_heads == 0, "Hidden size must be divisible by num_heads"
        self.qkv_layer = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Create positional embedding based on config
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "rope":
            self.pos_emb = create_positional_embedding(config)

    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_len, hidden_size = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply positional embeddings for RoPE
        if self.position_embedding_type == "rope":
            # Reshape for RoPE: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
            q_rope = q.permute(0, 2, 1, 3)
            k_rope = k.permute(0, 2, 1, 3)
            
            # Apply RoPE
            q_rope = self.pos_emb(q_rope, input_pos=position_ids)
            k_rope = self.pos_emb(k_rope, input_pos=position_ids)
            
            # Reshape back: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
            q = q_rope.permute(0, 2, 1, 3)
            k = k_rope.permute(0, 2, 1, 3)
        
        scale = (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            # Convert attention_mask to the correct format for additive attention
            # attention_mask comes as [batch_size, seq_len] with 1s for valid tokens, 0s for padding
            # We need to convert to [batch_size, 1, 1, seq_len] with 0s for valid, large negative for padding
            batch_size, seq_len = attention_mask.shape
            # Convert 1s to 0s and 0s to large negative values
            attention_mask = (1.0 - attention_mask) * -10000.0
            # Reshape to broadcast: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + attention_mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, hidden_size)
        output = self.output_proj(context)
        return output


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = MoE(config)
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None, position_ids=None):
        attn_output = self.attention(x, attention_mask, position_ids)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output, aux_loss = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x, aux_loss