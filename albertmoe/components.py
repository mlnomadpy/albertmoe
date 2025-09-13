"""
Core model components for AlbertMoE.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
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

    def forward(self, x):
        seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


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
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer("token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(input_ids.shape[0], -1)
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_rotary = getattr(config, 'use_rotary', True)
        
        if self.use_rotary:
            rotary_dim = getattr(config, 'rotary_dim', self.attention_head_size)
            self.rotary_emb = RotaryEmbedding(rotary_dim, config.max_position_embeddings)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        if self.use_rotary:
            cos, sin = self.rotary_emb(query_layer)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.moe = MoE(config)
        self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = attention_outputs[0]
        
        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        hidden_states = self.LayerNorm(attention_output + hidden_states)
        
        # MoE FFN
        moe_output, aux_loss = self.moe(hidden_states)
        hidden_states = self.ffn_layernorm(hidden_states + moe_output)
        
        outputs = (hidden_states, aux_loss)
        if output_attentions:
            outputs = outputs + (attention_outputs[1],)
        
        return outputs