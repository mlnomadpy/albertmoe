"""
Configuration classes for AlbertMoE models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AlbertMoEConfig:
    """Configuration class for ALBERT with Mixture of Experts."""
    
    # Model architecture
    vocab_size: int = 30000
    embedding_size: int = 128
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    
    # MoE specific
    num_experts: int = 8
    expert_capacity: int = 64
    top_k_experts: int = 2
    aux_loss_coefficient: float = 0.01
    
    # Training specific  
    use_rotary: bool = True
    rotary_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.rotary_dim is None:
            self.rotary_dim = self.hidden_size // self.num_attention_heads
        # Ensure rotary_dim is even for proper rotation
        if self.rotary_dim % 2 != 0:
            self.rotary_dim = self.rotary_dim - 1