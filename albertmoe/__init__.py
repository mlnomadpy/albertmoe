"""
AlbertMoE: A modular implementation of ALBERT with Mixture of Experts.

This package provides training and evaluation capabilities for ALBERT models
with Mixture of Experts (MoE) architecture, supporting both Causal Language 
Modeling (CLM) and Masked Language Modeling (MLM) objectives.
"""

__version__ = "0.1.0"
__author__ = "MLNomadPy"

from .models import AlbertForCausalLM, AlbertForMaskedLM, ALBERT, SentenceAlbert
from .optimizers import ChillAdam
from .training import CLMTrainer, MLMTrainer, BaseTrainer
from .config import AlbertMoEConfig
from .components import RotaryEmbedding, MoE, Expert, AlbertEmbeddings, MultiHeadAttention, AlbertLayer
from .evaluation import MTEBEvaluator, evaluate_model

__all__ = [
    "AlbertForCausalLM",
    "AlbertForMaskedLM",
    "ALBERT",
    "SentenceAlbert",
    "ChillAdam",
    "CLMTrainer",
    "MLMTrainer", 
    "BaseTrainer",
    "AlbertMoEConfig",
    "RotaryEmbedding",
    "MoE",
    "Expert",
    "AlbertEmbeddings",
    "MultiHeadAttention",
    "AlbertLayer",
    "MTEBEvaluator",
    "evaluate_model",
]