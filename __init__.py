"""Elastic Spectral Matryoshka Transformer (ESMT) package."""

from config import ESMTConfig, NanoGPTConfig, TrainConfig
from model import NanoGPT, SpectralGPT, count_parameters, create_matched_models
from spectral_layers import (
    CausalConv1d,
    MatryoshkaEmbedding,
    MatryoshkaPositionalEmbedding,
    SpectralGate,
    SpectralGatedLayer,
    SpectralMLP,
    SpectralNorm,
)
from spectral_init import (
    dct_transform_tensor,
    idct_transform_tensor,
    initialize_esmt_from_nanogpt,
    freeze_embeddings,
    unfreeze_embeddings,
    count_frozen_parameters,
    analyze_embedding_spectrum,
)

__all__ = [
    # Configs
    "ESMTConfig",
    "NanoGPTConfig",
    "TrainConfig",
    # Models
    "SpectralGPT",
    "NanoGPT",
    "count_parameters",
    "create_matched_models",
    # Layers
    "MatryoshkaEmbedding",
    "MatryoshkaPositionalEmbedding",
    "CausalConv1d",
    "SpectralNorm",
    "SpectralGate",
    "SpectralMLP",
    "SpectralGatedLayer",
    # Spectral initialization
    "dct_transform_tensor",
    "idct_transform_tensor",
    "initialize_esmt_from_nanogpt",
    "freeze_embeddings",
    "unfreeze_embeddings",
    "count_frozen_parameters",
    "analyze_embedding_spectrum",
]
