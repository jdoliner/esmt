"""Elastic Spectral Matryoshka Transformer (ESMT) package."""

from config import ESMTConfig, NanoGPTConfig, TrainConfig, ComplexESMTConfig
from model import NanoGPT, SpectralGPT, ComplexSpectralGPT, count_parameters, create_matched_models
from spectral_layers import (
    CausalConv1d,
    MatryoshkaEmbedding,
    MatryoshkaPositionalEmbedding,
    SpectralGate,
    SpectralGatedLayer,
    SpectralMLP,
    SpectralNorm,
)
from complex_layers import (
    ComplexLinear,
    ModReLU,
    ComplexLayerNorm,
    ComplexEmbedding,
    ComplexPositionalEncoding,
    ComplexAttention,
    ComplexFFN,
    ComplexSpectralBlock,
    ComplexToLogits,
    ComplexSpectralGate,
    ComplexSpectralGateDynamic,
)
from spectral_init import (
    dct_transform_tensor,
    idct_transform_tensor,
    initialize_esmt_from_nanogpt,
    freeze_embeddings,
    unfreeze_embeddings,
    count_frozen_parameters,
    analyze_embedding_spectrum,
    # Complex initialization
    fft_transform_tensor,
    ifft_transform_tensor,
    initialize_complex_esmt_from_nanogpt,
    freeze_complex_embeddings,
    unfreeze_complex_embeddings,
    analyze_complex_embedding_spectrum,
)

__all__ = [
    # Configs
    "ESMTConfig",
    "NanoGPTConfig",
    "TrainConfig",
    "ComplexESMTConfig",
    # Models
    "SpectralGPT",
    "NanoGPT",
    "ComplexSpectralGPT",
    "count_parameters",
    "create_matched_models",
    # Real-valued Layers
    "MatryoshkaEmbedding",
    "MatryoshkaPositionalEmbedding",
    "CausalConv1d",
    "SpectralNorm",
    "SpectralGate",
    "SpectralMLP",
    "SpectralGatedLayer",
    # Complex-valued Layers
    "ComplexLinear",
    "ModReLU",
    "ComplexLayerNorm",
    "ComplexEmbedding",
    "ComplexPositionalEncoding",
    "ComplexAttention",
    "ComplexFFN",
    "ComplexSpectralBlock",
    "ComplexToLogits",
    "ComplexSpectralGate",
    "ComplexSpectralGateDynamic",
    # Spectral initialization (DCT for real)
    "dct_transform_tensor",
    "idct_transform_tensor",
    "initialize_esmt_from_nanogpt",
    "freeze_embeddings",
    "unfreeze_embeddings",
    "count_frozen_parameters",
    "analyze_embedding_spectrum",
    # Spectral initialization (FFT for complex)
    "fft_transform_tensor",
    "ifft_transform_tensor",
    "initialize_complex_esmt_from_nanogpt",
    "freeze_complex_embeddings",
    "unfreeze_complex_embeddings",
    "analyze_complex_embedding_spectrum",
]
