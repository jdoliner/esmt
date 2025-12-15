"""Spectral-Augmented Transformer (SAT) model implementation.

This module implements a hybrid architecture combining Fourier Neural Operators (FNOs)
with Transformers. The FNO acts as a "world model" capturing global sequence dynamics
in the frequency domain, while the Transformer handles local token interactions.

Key components:
- Bfloat16 complex utilities: Represent complex numbers as (real, imag) pairs
- CumulativeFFT: Efficient causal FFT computation for all positions
- FNO blocks: Spectral convolutions with ModReLU activation
- Bridge: Converts spectral state to Transformer conditioning (AdaLN or cross-attention)
- AdaLN Transformer: Standard decoder with adaptive layer normalization

Architecture:
    Input tokens -> Embedding -> [Spectral Stream] -> Bridge -> [Token Stream] -> Logits
                                      |                              ^
                                      +---- conditions via AdaLN ----+
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SATConfig


# ==============================================================================
# Bfloat16 Complex Number Utilities
# ==============================================================================
# Complex numbers are represented as tensors with shape (..., 2) where
# [..., 0] is the real part and [..., 1] is the imaginary part.
# All operations preserve bfloat16 dtype for compatibility with mixed precision.


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two complex tensors represented as (..., 2) bfloat16 pairs.

    (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)

    Args:
        a: Complex tensor (..., 2)
        b: Complex tensor (..., 2), must be broadcastable with a

    Returns:
        Complex tensor (..., 2)
    """
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]

    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real

    return torch.stack([real, imag], dim=-1)


def complex_mul_scalar(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Multiply complex tensor by real scalar.

    Args:
        z: Complex tensor (..., 2)
        s: Real scalar tensor, broadcastable to z[..., 0]

    Returns:
        Complex tensor (..., 2)
    """
    return z * s.unsqueeze(-1)


def complex_abs(z: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitude of complex tensor.

    |z| = sqrt(real^2 + imag^2)

    Args:
        z: Complex tensor (..., 2)

    Returns:
        Real tensor (...) containing magnitudes
    """
    return torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-8)


def complex_abs_squared(z: torch.Tensor) -> torch.Tensor:
    """
    Compute squared magnitude of complex tensor (avoids sqrt).

    |z|^2 = real^2 + imag^2

    Args:
        z: Complex tensor (..., 2)

    Returns:
        Real tensor (...) containing squared magnitudes
    """
    return z[..., 0] ** 2 + z[..., 1] ** 2


def complex_conj(z: torch.Tensor) -> torch.Tensor:
    """
    Compute complex conjugate.

    conj(a + bi) = a - bi

    Args:
        z: Complex tensor (..., 2)

    Returns:
        Complex tensor (..., 2)
    """
    return torch.stack([z[..., 0], -z[..., 1]], dim=-1)


def complex_from_polar(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Create complex tensor from magnitude and phase.

    z = |z| * e^{i*phase} = |z| * (cos(phase) + i*sin(phase))

    Args:
        magnitude: Real tensor (...)
        phase: Real tensor (...), in radians

    Returns:
        Complex tensor (..., 2)
    """
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    return torch.stack([real, imag], dim=-1)


def complex_phase(z: torch.Tensor) -> torch.Tensor:
    """
    Compute phase angle of complex tensor.

    angle(z) = atan2(imag, real)

    Args:
        z: Complex tensor (..., 2)

    Returns:
        Real tensor (...) containing phases in [-pi, pi]
    """
    return torch.atan2(z[..., 1], z[..., 0])


def mod_relu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    ModReLU activation for complex numbers.

    ModReLU(z) = ReLU(|z| - b) * (z / |z|)
               = ReLU(|z| - b) * e^{i*angle(z)}

    This applies a learnable threshold to the magnitude while preserving phase.
    If |z| < b, the output is 0 (the signal is "noise").

    Args:
        z: Complex tensor (..., 2)
        bias: Real tensor, learnable bias (typically positive), broadcastable

    Returns:
        Complex tensor (..., 2)
    """
    mag = complex_abs(z)  # (...)
    phase = complex_phase(z)  # (...)

    # Apply ReLU to (magnitude - bias)
    new_mag = F.relu(mag - bias)  # (...)

    # Reconstruct with new magnitude and original phase
    return complex_from_polar(new_mag, phase)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert real tensor to complex by adding zero imaginary part.

    Args:
        x: Real tensor (...)

    Returns:
        Complex tensor (..., 2) with imag=0
    """
    zeros = torch.zeros_like(x)
    return torch.stack([x, zeros], dim=-1)


def rfft_to_complex_pair(fft_out: torch.Tensor) -> torch.Tensor:
    """
    Convert torch.fft.rfft output (complex64/128) to bfloat16 pair representation.

    Args:
        fft_out: Complex tensor from torch.fft.rfft

    Returns:
        Tensor (..., 2) in bfloat16 with real/imag parts
    """
    real = fft_out.real.to(torch.bfloat16)
    imag = fft_out.imag.to(torch.bfloat16)
    return torch.stack([real, imag], dim=-1)


def complex_pair_to_complex(z: torch.Tensor) -> torch.Tensor:
    """
    Convert bfloat16 pair representation to torch complex tensor.

    Args:
        z: Tensor (..., 2) with real/imag parts

    Returns:
        Complex tensor (...)
    """
    return torch.complex(z[..., 0].float(), z[..., 1].float())


# ==============================================================================
# Complex Linear Operations
# ==============================================================================


class ComplexLinearBF16(nn.Module):
    """
    Linear layer for complex-valued tensors using bfloat16 pairs.

    Computes: y = Wx + b where W, x, b are all complex (as bfloat16 pairs).

    The weight is stored as (out_features, in_features, 2) and bias as (out_features, 2).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight: (out_features, in_features, 2) for complex
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 2, dtype=torch.bfloat16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 2, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self._init_weights(zero_init)

    def _init_weights(self, zero_init: bool) -> None:
        if zero_init:
            # Zero initialization for residual exits
            nn.init.zeros_(self.weight)
        else:
            # Kaiming-style initialization for complex
            # Scale by 1/sqrt(2*fan_in) since we have real and imag parts
            fan_in = self.in_features
            std = 1.0 / math.sqrt(2.0 * fan_in)
            nn.init.normal_(self.weight[..., 0], mean=0.0, std=std)
            nn.init.normal_(self.weight[..., 1], mean=0.0, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor (..., in_features, 2)

        Returns:
            Complex tensor (..., out_features, 2)
        """
        # x: (..., in_features, 2)
        # weight: (out_features, in_features, 2)

        x_real, x_imag = x[..., 0], x[..., 1]  # (..., in_features)
        w_real, w_imag = self.weight[..., 0], self.weight[..., 1]  # (out, in)

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # out_real = x_real @ w_real.T - x_imag @ w_imag.T
        # out_imag = x_real @ w_imag.T + x_imag @ w_real.T
        out_real = F.linear(x_real, w_real) - F.linear(x_imag, w_imag)
        out_imag = F.linear(x_real, w_imag) + F.linear(x_imag, w_real)

        out = torch.stack([out_real, out_imag], dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out


# ==============================================================================
# Cumulative FFT Module
# ==============================================================================


class CumulativeFFT(nn.Module):
    """
    Compute cumulative FFT for causal spectral representation.

    For each position t, computes the FFT of tokens [0, 1, ..., t] with the rest
    zero-padded to the full sequence length. This is done efficiently by:

    1. Precomputing twiddle factors: e^{-2*pi*i*k*n/N} for all (n, k)
    2. Computing each token's contribution: embed[t] * twiddles[t]
    3. Cumulative sum over the sequence dimension

    This gives us O(T * D * K) compute instead of O(T^2 * log(T)) for naive approach.

    The output is a complex tensor in bfloat16 pair format.
    """

    def __init__(self, seq_len: int, k_max: int):
        """
        Args:
            seq_len: Maximum sequence length (N for FFT)
            k_max: Number of frequency modes to keep
        """
        super().__init__()
        self.seq_len = seq_len
        self.k_max = k_max

        # Precompute twiddle factors: e^{-2*pi*i*k*n/N}
        # Shape: (seq_len, k_max, 2) for bfloat16 complex pairs
        n = torch.arange(seq_len, dtype=torch.float32)
        k = torch.arange(k_max, dtype=torch.float32)

        # Compute angles: -2*pi*k*n/N
        # Using outer product: (seq_len,) x (k_max,) -> (seq_len, k_max)
        angles = -2 * math.pi * torch.outer(n, k) / seq_len

        # Convert to complex pairs
        twiddles_real = torch.cos(angles).to(torch.bfloat16)
        twiddles_imag = torch.sin(angles).to(torch.bfloat16)
        twiddles = torch.stack([twiddles_real, twiddles_imag], dim=-1)

        self.register_buffer("twiddles", twiddles)
        self.twiddles: torch.Tensor

        # Ortho normalization factor: 1/sqrt(N)
        self.register_buffer(
            "norm_factor", torch.tensor(1.0 / math.sqrt(seq_len), dtype=torch.bfloat16)
        )
        self.norm_factor: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative FFT for all positions.

        Args:
            x: Real-valued embeddings (batch, seq_len, d_model) in bfloat16

        Returns:
            Complex tensor (batch, seq_len, d_model, k_max, 2) where the value at
            position t contains the FFT of x[0:t+1] zero-padded to seq_len.
        """
        batch, seq_len, d_model = x.shape

        # Ensure we don't exceed precomputed twiddles
        assert seq_len <= self.seq_len, f"seq_len {seq_len} exceeds max {self.seq_len}"

        # Get twiddles for current sequence length
        twiddles = self.twiddles[:seq_len]  # (seq_len, k_max, 2)

        # Compute each token's contribution to all frequency bins
        # x: (batch, seq_len, d_model)
        # twiddles: (seq_len, k_max, 2)
        #
        # For each position n and frequency k:
        #   contribution[n, k] = x[n] * twiddle[n, k]
        #
        # We want: (batch, seq_len, d_model, k_max, 2)

        # Expand x to have frequency dimension
        # x_expanded: (batch, seq_len, d_model, 1)
        x_expanded = x.unsqueeze(-1)

        # Expand twiddles to broadcast with batch and d_model
        # twiddles_expanded: (1, seq_len, 1, k_max, 2)
        twiddles_expanded = twiddles.unsqueeze(0).unsqueeze(2)

        # Multiply real x by complex twiddles
        # Result: (batch, seq_len, d_model, k_max, 2)
        contributions_real = x_expanded * twiddles_expanded[..., 0]
        contributions_imag = x_expanded * twiddles_expanded[..., 1]
        contributions = torch.stack([contributions_real, contributions_imag], dim=-1)

        # Cumulative sum over sequence dimension gives FFT at each position
        # At position t, we have sum of contributions from 0 to t
        cumulative_fft = torch.cumsum(contributions, dim=1)

        # Apply ortho normalization
        cumulative_fft = cumulative_fft * self.norm_factor

        return cumulative_fft


# ==============================================================================
# FNO Layers
# ==============================================================================


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution layer for FNO.

    Performs element-wise multiplication with learnable complex weights in the
    frequency domain. This corresponds to convolution in the spatial domain.

    Weight shape: (d_out, d_in, k_max, 2) for complex weights per frequency mode.
    """

    def __init__(self, d_in: int, d_out: int, k_max: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.k_max = k_max

        # Complex weights: (d_out, d_in, k_max, 2)
        self.weight = nn.Parameter(torch.zeros(d_out, d_in, k_max, 2, dtype=torch.bfloat16))

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize with small complex values
        # Scale by 1/sqrt(d_in * k_max) for each output dimension
        std = 1.0 / math.sqrt(self.d_in * self.k_max)
        nn.init.normal_(self.weight[..., 0], mean=0.0, std=std)
        nn.init.normal_(self.weight[..., 1], mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor (batch, seq_len, d_in, k_max, 2)

        Returns:
            Complex tensor (batch, seq_len, d_out, k_max, 2)
        """
        # x: (batch, seq_len, d_in, k_max, 2)
        # weight: (d_out, d_in, k_max, 2)

        batch, seq_len, d_in, k_max, _ = x.shape

        # For each frequency mode k, we do a complex matrix-vector product:
        # out[b, t, d_out, k] = sum_d_in weight[d_out, d_in, k] * x[b, t, d_in, k]

        # Reshape for batch matmul
        # x: (batch * seq_len, k_max, d_in, 2) -> we need (batch*seq*k, d_in, 2)
        x_reshape = x.permute(0, 1, 3, 2, 4)  # (batch, seq, k_max, d_in, 2)
        x_reshape = x_reshape.reshape(batch * seq_len * k_max, d_in, 2)

        # weight: (d_out, d_in, k_max, 2) -> (k_max, d_out, d_in, 2)
        w_reshape = self.weight.permute(2, 0, 1, 3)  # (k_max, d_out, d_in, 2)

        # Complex matmul for each frequency mode
        x_real, x_imag = x_reshape[..., 0], x_reshape[..., 1]  # (batch*seq*k, d_in)

        # Expand weight for batch processing
        # We need to apply the same weight[k] to all (batch*seq) samples for that k
        out_list = []
        for k in range(k_max):
            # Get samples for this frequency
            start_idx = k
            indices = torch.arange(start_idx, batch * seq_len * k_max, k_max, device=x.device)
            x_k_real = x_real[indices]  # (batch*seq, d_in)
            x_k_imag = x_imag[indices]  # (batch*seq, d_in)

            w_k_real = w_reshape[k, :, :, 0]  # (d_out, d_in)
            w_k_imag = w_reshape[k, :, :, 1]  # (d_out, d_in)

            # Complex matmul: (d_out, d_in) @ (batch*seq, d_in).T -> (d_out, batch*seq)
            out_k_real = F.linear(x_k_real, w_k_real) - F.linear(x_k_imag, w_k_imag)
            out_k_imag = F.linear(x_k_real, w_k_imag) + F.linear(x_k_imag, w_k_real)

            out_list.append(torch.stack([out_k_real, out_k_imag], dim=-1))

        # Stack and reshape back
        # out_list[k]: (batch*seq, d_out, 2)
        out = torch.stack(out_list, dim=2)  # (batch*seq, d_out, k_max, 2)
        out = out.reshape(batch, seq_len, self.d_out, k_max, 2)

        return out


class FNOBlock(nn.Module):
    """
    Fourier Neural Operator block.

    Architecture:
        x -> SpectralConv -> + -> ModReLU -> out
             |               ^
             +-- residual ---+

    The residual is in the frequency domain (simple addition).
    """

    def __init__(self, d_model: int, k_max: int):
        super().__init__()
        self.d_model = d_model
        self.k_max = k_max

        # Spectral convolution (complex)
        self.spectral_conv = SpectralConv1d(d_model, d_model, k_max)

        # ModReLU bias (one per feature dimension, shared across frequencies)
        self.modrelu_bias = nn.Parameter(torch.full((d_model,), 0.5, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor (batch, seq_len, d_model, k_max, 2)

        Returns:
            Complex tensor (batch, seq_len, d_model, k_max, 2)
        """
        # Spectral convolution
        out = self.spectral_conv(x)

        # Residual connection (in frequency domain)
        out = out + x

        # ModReLU activation
        # Apply per-feature bias, broadcast across batch, seq, and frequency
        # bias shape for broadcasting: (1, 1, d_model, 1)
        bias = self.modrelu_bias.view(1, 1, -1, 1)
        out = mod_relu(out, bias)

        return out


# ==============================================================================
# Bridge: Spectral to Transformer Conditioning
# ==============================================================================


class AdaLNBridge(nn.Module):
    """
    Bridge that converts spectral state to AdaLN parameters (gamma, beta).

    Architecture:
        spectral (B, T, D_spec, K, 2)
        -> magnitude mean over K -> (B, T, D_spec)
        -> shared_proj -> (B, T, D)
        -> per-layer proj -> (gamma_i, beta_i) for each layer

    The final projection is zero-initialized so that at init:
        gamma = 1, beta = 0 (standard LayerNorm behavior)
    """

    def __init__(self, d_spectral: int, d_model: int, n_layers: int):
        super().__init__()
        self.d_spectral = d_spectral
        self.d_model = d_model
        self.n_layers = n_layers

        # Shared projection from spectral to model dimension
        self.shared_proj = nn.Linear(d_spectral, d_model)
        self.act = nn.SiLU()

        # Per-layer projections to (gamma, beta)
        self.layer_projs = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard init for shared projection
        nn.init.normal_(self.shared_proj.weight, std=0.02)
        nn.init.zeros_(self.shared_proj.bias)

        # Zero-init for layer projections so gamma=1, beta=0 at start
        for i in range(len(self.layer_projs)):
            proj = self.layer_projs[i]
            assert isinstance(proj, nn.Linear), "Layer proj should be Linear"
            nn.init.zeros_(proj.weight)
            # Initialize bias so that gamma=1, beta=0
            # Output is [gamma, beta] concatenated, each of size d_model
            # Set gamma bias to 1, beta bias to 0
            assert proj.bias is not None, "Layer proj should have bias"
            proj.bias.data[: self.d_model] = 1.0  # gamma
            proj.bias.data[self.d_model :] = 0.0  # beta

    def forward(self, spectral: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            spectral: Complex tensor (batch, seq_len, d_spectral, k_max, 2)

        Returns:
            List of (gamma, beta) tuples for each layer, each shape (batch, seq_len, d_model)
        """
        # Compute magnitude and mean over frequency dimension
        # spectral: (B, T, D_spec, K, 2)
        magnitude = complex_abs(spectral)  # (B, T, D_spec, K)
        pooled = magnitude.mean(dim=-1)  # (B, T, D_spec)

        # Convert to float32 for projection
        pooled = pooled.float()

        # Shared projection
        hidden = self.shared_proj(pooled)  # (B, T, D)
        hidden = self.act(hidden)

        # Per-layer projections
        conditioning = []
        for proj in self.layer_projs:
            gamma_beta = proj(hidden)  # (B, T, 2*D)
            gamma = gamma_beta[..., : self.d_model]  # (B, T, D)
            beta = gamma_beta[..., self.d_model :]  # (B, T, D)
            conditioning.append((gamma, beta))

        return conditioning


class SpectralCrossAttentionBridge(nn.Module):
    """
    Bridge that prepares spectral tokens for cross-attention.

    The spectral state (B, T, D_spec, K, 2) is projected to key/value tokens
    that the transformer can attend to. Each position t attends to its own
    K spectral tokens (the frequency modes at that position).

    This allows position-specific spectral influence rather than global modulation.
    """

    def __init__(self, d_spectral: int, d_model: int, n_heads: int):
        super().__init__()
        self.d_spectral = d_spectral
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Project complex spectral features to real K, V
        # Input: magnitude of spectral (we use magnitude, not concat real/imag)
        self.k_proj = nn.Linear(d_spectral, d_model, bias=False)
        self.v_proj = nn.Linear(d_spectral, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)

    def forward(self, spectral: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectral: Complex tensor (batch, seq_len, d_spectral, k_max, 2)

        Returns:
            k: Key tensor (batch, seq_len, k_max, d_model)
            v: Value tensor (batch, seq_len, k_max, d_model)
        """
        # Use magnitude as features
        magnitude = complex_abs(spectral)  # (B, T, D_spec, K)

        # Transpose to (B, T, K, D_spec) for projection
        magnitude = magnitude.permute(0, 1, 3, 2)  # (B, T, K, D_spec)

        # Convert to float32 for projection
        magnitude = magnitude.float()

        # Project to K, V
        k = self.k_proj(magnitude)  # (B, T, K, D)
        v = self.v_proj(magnitude)  # (B, T, K, D)

        return k, v


# ==============================================================================
# Transformer Components with AdaLN
# ==============================================================================


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).

    Instead of learned scale/shift, uses conditioning from external source:
        AdaLN(x, gamma, beta) = gamma * LayerNorm(x) + beta

    The gamma and beta are provided externally (from the spectral bridge).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Note: No learnable parameters - gamma/beta come from conditioning

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            gamma: Scale tensor (batch, seq_len, d_model) or (batch, 1, d_model)
            beta: Shift tensor (batch, seq_len, d_model) or (batch, 1, d_model)

        Returns:
            Normalized and modulated tensor (batch, seq_len, d_model)
        """
        # Standard layer norm (without affine)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply conditioning
        return gamma * x_norm + beta


class CausalSelfAttentionAdaLN(nn.Module):
    """
    Causal self-attention with AdaLN pre-normalization.
    """

    def __init__(self, config: SATConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Pre-norm with AdaLN
        self.adaln = AdaptiveLayerNorm(config.d_model, eps=config.eps)

        # QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                1, 1, config.seq_len, config.seq_len
            ),
        )
        self.mask: torch.Tensor

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            gamma, beta: AdaLN conditioning (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # AdaLN pre-norm
        x_norm = self.adaln(x, gamma, beta)

        # Compute Q, K, V
        qkv = self.qkv(x_norm)  # (batch, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Attend to values
        out = attn @ v  # (batch, heads, seq, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.proj(out)

        return out


class SpectralCrossAttention(nn.Module):
    """
    Cross-attention from transformer to spectral tokens.

    Each transformer position queries its corresponding spectral tokens
    (the K frequency modes at that position).
    """

    def __init__(self, config: SATConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.k_max = config.k_max

        # Query projection (from transformer hidden state)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q_proj.weight, std=0.02)
        # Zero-init output projection for residual
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Transformer hidden state (batch, seq_len, d_model)
            k: Spectral keys (batch, seq_len, k_max, d_model)
            v: Spectral values (batch, seq_len, k_max, d_model)

        Returns:
            Cross-attention output (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        k_max = k.shape[2]

        # Project query
        q = self.q_proj(x)  # (B, T, D)

        # Reshape for multi-head attention
        # q: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # k, v: (B, T, K, D) -> (B, T, K, n_heads, head_dim) -> (B, n_heads, T, K, head_dim)
        k = k.view(batch, seq_len, k_max, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(batch, seq_len, k_max, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention: q (B, n_heads, T, head_dim) @ k.T (B, n_heads, T, head_dim, K)
        # -> (B, n_heads, T, K)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhti,bhtki->bhtk", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Attend to values: attn (B, n_heads, T, K) @ v (B, n_heads, T, K, head_dim)
        # -> (B, n_heads, T, head_dim)
        out = torch.einsum("bhtk,bhtki->bhti", attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.out_proj(out)

        return out


class MLPAdaLN(nn.Module):
    """
    MLP block with AdaLN pre-normalization.
    """

    def __init__(self, config: SATConfig):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.mlp_ratio

        # Pre-norm with AdaLN
        self.adaln = AdaptiveLayerNorm(config.d_model, eps=config.eps)

        # MLP layers
        self.fc1 = nn.Linear(config.d_model, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, config.d_model)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            gamma, beta: AdaLN conditioning (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x_norm = self.adaln(x, gamma, beta)
        x = self.fc1(x_norm)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SATBlock(nn.Module):
    """
    Spectral-Augmented Transformer block.

    Architecture depends on integration_mode:
    - "adaln": Pre-norm attention and MLP use AdaLN with spectral conditioning
    - "cross_attention": Standard pre-norm + cross-attention to spectral tokens
    - "both": AdaLN + cross-attention
    """

    def __init__(self, config: SATConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.integration_mode = config.integration_mode

        # Self-attention with AdaLN (always used)
        self.attn = CausalSelfAttentionAdaLN(config)

        # MLP with AdaLN
        self.mlp = MLPAdaLN(config)

        # Cross-attention (optional)
        if config.integration_mode in ["cross_attention", "both"]:
            self.cross_attn = SpectralCrossAttention(config)
            self.cross_attn_ln = nn.LayerNorm(config.d_model, eps=config.eps)
        else:
            self.cross_attn = None
            self.cross_attn_ln = None

    def forward(
        self,
        x: torch.Tensor,
        gamma_attn: torch.Tensor,
        beta_attn: torch.Tensor,
        gamma_mlp: torch.Tensor,
        beta_mlp: torch.Tensor,
        spectral_k: torch.Tensor | None = None,
        spectral_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            gamma_attn, beta_attn: AdaLN conditioning for attention
            gamma_mlp, beta_mlp: AdaLN conditioning for MLP
            spectral_k, spectral_v: Spectral KV for cross-attention (optional)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual
        x = x + self.attn(x, gamma_attn, beta_attn)

        # Cross-attention (if enabled)
        if (
            self.cross_attn is not None
            and self.cross_attn_ln is not None
            and spectral_k is not None
        ):
            x_norm = self.cross_attn_ln(x)
            x = x + self.cross_attn(x_norm, spectral_k, spectral_v)

        # MLP with residual
        x = x + self.mlp(x, gamma_mlp, beta_mlp)

        return x


# ==============================================================================
# Main Model
# ==============================================================================


class SpectralAugmentedTransformer(nn.Module):
    """
    Spectral-Augmented Transformer (SAT).

    Combines a Fourier Neural Operator (FNO) "world model" with a Transformer decoder.
    The FNO processes input in the frequency domain and conditions the Transformer
    via Adaptive Layer Norm (AdaLN) and/or cross-attention.

    Architecture:
        Input tokens
        -> Embedding
        -> Spectral Stream (cumulative FFT -> FNO blocks)
        -> Bridge (AdaLN params and/or cross-attention KV)
        -> Token Stream (Transformer with AdaLN conditioning)
        -> Output logits
    """

    def __init__(self, config: SATConfig):
        super().__init__()
        self.config = config

        # Ensure config values are set (they should be after __post_init__)
        assert config.d_spectral is not None, "d_spectral must be set"
        assert config.n_fno_layers is not None, "n_fno_layers must be set"
        assert config.k_max is not None, "k_max must be set"

        # Token embedding (same as NanoGPT)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)

        # =====================================================================
        # Spectral Stream
        # =====================================================================

        # Project to spectral dimension
        self.spectral_proj_in = nn.Linear(config.d_model, config.d_spectral)

        # Cumulative FFT
        self.cumulative_fft = CumulativeFFT(config.seq_len, config.k_max)

        # FNO blocks
        self.fno_blocks = nn.ModuleList(
            [FNOBlock(config.d_spectral, config.k_max) for _ in range(config.n_fno_layers)]
        )

        # =====================================================================
        # Bridge
        # =====================================================================

        # AdaLN bridge (always created, provides default gamma=1, beta=0 if not used)
        self.adaln_bridge = AdaLNBridge(config.d_spectral, config.d_model, config.n_layers)

        # Cross-attention bridge (optional)
        if config.integration_mode in ["cross_attention", "both"]:
            self.cross_attn_bridge = SpectralCrossAttentionBridge(
                config.d_spectral, config.d_model, config.n_heads
            )
        else:
            self.cross_attn_bridge = None

        # =====================================================================
        # Token Stream (Transformer)
        # =====================================================================

        self.blocks = nn.ModuleList([SATBlock(config, layer_idx=i) for i in range(config.n_layers)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.eps)

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_spectral: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input token indices (batch, seq_len)
            return_spectral: If True, also return spectral state for auxiliary loss

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            spectral (optional): FNO output (batch, seq_len, d_spectral, k_max, 2)
        """
        batch, seq_len = x.shape
        device = x.device

        # =====================================================================
        # Embedding
        # =====================================================================

        tok_emb = self.token_emb(x)  # (B, T, D)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (T, D)

        h = tok_emb + pos_emb  # (B, T, D)

        # =====================================================================
        # Spectral Stream
        # =====================================================================

        # Project to spectral dimension
        h_spectral = self.spectral_proj_in(h)  # (B, T, D_spec)

        # Ensure bfloat16 for FFT
        h_spectral = h_spectral.to(torch.bfloat16)

        # Cumulative FFT
        spectral = self.cumulative_fft(h_spectral)  # (B, T, D_spec, K, 2)

        # FNO blocks
        for fno_block in self.fno_blocks:
            spectral = fno_block(spectral)

        # =====================================================================
        # Bridge
        # =====================================================================

        # Get AdaLN conditioning
        # Each element is (gamma, beta) tuple for one layer
        adaln_conditioning = self.adaln_bridge(spectral)

        # Get cross-attention KV (if enabled)
        if self.cross_attn_bridge is not None:
            spectral_k, spectral_v = self.cross_attn_bridge(spectral)
        else:
            spectral_k, spectral_v = None, None

        # =====================================================================
        # Token Stream (Transformer)
        # =====================================================================

        # Convert h back to float32 for transformer (it's still float32 from embedding)
        for i, block in enumerate(self.blocks):
            gamma_attn, beta_attn = adaln_conditioning[i]
            # For MLP, we reuse the same conditioning (could add separate conditioning)
            gamma_mlp, beta_mlp = gamma_attn, beta_attn

            h = block(
                h,
                gamma_attn.float(),
                beta_attn.float(),
                gamma_mlp.float(),
                beta_mlp.float(),
                spectral_k,
                spectral_v,
            )

        # Final layer norm and projection
        h = self.ln_f(h)
        logits = self.lm_head(h)

        if return_spectral:
            return logits, spectral
        return logits

    def compute_auxiliary_loss(
        self,
        spectral: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary FNO prediction loss.

        The FNO should predict the next step's spectral state:
            L_aux = weighted_MSE(FNO_output[t], FFT[0:t+1])

        We use sqrt(k+1) weighting to counteract pink noise bias.

        Args:
            spectral: FNO output (batch, seq_len, d_spectral, k_max, 2)
            x: Original input tokens (batch, seq_len)

        Returns:
            Auxiliary loss scalar
        """
        batch, seq_len = x.shape

        # Get target: FFT of full sequence at each position (shifted by 1)
        # We want target[t] = FFT(x[0:t+1])
        # This is just the cumulative FFT shifted by 1 position

        # Re-embed and project
        tok_emb = self.token_emb(x)
        h_spectral = self.spectral_proj_in(tok_emb).to(torch.bfloat16)

        # Get cumulative FFT (this is the target, detached)
        with torch.no_grad():
            target_fft = self.cumulative_fft(h_spectral)  # (B, T, D_spec, K, 2)

        # FNO output at position t should predict FFT at position t+1
        # So we compare spectral[:, :-1] with target_fft[:, 1:]
        pred = spectral[:, :-1]  # (B, T-1, D_spec, K, 2)
        target = target_fft[:, 1:]  # (B, T-1, D_spec, K, 2)

        # Compute weighted MSE
        # Weight by sqrt(k+1) to counteract pink noise
        k_max = self.config.k_max
        assert k_max is not None
        k_indices = torch.arange(k_max, device=x.device, dtype=torch.float32)
        weights = torch.sqrt(k_indices + 1)  # (K,)
        weights = weights / weights.sum()  # Normalize

        # Expand weights for broadcasting: (1, 1, 1, K, 1)
        weights = weights.view(1, 1, 1, -1, 1)

        # Weighted MSE
        diff_squared = (pred - target) ** 2  # (B, T-1, D_spec, K, 2)
        weighted_diff = diff_squared * weights
        loss = weighted_diff.mean()

        return loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
