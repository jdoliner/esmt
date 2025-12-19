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
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from config import SATConfig, SITConfig


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

    # Apply ReLU to (magnitude - bias)
    new_mag = F.relu(mag - bias)  # (...)

    # Instead of computing phase and using cos/sin (unstable for small mag),
    # we directly scale the original complex number: z * (new_mag / mag)
    # This avoids the phase computation entirely.
    # Add eps to avoid division by zero
    scale = new_mag / (mag + 1e-8)

    # Scale both real and imaginary parts
    return z * scale.unsqueeze(-1)


def mod_softplus(z: torch.Tensor, bias: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Soft version of ModReLU that doesn't have the "dead neuron" problem.

    Uses softplus instead of ReLU on the magnitude, ensuring gradients always flow.
    The bias acts as a "soft threshold" rather than hard cutoff.

    ModSoftplus(z) = softplus(|z| - b) * (z / |z|)

    For large |z|, this approximates ModReLU.
    For small |z|, gradients still flow (no dead neurons).

    Args:
        z: Complex tensor (..., 2)
        bias: Real tensor, soft threshold bias
        beta: Softplus beta parameter (higher = sharper transition)

    Returns:
        Complex tensor (..., 2)
    """
    mag = complex_abs(z)  # (...)

    # Apply softplus to (magnitude - bias)
    # softplus(x) = (1/beta) * log(1 + exp(beta * x))
    new_mag = F.softplus(mag - bias, beta=beta)  # (...)

    # Scale original complex number by ratio
    scale = new_mag / (mag + 1e-8)

    return z * scale.unsqueeze(-1)


def mod_elu(z: torch.Tensor, bias: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    ELU-based complex activation that never completely kills gradients.

    For |z| > bias: output magnitude = |z| - bias (like ModReLU)
    For |z| < bias: output magnitude = alpha * (exp(|z| - bias) - 1) (small negative -> small positive)

    This ensures gradients always flow, preventing dead neurons.

    Args:
        z: Complex tensor (..., 2)
        bias: Real tensor, threshold bias
        alpha: ELU alpha parameter

    Returns:
        Complex tensor (..., 2)
    """
    mag = complex_abs(z)  # (...)

    # Apply ELU to (magnitude - bias), then shift to ensure positive output
    # ELU: x if x > 0, else alpha * (exp(x) - 1)
    diff = mag - bias
    new_mag = torch.where(
        diff > 0,
        diff,
        alpha * (torch.exp(diff.clamp(max=10)) - 1) + alpha,  # +alpha to keep positive
    )
    new_mag = new_mag.clamp(min=0)  # Ensure non-negative

    # Scale original complex number
    scale = new_mag / (mag + 1e-8)

    return z * scale.unsqueeze(-1)


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
# Initialization Utilities for BF16 Complex Numbers
# ==============================================================================


def bf16_complex_unitary_init(weight: torch.Tensor) -> None:
    """
    Initialize complex weight tensor (*, 2) with unitary/isometric matrix.

    Uses QR decomposition to create norm-preserving weight matrices.
    This prevents gradient vanishing/exploding in deep networks.

    Based on complex_unitary_init from complex_layers.py but adapted for
    bfloat16 pair representation.

    Args:
        weight: Tensor of shape (out_features, in_features, 2)
    """
    out_features, in_features = weight.shape[0], weight.shape[1]
    max_dim = max(out_features, in_features)

    with torch.no_grad():
        # Generate random complex Gaussian matrix in float32
        real_part = torch.randn(max_dim, max_dim, dtype=torch.float32)
        imag_part = torch.randn(max_dim, max_dim, dtype=torch.float32)
        random_matrix = torch.complex(real_part, imag_part)

        # QR decomposition gives orthonormal columns
        Q, R = torch.linalg.qr(random_matrix)

        # Fix phase ambiguity
        diag_R = torch.diag(R)
        phase_correction = diag_R / diag_R.abs().clamp(min=1e-10)
        Q = Q * phase_correction.unsqueeze(0)

        # Extract the portion we need
        Q_slice = Q[:out_features, :in_features]

        # Convert to bfloat16 pairs
        weight[..., 0].copy_(Q_slice.real.to(torch.bfloat16))
        weight[..., 1].copy_(Q_slice.imag.to(torch.bfloat16))


def bf16_complex_small_init(weight: torch.Tensor, scale: float = 1e-5) -> None:
    """
    Initialize complex weight tensor with small magnitude values.

    Used for residual exit layers (output projections) to ensure the block
    starts as approximately identity: y = x + epsilon

    Based on complex_small_init from complex_layers.py.

    Args:
        weight: Tensor of shape (*, 2)
        scale: Standard deviation for real and imaginary parts
    """
    with torch.no_grad():
        real_part = torch.randn(weight.shape[:-1], dtype=torch.float32) * scale
        imag_part = torch.randn(weight.shape[:-1], dtype=torch.float32) * scale
        weight[..., 0].copy_(real_part.to(torch.bfloat16))
        weight[..., 1].copy_(imag_part.to(torch.bfloat16))


# ==============================================================================
# Complex Linear Operations
# ==============================================================================


class ComplexLinearBF16(nn.Module):
    """
    Linear layer for complex-valued tensors using bfloat16 pairs.

    Computes: y = Wx + b where W, x, b are all complex (as bfloat16 pairs).

    The weight is stored as (out_features, in_features, 2) and bias as (out_features, 2).

    Initialization (following best practices from complex_layers.py):
        - Default: Unitary/isometric initialization (preserves norms)
        - residual_exit=True: Near-zero initialization for output projections
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        residual_exit: bool = False,
        residual_exit_scale: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual_exit = residual_exit
        self.residual_exit_scale = residual_exit_scale

        # Weight: (out_features, in_features, 2) for complex
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 2, dtype=torch.bfloat16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 2, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.residual_exit:
            # Near-zero initialization for residual exit layers
            # Ensures block starts as approximately identity: y = x + epsilon
            bf16_complex_small_init(self.weight, scale=self.residual_exit_scale)
        else:
            # Unitary/isometric initialization
            # Preserves norms and prevents gradient vanishing/exploding
            bf16_complex_unitary_init(self.weight)

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

    Normalization options:
    - "ortho": Standard 1/sqrt(N) normalization (default)
    - "position": Per-position normalization by 1/sqrt(t+1) to keep magnitudes stable
    """

    def __init__(self, seq_len: int, k_max: int, normalization: str = "position"):
        """
        Args:
            seq_len: Maximum sequence length (N for FFT)
            k_max: Number of frequency modes to keep
            normalization: "ortho" for standard 1/sqrt(N), "position" for per-position 1/sqrt(t+1)
        """
        super().__init__()
        self.seq_len = seq_len
        self.k_max = k_max
        self.normalization = normalization

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

        # Per-position normalization factors: 1/sqrt(t+1) for t=0,1,...,seq_len-1
        # This keeps the magnitude roughly constant regardless of position
        pos_norm = 1.0 / torch.sqrt(torch.arange(1, seq_len + 1, dtype=torch.float32))
        self.register_buffer("pos_norm", pos_norm.to(torch.bfloat16))
        self.pos_norm: torch.Tensor

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

        # Apply normalization
        if self.normalization == "position":
            # Per-position normalization: 1/sqrt(t+1)
            # This keeps magnitude roughly constant regardless of position
            # Shape: (1, seq_len, 1, 1, 1) for broadcasting
            pos_norm = self.pos_norm[:seq_len].view(1, seq_len, 1, 1, 1)
            cumulative_fft = cumulative_fft * pos_norm
        else:
            # Standard ortho normalization: 1/sqrt(N)
            cumulative_fft = cumulative_fft * self.norm_factor

        return cumulative_fft


class CumulativeIFFT(nn.Module):
    """
    Compute inverse FFT for each position's cumulative spectral representation.

    For each position t, we have a spectral representation (from cumulative FFT + FNO).
    This module applies iFFT to convert back to time domain, producing a length-N
    signal for each position t.

    The output at [t, t'] represents "position t's belief about position t'",
    derived only from tokens [0:t+1].

    This is the inverse of CumulativeFFT, completing the spectral round-trip:
        time -> cumFFT -> FNO -> cumIFFT -> time
    """

    def __init__(self, seq_len: int, k_max: int, normalization: str = "position"):
        """
        Args:
            seq_len: Maximum sequence length (N for FFT)
            k_max: Number of frequency modes kept in spectral representation
            normalization: Must match CumulativeFFT normalization for proper inversion
        """
        super().__init__()
        self.seq_len = seq_len
        self.k_max = k_max
        self.normalization = normalization

        # Precompute inverse twiddle factors: e^{+2*pi*i*k*n/N}
        # Note: positive sign for inverse FFT
        # Shape: (seq_len, k_max, 2) for bfloat16 complex pairs
        n = torch.arange(seq_len, dtype=torch.float32)
        k = torch.arange(k_max, dtype=torch.float32)

        # Compute angles: +2*pi*k*n/N (positive for inverse)
        angles = 2 * math.pi * torch.outer(n, k) / seq_len

        # Convert to complex pairs
        inv_twiddles_real = torch.cos(angles).to(torch.bfloat16)
        inv_twiddles_imag = torch.sin(angles).to(torch.bfloat16)
        inv_twiddles = torch.stack([inv_twiddles_real, inv_twiddles_imag], dim=-1)

        self.register_buffer("inv_twiddles", inv_twiddles)
        self.inv_twiddles: torch.Tensor

        # Normalization factors (inverse of forward FFT normalization)
        # For ortho: multiply by sqrt(N) to undo 1/sqrt(N)
        self.register_buffer(
            "inv_norm_factor", torch.tensor(math.sqrt(seq_len), dtype=torch.bfloat16)
        )
        self.inv_norm_factor: torch.Tensor

        # For position normalization: multiply by sqrt(t+1) to undo 1/sqrt(t+1)
        pos_inv_norm = torch.sqrt(torch.arange(1, seq_len + 1, dtype=torch.float32))
        self.register_buffer("pos_inv_norm", pos_inv_norm.to(torch.bfloat16))
        self.pos_inv_norm: torch.Tensor

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse FFT for each position's spectral representation.

        Args:
            spectral: Complex tensor (batch, T, d_spectral, k_max, 2)
                     T is the "source" position (which cumulative FFT this came from)

        Returns:
            Real tensor (batch, T, seq_len, d_spectral)
            - Dimension 1 (T): source position (the cumulative FFT position)
            - Dimension 2 (seq_len): target position (iFFT output position)
            - For position t, output[t, t'] is the prediction for position t'
              based on tokens [0:t+1]
        """
        batch, T, d_spectral, k_max, _ = spectral.shape

        # Undo the forward normalization first
        if self.normalization == "position":
            # Undo per-position normalization: multiply by sqrt(t+1)
            pos_inv_norm = self.pos_inv_norm[:T].view(1, T, 1, 1, 1)
            spectral = spectral * pos_inv_norm
        else:
            # Undo ortho normalization: multiply by sqrt(N)
            spectral = spectral * self.inv_norm_factor

        # Get inverse twiddles: (seq_len, k_max, 2)
        inv_twiddles = self.inv_twiddles  # (seq_len, k_max, 2)

        # Compute iFFT via sum over frequency modes:
        # x[n] = (1/N) * sum_k X[k] * e^{+2*pi*i*k*n/N}
        #
        # spectral: (batch, T, d_spectral, k_max, 2)
        # inv_twiddles: (seq_len, k_max, 2)
        #
        # For each output position n, we sum over k:
        #   output[b, t, n, d] = sum_k spectral[b, t, d, k] * inv_twiddles[n, k]
        #
        # This is a complex multiplication followed by sum over k

        # Expand for broadcasting
        # spectral: (batch, T, d_spectral, k_max, 2) -> (batch, T, 1, d_spectral, k_max, 2)
        spectral_expanded = spectral.unsqueeze(2)

        # inv_twiddles: (seq_len, k_max, 2) -> (1, 1, seq_len, 1, k_max, 2)
        inv_twiddles_expanded = inv_twiddles.unsqueeze(0).unsqueeze(0).unsqueeze(3)

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        s_real = spectral_expanded[..., 0]  # (batch, T, 1, d_spectral, k_max)
        s_imag = spectral_expanded[..., 1]
        t_real = inv_twiddles_expanded[..., 0]  # (1, 1, seq_len, 1, k_max)
        t_imag = inv_twiddles_expanded[..., 1]

        # Product real and imag parts
        prod_real = s_real * t_real - s_imag * t_imag  # (batch, T, seq_len, d_spectral, k_max)
        prod_imag = s_real * t_imag + s_imag * t_real

        # Sum over frequency modes (k dimension)
        # The result should be approximately real for real input signals
        output_real = prod_real.sum(dim=-1)  # (batch, T, seq_len, d_spectral)
        output_imag = prod_imag.sum(dim=-1)

        # Apply 1/N normalization for proper inverse
        output_real = output_real / self.seq_len
        output_imag = output_imag / self.seq_len

        # For real-valued original signals, imaginary part should be ~0
        # We return only the real part
        # (In practice, imaginary part won't be exactly 0 due to k_max truncation)
        return output_real


# ==============================================================================
# FNO Layers
# ==============================================================================


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution layer for FNO.

    Performs element-wise multiplication with learnable complex weights in the
    frequency domain. This corresponds to convolution in the spatial domain.

    Weight shape: (d_out, d_in, k_max, 2) for complex weights per frequency mode.

    Initialization: Each frequency mode gets a separate unitary/isometric matrix
    to preserve norms and prevent gradient issues.
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
        # Initialize each frequency mode with a unitary/isometric matrix
        # This preserves norms and prevents gradient vanishing/exploding
        with torch.no_grad():
            for k in range(self.k_max):
                # Get a (d_out, d_in) slice for this frequency
                weight_k = self.weight[:, :, k, :]  # (d_out, d_in, 2)
                bf16_complex_unitary_init(weight_k)

            # Scale down slightly to prevent output explosion when summing across frequencies
            # The unitary init preserves norm, but we're summing k_max contributions
            scale = 1.0 / math.sqrt(self.k_max)
            self.weight.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor (batch, seq_len, d_in, k_max, 2)

        Returns:
            Complex tensor (batch, seq_len, d_out, k_max, 2)
        """
        # x: (batch, seq_len, d_in, k_max, 2)
        # weight: (d_out, d_in, k_max, 2)

        B, T, d_in, K, _ = x.shape

        # For each frequency mode k, we do a complex matrix-vector product:
        # out[b, t, d_out, k] = sum_d_in weight[d_out, d_in, k] * x[b, t, d_in, k]
        #
        # This is equivalent to K independent matrix multiplications.
        # We use bmm by treating K as the batch dimension.

        x_real, x_imag = x[..., 0], x[..., 1]  # (B, T, d_in, K)
        w_real = self.weight[..., 0]  # (d_out, d_in, K)
        w_imag = self.weight[..., 1]

        # Reshape: x -> (K, B*T, d_in), w -> (K, d_out, d_in)
        x_real = x_real.permute(3, 0, 1, 2).reshape(K, B * T, d_in)  # (K, B*T, d_in)
        x_imag = x_imag.permute(3, 0, 1, 2).reshape(K, B * T, d_in)
        w_real = w_real.permute(2, 0, 1)  # (K, d_out, d_in)
        w_imag = w_imag.permute(2, 0, 1)

        # Batched matmul: (K, B*T, d_in) @ (K, d_in, d_out) -> (K, B*T, d_out)
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = torch.bmm(x_real, w_real.transpose(1, 2)) - torch.bmm(
            x_imag, w_imag.transpose(1, 2)
        )
        out_imag = torch.bmm(x_real, w_imag.transpose(1, 2)) + torch.bmm(
            x_imag, w_real.transpose(1, 2)
        )

        # Reshape back: (K, B*T, d_out) -> (B, T, d_out, K)
        out_real = out_real.reshape(K, B, T, self.d_out).permute(1, 2, 3, 0)
        out_imag = out_imag.reshape(K, B, T, self.d_out).permute(1, 2, 3, 0)

        return torch.stack([out_real, out_imag], dim=-1)


class FNOBlock(nn.Module):
    """
    Fourier Neural Operator block.

    Architecture:
        x -> SpectralConv -> + -> Activation -> (optional gate) -> out
             |               ^
             +-- residual ---+

    The residual is in the frequency domain (simple addition).

    Activation options:
    - "modrelu": Hard threshold (can cause dead neurons)
    - "modsoftplus": Soft threshold (gradients always flow)
    - "modelu": ELU-based (gradients always flow)

    Output gating (optional):
    - Learnable scalar gate initialized to small value
    - Prevents early explosion by keeping outputs small initially
    """

    def __init__(
        self,
        d_model: int,
        k_max: int,
        activation: str = "modsoftplus",
        use_output_gate: bool = True,
        gate_init: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_max = k_max
        self.activation = activation
        self.use_output_gate = use_output_gate

        # Spectral convolution (complex)
        self.spectral_conv = SpectralConv1d(d_model, d_model, k_max)

        # Activation bias (one per feature dimension, shared across frequencies)
        # Initialize to 0 so all signals pass through initially.
        # The model can learn to increase this to filter noise.
        self.modrelu_bias = nn.Parameter(torch.zeros(d_model, dtype=torch.bfloat16))

        # Output gate: learnable scalar that starts small
        # This prevents the FNO from contributing too much early in training
        if use_output_gate:
            self.output_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.bfloat16))
        else:
            self.output_gate = None

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

        # Apply activation
        # bias shape for broadcasting: (1, 1, d_model, 1)
        bias = self.modrelu_bias.view(1, 1, -1, 1)

        if self.activation == "modrelu":
            out = mod_relu(out, bias)
        elif self.activation == "modsoftplus":
            out = mod_softplus(out, bias, beta=2.0)
        elif self.activation == "modelu":
            out = mod_elu(out, bias)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Apply output gate if enabled
        if self.output_gate is not None:
            # Sigmoid to keep gate in (0, 1)
            gate = torch.sigmoid(self.output_gate)
            out = out * gate

        return out


# ==============================================================================
# Spectral Normalization
# ==============================================================================


def spectral_clip(z: torch.Tensor, max_magnitude: float = 10.0) -> torch.Tensor:
    """
    Clip complex magnitudes while preserving phase.

    This prevents extreme outliers from destabilizing training.

    Args:
        z: Complex tensor (..., 2) in bfloat16 pair format
        max_magnitude: Maximum allowed magnitude

    Returns:
        Clipped complex tensor (..., 2)
    """
    mag = complex_abs(z)  # (...)

    # Compute clipping scale: min(1, max_magnitude / mag)
    # This clips magnitudes > max_magnitude while leaving smaller ones unchanged
    scale = torch.clamp(max_magnitude / (mag + 1e-8), max=1.0)

    return z * scale.unsqueeze(-1)


def normalize_complex(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize complex tensor to unit RMS magnitude while preserving phase.

    This scales the tensor so that the root-mean-square of magnitudes is 1.
    Useful for normalizing delta predictions to a consistent scale.

    Args:
        z: Complex tensor (..., 2) in bfloat16 pair format
        eps: Small constant for numerical stability

    Returns:
        Normalized complex tensor (..., 2) with RMS magnitude ≈ 1
    """
    mag = complex_abs(z)  # (...)
    rms = torch.sqrt((mag**2).mean() + eps)
    return z / rms.unsqueeze(-1) if rms.ndim == 0 else z / rms


class SpectralLayerNorm(nn.Module):
    """
    Layer normalization for complex spectral tensors.

    Normalizes magnitude while preserving phase. This prevents magnitude explosion
    while maintaining the spectral structure.

    Two modes:
    - "magnitude": Normalize magnitude statistics, preserve phase
    - "rms": RMS normalization of magnitude (simpler, no mean centering)
    """

    def __init__(
        self,
        d_model: int,
        k_max: int,
        eps: float = 1e-6,
        mode: str = "rms",
        learnable: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_max = k_max
        self.eps = eps
        self.mode = mode
        self.learnable = learnable

        if learnable:
            # Learnable scale per (feature, frequency) - applied to magnitude
            self.scale = nn.Parameter(torch.ones(d_model, k_max, dtype=torch.bfloat16))
        else:
            self.register_parameter("scale", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor (batch, seq_len, d_model, k_max, 2)

        Returns:
            Normalized complex tensor (batch, seq_len, d_model, k_max, 2)
        """
        mag = complex_abs(z)  # (batch, seq_len, d_model, k_max)

        if self.mode == "rms":
            # RMS normalization: divide by RMS of magnitude
            # Compute RMS over feature and frequency dimensions
            rms = torch.sqrt((mag**2).mean(dim=(-2, -1), keepdim=True) + self.eps)
            mag_normalized = mag / rms
        else:  # magnitude
            # Standard LayerNorm on magnitude
            mean = mag.mean(dim=(-2, -1), keepdim=True)
            var = ((mag - mean) ** 2).mean(dim=(-2, -1), keepdim=True)
            mag_normalized = (mag - mean) / torch.sqrt(var + self.eps)
            # Shift to positive (since magnitude should be positive)
            mag_normalized = mag_normalized + 1.0  # Center around 1

        # Scale normalized magnitude back to original complex
        # z_new = z * (mag_normalized / mag)
        scale_factor = mag_normalized / (mag + self.eps)
        z_normalized = z * scale_factor.unsqueeze(-1)

        # Apply learnable scale
        if self.scale is not None:
            z_normalized = z_normalized * self.scale.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        return z_normalized


# ==============================================================================
# Bridge: Spectral to Transformer Conditioning
# ==============================================================================


class AdaLNBridge(nn.Module):
    """
    Bridge that converts spectral state to AdaLN parameters (gamma, beta).

    Architecture:
        spectral (B, T, D_spec, K, 2)
        -> flatten to (B, T, D_spec * K * 2) preserving both magnitude and phase
        -> two-layer MLP -> (B, T, D)
        -> per-layer proj -> (gamma_i, beta_i) for each layer

    The final projection is zero-initialized so that at init:
        gamma = 1, beta = 0 (standard LayerNorm behavior)
    """

    def __init__(self, d_spectral: int, d_model: int, n_layers: int, k_max: int):
        super().__init__()
        self.d_spectral = d_spectral
        self.d_model = d_model
        self.n_layers = n_layers
        self.k_max = k_max

        # Input dimension: flatten full complex spectral tensor
        # This preserves both magnitude and phase information
        input_dim = d_spectral * k_max * 2

        # Two-layer MLP to project from flattened spectral to model dimension
        # Use an intermediate dimension to avoid too aggressive compression
        hidden_dim = max(d_model * 4, input_dim // 4)

        # LayerNorm on input to handle varying spectral magnitudes
        self.ln_in = nn.LayerNorm(input_dim)

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.SiLU()
        self.proj2 = nn.Linear(hidden_dim, d_model)
        self.act2 = nn.SiLU()

        # LayerNorm to stabilize the hidden representation before gamma/beta projection
        # This prevents extreme values from compounding through transformer layers
        self.ln = nn.LayerNorm(d_model)

        # Per-layer projections to (gamma, beta)
        self.layer_projs = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])

        # Scale factor for beta to prevent large shifts
        # Beta is scaled by tanh to keep it bounded
        self.beta_scale = 1.0

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard init for shared projections
        nn.init.normal_(self.proj1.weight, std=0.02)
        nn.init.zeros_(self.proj1.bias)
        nn.init.normal_(self.proj2.weight, std=0.02)
        nn.init.zeros_(self.proj2.bias)

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
        batch, seq_len, d_spectral, k_max, _ = spectral.shape

        # Flatten the full complex tensor to preserve magnitude and phase
        # spectral: (B, T, D_spec, K, 2) -> (B, T, D_spec * K * 2)
        flat = spectral.view(batch, seq_len, -1)

        # Convert to float32 for projection
        flat = flat.float()

        # Normalize input to handle varying spectral magnitudes
        flat = self.ln_in(flat)

        # Two-layer MLP
        hidden = self.proj1(flat)  # (B, T, hidden_dim)
        hidden = self.act1(hidden)
        hidden = self.proj2(hidden)  # (B, T, D)
        hidden = self.act2(hidden)

        # Normalize to prevent extreme gamma/beta values
        hidden = self.ln(hidden)

        # Per-layer projections
        conditioning = []
        for proj in self.layer_projs:
            gamma_beta = proj(hidden)  # (B, T, 2*D)
            gamma = gamma_beta[..., : self.d_model]  # (B, T, D)
            beta = gamma_beta[..., self.d_model :]  # (B, T, D)
            # Bound beta with tanh to prevent large shifts
            beta = self.beta_scale * torch.tanh(beta / self.beta_scale)
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

    Instead of learned affine parameters, AdaLN uses input-dependent
    gamma and beta provided by an external conditioning signal.

    AdaLN(x, γ, β) = γ * LayerNorm(x) + β

    With gamma_scale option:
    - Constrains gamma to be 1 + scale * tanh(gamma_input)
    - This keeps gamma in range [1-scale, 1+scale], preventing drift
    """

    def __init__(self, d_model: int, eps: float = 1e-6, gamma_scale: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma_scale = gamma_scale

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            gamma: Scale tensor (batch, seq_len, d_model) or (batch, 1, d_model)
                   If gamma_scale > 0, this is treated as pre-activation and
                   transformed to 1 + gamma_scale * tanh(gamma)
            beta: Shift tensor (batch, seq_len, d_model) or (batch, 1, d_model)

        Returns:
            Normalized and modulated tensor (batch, seq_len, d_model)
        """
        # Standard layer norm (without affine)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Constrain gamma to stay close to 1
        # gamma comes in centered around 1 from bridge, so subtract 1 before tanh
        if self.gamma_scale > 0:
            gamma_centered = gamma - 1.0  # Now centered around 0
            gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_centered / self.gamma_scale)

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
        self.adaln = AdaptiveLayerNorm(
            config.d_model, eps=config.eps, gamma_scale=config.adaln_gamma_scale
        )

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
        self.adaln = AdaptiveLayerNorm(
            config.d_model, eps=config.eps, gamma_scale=config.adaln_gamma_scale
        )

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
        self.cumulative_fft = CumulativeFFT(
            config.seq_len, config.k_max, normalization=config.fft_normalization
        )

        # FNO blocks
        self.fno_blocks = nn.ModuleList(
            [
                FNOBlock(
                    config.d_spectral,
                    config.k_max,
                    activation=config.fno_activation,
                    use_output_gate=config.fno_output_gate,
                    gate_init=config.fno_gate_init,
                )
                for _ in range(config.n_fno_layers)
            ]
        )

        # Spectral layer normalization (optional, applied after FNO blocks)
        if config.spectral_layernorm is not None:
            self.spectral_ln = SpectralLayerNorm(
                config.d_spectral,
                config.k_max,
                mode=config.spectral_layernorm,
                learnable=True,
            )
        else:
            self.spectral_ln = None

        # Store clipping threshold
        self.spectral_clip_magnitude = config.spectral_clip_magnitude

        # =====================================================================
        # Bridge
        # =====================================================================

        # AdaLN bridge (always created, provides default gamma=1, beta=0 if not used)
        self.adaln_bridge = AdaLNBridge(
            config.d_spectral, config.d_model, config.n_layers, config.k_max
        )

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

        # Re-apply custom initialization for components that need special init
        # (the apply() above resets all Linear biases to 0, but bridge needs gamma=1)
        self.adaln_bridge._init_weights()

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
        # Note: We use tok_emb without pos_emb because the cumulative FFT
        # naturally encodes position through phase relationships
        h_spectral = self.spectral_proj_in(h)  # (B, T, D_spec)

        # Ensure bfloat16 for FFT
        h_spectral = h_spectral.to(torch.bfloat16)

        # Cumulative FFT
        spectral = self.cumulative_fft(h_spectral)  # (B, T, D_spec, K, 2)

        # Optional: clip spectral magnitudes to prevent outliers
        if self.spectral_clip_magnitude is not None:
            spectral = spectral_clip(spectral, self.spectral_clip_magnitude)

        # Optional: spectral layer normalization after FNO blocks
        if self.spectral_ln is not None:
            spectral = self.spectral_ln(spectral)

        # FNO blocks - output is interpreted as predicted delta (FFT[t+1] - FFT[t])
        for fno_block in self.fno_blocks:
            spectral = fno_block(spectral)

            # Optional: clip after each FNO block to prevent accumulation
            if self.spectral_clip_magnitude is not None:
                spectral = spectral_clip(spectral, self.spectral_clip_magnitude)

        # =====================================================================
        # Bridge
        # =====================================================================

        # Normalize the FNO output (predicted delta) before passing to AdaLN
        # This ensures consistent scale for the bridge network regardless of delta magnitude
        spectral_normalized = normalize_complex(spectral)

        # Get AdaLN conditioning from normalized predicted delta
        # Each element is (gamma, beta) tuple for one layer
        adaln_conditioning = self.adaln_bridge(spectral_normalized)

        # Ablation: shuffle conditioning across batch to break input-output correlation
        # This tests whether the FNO signal is actually useful or just learned noise
        if self.config.ablate_adaln_shuffle and self.training:
            # Generate a random permutation of batch indices
            perm = torch.randperm(batch, device=device)
            # Shuffle gamma/beta for each layer
            adaln_conditioning = [(gamma[perm], beta[perm]) for gamma, beta in adaln_conditioning]

        # Get cross-attention KV (if enabled) - also use normalized spectral
        if self.cross_attn_bridge is not None:
            spectral_k, spectral_v = self.cross_attn_bridge(spectral_normalized)
            # Also shuffle cross-attention KV if ablating
            if self.config.ablate_adaln_shuffle and self.training:
                # perm was defined above when ablate_adaln_shuffle is True
                shuffle_perm = torch.randperm(batch, device=device)
                spectral_k = spectral_k[shuffle_perm]
                spectral_v = spectral_v[shuffle_perm]
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

        The FNO should predict the normalized delta between consecutive FFT positions:
            L_aux = MSE(FNO_output[t], normalize(FFT[t+1] - FFT[t]))

        By predicting the delta instead of the absolute FFT, the FNO is forced to
        learn the actual dynamics rather than just approximating an identity function.
        The delta is normalized to ensure consistent gradient scale.

        Args:
            spectral: FNO output (batch, seq_len, d_spectral, k_max, 2)
            x: Original input tokens (batch, seq_len)

        Returns:
            Auxiliary loss scalar
        """
        batch, seq_len = x.shape
        device = x.device

        # Re-embed and project
        tok_emb = self.token_emb(x)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (T, D)

        h = tok_emb + pos_emb  # (B, T, D)
        h_spectral = self.spectral_proj_in(tok_emb + pos_emb).to(torch.bfloat16)

        # Get cumulative FFT
        with torch.no_grad():
            target_fft = self.cumulative_fft(h_spectral)  # (B, T, D_spec, K, 2)

            # Apply same clipping as forward pass
            if self.spectral_clip_magnitude is not None:
                target_fft = spectral_clip(target_fft, self.spectral_clip_magnitude)

            # Apply same layer norm as forward pass
            if self.spectral_ln is not None:
                target_fft = self.spectral_ln(target_fft)

            # Compute delta: FFT[t+1] - FFT[t]
            # delta[t] = target_fft[t+1] - target_fft[t]
            delta = target_fft[:, 1:] - target_fft[:, :-1]  # (B, T-1, D_spec, K, 2)

            # Normalize the delta to unit RMS magnitude
            target_delta_normalized = normalize_complex(delta)

        # FNO output at position t should predict normalized delta
        # pred[t] should match normalize(FFT[t+1] - FFT[t])
        pred = spectral[:, :-1]  # (B, T-1, D_spec, K, 2)

        # Normalize prediction as well for consistent comparison
        pred_normalized = normalize_complex(pred)

        # MSE loss between normalized predictions and normalized targets
        pred_f = pred_normalized.float()
        target_f = target_delta_normalized.float()
        diff_squared = (pred_f - target_f) ** 2  # (B, T-1, D_spec, K, 2)

        loss = diff_squared.mean()

        return loss

    def compute_auxiliary_loss_with_diagnostics(
        self,
        spectral: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute auxiliary FNO prediction loss with detailed diagnostics.

        Same as compute_auxiliary_loss but returns diagnostic statistics
        to help debug the FNO training dynamics.

        Returns:
            Tuple of (loss, diagnostics_dict)
        """
        batch, seq_len = x.shape
        device = x.device

        # Re-embed and project
        tok_emb = self.token_emb(x)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (T, D)
        h_spectral = self.spectral_proj_in(tok_emb + pos_emb).to(torch.bfloat16)

        # Get cumulative FFT
        with torch.no_grad():
            target_fft = self.cumulative_fft(h_spectral)  # (B, T, D_spec, K, 2)

            # Apply same clipping as forward pass
            if self.spectral_clip_magnitude is not None:
                target_fft = spectral_clip(target_fft, self.spectral_clip_magnitude)

            # Apply same layer norm as forward pass
            if self.spectral_ln is not None:
                target_fft = self.spectral_ln(target_fft)

            # Compute delta: FFT[t+1] - FFT[t]
            delta = target_fft[:, 1:] - target_fft[:, :-1]  # (B, T-1, D_spec, K, 2)

            # Normalize the delta
            target_delta_normalized = normalize_complex(delta)

        # FNO output at position t should predict normalized delta
        pred = spectral[:, :-1]  # (B, T-1, D_spec, K, 2)
        pred_normalized = normalize_complex(pred)

        # Convert to float for diagnostics
        pred_f = pred_normalized.float()
        target_f = target_delta_normalized.float()

        # Also get unnormalized versions for magnitude diagnostics
        pred_unnorm_f = pred.float()
        delta_unnorm_f = delta.float()

        # Compute magnitudes (on unnormalized for interpretability)
        pred_mag = torch.sqrt(pred_unnorm_f[..., 0] ** 2 + pred_unnorm_f[..., 1] ** 2)
        delta_mag = torch.sqrt(delta_unnorm_f[..., 0] ** 2 + delta_unnorm_f[..., 1] ** 2)

        # Step 1: Scale comparison (unnormalized pred vs unnormalized delta)
        diagnostics = {
            "pred_mag_mean": pred_mag.mean().item(),
            "pred_mag_std": pred_mag.std().item(),
            "pred_mag_max": pred_mag.max().item(),
            "target_delta_mag_mean": delta_mag.mean().item(),
            "target_delta_mag_std": delta_mag.std().item(),
            "target_delta_mag_max": delta_mag.max().item(),
            "scale_ratio": pred_mag.mean().item() / (delta_mag.mean().item() + 1e-8),
        }

        # Step 2: Check if FNO is outputting near-constant values
        pred_variance_across_samples = pred_mag.var(dim=(0, 1)).mean().item()
        pred_variance_across_features = pred_mag.var(dim=(2, 3)).mean().item()
        diagnostics["pred_var_across_samples"] = pred_variance_across_samples
        diagnostics["pred_var_across_features"] = pred_variance_across_features

        # Step 3: Baseline comparison
        # The "zero baseline" for delta prediction: predict delta = 0
        # MSE of predicting zero = mean(delta^2) = variance of delta
        zero_baseline_mse = (delta_unnorm_f**2).mean().item()
        diagnostics["zero_baseline_mse"] = zero_baseline_mse

        # Normalized prediction error
        pred_error = ((pred_f - target_f) ** 2).mean().item()
        diagnostics["pred_mse"] = pred_error

        # For normalized comparison, zero baseline would be MSE(0, normalized_delta)
        # Since normalized_delta has RMS=1, this is approximately 1.0
        zero_baseline_normalized = (target_f**2).mean().item()
        diagnostics["zero_baseline_normalized_mse"] = zero_baseline_normalized

        # How much better is our prediction than predicting zero?
        # < 1.0 means FNO is learning something useful
        diagnostics["pred_vs_zero_ratio"] = pred_error / (zero_baseline_normalized + 1e-8)

        # Cosine similarity between prediction and target (direction accuracy)
        # Flatten for cosine computation
        pred_flat = pred_f.reshape(-1)
        target_flat = target_f.reshape(-1)
        cosine_sim = (pred_flat * target_flat).sum() / (
            pred_flat.norm() * target_flat.norm() + 1e-8
        )
        diagnostics["cosine_similarity"] = cosine_sim.item()

        # Compute loss (same as main function)
        diff_squared = (pred_f - target_f) ** 2
        loss = diff_squared.mean()

        diagnostics["uniform_loss"] = loss.item()

        return loss, diagnostics


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Spectral Injection Transformer (SIT) - Sparse Cutoff Version
# ==============================================================================


class CausalSelfAttention(nn.Module):
    """
    Standard causal self-attention (same as NanoGPT).

    Used in SIT with sparse cutoffs, which uses a standard causal transformer
    augmented with spectral predictions at fixed cutoff points.
    """

    def __init__(self, config: SITConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Use scaled_dot_product_attention with causal mask (Flash Attention)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.proj(out)

        return out


class CausalTransformerBlock(nn.Module):
    """
    Standard causal transformer block.

    Uses pre-norm architecture with standard LayerNorm.
    """

    def __init__(self, config: SITConfig):
        super().__init__()
        self.config = config

        # Pre-norm
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.eps)

        # Attention
        self.attn = CausalSelfAttention(config)

        # MLP
        hidden_dim = config.d_model * config.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class SpectralInjectionTransformer(nn.Module):
    """
    Spectral Injection Transformer (SIT) with Sparse Cutoffs.

    This model injects spectral predictions at a small number of cutoff points,
    rather than at every position. This provides long-range spectral signal
    while maintaining O(B * T * D) memory - same as a standard transformer.

    Architecture:
    1. Compute token + positional embeddings as usual
    2. At each cutoff point c, compute: FFT[0:c] -> FNO -> iFFT -> predictions
    3. For positions >= c, add the spectral predictions to their embeddings
    4. Each position sees only the most recent cutoff's spectral signal
    5. Standard causal transformer processes the augmented embeddings

    Example with num_cutoffs=3, seq_len=512 (cutoffs at [128, 256, 384]):
    - Position 130 sees: tok_emb[130] + pos_emb[130] + spectral_from_FFT[0:128]
    - Position 260 sees: tok_emb[260] + pos_emb[260] + spectral_from_FFT[0:256]
    - Position 100 sees: tok_emb[100] + pos_emb[100] (no spectral, before first cutoff)

    Key insight: The FNO captures long-range dependencies much better than running
    it for every position. Running it 3 times is almost as informative since the
    inputs to the FNO are similar for nearby positions anyway.
    """

    def __init__(self, config: SITConfig):
        super().__init__()
        self.config = config

        # Ensure config values are set
        assert config.d_spectral is not None
        assert config.n_fno_layers is not None
        assert config.k_max is not None

        # Compute cutoff positions
        self.cutoff_positions = config.get_cutoff_positions()

        # =====================================================================
        # Embeddings
        # =====================================================================
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)

        # =====================================================================
        # Spectral Stream (shared across all cutoffs)
        # =====================================================================

        # Project to spectral dimension
        self.spectral_proj_in = nn.Linear(config.d_model, config.d_spectral)

        # FNO blocks (process spectral representations)
        self.fno_blocks = nn.ModuleList(
            [
                FNOBlock(
                    config.d_spectral,
                    config.k_max,
                    activation=config.fno_activation,
                    use_output_gate=config.fno_output_gate,
                    gate_init=config.fno_gate_init,
                )
                for _ in range(config.n_fno_layers)
            ]
        )

        # Project spectral back to model dimension
        self.spectral_proj_out = nn.Linear(config.d_spectral, config.d_model)

        # Learnable gate for spectral injection (starts small)
        self.injection_gate = nn.Parameter(
            torch.tensor(config.injection_gate_init, dtype=torch.bfloat16)
        )

        # =====================================================================
        # Transformer (Causal)
        # =====================================================================
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute FFT twiddle factors for each cutoff position
        # This allows efficient FFT computation at specific positions
        self._precompute_fft_twiddles()

        # Initialize weights
        self.apply(self._init_weights)

    def _precompute_fft_twiddles(self) -> None:
        """Precompute FFT twiddle factors for each cutoff position."""
        assert self.config.k_max is not None, "k_max must be set"
        k_max = self.config.k_max
        seq_len = self.config.seq_len

        # For each cutoff c, we need twiddles for positions [0:c]
        # twiddles[n, k] = e^{-2*pi*i*k*n/seq_len}
        # We compute these with respect to full seq_len for consistent frequency bins
        n = torch.arange(seq_len, dtype=torch.float32)
        k = torch.arange(k_max, dtype=torch.float32)
        angles = -2 * math.pi * torch.outer(n, k) / seq_len

        twiddles_real = torch.cos(angles).to(torch.bfloat16)
        twiddles_imag = torch.sin(angles).to(torch.bfloat16)
        twiddles = torch.stack([twiddles_real, twiddles_imag], dim=-1)  # (seq_len, k_max, 2)

        self.register_buffer("fft_twiddles", twiddles)
        self.fft_twiddles: torch.Tensor

        # Inverse twiddles for iFFT
        inv_angles = 2 * math.pi * torch.outer(n, k) / seq_len
        inv_twiddles_real = torch.cos(inv_angles).to(torch.bfloat16)
        inv_twiddles_imag = torch.sin(inv_angles).to(torch.bfloat16)
        inv_twiddles = torch.stack([inv_twiddles_real, inv_twiddles_imag], dim=-1)

        self.register_buffer("ifft_twiddles", inv_twiddles)
        self.ifft_twiddles: torch.Tensor

    def _compute_fft_at_cutoff(self, h_spectral: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Compute FFT of tokens [0:cutoff] (exclusive).

        Args:
            h_spectral: Spectral embeddings (batch, seq_len, d_spectral) in bfloat16
            cutoff: Cutoff position (exclusive)

        Returns:
            Complex FFT output (batch, d_spectral, k_max, 2)
        """
        batch, seq_len, d_spectral = h_spectral.shape
        assert self.config.k_max is not None
        k_max = self.config.k_max

        # Get embeddings for positions [0:cutoff]
        h_cut = h_spectral[:, :cutoff, :]  # (B, cutoff, d_spectral)

        # Get twiddles for positions [0:cutoff]
        twiddles = self.fft_twiddles[:cutoff]  # (cutoff, k_max, 2)

        # Compute FFT: X[k] = sum_n x[n] * e^{-2*pi*i*k*n/N}
        # h_cut: (B, cutoff, d_spectral)
        # twiddles: (cutoff, k_max, 2)
        # Output: (B, d_spectral, k_max, 2)

        # Use einsum for clarity
        # h_cut[b, n, d] * twiddles[n, k, (real/imag)]
        # -> sum over n -> (B, d_spectral, k_max)

        # Real part of FFT: sum_n h[n] * cos(angle[n,k])
        # Imag part of FFT: sum_n h[n] * sin(angle[n,k])
        twiddles_real = twiddles[..., 0]  # (cutoff, k_max)
        twiddles_imag = twiddles[..., 1]  # (cutoff, k_max)

        # Einsum: b=batch, n=position, d=d_spectral, k=frequency
        # h_cut: (b, n, d), twiddles: (n, k) -> (b, d, k)
        fft_real = torch.einsum("bnd,nk->bdk", h_cut, twiddles_real)
        fft_imag = torch.einsum("bnd,nk->bdk", h_cut, twiddles_imag)

        # Normalize by sqrt(cutoff) for stable magnitudes
        norm = math.sqrt(cutoff)
        fft_real = fft_real / norm
        fft_imag = fft_imag / norm

        return torch.stack([fft_real, fft_imag], dim=-1)  # (B, d_spectral, k_max, 2)

    def _compute_ifft(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse FFT to get time-domain predictions for all positions.

        Args:
            spectral: Complex spectral representation (batch, d_spectral, k_max, 2)

        Returns:
            Real time-domain predictions (batch, seq_len, d_spectral)
        """
        batch, d_spectral, k_max, _ = spectral.shape
        seq_len = self.config.seq_len

        # inv_twiddles: (seq_len, k_max, 2)
        # spectral: (B, d_spectral, k_max, 2)
        # Output: (B, seq_len, d_spectral)

        # iFFT: x[n] = sum_k X[k] * e^{+2*pi*i*k*n/N}
        # Complex multiplication: (X_r + i*X_i) * (cos + i*sin) = (X_r*cos - X_i*sin) + i*(...)
        # We only need the real part for output

        s_real = spectral[..., 0]  # (B, d_spectral, k_max)
        s_imag = spectral[..., 1]
        t_real = self.ifft_twiddles[..., 0]  # (seq_len, k_max)
        t_imag = self.ifft_twiddles[..., 1]

        # Real part of iFFT output: sum_k (X_r * cos - X_i * sin)
        # Using einsum: b=batch, d=d_spectral, k=frequency, n=position
        # s_real: (b, d, k), t_real: (n, k) -> (b, n, d)
        output_real_part1 = torch.einsum("bdk,nk->bnd", s_real, t_real)
        output_real_part2 = torch.einsum("bdk,nk->bnd", s_imag, t_imag)
        output_real = output_real_part1 - output_real_part2

        # Normalize by seq_len (standard iFFT normalization)
        output_real = output_real / seq_len

        return output_real

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices (batch, seq_len)

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        batch, seq_len = x.shape
        device = x.device

        # =====================================================================
        # Embeddings
        # =====================================================================

        tok_emb = self.token_emb(x)  # (B, T, D)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (T, D)

        # Base embeddings
        h = tok_emb + pos_emb  # (B, T, D)

        # =====================================================================
        # Spectral Injection at Cutoff Points
        # =====================================================================

        # Project to spectral dimension for FFT
        h_spectral = self.spectral_proj_in(h).to(torch.bfloat16)  # (B, T, D_spec)

        # Compute injection gate
        gate = torch.sigmoid(self.injection_gate)

        # Initialize spectral contribution as zeros
        spectral_contrib = torch.zeros_like(h)  # (B, T, D)

        # Process each cutoff
        for i, cutoff in enumerate(self.cutoff_positions):
            if cutoff >= seq_len:
                continue  # Skip cutoffs beyond current sequence length

            # Compute FFT of tokens [0:cutoff]
            fft_out = self._compute_fft_at_cutoff(h_spectral, cutoff)  # (B, d_spectral, k_max, 2)

            # Process through FNO
            # Reshape to (B, 1, d_spectral, k_max, 2) for FNO blocks
            spectral = fft_out.unsqueeze(1)
            for fno_block in self.fno_blocks:
                spectral = fno_block(spectral)
            spectral = spectral.squeeze(1)  # (B, d_spectral, k_max, 2)

            # Inverse FFT to get time-domain predictions
            time_pred = self._compute_ifft(spectral)  # (B, seq_len, d_spectral)

            # Project back to model dimension
            time_pred = self.spectral_proj_out(time_pred.float())  # (B, seq_len, D)

            # Determine which positions receive this spectral signal
            # Positions in [cutoff, next_cutoff) should use this prediction
            if i + 1 < len(self.cutoff_positions):
                next_cutoff = self.cutoff_positions[i + 1]
            else:
                next_cutoff = seq_len

            # Create mask for positions [cutoff, next_cutoff)
            # Shape: (1, seq_len, 1) for broadcasting
            pos_indices = torch.arange(seq_len, device=device)
            mask = ((pos_indices >= cutoff) & (pos_indices < next_cutoff)).float()
            mask = mask.view(1, seq_len, 1)  # (1, T, 1)

            # Add spectral contribution for these positions
            spectral_contrib = spectral_contrib + mask * time_pred

        # Apply gate and add to embeddings
        h = h + gate * spectral_contrib

        # =====================================================================
        # Transformer Forward (Causal)
        # =====================================================================

        for block in self.blocks:
            h = block(h)

        # Final layer norm and output projection
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, vocab)

        return logits
