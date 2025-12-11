"""Complex-valued layers for the Complex Spectral Transformer.

This module implements complex-valued neural network components based on:
- Deep Complex Networks (Trabelsi et al., ICLR 2018)
- Unitary RNNs literature
- Holographic Reduced Representations

Key components:
- ComplexLinear: Linear layer for complex tensors with proper initialization
- ModReLU: Magnitude-only activation that preserves phase
- ComplexLayerNorm: Normalization for complex tensors
- ComplexEmbedding: Learnable complex embeddings
- ComplexPositionalEncoding: Phase-ramp positional encoding (shift theorem)
- ComplexAttention: Attention with magnitude-based or phase-aware softmax
- ComplexFFN: Feed-forward network with ModReLU activation
- ComplexSpectralBlock: Full transformer block for complex representations
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Initialization Utilities
# ==============================================================================


def complex_kaiming_init(tensor: torch.Tensor, mode: str = "fan_in") -> torch.Tensor:
    """
    Initialize complex tensor using Rayleigh distribution for magnitude
    and uniform distribution for phase.
    
    This is the complex analog of Kaiming initialization.
    
    Args:
        tensor: Complex tensor to initialize in-place
        mode: 'fan_in' or 'fan_out' for variance scaling
    
    Returns:
        The initialized tensor
    """
    # Calculate fan for variance scaling
    if tensor.dim() < 2:
        fan = tensor.numel()
    else:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
        fan = fan_in if mode == "fan_in" else fan_out
    
    # Variance for each real/imag component
    # For complex: Var(|z|²) = Var(x) + Var(y) = 2*sigma²
    # We want total variance = 1/fan, so sigma = sqrt(1/(2*fan))
    sigma = math.sqrt(1.0 / (2.0 * fan))
    
    # Generate magnitude (Rayleigh) and phase (uniform)
    # Rayleigh with scale sigma: magnitude = sigma * sqrt(-2 * log(U))
    # Simpler: just use complex normal and scale
    with torch.no_grad():
        real_part = torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32) * sigma
        imag_part = torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32) * sigma
        tensor.copy_(torch.complex(real_part, imag_part))
    
    return tensor


def complex_xavier_init(tensor: torch.Tensor) -> torch.Tensor:
    """
    Initialize complex tensor using Xavier/Glorot style initialization.
    
    Uses average of fan_in and fan_out for variance scaling.
    """
    if tensor.dim() < 2:
        fan = tensor.numel()
    else:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
        fan = (fan_in + fan_out) / 2.0
    
    sigma = math.sqrt(1.0 / (2.0 * fan))
    
    with torch.no_grad():
        real_part = torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32) * sigma
        imag_part = torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32) * sigma
        tensor.copy_(torch.complex(real_part, imag_part))
    
    return tensor


def complex_uniform_phase_init(tensor: torch.Tensor, magnitude: float = 0.02) -> torch.Tensor:
    """
    Initialize with fixed magnitude and uniform random phase.
    
    This creates "unit-like" initialization where all weights have
    similar magnitude but random directions in the complex plane.
    
    Args:
        tensor: Complex tensor to initialize
        magnitude: Fixed magnitude for all weights
    """
    with torch.no_grad():
        phase = torch.rand(tensor.shape, device=tensor.device, dtype=torch.float32) * 2 * math.pi - math.pi  # [-π, π]
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        tensor.copy_(torch.complex(real_part, imag_part))
    return tensor


# ==============================================================================
# Core Complex Layers
# ==============================================================================


class ComplexLinear(nn.Module):
    """
    Linear layer for complex-valued tensors.
    
    Computes: y = Wx + b where W, x, b are all complex.
    
    Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    
    We implement this using the formula:
        real_out = x.real @ W.real - x.imag @ W.imag + b.real
        imag_out = x.real @ W.imag + x.imag @ W.real + b.imag
    
    This is mathematically equivalent to PyTorch's native complex matmul
    but gives us more control over the implementation.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        init: Literal["kaiming", "xavier", "uniform_phase"] = "kaiming"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Complex weight matrix
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.complex64)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.complex64)
            )
        else:
            self.register_parameter("bias", None)
        
        # Initialize
        self._init_weights(init)
    
    def _init_weights(self, init: str) -> None:
        if init == "kaiming":
            complex_kaiming_init(self.weight)
        elif init == "xavier":
            complex_xavier_init(self.weight)
        elif init == "uniform_phase":
            complex_uniform_phase_init(self.weight, magnitude=0.02)
        else:
            raise ValueError(f"Unknown init: {init}")
        
        if self.bias is not None:
            # Initialize bias to zero
            self.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [..., in_features]
        Returns:
            Complex tensor [..., out_features]
        """
        # Use PyTorch's native complex matmul (works in recent versions)
        # Falls back to manual implementation if needed
        try:
            out = F.linear(x, self.weight, self.bias)
        except RuntimeError:
            # Manual fallback for complex matmul
            out_real = (
                F.linear(x.real, self.weight.real) - 
                F.linear(x.imag, self.weight.imag)
            )
            out_imag = (
                F.linear(x.real, self.weight.imag) + 
                F.linear(x.imag, self.weight.real)
            )
            out = torch.complex(out_real, out_imag)
            
            if self.bias is not None:
                out = out + self.bias
        
        return out
    
    def forward_sliced(self, x: torch.Tensor, in_cutoff: int, out_cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        weight_sliced = self.weight[:out_cutoff, :in_cutoff]
        bias_sliced = self.bias[:out_cutoff] if self.bias is not None else None
        
        try:
            out = F.linear(x, weight_sliced, bias_sliced)
        except RuntimeError:
            out_real = (
                F.linear(x.real, weight_sliced.real) - 
                F.linear(x.imag, weight_sliced.imag)
            )
            out_imag = (
                F.linear(x.real, weight_sliced.imag) + 
                F.linear(x.imag, weight_sliced.real)
            )
            out = torch.complex(out_real, out_imag)
            if bias_sliced is not None:
                out = out + bias_sliced
        
        return out


class ModReLU(nn.Module):
    """
    Modulus ReLU activation for complex numbers.
    
    From Deep Complex Networks (Trabelsi et al., 2018):
    ModReLU(z) = ReLU(|z| + b) * (z / |z|)
               = ReLU(|z| + b) * e^{i*angle(z)}
    
    This applies a learnable threshold to the magnitude while preserving phase.
    If |z| + b < 0, the output is 0 (the signal is "noise").
    If |z| + b >= 0, the output has magnitude (|z| + b) and original phase.
    
    The bias b is typically initialized to a small negative value so that
    only signals with sufficient magnitude pass through.
    """
    
    def __init__(self, features: int, bias_init: float = -0.1):
        super().__init__()
        # Learnable bias per feature (applied to magnitude)
        self.bias = nn.Parameter(torch.full((features,), bias_init))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor [..., features]
        Returns:
            Complex tensor [..., features]
        """
        # Compute magnitude and phase
        mag = torch.abs(z)  # [..., features]
        phase = torch.angle(z)  # [..., features]
        
        # Apply ReLU to (magnitude + bias)
        new_mag = F.relu(mag + self.bias)
        
        # Reconstruct complex number with new magnitude and original phase
        return torch.polar(new_mag, phase)
    
    def forward_sliced(self, z: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced bias for Matryoshka inference."""
        mag = torch.abs(z)
        phase = torch.angle(z)
        new_mag = F.relu(mag + self.bias[:cutoff])
        return torch.polar(new_mag, phase)


class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex tensors.
    
    Two modes:
    1. "magnitude" (default): Normalize by magnitude, preserve relative phases
       - Compute mean magnitude, normalize so mean |z| = 1
       - Apply learnable complex scale and bias
       
    2. "split" (Trabelsi et al.): Normalize real and imaginary parts separately
       - Treat as 2D vector, normalize each component
       - This can distort phase relationships
    
    The "magnitude" mode is preferred for preserving spectral structure.
    """
    
    def __init__(
        self, 
        normalized_shape: int, 
        eps: float = 1e-6,
        mode: Literal["magnitude", "split"] = "magnitude"
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mode = mode
        
        # Learnable scale (complex) and bias (complex)
        # Initialize scale to 1+0j, bias to 0+0j
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor [..., normalized_shape]
        Returns:
            Normalized complex tensor [..., normalized_shape]
        """
        if self.mode == "magnitude":
            return self._forward_magnitude(z)
        else:
            return self._forward_split(z)
    
    def _forward_magnitude(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize by magnitude, preserving relative phases."""
        # Compute magnitude
        mag = torch.abs(z)  # [..., D]
        
        # Compute mean and variance of magnitude across feature dim
        mean_mag = mag.mean(dim=-1, keepdim=True)
        var_mag = ((mag - mean_mag) ** 2).mean(dim=-1, keepdim=True)
        
        # Normalize magnitude
        mag_normalized = (mag - mean_mag) / torch.sqrt(var_mag + self.eps)
        
        # Reconstruct with normalized magnitude but original phase
        phase = torch.angle(z)
        z_normalized = torch.polar(mag_normalized.abs(), phase)  # abs() to handle negative normalized values
        
        # Alternative: scale the original complex number by ratio
        # This preserves phase better when mean_mag is large
        scale_factor = mag_normalized / (mag + self.eps)
        z_normalized = z * scale_factor
        
        # Apply learnable affine transform
        return z_normalized * self.weight + self.bias
    
    def _forward_split(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize real and imaginary parts separately (Trabelsi style)."""
        # Normalize real part
        real_mean = z.real.mean(dim=-1, keepdim=True)
        real_var = ((z.real - real_mean) ** 2).mean(dim=-1, keepdim=True)
        real_norm = (z.real - real_mean) / torch.sqrt(real_var + self.eps)
        
        # Normalize imaginary part
        imag_mean = z.imag.mean(dim=-1, keepdim=True)
        imag_var = ((z.imag - imag_mean) ** 2).mean(dim=-1, keepdim=True)
        imag_norm = (z.imag - imag_mean) / torch.sqrt(imag_var + self.eps)
        
        z_normalized = torch.complex(real_norm, imag_norm)
        
        # Apply learnable affine transform
        return z_normalized * self.weight + self.bias
    
    def forward_sliced(self, z: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced parameters for Matryoshka inference."""
        if self.mode == "magnitude":
            mag = torch.abs(z)
            mean_mag = mag.mean(dim=-1, keepdim=True)
            var_mag = ((mag - mean_mag) ** 2).mean(dim=-1, keepdim=True)
            mag_normalized = (mag - mean_mag) / torch.sqrt(var_mag + self.eps)
            scale_factor = mag_normalized / (mag + self.eps)
            z_normalized = z * scale_factor
        else:
            real_mean = z.real.mean(dim=-1, keepdim=True)
            real_var = ((z.real - real_mean) ** 2).mean(dim=-1, keepdim=True)
            real_norm = (z.real - real_mean) / torch.sqrt(real_var + self.eps)
            
            imag_mean = z.imag.mean(dim=-1, keepdim=True)
            imag_var = ((z.imag - imag_mean) ** 2).mean(dim=-1, keepdim=True)
            imag_norm = (z.imag - imag_mean) / torch.sqrt(imag_var + self.eps)
            
            z_normalized = torch.complex(real_norm, imag_norm)
        
        return z_normalized * self.weight[:cutoff] + self.bias[:cutoff]


# ==============================================================================
# Embeddings and Positional Encoding
# ==============================================================================


class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding layer.
    
    Each token maps to a complex vector where both real and imaginary
    parts are learned. This gives the model 2x the representational
    capacity per dimension.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Learned complex embeddings
        self.embedding = nn.Parameter(
            torch.zeros(vocab_size, d_model, dtype=torch.complex64)
        )
        
        # Initialize with small magnitude, random phase
        self._init_embeddings()
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings with Gaussian real/imag parts."""
        # Standard deviation matching GPT-2 style init
        std = 0.02
        real_part = torch.randn(self.vocab_size, self.d_model) * std
        imag_part = torch.randn(self.vocab_size, self.d_model) * std
        self.embedding.data = torch.complex(real_part, imag_part)
    
    def forward(self, x: torch.Tensor, bandwidth_ratio: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]
            bandwidth_ratio: Fraction of dimensions to use (0.0 to 1.0)
        
        Returns:
            Complex embeddings [batch, seq_len, d_model] or truncated
        """
        embeddings = self.embedding[x]  # [batch, seq, d_model]
        
        if bandwidth_ratio < 1.0:
            # Truncate at frequency boundary (must be even to preserve complex pairs)
            cutoff = int(self.d_model * bandwidth_ratio)
            cutoff = max(2, cutoff - cutoff % 2)  # Ensure even
            embeddings = embeddings[..., :cutoff]
        
        return embeddings


class ComplexPositionalEncoding(nn.Module):
    """
    Positional encoding via phase rotation (Shift Theorem).
    
    Instead of adding a learned positional embedding, we rotate the
    frequency components by position-dependent phases.
    
    The shift theorem states: shifting a signal by Δt is equivalent to
    multiplying its k-th frequency component by e^{-i * 2π * k * Δt / N}
    
    This means position is encoded "for free" in the phase of each frequency.
    The model can recover relative positions by multiplying conjugates.
    
    Properties:
    - Position 0: no rotation (identity)
    - Position p: rotate dimension k by angle proportional to k*p
    - Low frequencies (small k) rotate slowly → coarse position info
    - High frequencies (large k) rotate fast → fine position info
    """
    
    def __init__(self, d_model: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Frequency for each dimension (exponentially spaced)
        # Dim 0 has lowest frequency, dim D-1 has highest
        freqs = torch.pow(base, -torch.arange(0, d_model, dtype=torch.float32) / d_model)
        self.register_buffer("freqs", freqs)
        self.freqs: torch.Tensor  # Type hint for buffer
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: int | None = None,
        bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Apply positional encoding via phase rotation.
        
        Args:
            x: Complex tensor [batch, seq_len, d_model]
            seq_len: Override sequence length (uses x.shape[1] if None)
            bandwidth_ratio: Fraction of dimensions to use
        
        Returns:
            Position-encoded complex tensor
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        d_model = x.shape[-1]
        
        # Get frequencies for current dimension
        freqs = self.freqs[:d_model]
        
        # Create position indices [0, 1, ..., seq_len-1]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute phase angles: theta[p, k] = pos[p] * freq[k]
        # Shape: [seq_len, d_model]
        theta = torch.outer(pos, freqs)
        
        # Create rotation vector: e^{-i * theta}
        # Negative sign encodes "time passed" (standard convention)
        rotator = torch.polar(torch.ones_like(theta), -theta)
        
        # Apply rotation (broadcast over batch)
        # x: [batch, seq, d_model], rotator: [seq, d_model]
        return x * rotator.unsqueeze(0)
    
    def forward_sliced(
        self, 
        x: torch.Tensor, 
        cutoff: int,
        seq_len: int | None = None
    ) -> torch.Tensor:
        """Forward with truncated dimensions for Matryoshka inference."""
        if seq_len is None:
            seq_len = x.shape[1]
        
        freqs = self.freqs[:cutoff]
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        theta = torch.outer(pos, freqs)
        rotator = torch.polar(torch.ones_like(theta), -theta)
        
        return x * rotator.unsqueeze(0)


# ==============================================================================
# Complex Attention
# ==============================================================================


class ComplexAttention(nn.Module):
    """
    Complex-valued self-attention.
    
    Two modes:
    1. "magnitude" (default): Attention weights based on magnitude of Q·K†
       - score = |Q @ K†| / sqrt(d)
       - This preserves the intuition that "similar" vectors have large dot product
       
    2. "phase_aware": Attention incorporates phase difference
       - score = |Q @ K†| * cos(angle(Q @ K†)) / sqrt(d)
       - Vectors with aligned phases get higher attention
       - This could help capture syntactic relationships
    
    The values are complex, so the output is a weighted sum of complex vectors.
    """
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        seq_len: int,
        attention_mode: Literal["magnitude", "phase_aware"] = "magnitude",
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_mode = attention_mode
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV projection (complex)
        self.qkv = ComplexLinear(d_model, 3 * d_model, bias=False)
        
        # Output projection (complex)
        self.proj = ComplexLinear(d_model, d_model, bias=False)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        )
        self.mask: torch.Tensor
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [batch, seq_len, d_model]
        Returns:
            Complex tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Compute Q, K, V (all complex)
        qkv = self.qkv(x)  # [batch, seq, 3 * d_model]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to heads: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Q @ K† (conjugate transpose of K)
        # For complex: (Q @ K†)[i,j] = sum_d Q[i,d] * conj(K[j,d])
        k_conj = torch.conj(k)
        scores = q @ k_conj.transpose(-2, -1)  # [batch, heads, seq, seq]
        
        # Scale
        scale = 1.0 / math.sqrt(self.head_dim)
        
        if self.attention_mode == "magnitude":
            # Use magnitude of complex scores
            attn_weights = torch.abs(scores) * scale
        else:  # phase_aware
            # Use magnitude * cos(phase) - rewards aligned phases
            magnitude = torch.abs(scores)
            phase = torch.angle(scores)
            attn_weights = magnitude * torch.cos(phase) * scale
        
        # Apply causal mask
        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, 
            float("-inf")
        )
        
        # Softmax over keys (real-valued operation)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attend to values (complex)
        # attn_weights is real [batch, heads, seq, seq]
        # v is complex [batch, heads, seq, head_dim]
        # Output: weighted sum of complex values
        out = attn_weights.to(v.dtype) @ v  # [batch, heads, seq, head_dim]
        
        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # Output projection
        out = self.proj(out)
        
        return out
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        batch, seq_len, _ = x.shape
        
        # Ensure cutoff respects head boundaries
        n_heads_active = max(1, cutoff // self.head_dim)
        head_dim_active = self.head_dim
        
        # Handle case where cutoff < head_dim
        if cutoff < self.head_dim:
            n_heads_active = 1
            head_dim_active = cutoff
        
        aligned_dim = n_heads_active * head_dim_active
        
        # Slice QKV weights
        qkv = self.qkv.forward_sliced(x, cutoff, 3 * aligned_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # type: ignore[union-attr]
        
        # Reshape to heads
        q = q.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        k = k.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        v = v.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        
        # Attention scores
        k_conj = torch.conj(k)
        scores = q @ k_conj.transpose(-2, -1)
        scale = 1.0 / math.sqrt(head_dim_active)
        
        if self.attention_mode == "magnitude":
            attn_weights = torch.abs(scores) * scale
        else:
            magnitude = torch.abs(scores)
            phase = torch.angle(scores)
            attn_weights = magnitude * torch.cos(phase) * scale
        
        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = attn_weights.to(v.dtype) @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, aligned_dim)
        
        # Output projection
        out = self.proj.forward_sliced(out, aligned_dim, cutoff)
        
        return out


# ==============================================================================
# Complex FFN
# ==============================================================================


class ComplexFFN(nn.Module):
    """
    Complex-valued feed-forward network with ModReLU activation.
    
    Architecture: Linear -> ModReLU -> Linear
    
    The ModReLU preserves phase while applying a threshold to magnitude,
    acting as a learned noise gate.
    """
    
    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * mlp_ratio
        
        self.fc1 = ComplexLinear(d_model, self.hidden_dim)
        self.act = ModReLU(self.hidden_dim)
        self.fc2 = ComplexLinear(self.hidden_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [..., d_model]
        Returns:
            Complex tensor [..., d_model]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        hidden_cutoff = cutoff * (self.hidden_dim // self.d_model)
        
        x = self.fc1.forward_sliced(x, cutoff, hidden_cutoff)
        x = self.act.forward_sliced(x, hidden_cutoff)
        x = self.fc2.forward_sliced(x, hidden_cutoff, cutoff)
        
        return x


# ==============================================================================
# Complex Spectral Block
# ==============================================================================


class ComplexSpectralBlock(nn.Module):
    """
    Full transformer block for complex representations.
    
    Architecture with multiplicative residuals:
        x = x * (1 + attention(norm(x)))
        x = x * (1 + ffn(norm(x)))
    
    Multiplicative residuals are more natural for complex numbers:
    - Addition can cause phase cancellation
    - Multiplication preserves the "base" phase while modulating it
    - (1 + small_change) ≈ exp(small_change) for small values
    
    The residual_mode flag allows switching between:
    - "multiplicative": x * (1 + f(x))
    - "additive": x + f(x) (standard transformer residual)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        mlp_ratio: int = 4,
        attention_mode: Literal["magnitude", "phase_aware"] = "magnitude",
        layernorm_mode: Literal["magnitude", "split"] = "magnitude",
        residual_mode: Literal["multiplicative", "additive"] = "multiplicative",
        eps: float = 1e-6,
        dropout: float = 0.0
    ):
        super().__init__()
        self.residual_mode = residual_mode
        
        # Layer norms
        self.ln1 = ComplexLayerNorm(d_model, eps=eps, mode=layernorm_mode)
        self.ln2 = ComplexLayerNorm(d_model, eps=eps, mode=layernorm_mode)
        
        # Attention
        self.attn = ComplexAttention(
            d_model=d_model,
            n_heads=n_heads,
            seq_len=seq_len,
            attention_mode=attention_mode,
            dropout=dropout
        )
        
        # FFN
        self.ffn = ComplexFFN(d_model=d_model, mlp_ratio=mlp_ratio)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [batch, seq_len, d_model]
        Returns:
            Complex tensor [batch, seq_len, d_model]
        """
        if self.residual_mode == "multiplicative":
            # Multiplicative residual: x * (1 + f(x))
            x = x * (1 + self.attn(self.ln1(x)))
            x = x * (1 + self.ffn(self.ln2(x)))
        else:
            # Standard additive residual
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        
        return x
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        if self.residual_mode == "multiplicative":
            ln1_out = self.ln1.forward_sliced(x, cutoff)
            x = x * (1 + self.attn.forward_sliced(ln1_out, cutoff))
            
            ln2_out = self.ln2.forward_sliced(x, cutoff)
            x = x * (1 + self.ffn.forward_sliced(ln2_out, cutoff))
        else:
            ln1_out = self.ln1.forward_sliced(x, cutoff)
            x = x + self.attn.forward_sliced(ln1_out, cutoff)
            
            ln2_out = self.ln2.forward_sliced(x, cutoff)
            x = x + self.ffn.forward_sliced(ln2_out, cutoff)
        
        return x


# ==============================================================================
# Output Projection
# ==============================================================================


class ComplexToLogits(nn.Module):
    """
    Project complex hidden states to real-valued logits.
    
    Uses both magnitude and phase information:
        features = concat(|z|, angle(z))  # [batch, seq, 2*d_model]
        logits = Linear(features)          # [batch, seq, vocab_size]
    
    This preserves the phase information (potentially syntactic) while
    converting to the real domain needed for cross-entropy loss.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Project concatenated magnitude+phase to vocab
        # Input: 2*d_model (magnitude + phase)
        # Output: vocab_size
        self.proj = nn.Linear(2 * d_model, vocab_size, bias=False)
        
        # Initialize
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor [batch, seq_len, d_model]
        Returns:
            Real logits [batch, seq_len, vocab_size]
        """
        # Extract magnitude and phase
        magnitude = torch.abs(z)  # [batch, seq, d_model]
        phase = torch.angle(z)    # [batch, seq, d_model], in [-π, π]
        
        # Normalize phase to [0, 1] for better numerical behavior
        phase_normalized = (phase + math.pi) / (2 * math.pi)
        
        # Concatenate
        features = torch.cat([magnitude, phase_normalized], dim=-1)  # [batch, seq, 2*d_model]
        
        # Project to logits
        logits = self.proj(features)
        
        return logits
    
    def forward_sliced(self, z: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        magnitude = torch.abs(z)
        phase = torch.angle(z)
        phase_normalized = (phase + math.pi) / (2 * math.pi)
        
        features = torch.cat([magnitude, phase_normalized], dim=-1)  # [batch, seq, 2*cutoff]
        
        # Slice projection weights
        # Weight shape: [vocab_size, 2*d_model]
        # We need: [vocab_size, 2*cutoff]
        weight_sliced = self.proj.weight[:, :2*cutoff]
        logits = F.linear(features, weight_sliced)
        
        return logits


# ==============================================================================
# Spectral Gate (Optional)
# ==============================================================================


class ComplexSpectralGate(nn.Module):
    """
    Learnable complex filter for frequency-selective gating.
    
    Each frequency dimension has a learned complex weight that can:
    - Scale magnitude (amplitude control)
    - Rotate phase (phase shift)
    
    This allows the model to learn which frequencies are important
    and how to align their phases.
    
    For data-dependent gating, use ComplexSpectralGateDynamic instead.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable complex weights (one per frequency)
        # Initialize with magnitude=1, random phase
        self.weight = nn.Parameter(torch.zeros(d_model, dtype=torch.complex64))
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize with unit magnitude and random phase."""
        phase = torch.rand(self.d_model) * 2 * math.pi - math.pi
        self.weight.data = torch.polar(torch.ones(self.d_model), phase)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [..., d_model]
        Returns:
            Gated complex tensor [..., d_model]
        """
        return x * self.weight
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        return x * self.weight[:cutoff]


class ComplexSpectralGateDynamic(nn.Module):
    """
    Data-dependent complex gating.
    
    The gate is predicted from the input:
        gate = ComplexLinear(x)
        output = x * gate
    
    This allows the model to dynamically filter frequencies based on content.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gate_proj = ComplexLinear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [..., d_model]
        Returns:
            Gated complex tensor [..., d_model]
        """
        gate = self.gate_proj(x)
        return x * gate
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for Matryoshka inference."""
        gate = self.gate_proj.forward_sliced(x, cutoff, cutoff)
        return x * gate
