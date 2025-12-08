"""Spectral layers for the Elastic Spectral Matryoshka Transformer (ESMT).

This module contains both baseline components (matching NanoGPT) and experimental
spectral components that exploit the wave-like properties of the representation.

Experimental components:
- SpectralBlurMLP: Uses 1D convolution for local feature mixing
- HarmonicMixing: Connects features at octave intervals (f <-> 2f)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ESMTConfig


class SpectralAttention(nn.Module):
    """
    Standard causal self-attention (identical to NanoGPT's CausalSelfAttention).
    """

    def __init__(self, config: ESMTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # QKV projection (combined for efficiency)
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
        self.mask: torch.Tensor  # Type hint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [batch, seq, 3 * d_model]
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
        out = attn @ v  # [batch, heads, seq, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.proj(out)

        return out

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for elastic inference."""
        batch, seq_len, _ = x.shape

        # Calculate aligned dimensions for attention
        # We need n_heads * head_dim to equal our working dimension
        n_heads_active = max(1, cutoff // self.head_dim)
        # Keep original head_dim, align total dimension to head boundary
        aligned_dim = n_heads_active * self.head_dim

        # Handle case where cutoff < head_dim
        if cutoff < self.head_dim:
            n_heads_active = 1
            aligned_dim = cutoff
            head_dim_active = cutoff
        else:
            head_dim_active = self.head_dim

        # Project input to aligned QKV dimension
        qkv_weight = self.qkv.weight[: 3 * aligned_dim, :cutoff]
        qkv = F.linear(x, qkv_weight)

        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads
        q = q.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        k = k.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        v = v.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)

        # Attention
        scale = 1.0 / math.sqrt(head_dim_active)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape back to aligned_dim
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, aligned_dim)

        # Project back to cutoff dimensions
        proj_weight = self.proj.weight[:cutoff, :aligned_dim]
        out = F.linear(out, proj_weight)

        return out


class SpectralMLP(nn.Module):
    """
    Standard transformer MLP (identical to NanoGPT's TransformerMLP).
    """

    def __init__(self, config: ESMTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.mlp_ratio

        self.fc1 = nn.Linear(config.d_model, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, config.d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        hidden_cutoff = cutoff * (self.hidden_dim // self.d_model)

        x = F.linear(
            x,
            self.fc1.weight[:hidden_cutoff, :cutoff],
            self.fc1.bias[:hidden_cutoff] if self.fc1.bias is not None else None,
        )
        x = self.act(x)
        x = F.linear(
            x,
            self.fc2.weight[:cutoff, :hidden_cutoff],
            self.fc2.bias[:cutoff] if self.fc2.bias is not None else None,
        )
        return x


class SpectralBlock(nn.Module):
    """
    Standard transformer block (identical to NanoGPT's TransformerBlock).

    Pre-norm architecture: LN -> Attn -> Residual -> LN -> MLP -> Residual
    """

    def __init__(self, config: ESMTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.attn = SpectralAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.mlp = SpectralMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        # LayerNorm sliced
        ln1_out = F.layer_norm(
            x,
            (cutoff,),
            self.ln1.weight[:cutoff],
            self.ln1.bias[:cutoff] if self.ln1.bias is not None else None,
            self.ln1.eps,
        )
        x = x + self.attn.forward_sliced(ln1_out, cutoff)

        ln2_out = F.layer_norm(
            x,
            (cutoff,),
            self.ln2.weight[:cutoff],
            self.ln2.bias[:cutoff] if self.ln2.bias is not None else None,
            self.ln2.eps,
        )
        x = x + self.mlp.forward_sliced(ln2_out, cutoff)
        return x


# Keep these for backwards compatibility with model.py imports
# They just forward to the new classes
SpectralGatedLayer = SpectralBlock


class MatryoshkaEmbedding(nn.Module):
    """
    Standard embedding layer with truncation support for elastic inference.
    
    Unlike the previous version, this uses standard initialization
    (no spectral decay bias) to match NanoGPT exactly.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Standard initialization (same as NanoGPT)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        embeddings = self.embedding(x)

        if bandwidth_ratio < 1.0:
            cutoff = int(self.d_model * bandwidth_ratio)
            cutoff = max(1, cutoff)
            embeddings = embeddings[:, :, :cutoff]

        return embeddings


class MatryoshkaPositionalEmbedding(nn.Module):
    """
    Standard positional embedding with truncation support for elastic inference.
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(seq_len, d_model)
        
        # Standard initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self, seq_len: int, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        embeddings = self.embedding(positions).unsqueeze(0)

        if bandwidth_ratio < 1.0:
            cutoff = int(self.d_model * bandwidth_ratio)
            cutoff = max(1, cutoff)
            embeddings = embeddings[:, :, :cutoff]

        return embeddings


# Keeping SpectralNorm for compatibility, but it's not used in the baseline
class SpectralNorm(nn.Module):
    """L2 normalization (not used in baseline, kept for compatibility)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)


# ==============================================================================
# Experimental Spectral Components
# ==============================================================================


class SpectralBlurMLP(nn.Module):
    """
    MLP with local feature mixing via 1D convolution.
    
    The key insight: In a spectral/wave representation, nearby frequencies are
    related (spectral leakage). Instead of dense mixing (every feature talks to
    every feature), we use local convolution (features only talk to neighbors).
    
    This forces the model to organize information smoothly along the feature
    dimension, creating a "semantic gradient" where related concepts are placed
    at adjacent frequencies.
    
    Architecture:
        1. Conv1D across feature dimension (local mixing)
        2. Standard MLP expansion with GELU
        3. Optional second Conv1D for output smoothing
    """

    def __init__(self, config: ESMTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.mlp_ratio
        self.kernel_size = getattr(config, 'blur_kernel_size', 3)
        
        # Local feature mixing via depthwise-separable convolution
        # Using groups=1 for cross-channel mixing (each output depends on kernel_size inputs)
        padding = self.kernel_size // 2
        self.feature_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1, 
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False
        )
        # Initialize close to identity (small perturbation around center)
        nn.init.zeros_(self.feature_conv.weight)
        center = self.kernel_size // 2
        self.feature_conv.weight.data[0, 0, center] = 1.0
        
        # Standard MLP layers
        self.fc1 = nn.Linear(config.d_model, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, config.d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model]
        """
        batch, seq, d = x.shape
        
        # Apply 1D conv across feature dimension
        # Reshape: [batch, seq, d] -> [batch*seq, 1, d]
        x_flat = x.view(batch * seq, 1, d)
        x_blur = self.feature_conv(x_flat)  # [batch*seq, 1, d]
        x_blur = x_blur.view(batch, seq, d)
        
        # Standard MLP
        x_blur = self.fc1(x_blur)
        x_blur = self.act(x_blur)
        x_blur = self.fc2(x_blur)
        
        return x_blur

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced weights for elastic inference."""
        batch, seq, _ = x.shape
        hidden_cutoff = cutoff * (self.hidden_dim // self.d_model)
        
        # Apply conv (works on any dimension, just slice output)
        x_flat = x.view(batch * seq, 1, cutoff)
        
        # For sliced inference, we need to handle the conv carefully
        # The conv kernel is the same, but we apply it to fewer features
        # We use the same kernel but the effective receptive field stays local
        x_blur = F.conv1d(
            x_flat, 
            self.feature_conv.weight,
            padding=self.kernel_size // 2
        )
        x_blur = x_blur[..., :cutoff]  # Ensure output matches cutoff
        x_blur = x_blur.view(batch, seq, cutoff)
        
        # Sliced MLP
        x_blur = F.linear(
            x_blur,
            self.fc1.weight[:hidden_cutoff, :cutoff],
            self.fc1.bias[:hidden_cutoff] if self.fc1.bias is not None else None,
        )
        x_blur = self.act(x_blur)
        x_blur = F.linear(
            x_blur,
            self.fc2.weight[:cutoff, :hidden_cutoff],
            self.fc2.bias[:cutoff] if self.fc2.bias is not None else None,
        )
        
        return x_blur


class HarmonicMixing(nn.Module):
    """
    Mix features at octave intervals (f <-> 2f <-> 4f).
    
    The key insight: In physics and music, frequencies are related not just to
    neighbors but to harmonics (f, 2f, 4f, ...). Language has similar multi-scale
    structure: words -> sentences -> paragraphs -> documents.
    
    By wiring f <-> 2f, we give the model a dedicated highway to move information
    between different structural scales without passing through intermediate steps.
    
    Implementation:
        - For each feature index i, we connect it to 2i and i//2
        - Connections are weighted and learnable
        - The mixing is additive (residual-style)
    """

    def __init__(self, d_model: int, n_octaves: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_octaves = n_octaves
        
        # Learnable mixing weights for each octave level
        # Initialize small so harmonic mixing is initially weak
        self.up_weights = nn.Parameter(torch.ones(n_octaves) * 0.1)
        self.down_weights = nn.Parameter(torch.ones(n_octaves) * 0.1)
        
        # Pre-compute index mappings for efficiency
        # We'll build these as buffers
        self._build_index_maps()

    def _build_index_maps(self):
        """Pre-compute octave index mappings."""
        # For each octave level, compute source->target index pairs
        up_indices = []  # low freq -> high freq (i -> 2i)
        down_indices = []  # high freq -> low freq (i -> i//2)
        
        for octave in range(1, self.n_octaves + 1):
            stride = 2 ** octave
            
            # Upsample: feature[i] contributes to feature[i * stride]
            up_src, up_tgt = [], []
            for i in range(self.d_model // stride):
                tgt_idx = i * stride
                if tgt_idx < self.d_model:
                    up_src.append(i)
                    up_tgt.append(tgt_idx)
            
            if up_src:
                self.register_buffer(
                    f'up_src_{octave}', 
                    torch.tensor(up_src, dtype=torch.long)
                )
                self.register_buffer(
                    f'up_tgt_{octave}', 
                    torch.tensor(up_tgt, dtype=torch.long)
                )
            
            # Downsample: feature[i * stride] contributes to feature[i]
            down_src, down_tgt = [], []
            for i in range(stride, self.d_model, 1):
                src_idx = i
                tgt_idx = i // stride
                down_src.append(src_idx)
                down_tgt.append(tgt_idx)
            
            if down_src:
                self.register_buffer(
                    f'down_src_{octave}', 
                    torch.tensor(down_src, dtype=torch.long)
                )
                self.register_buffer(
                    f'down_tgt_{octave}', 
                    torch.tensor(down_tgt, dtype=torch.long)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply harmonic mixing.
        
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model] with harmonic connections mixed in
        """
        out = x.clone()
        
        for octave in range(1, self.n_octaves + 1):
            # Get weights (sigmoid to keep bounded)
            up_w = torch.sigmoid(self.up_weights[octave - 1])
            down_w = torch.sigmoid(self.down_weights[octave - 1])
            
            # Upsample contribution: low-freq info flows to high-freq positions
            up_src = getattr(self, f'up_src_{octave}', None)
            up_tgt = getattr(self, f'up_tgt_{octave}', None)
            if up_src is not None and up_tgt is not None:
                # x[..., up_tgt] += up_w * x[..., up_src]
                contribution = up_w * x[..., up_src]
                out = out.index_add(-1, up_tgt, contribution)
            
            # Downsample contribution: high-freq info aggregates to low-freq
            down_src = getattr(self, f'down_src_{octave}', None)
            down_tgt = getattr(self, f'down_tgt_{octave}', None)
            if down_src is not None and down_tgt is not None:
                contribution = down_w * x[..., down_src]
                out = out.index_add(-1, down_tgt, contribution)
        
        return out

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """Forward with sliced dimensions for elastic inference."""
        out = x.clone()
        
        for octave in range(1, self.n_octaves + 1):
            up_w = torch.sigmoid(self.up_weights[octave - 1])
            down_w = torch.sigmoid(self.down_weights[octave - 1])
            
            # Filter indices to those within cutoff
            up_src = getattr(self, f'up_src_{octave}', None)
            up_tgt = getattr(self, f'up_tgt_{octave}', None)
            if up_src is not None and up_tgt is not None:
                # Only use indices within cutoff
                mask = (up_src < cutoff) & (up_tgt < cutoff)
                if mask.any():
                    src_filtered = up_src[mask]
                    tgt_filtered = up_tgt[mask]
                    contribution = up_w * x[..., src_filtered]
                    out = out.index_add(-1, tgt_filtered, contribution)
            
            down_src = getattr(self, f'down_src_{octave}', None)
            down_tgt = getattr(self, f'down_tgt_{octave}', None)
            if down_src is not None and down_tgt is not None:
                mask = (down_src < cutoff) & (down_tgt < cutoff)
                if mask.any():
                    src_filtered = down_src[mask]
                    tgt_filtered = down_tgt[mask]
                    contribution = down_w * x[..., src_filtered]
                    out = out.index_add(-1, tgt_filtered, contribution)
        
        return out


class SpectralBlockV2(nn.Module):
    """
    Enhanced spectral block with experimental components.
    
    This block can optionally use:
    - SpectralBlurMLP instead of standard MLP (local feature mixing)
    - HarmonicMixing after MLP (octave skip connections)
    
    The block architecture is:
        LN -> Attn -> Residual -> LN -> MLP -> [Harmonic] -> Residual
    """

    def __init__(self, config: ESMTConfig):
        super().__init__()
        self.config = config
        
        # Standard components
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.attn = SpectralAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.eps)
        
        # MLP: either standard or spectral blur
        use_blur = getattr(config, 'use_spectral_blur', False)
        if use_blur:
            self.mlp = SpectralBlurMLP(config)
        else:
            self.mlp = SpectralMLP(config)
        
        # Optional harmonic mixing
        use_harmonic = getattr(config, 'use_harmonic_mixing', False)
        n_octaves = getattr(config, 'n_octaves', 3)
        if use_harmonic:
            self.harmonic = HarmonicMixing(config.d_model, n_octaves=n_octaves)
        else:
            self.harmonic = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        x = x + self.attn(self.ln1(x))
        
        # MLP
        mlp_out = self.mlp(self.ln2(x))
        
        # Optional harmonic mixing
        if self.harmonic is not None:
            mlp_out = self.harmonic(mlp_out)
        
        x = x + mlp_out
        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        # LayerNorm sliced
        ln1_out = F.layer_norm(
            x,
            (cutoff,),
            self.ln1.weight[:cutoff],
            self.ln1.bias[:cutoff] if self.ln1.bias is not None else None,
            self.ln1.eps,
        )
        x = x + self.attn.forward_sliced(ln1_out, cutoff)

        ln2_out = F.layer_norm(
            x,
            (cutoff,),
            self.ln2.weight[:cutoff],
            self.ln2.bias[:cutoff] if self.ln2.bias is not None else None,
            self.ln2.eps,
        )
        mlp_out = self.mlp.forward_sliced(ln2_out, cutoff)
        
        # Optional harmonic mixing (sliced)
        if self.harmonic is not None:
            mlp_out = self.harmonic.forward_sliced(mlp_out, cutoff)
        
        x = x + mlp_out
        return x
