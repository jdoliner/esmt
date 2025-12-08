"""Spectral layers for the Elastic Spectral Matryoshka Transformer (ESMT).

This is a direct replica of NanoGPT's architecture, just with 'Spectral' naming.
The goal is to have a known-working baseline that matches NanoGPT's performance,
then make incremental changes to add spectral/Matryoshka features.
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

        n_heads_active = cutoff // self.head_dim
        if n_heads_active == 0:
            n_heads_active = 1
        head_dim_active = cutoff // n_heads_active

        # Slice QKV weights
        qkv_weight = self.qkv.weight[: 3 * cutoff, :cutoff]
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

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, cutoff)

        # Project
        proj_weight = self.proj.weight[:cutoff, :cutoff]
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
