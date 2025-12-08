"""Model definitions for ESMT and NanoGPT baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ESMTConfig, NanoGPTConfig
from spectral_layers import (
    MatryoshkaEmbedding,
    MatryoshkaPositionalEmbedding,
    SpectralBlock,
    SpectralBlockV2,
)


class SpectralGPT(nn.Module):
    """
    Elastic Spectral Matryoshka Transformer (ESMT).

    This is now a direct replica of NanoGPT's architecture with support for
    elastic inference via bandwidth truncation. The goal is to match NanoGPT's
    performance exactly, then add spectral features incrementally.
    
    Experimental features (controlled via config):
    - use_spectral_blur: Use SpectralBlurMLP with local conv instead of dense MLP
    - use_harmonic_mixing: Add octave skip connections (f <-> 2f)
    """

    def __init__(self, config: ESMTConfig):
        """
        Initialize SpectralGPT.

        Args:
            config: ESMT configuration
        """
        super().__init__()
        self.config = config

        # Token and positional embeddings (with truncation support)
        self.token_emb = MatryoshkaEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = MatryoshkaPositionalEmbedding(config.seq_len, config.d_model)

        # Stack of transformer blocks
        # Use SpectralBlockV2 if any experimental features are enabled
        use_experimental = config.use_spectral_blur or config.use_harmonic_mixing
        if use_experimental:
            self.layers = nn.ModuleList(
                [SpectralBlockV2(config) for _ in range(config.n_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [SpectralBlock(config) for _ in range(config.n_layers)]
            )

        # Final layer norm (same as NanoGPT)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.eps)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small values."""
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
        self, x: torch.Tensor, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with optional bandwidth truncation.

        Args:
            x: Input token indices [batch, seq_len]
            bandwidth_ratio: Fraction of dimensions to use (0.25 to 1.0)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = x.shape
        cutoff = int(self.config.d_model * bandwidth_ratio)

        # Get embeddings (truncated if bandwidth < 1.0)
        tok_emb = self.token_emb(x, bandwidth_ratio)
        pos_emb = self.pos_emb(seq_len, bandwidth_ratio)

        h = tok_emb + pos_emb

        # Pass through layers
        if bandwidth_ratio < 1.0:
            for layer in self.layers:
                h = layer.forward_sliced(h, cutoff)
            # Final layer norm (sliced)
            h = F.layer_norm(
                h,
                (cutoff,),
                self.ln_f.weight[:cutoff],
                self.ln_f.bias[:cutoff] if self.ln_f.bias is not None else None,
                self.ln_f.eps,
            )
            # Slice lm_head weights
            logits = F.linear(h, self.lm_head.weight[:, :cutoff])
        else:
            for layer in self.layers:
                h = layer(h)
            h = self.ln_f(h)
            logits = self.lm_head(h)

        return logits

    def get_spectral_regularization_loss(self) -> torch.Tensor:
        """
        Returns zero (no regularization in baseline).
        
        Kept for API compatibility with training code.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ==============================================================================
# NanoGPT Baseline Implementation
# ==============================================================================


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention for NanoGPT baseline."""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

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
        """Forward with sliced weights for baseline comparison."""
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


class TransformerMLP(nn.Module):
    """Standard transformer MLP for NanoGPT baseline."""

    def __init__(self, config: NanoGPTConfig):
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


class TransformerBlock(nn.Module):
    """Standard transformer block for NanoGPT baseline."""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.mlp = TransformerMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        # LayerNorm sliced (just slice the weight/bias)
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


class NanoGPT(nn.Module):
    """
    NanoGPT baseline model.

    Standard GPT-2 style architecture for comparison with ESMT.
    """

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.eps)

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        # self.lm_head.weight = self.token_emb.weight

        # Initialize
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
        self, x: torch.Tensor, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices [batch, seq_len]
            bandwidth_ratio: Fraction of dimensions to use (for sliced comparison)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = x.shape
        device = x.device

        # Embeddings
        tok_emb = self.token_emb(x)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)

        h = tok_emb + pos_emb

        if bandwidth_ratio < 1.0:
            return self._forward_sliced(h, bandwidth_ratio)

        # Standard forward
        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        return logits

    def _forward_sliced(
        self, h: torch.Tensor, bandwidth_ratio: float
    ) -> torch.Tensor:
        """Forward with sliced weights (for baseline comparison)."""
        cutoff = int(self.config.d_model * bandwidth_ratio)

        # Slice embeddings
        h = h[:, :, :cutoff]

        # Pass through blocks
        for block in self.blocks:
            h = block.forward_sliced(h, cutoff)

        # Final layer norm (sliced)
        h = F.layer_norm(
            h,
            (cutoff,),
            self.ln_f.weight[:cutoff],
            self.ln_f.bias[:cutoff] if self.ln_f.bias is not None else None,
            self.ln_f.eps,
        )

        # Project to vocab (sliced)
        logits = F.linear(h, self.lm_head.weight[:, :cutoff])

        return logits


# ==============================================================================
# Parameter Counting and Model Matching
# ==============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_parameter_counts(
    esmt_config: ESMTConfig, nano_config: NanoGPTConfig, tolerance: float = 0.05
) -> tuple[ESMTConfig, NanoGPTConfig]:
    """
    Adjust NanoGPT d_model to match ESMT parameter count.

    Args:
        esmt_config: ESMT configuration
        nano_config: NanoGPT configuration
        tolerance: Acceptable parameter count difference (default 5%)

    Returns:
        Updated configs with matched parameter counts
    """
    # Create models to count parameters
    esmt = SpectralGPT(esmt_config)
    esmt_params = count_parameters(esmt)

    # Binary search for matching d_model
    low, high = 64, 1024
    best_d_model = nano_config.d_model
    best_diff = float("inf")

    while low <= high:
        mid = ((low + high) // 2 // nano_config.n_heads) * nano_config.n_heads  # Align to heads
        if mid == 0:
            mid = nano_config.n_heads

        test_config = NanoGPTConfig(
            d_model=mid,
            n_layers=nano_config.n_layers,
            n_heads=nano_config.n_heads,
            vocab_size=nano_config.vocab_size,
            seq_len=nano_config.seq_len,
            mlp_ratio=nano_config.mlp_ratio,
        )

        try:
            nano = NanoGPT(test_config)
            nano_params = count_parameters(nano)
        except AssertionError:
            low = mid + nano_config.n_heads
            continue

        diff = abs(nano_params - esmt_params) / esmt_params

        if diff < best_diff:
            best_diff = diff
            best_d_model = mid

        if nano_params < esmt_params:
            low = mid + nano_config.n_heads
        else:
            high = mid - nano_config.n_heads

    # Update NanoGPT config
    nano_config.d_model = best_d_model

    # Verify final counts
    nano = NanoGPT(nano_config)
    final_nano_params = count_parameters(nano)

    print(f"ESMT parameters: {esmt_params:,}")
    print(f"NanoGPT parameters: {final_nano_params:,}")
    print(f"Difference: {abs(final_nano_params - esmt_params) / esmt_params * 100:.2f}%")

    if abs(final_nano_params - esmt_params) / esmt_params > tolerance:
        print(f"Warning: Parameter counts differ by more than {tolerance * 100}%")

    return esmt_config, nano_config


def create_matched_models(
    esmt_config: ESMTConfig | None = None,
    nano_config: NanoGPTConfig | None = None,
) -> tuple[SpectralGPT, NanoGPT, ESMTConfig, NanoGPTConfig]:
    """
    Create ESMT and NanoGPT models with matched parameter counts.

    Args:
        esmt_config: Optional ESMT config (uses defaults if None)
        nano_config: Optional NanoGPT config (uses defaults if None)

    Returns:
        Tuple of (esmt_model, nano_model, esmt_config, nano_config)
    """
    if esmt_config is None:
        esmt_config = ESMTConfig()
    if nano_config is None:
        nano_config = NanoGPTConfig()

    # Match parameter counts
    esmt_config, nano_config = match_parameter_counts(esmt_config, nano_config)

    # Create models
    esmt = SpectralGPT(esmt_config)
    nano = NanoGPT(nano_config)

    return esmt, nano, esmt_config, nano_config
