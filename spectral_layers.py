"""Spectral layers for the Elastic Spectral Matryoshka Transformer (ESMT)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ESMTConfig


class MatryoshkaEmbedding(nn.Module):
    """
    Embedding layer with Matryoshka (nested) truncation support.

    The embedding weights are interpreted as frequency coefficients (DCT-like),
    where index 0 is the DC component (lowest frequency) and index D-1 is the
    highest frequency (Nyquist).
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize MatryoshkaEmbedding.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension (interpreted as spectral coefficients)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Initialize with smaller values for high frequencies to encourage
        # the model to use low frequencies first
        self._init_spectral_weights()

    def _init_spectral_weights(self) -> None:
        """Initialize weights with spectral decay bias."""
        # Standard initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Apply soft decay to high frequencies to bias learning toward low-freq
        with torch.no_grad():
            decay = torch.linspace(1.0, 0.5, self.d_model)
            self.embedding.weight.mul_(decay.unsqueeze(0))

    def forward(
        self, x: torch.Tensor, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with optional bandwidth truncation.

        Args:
            x: Input token indices [batch, seq_len]
            bandwidth_ratio: Fraction of dimensions to keep (0.25 to 1.0)

        Returns:
            Embedded tokens [batch, seq_len, cutoff] where cutoff = d_model * bandwidth_ratio
        """
        # Get full embeddings
        embeddings = self.embedding(x)  # [batch, seq, d_model]

        # Apply truncation if not full bandwidth
        if bandwidth_ratio < 1.0:
            cutoff = int(self.d_model * bandwidth_ratio)
            # Ensure cutoff is at least 1 and aligned to head boundaries if possible
            cutoff = max(1, cutoff)
            embeddings = embeddings[:, :, :cutoff]

        return embeddings


class MatryoshkaPositionalEmbedding(nn.Module):
    """Learned positional embeddings with Matryoshka truncation support."""

    def __init__(self, seq_len: int, d_model: int):
        """
        Initialize positional embeddings.

        Args:
            seq_len: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(seq_len, d_model)

        # Initialize
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self, seq_len: int, bandwidth_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Get positional embeddings for a sequence.

        Args:
            seq_len: Length of the sequence
            bandwidth_ratio: Fraction of dimensions to keep

        Returns:
            Positional embeddings [1, seq_len, cutoff]
        """
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        embeddings = self.embedding(positions).unsqueeze(0)  # [1, seq_len, d_model]

        if bandwidth_ratio < 1.0:
            cutoff = int(self.d_model * bandwidth_ratio)
            cutoff = max(1, cutoff)
            embeddings = embeddings[:, :, :cutoff]

        return embeddings


class SpectralNorm(nn.Module):
    """
    Spectral normalization layer.

    Normalizes the input such that Parseval's energy is bounded.
    Unlike LayerNorm, this does not center the mean.

    x_norm = x / (||x|| + eps)
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize SpectralNorm.

        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input by its L2 norm.

        Args:
            x: Input tensor [..., D]

        Returns:
            Normalized tensor with same shape
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)


class SpectralAttention(nn.Module):
    """
    Attention mechanism with optional frequency-domain structure.

    This is standard multi-head attention with two optional learnable additions:
    1. Frequency weights: per-dimension scaling of Q before dot product
    2. Frequency pooling: cross-frequency interaction in K

    Both are initialized to behave like standard attention (uniform weights,
    center-only pooling) so the model starts equivalent to a standard transformer
    and can learn spectral structure if it helps.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        pool_width: int = 3,
    ):
        """
        Initialize SpectralAttention.

        Args:
            d_model: Model dimension (total across all heads)
            n_heads: Number of attention heads
            seq_len: Maximum sequence length (for causal mask buffer)
            pool_width: Width of frequency pooling kernel (must be odd)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert pool_width % 2 == 1, "pool_width must be odd for symmetric pooling"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pool_width = pool_width
        self.pool_pad = pool_width // 2

        # Standard Q, K, V projections (like transformer attention)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable frequency weights (shared across heads)
        # Initialize to 1.0 (uniform) - model starts as standard attention
        # No constraints - model is free to learn any weighting
        self.freq_weights = nn.Parameter(torch.ones(d_model))

        # Learnable pool weights
        # Initialize to [0, 1, 0] (center only) - model starts as standard attention
        # No constraints - model is free to learn cross-frequency interactions
        pool_init = torch.zeros(pool_width)
        pool_init[pool_width // 2] = 1.0  # Only center weight active
        self.pool_weights = nn.Parameter(pool_init)

        # Causal mask buffer
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(
            1, 1, seq_len, seq_len
        )
        self.register_buffer("causal_mask", causal_mask)
        self.causal_mask: torch.Tensor  # Type hint for the buffer

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_v.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)

    def get_freq_weights(self, cutoff: int | None = None) -> torch.Tensor:
        """
        Get frequency weights for Q scaling.

        Args:
            cutoff: Optional dimension cutoff for elastic inference

        Returns:
            Frequency weights of shape [cutoff] or [d_model]
        """
        if cutoff is None:
            cutoff = self.d_model
        return self.freq_weights[:cutoff]

    def get_pool_weights(self) -> torch.Tensor:
        """
        Get pool weights for cross-frequency K interaction.

        Returns:
            Pool weights of shape [pool_width]
        """
        return self.pool_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spectral attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Get frequency and pool weights
        freq_weights = self.get_freq_weights()  # [d_model]
        pool_weights = self.get_pool_weights()  # [pool_width]

        # Compute attention scores with spectral modifications
        scores = self._compute_spectral_scores(Q, K, freq_weights, pool_weights)

        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Softmax and attend to values
        attn = F.softmax(scores, dim=-1)

        # Reshape V to heads and apply attention
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # attn: [batch, n_heads, seq, seq], V: [batch, n_heads, seq, head_dim]
        out = torch.matmul(attn, V)  # [batch, n_heads, seq, head_dim]

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.W_o(out)

        return out

    def _compute_spectral_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        freq_weights: torch.Tensor,
        pool_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention scores with frequency weighting and pooling.

        The score between tokens t1 and t2 is:
            score[t1, t2] = sum_f sum_o Q[t1,f] * K[t2,f+o] * freq_weight[f] * pool_weight[o]

        This captures:
        - Frequency importance (lower freq weighted more)
        - Cross-frequency similarity (adjacent frequencies interact)

        Args:
            Q: Query tensor [batch, seq, d_model]
            K: Key tensor [batch, seq, d_model]
            freq_weights: Monotonically non-increasing weights [d_model]
            pool_weights: Symmetric pool weights [pool_width]

        Returns:
            Attention scores [batch, n_heads, seq, seq]
        """
        batch, seq_len, d_model = Q.shape

        # Apply frequency weighting to Q
        Q_weighted = Q * freq_weights  # [batch, seq, d_model]

        # Reshape to heads for final output shape
        Q_heads = Q_weighted.view(batch, seq_len, self.n_heads, self.head_dim)
        Q_heads = Q_heads.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

        # Pad K in frequency dimension for pooling
        # K: [batch, seq, d_model] -> pad last dim
        K_padded = F.pad(K, (self.pool_pad, self.pool_pad), mode="constant", value=0)

        # Compute pooled scores by summing over frequency offsets
        scores = torch.zeros(batch, self.n_heads, seq_len, seq_len, device=Q.device)

        for offset_idx, offset in enumerate(range(-self.pool_pad, self.pool_pad + 1)):
            # Extract shifted K
            start = self.pool_pad + offset
            end = start + d_model
            K_shifted = K_padded[:, :, start:end]  # [batch, seq, d_model]

            # Reshape to heads
            K_heads = K_shifted.view(batch, seq_len, self.n_heads, self.head_dim)
            K_heads = K_heads.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

            # Standard dot product attention for this offset
            offset_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))

            # Weight by pool weight for this offset
            scores = scores + pool_weights[offset_idx] * offset_scores

        # Scale by sqrt(head_dim) as in standard attention
        scores = scores / (self.head_dim**0.5)

        return scores

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Forward pass with bandwidth truncation for elastic inference.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active dimensions

        Returns:
            Output tensor [batch, seq_len, cutoff]
        """
        batch, seq_len, _ = x.shape

        # Calculate active heads - must align to head boundaries
        # We use as many complete heads as fit in the cutoff
        n_heads_active = max(1, cutoff // self.head_dim)
        # Keep the original head_dim (don't change it based on cutoff)
        head_dim_active = self.head_dim
        # The actual dimension we'll use for attention (aligned to head boundary)
        aligned_dim = n_heads_active * head_dim_active

        # If cutoff is smaller than one head, use cutoff as a single "head"
        if cutoff < self.head_dim:
            n_heads_active = 1
            head_dim_active = cutoff
            aligned_dim = cutoff

        # Slice projection weights to aligned dimensions
        Q = F.linear(x, self.W_q.weight[:aligned_dim, :cutoff])
        K = F.linear(x, self.W_k.weight[:aligned_dim, :cutoff])
        V = F.linear(x, self.W_v.weight[:aligned_dim, :cutoff])

        # Get sliced frequency and pool weights (for aligned_dim)
        freq_weights = self.get_freq_weights(aligned_dim)
        pool_weights = self.get_pool_weights()

        # Compute spectral scores (sliced version)
        scores = self._compute_spectral_scores_sliced(
            Q, K, freq_weights, pool_weights, n_heads_active, head_dim_active
        )

        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Softmax and attend
        attn = F.softmax(scores, dim=-1)

        # Reshape V to heads
        V = V.view(batch, seq_len, n_heads_active, head_dim_active).transpose(1, 2)
        out = torch.matmul(attn, V)

        # Reshape back to aligned_dim
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, aligned_dim)

        # Project output back to cutoff dimensions
        out = F.linear(out, self.W_o.weight[:cutoff, :aligned_dim])

        return out

    def _compute_spectral_scores_sliced(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        freq_weights: torch.Tensor,
        pool_weights: torch.Tensor,
        n_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        Compute spectral attention scores with sliced dimensions.

        Args:
            Q: Query tensor [batch, seq, aligned_dim]
            K: Key tensor [batch, seq, aligned_dim]
            freq_weights: Sliced frequency weights [aligned_dim]
            pool_weights: Pool weights [pool_width]
            n_heads: Number of active heads
            head_dim: Active head dimension

        Returns:
            Attention scores [batch, n_heads, seq, seq]
        """
        batch, seq_len, aligned_dim = Q.shape

        # Apply frequency weighting
        Q_weighted = Q * freq_weights

        # Reshape to heads
        Q_heads = Q_weighted.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        # Pad K in frequency dimension for pooling
        K_padded = F.pad(K, (self.pool_pad, self.pool_pad), mode="constant", value=0)

        # Accumulate scores across frequency offsets
        scores = torch.zeros(batch, n_heads, seq_len, seq_len, device=Q.device)

        for offset_idx, offset in enumerate(range(-self.pool_pad, self.pool_pad + 1)):
            start = self.pool_pad + offset
            end = start + aligned_dim
            K_shifted = K_padded[:, :, start:end]

            K_heads = K_shifted.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            offset_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
            scores = scores + pool_weights[offset_idx] * offset_scores

        scores = scores / (head_dim**0.5)
        return scores

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Regularization loss (disabled - returns 0).

        The spectral structure is now optional and learnable, so we don't
        penalize any particular configuration. The model is free to learn
        standard attention (uniform weights) or spectral attention as needed.

        Returns:
            Zero tensor (no regularization)
        """
        return torch.tensor(0.0, device=self.freq_weights.device)


class SpectralGate(nn.Module):
    """
    Multi-head spectral gating module.

    Acts as a learnable "graphic equalizer" that amplifies or dampens
    specific frequency bands via element-wise multiplication.
    """

    def __init__(self, n_heads: int, head_dim: int):
        """
        Initialize SpectralGate.

        Args:
            n_heads: Number of spectral filter heads
            head_dim: Dimension per head
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim

        # Learnable filter per head: [n_heads, head_dim]
        self.filter = nn.Parameter(torch.ones(n_heads, head_dim))

        # Initialize filters close to identity (1.0) with small perturbations
        nn.init.normal_(self.filter, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral gating.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Gated tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Reshape to heads: [batch, seq_len, n_heads, head_dim]
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)

        # Element-wise multiplication with filter (broadcast over batch and seq)
        x = x * self.filter  # filter broadcasts: [1, 1, n_heads, head_dim]

        # Reshape back: [batch, seq_len, d_model]
        return x.view(batch, seq_len, d_model)

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Apply spectral gating with bandwidth truncation.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active dimensions

        Returns:
            Gated tensor [batch, seq_len, cutoff]
        """
        batch, seq_len, _ = x.shape

        # Calculate how many complete heads fit in cutoff
        n_heads_active = cutoff // self.head_dim
        remainder = cutoff % self.head_dim

        if n_heads_active == 0:
            # Less than one full head, use partial first head
            filter_sliced = self.filter[0, :cutoff]
            return x * filter_sliced

        # Full heads portion
        if remainder == 0:
            # Exact head boundary
            x = x.view(batch, seq_len, n_heads_active, self.head_dim)
            filter_sliced = self.filter[:n_heads_active]
            x = x * filter_sliced
            return x.view(batch, seq_len, cutoff)
        else:
            # Partial head at the end
            full_dim = n_heads_active * self.head_dim

            # Process full heads
            x_full = x[:, :, :full_dim].view(batch, seq_len, n_heads_active, self.head_dim)
            filter_full = self.filter[:n_heads_active]
            x_full = (x_full * filter_full).view(batch, seq_len, full_dim)

            # Process partial head
            x_partial = x[:, :, full_dim:]
            filter_partial = self.filter[n_heads_active, :remainder]
            x_partial = x_partial * filter_partial

            return torch.cat([x_full, x_partial], dim=-1)


class SpectralMLP(nn.Module):
    """
    Feed-forward network (mixer) for cross-band frequency mixing.

    Standard MLP with expansion: Linear(D, 4D) -> SiLU -> Linear(4D, D)
    """

    def __init__(self, d_model: int, mlp_ratio: int = 4):
        """
        Initialize SpectralMLP.

        Args:
            d_model: Input/output dimension
            mlp_ratio: Expansion ratio for hidden layer
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * mlp_ratio

        self.fc1 = nn.Linear(d_model, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Forward pass with sliced weights for elastic inference.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active input dimensions

        Returns:
            Output tensor [batch, seq_len, cutoff]
        """
        hidden_cutoff = cutoff * (self.hidden_dim // self.d_model)

        # fc1: [cutoff] -> [hidden_cutoff]
        x = F.linear(
            x,
            self.fc1.weight[:hidden_cutoff, :cutoff],
            self.fc1.bias[:hidden_cutoff] if self.fc1.bias is not None else None,
        )
        x = self.act(x)

        # fc2: [hidden_cutoff] -> [cutoff]
        x = F.linear(
            x,
            self.fc2.weight[:cutoff, :hidden_cutoff],
            self.fc2.bias[:cutoff] if self.fc2.bias is not None else None,
        )
        return x


class SpectralGatedLayer(nn.Module):
    """
    Single Spectral Gated Layer (SGL) block.

    Architecture: SpectralAttention -> SpectralGate -> SpectralMLP
    This is analogous to standard transformer: Attention -> MLP

    1. SpectralAttention (cross-token mixing with frequency-aware similarity)
    2. SpectralGate (frequency filtering / "graphic equalizer")
    3. SpectralMLP (cross-band mixing)
    4. SpectralNorm + Residual connections
    """

    def __init__(self, config: ESMTConfig):
        """
        Initialize SpectralGatedLayer.

        Args:
            config: ESMT configuration
        """
        super().__init__()
        self.config = config

        # Cross-token mixing via spectral attention
        self.attn = SpectralAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
            pool_width=config.pool_width,
        )

        # Spectral gating (multi-head)
        self.gate = SpectralGate(config.n_heads, config.head_dim)

        # Cross-band mixer (MLP)
        self.mlp = SpectralMLP(config.d_model, config.mlp_ratio)

        # Normalization
        self.norm1 = SpectralNorm(config.eps)
        self.norm2 = SpectralNorm(config.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Cross-token mixing via spectral attention
        h = self.attn(x)

        # Spectral gating
        h = self.gate(h)

        # First residual + norm
        x = self.norm1(x + h)

        # Cross-band mixing (MLP)
        h = self.mlp(x)

        # Second residual + norm
        x = self.norm2(x + h)

        return x

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Forward pass with bandwidth truncation.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active dimensions

        Returns:
            Output tensor [batch, seq_len, cutoff]
        """
        # Cross-token mixing (sliced)
        h = self.attn.forward_sliced(x, cutoff)

        # Spectral gating (sliced)
        h = self.gate.forward_sliced(h, cutoff)

        # First residual + norm
        x = self.norm1(x + h)

        # Cross-band mixing (sliced)
        h = self.mlp.forward_sliced(x, cutoff)

        # Second residual + norm
        x = self.norm2(x + h)

        return x

    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from the attention layer."""
        return self.attn.get_regularization_loss()
