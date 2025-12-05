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


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution for time-mixing.

    Applies convolution along the sequence dimension with left-padding only
    to ensure causality (token t cannot see future tokens).
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        """
        Initialize CausalConv1d.

        Args:
            d_model: Number of input/output channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Left padding for causality

        # Depthwise convolution (each channel convolved independently)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,  # We'll handle padding manually
            groups=d_model,  # Depthwise
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal convolution.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Convolved tensor [batch, seq_len, d_model]
        """
        # Transpose for conv1d: [batch, d_model, seq_len]
        x = x.transpose(1, 2)

        # Apply left-only padding for causality
        x = F.pad(x, (self.padding, 0))

        # Convolve
        x = self.conv(x)

        # Transpose back: [batch, seq_len, d_model]
        return x.transpose(1, 2)

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Apply causal convolution with sliced weights.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active dimensions

        Returns:
            Convolved tensor [batch, seq_len, cutoff]
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)  # [batch, cutoff, seq_len]

        # Apply left-only padding
        x = F.pad(x, (self.padding, 0))

        # Use functional conv with sliced weights
        weight = self.conv.weight[:cutoff, :, :]  # [cutoff, 1, kernel_size]
        bias = self.conv.bias[:cutoff] if self.conv.bias is not None else None

        x = F.conv1d(x, weight, bias, groups=cutoff)

        return x.transpose(1, 2)  # [batch, seq_len, cutoff]


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

    Replaces the standard Self-Attention + MLP block with:
    1. CausalConv1d (time-mixing)
    2. SpectralGate (frequency filtering)
    3. SpectralMLP (cross-band mixing)
    4. SpectralNorm
    5. Residual connection
    """

    def __init__(self, config: ESMTConfig):
        """
        Initialize SpectralGatedLayer.

        Args:
            config: ESMT configuration
        """
        super().__init__()
        self.config = config

        # Time-mixing via causal convolution
        self.conv = CausalConv1d(config.d_model, config.conv_kernel_size)

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
        # Time-mixing
        h = self.conv(x)

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
        # Time-mixing (sliced)
        h = self.conv.forward_sliced(x, cutoff)

        # Spectral gating (sliced)
        h = self.gate.forward_sliced(h, cutoff)

        # First residual + norm
        x = self.norm1(x + h)

        # Cross-band mixing (sliced)
        h = self.mlp.forward_sliced(x, cutoff)

        # Second residual + norm
        x = self.norm2(x + h)

        return x
