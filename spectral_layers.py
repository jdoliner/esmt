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
    Multi-head spectral gating module with input-dependent filtering.

    Computes dynamic filter weights based on the input, allowing the model
    to adaptively amplify or dampen specific frequency bands per-token.
    
    This is analogous to how attention computes input-dependent weights,
    but operates on frequency bands rather than token positions.
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

        # Linear projection to compute input-dependent filter weights
        # Input: [batch, seq, d_model] -> Output: [batch, seq, d_model]
        self.filter_proj = nn.Linear(self.d_model, self.d_model)
        
        # Initialize to produce values near 1.0 (identity-like behavior initially)
        # Set bias to ~2.0 so sigmoid(2.0) ≈ 0.88, close to the ~0.85 the static 
        # filters were learning. Weights near zero so initial output is bias-dominated.
        nn.init.normal_(self.filter_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.filter_proj.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply input-dependent spectral gating.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Gated tensor [batch, seq_len, d_model]
        """
        # Compute input-dependent filter weights
        # sigmoid keeps weights positive and bounded [0, 1]
        # We scale by 2 to allow weights in range [0, 2] for amplification
        filter_weights = 2.0 * torch.sigmoid(self.filter_proj(x))
        
        # Element-wise multiplication (input-dependent gating)
        return x * filter_weights

    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Apply input-dependent spectral gating with bandwidth truncation.

        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active dimensions

        Returns:
            Gated tensor [batch, seq_len, cutoff]
        """
        # Slice the projection weights for elastic inference
        weight_sliced = self.filter_proj.weight[:cutoff, :cutoff]
        bias_sliced = self.filter_proj.bias[:cutoff]
        
        # Compute input-dependent filter weights with sliced projection
        filter_logits = F.linear(x, weight_sliced, bias_sliced)
        filter_weights = 2.0 * torch.sigmoid(filter_logits)
        
        # Element-wise multiplication
        return x * filter_weights


class DualSpectralAttention(nn.Module):
    """
    Dual-path spectral attention combining cross-frequency and frequency-shift similarity.
    
    Path 1 (Cross-frequency): Learns which query frequencies should attend to which 
    key frequencies via a learned mixing matrix. Captures structured f->g relationships.
    
    Path 2 (Spectral convolution): Computes frequency-shift similarity by correlating
    spectra at different shifts. Captures patterns where related tokens have similar
    spectral shapes but shifted in frequency (e.g., morphological variants).
    
    Both paths operate directly on frequency-domain representations (no FFT needed
    since embeddings are already spectral coefficients).
    """
    
    def __init__(self, d_model: int, n_heads: int, max_shift: int | None = None):
        """
        Initialize DualSpectralAttention.
        
        Args:
            d_model: Model dimension (number of frequency bands)
            n_heads: Number of attention heads
            max_shift: Maximum frequency shift to consider (default: head_dim // 2)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_shift = max_shift if max_shift is not None else max(1, self.head_dim // 2)
        
        # Q, K, V projections (shared between both paths)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Path 1: Cross-frequency mixing matrix per head
        # W_cross[h, f, g] = importance of query_freq_f attending to key_freq_g
        self.W_cross = nn.Parameter(torch.zeros(n_heads, self.head_dim, self.head_dim))
        # Initialize as identity (same-frequency attention) with small noise
        for h in range(n_heads):
            nn.init.eye_(self.W_cross.data[h])
        self.W_cross.data += torch.randn_like(self.W_cross) * 0.01
        
        # Path 2: Learned weights for each frequency shift
        # shift_weights[h, tau] = importance of shift tau for head h
        self.shift_weights = nn.Parameter(torch.zeros(n_heads, self.max_shift))
        # Initialize with exponential decay (small shifts more important)
        for h in range(n_heads):
            self.shift_weights.data[h] = torch.exp(
                -torch.arange(self.max_shift).float() / (self.max_shift / 4)
            )
        
        # Learned gate to combine paths (one per head for flexibility)
        # Initialized to 0.5 (equal weighting)
        self.gate = nn.Parameter(torch.zeros(n_heads))
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5
        
    def _cross_frequency_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-frequency attention scores.
        
        Q: [batch, seq_q, heads, head_dim]
        K: [batch, seq_k, heads, head_dim]
        
        Returns: [batch, heads, seq_q, seq_k]
        """
        # Q @ W_cross: [batch, seq_q, heads, head_dim]
        Q_transformed = torch.einsum('bqhf,hfg->bqhg', Q, self.W_cross)
        
        # (Q @ W_cross) @ K^T: [batch, heads, seq_q, seq_k]
        scores = torch.einsum('bqhd,bkhd->bhqk', Q_transformed, K)
        
        return scores
    
    def _spectral_conv_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-shift similarity scores.
        
        For each (query, key) pair, computes correlation at multiple frequency shifts
        and returns weighted sum based on learned shift importance.
        
        Q: [batch, seq_q, heads, head_dim]
        K: [batch, seq_k, heads, head_dim]
        
        Returns: [batch, heads, seq_q, seq_k]
        """
        batch, seq_q, heads, freq = Q.shape
        seq_k = K.shape[1]
        
        # Rearrange for easier manipulation
        Q = Q.permute(0, 2, 1, 3)  # [batch, heads, seq_q, freq]
        K = K.permute(0, 2, 1, 3)  # [batch, heads, seq_k, freq]
        
        # Build shifted versions of K and compute dot products
        # K_shifted[tau] has K's frequencies shifted left by tau
        # (i.e., K_shifted[..., f] = K[..., f + tau])
        
        dots = []
        for tau in range(self.max_shift):
            if tau == 0:
                # No shift: standard dot product
                dot = torch.einsum('bhqf,bhkf->bhqk', Q, K)
            else:
                # Shift: Q[0:freq-tau] dot K[tau:freq]
                overlap = freq - tau
                Q_slice = Q[..., :overlap]
                K_slice = K[..., tau:tau + overlap]
                dot = torch.einsum('bhqf,bhkf->bhqk', Q_slice, K_slice)
            dots.append(dot)
        
        # Stack shifts: [batch, heads, seq_q, seq_k, max_shift]
        dots = torch.stack(dots, dim=-1)
        
        # Weight by learned shift importance and sum
        scores = torch.einsum('bhqks,hs->bhqk', dots, self.shift_weights)
        
        return scores
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual-path spectral attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.head_dim)
        
        # Compute scores from both paths
        scores_cross = self._cross_frequency_scores(Q, K)  # [batch, heads, seq_q, seq_k]
        scores_conv = self._spectral_conv_scores(Q, K)     # [batch, heads, seq_q, seq_k]
        
        # Combine paths with learned gating (per head)
        gate = torch.sigmoid(self.gate)  # [heads]
        gate = gate.view(1, self.n_heads, 1, 1)
        scores = gate * scores_cross + (1 - gate) * scores_conv
        
        # Scale
        scores = scores * self.scale
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax over keys
        attn = torch.softmax(scores, dim=-1)
        
        # Aggregate values
        # attn: [batch, heads, seq_q, seq_k]
        # V: [batch, seq_k, heads, head_dim]
        V_permuted = V.permute(0, 2, 1, 3)  # [batch, heads, seq_k, head_dim]
        out = torch.einsum('bhqk,bhkd->bhqd', attn, V_permuted)
        
        # Reshape and output projection
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return self.W_o(out)
    
    def forward_sliced(self, x: torch.Tensor, cutoff: int) -> torch.Tensor:
        """
        Forward pass with bandwidth truncation for elastic inference.
        
        Args:
            x: Input tensor [batch, seq_len, cutoff]
            cutoff: Number of active frequency dimensions
            
        Returns:
            Output tensor [batch, seq_len, cutoff]
        """
        batch, seq_len, _ = x.shape
        
        # Calculate active heads and head_dim for this cutoff
        n_heads_active = cutoff // self.head_dim
        if n_heads_active == 0:
            n_heads_active = 1
            head_dim_active = cutoff
        else:
            head_dim_active = self.head_dim
        
        active_dim = n_heads_active * head_dim_active
        
        # Slice projection weights
        Q = F.linear(x, self.W_q.weight[:active_dim, :cutoff], 
                     self.W_q.bias[:active_dim] if self.W_q.bias is not None else None)
        K = F.linear(x, self.W_k.weight[:active_dim, :cutoff],
                     self.W_k.bias[:active_dim] if self.W_k.bias is not None else None)
        V = F.linear(x, self.W_v.weight[:active_dim, :cutoff],
                     self.W_v.bias[:active_dim] if self.W_v.bias is not None else None)
        
        Q = Q.view(batch, seq_len, n_heads_active, head_dim_active)
        K = K.view(batch, seq_len, n_heads_active, head_dim_active)
        V = V.view(batch, seq_len, n_heads_active, head_dim_active)
        
        # Cross-frequency scores with sliced W_cross
        W_cross_sliced = self.W_cross[:n_heads_active, :head_dim_active, :head_dim_active]
        Q_transformed = torch.einsum('bqhf,hfg->bqhg', Q, W_cross_sliced)
        scores_cross = torch.einsum('bqhd,bkhd->bhqk', Q_transformed, K)
        
        # Spectral conv scores with sliced operations
        max_shift_active = min(self.max_shift, head_dim_active)
        Q_perm = Q.permute(0, 2, 1, 3)
        K_perm = K.permute(0, 2, 1, 3)
        
        dots = []
        for tau in range(max_shift_active):
            if tau == 0:
                dot = torch.einsum('bhqf,bhkf->bhqk', Q_perm, K_perm)
            else:
                overlap = head_dim_active - tau
                dot = torch.einsum('bhqf,bhkf->bhqk', 
                                   Q_perm[..., :overlap], K_perm[..., tau:tau + overlap])
            dots.append(dot)
        
        dots = torch.stack(dots, dim=-1)
        shift_weights_sliced = self.shift_weights[:n_heads_active, :max_shift_active]
        scores_conv = torch.einsum('bhqks,hs->bhqk', dots, shift_weights_sliced)
        
        # Combine paths
        gate = torch.sigmoid(self.gate[:n_heads_active]).view(1, n_heads_active, 1, 1)
        scores = gate * scores_cross + (1 - gate) * scores_conv
        
        # Scale
        scale = head_dim_active ** -0.5
        scores = scores * scale
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and aggregate
        attn = torch.softmax(scores, dim=-1)
        V_perm = V.permute(0, 2, 1, 3)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, V_perm)
        
        # Output projection with sliced weights
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, active_dim)
        
        # Handle case where active_dim < cutoff (pad with zeros or project appropriately)
        if active_dim < cutoff:
            out = F.pad(out, (0, cutoff - active_dim))
        
        out = F.linear(out, self.W_o.weight[:cutoff, :cutoff],
                       self.W_o.bias[:cutoff] if self.W_o.bias is not None else None)
        
        return out


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

    Architecture:
    1. DualSpectralAttention (sequence mixing via spectral attention)
    2. SpectralMLP (cross-band frequency mixing)
    3. SpectralNorm + Residual connections
    
    The DualSpectralAttention replaces the previous CausalConv1d + SpectralGate,
    providing both sequence mixing and frequency-aware attention in a single operation.
    """

    def __init__(self, config: ESMTConfig):
        """
        Initialize SpectralGatedLayer.

        Args:
            config: ESMT configuration
        """
        super().__init__()
        self.config = config

        # Dual-path spectral attention (handles sequence mixing with spectral awareness)
        self.attn = DualSpectralAttention(config.d_model, config.n_heads)

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
        # Spectral attention
        h = self.attn(x)

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
        # Spectral attention (sliced)
        h = self.attn.forward_sliced(x, cutoff)

        # First residual + norm
        x = self.norm1(x + h)

        # Cross-band mixing (sliced)
        h = self.mlp.forward_sliced(x, cutoff)

        # Second residual + norm
        x = self.norm2(x + h)

        return x
