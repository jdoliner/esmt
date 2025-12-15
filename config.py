"""Configuration dataclasses for ESMT and training."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ESMTConfig:
    """Configuration for the Elastic Spectral Matryoshka Transformer."""

    # Default config targets ~6-7M parameters to fit within 5-10M budget
    d_model: int = 64  # Hidden dimension (must be even, divisible by n_heads)
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    seq_len: int = 512  # Maximum sequence length
    mlp_ratio: int = 4  # MLP expansion ratio (d_model -> mlp_ratio * d_model)
    dropout: float = 0.0  # Dropout rate (0 for this experiment)
    eps: float = 1e-6  # LayerNorm epsilon

    # ===========================================================================
    # Experimental Spectral Features
    # ===========================================================================

    # Spectral Blur: Replace dense feature mixing with local 1D convolution
    # Forces the model to organize information smoothly along the feature dimension
    use_spectral_blur: bool = False
    blur_kernel_size: int = 3  # Kernel size for feature-axis convolution (3, 5, or 7)

    # Harmonic Mixing: Connect features at octave intervals (f <-> 2f <-> 4f)
    # Enables multi-scale reasoning by wiring different "resolution levels"
    use_harmonic_mixing: bool = False
    n_octaves: int = 3  # Number of octave levels to connect (1, 2, or 3)

    # ===========================================================================
    # Spectral Initialization from Pretrained NanoGPT
    # ===========================================================================

    # Initialize embeddings from a pretrained NanoGPT checkpoint
    # The embeddings are DCT-transformed to create an explicitly spectral basis
    spectral_init_checkpoint: str | None = None  # Path to NanoGPT checkpoint

    # Which components to DCT-transform (only used when spectral_init_checkpoint is set)
    dct_token_emb: bool = True  # DCT token embeddings
    dct_pos_emb: bool = True  # DCT positional embeddings
    dct_lm_head: bool = True  # DCT output projection (lm_head)

    # Whether to freeze DCT'd embeddings during training
    # If True, only the transformer layers train; embeddings stay fixed
    freeze_embeddings: bool = False

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.blur_kernel_size % 2 == 1, "blur_kernel_size must be odd"
        assert 1 <= self.n_octaves <= 4, "n_octaves must be between 1 and 4"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def experiment_summary(self) -> str:
        """Return a summary of enabled experimental features."""
        features = []
        if self.use_spectral_blur:
            features.append(f"SpectralBlur(k={self.blur_kernel_size})")
        if self.use_harmonic_mixing:
            features.append(f"Harmonic(oct={self.n_octaves})")
        if self.spectral_init_checkpoint:
            dct_parts = []
            if self.dct_token_emb:
                dct_parts.append("tok")
            if self.dct_pos_emb:
                dct_parts.append("pos")
            if self.dct_lm_head:
                dct_parts.append("head")
            freeze_str = ",frozen" if self.freeze_embeddings else ""
            features.append(f"SpectralInit({'+'.join(dct_parts)}{freeze_str})")
        return ", ".join(features) if features else "Baseline (no experiments)"


@dataclass
class ComplexESMTConfig:
    """Configuration for the Complex Spectral Transformer.

    This config is for the complex-valued variant of ESMT that uses:
    - Complex embeddings and positional encoding (shift theorem)
    - Complex attention with magnitude-based or phase-aware softmax
    - Complex FFN with ModReLU activation
    - Multiplicative or additive residual connections
    """

    # Model dimensions
    d_model: int = 64  # Hidden dimension (must be even for complex pairs)
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    seq_len: int = 512  # Maximum sequence length
    mlp_ratio: int = 4  # MLP expansion ratio
    dropout: float = 0.0  # Dropout rate
    eps: float = 1e-6  # LayerNorm epsilon

    # ===========================================================================
    # Complex-Specific Options
    # ===========================================================================

    # Attention mode:
    # - "magnitude": Attention weights = |Q @ K†| / sqrt(d)
    # - "phase_aware": Attention weights = |Q @ K†| * cos(angle(Q @ K†)) / sqrt(d)
    attention_mode: Literal["magnitude", "phase_aware"] = "magnitude"

    # LayerNorm mode:
    # - "magnitude": Normalize magnitude only, preserve relative phases
    # - "split": Normalize real and imaginary parts separately (Trabelsi et al.)
    layernorm_mode: Literal["magnitude", "split"] = "magnitude"

    # Residual connection mode:
    # - "multiplicative": x * (1 + f(x)) - more natural for complex numbers
    # - "additive": x + f(x) - standard transformer residual
    residual_mode: Literal["multiplicative", "additive"] = "multiplicative"

    # Positional encoding base frequency (higher = slower rotation for low dims)
    pos_encoding_base: float = 10000.0

    # Maximum sequence length for positional encoding (can be > seq_len for OOD testing)
    max_pos_len: int = 8192

    # ===========================================================================
    # Initialization from Pretrained NanoGPT
    # ===========================================================================

    # Path to pretrained NanoGPT checkpoint for FFT initialization
    # If set, embeddings are initialized by applying FFT to NanoGPT embeddings
    fft_init_checkpoint: str | None = None

    # Whether to freeze FFT'd embeddings during training
    freeze_embeddings: bool = False

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even for complex pairs"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def experiment_summary(self) -> str:
        """Return a summary of configuration choices."""
        parts = [
            f"attn={self.attention_mode}",
            f"ln={self.layernorm_mode}",
            f"res={self.residual_mode}",
        ]
        if self.fft_init_checkpoint:
            freeze_str = ",frozen" if self.freeze_embeddings else ""
            parts.append(f"FFT_init{freeze_str}")
        return f"ComplexESMT({', '.join(parts)})"


@dataclass
class SATConfig:
    """Configuration for the Spectral-Augmented Transformer (SAT).

    This model combines a Fourier Neural Operator (FNO) "world model" with a
    standard Transformer decoder. The FNO processes the input in the frequency
    domain and conditions the Transformer via Adaptive Layer Norm (AdaLN) or
    cross-attention.

    Key design choices:
    - Complex numbers represented as bfloat16 pairs (real, imag) stacked on last dim
    - Cumulative FFT for causal spectral representation
    - FNO captures global dynamics, Transformer handles local token interactions
    - Two integration modes: AdaLN (global modulation) or cross-attention (per-position)
    """

    # ===========================================================================
    # Transformer (Token Model) Configuration
    # ===========================================================================
    d_model: int = 64  # Transformer hidden dimension
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    seq_len: int = 512  # Maximum sequence length (should be power of 2 for FFT)
    mlp_ratio: int = 4  # MLP expansion ratio
    dropout: float = 0.0  # Dropout rate
    eps: float = 1e-6  # LayerNorm epsilon

    # ===========================================================================
    # Spectral Stream (FNO World Model) Configuration
    # ===========================================================================
    d_spectral: int | None = None  # Spectral dimension (default: d_model // 4)
    n_fno_layers: int | None = None  # Number of FNO layers (default: n_layers)
    k_max: int | None = None  # Number of frequency modes (default: seq_len // 8)

    # FNO activation function:
    # - "modrelu": Hard threshold (can cause dead neurons)
    # - "modsoftplus": Soft threshold (gradients always flow) [default]
    # - "modelu": ELU-based (gradients always flow)
    fno_activation: Literal["modrelu", "modsoftplus", "modelu"] = "modsoftplus"

    # Output gating for FNO blocks: prevents early explosion
    fno_output_gate: bool = True
    fno_gate_init: float = 0.1  # Initial gate value (sigmoid applied)

    # Cumulative FFT normalization:
    # - "ortho": Standard 1/sqrt(N) normalization
    # - "position": Per-position 1/sqrt(t+1) to keep magnitudes stable [default]
    fft_normalization: Literal["ortho", "position"] = "position"

    # Spectral magnitude clipping: prevents extreme outliers
    # Set to None to disable, or a float like 5.0 to clip magnitudes
    spectral_clip_magnitude: float | None = 5.0

    # Spectral layer normalization after FNO blocks
    # - None: No normalization
    # - "rms": RMS normalization (simpler, recommended)
    # - "magnitude": Full LayerNorm on magnitude
    spectral_layernorm: Literal["rms", "magnitude"] | None = "rms"

    # ===========================================================================
    # AdaLN Stability
    # ===========================================================================
    # Constrain gamma to stay close to 1 to prevent drift
    # gamma = 1 + adaln_gamma_scale * tanh((gamma_raw - 1) / adaln_gamma_scale)
    # This keeps gamma in range [1 - scale, 1 + scale]
    # Set to 0 to disable constraint (use raw gamma)
    adaln_gamma_scale: float = 0.1

    # ===========================================================================
    # Integration Mode
    # ===========================================================================
    # How the spectral stream conditions the transformer:
    # - "adaln": Global modulation via Adaptive Layer Norm (γ, β per layer)
    # - "cross_attention": Per-position attention to spectral tokens
    # - "both": Use both AdaLN and cross-attention
    integration_mode: Literal["adaln", "cross_attention", "both"] = "adaln"

    # ===========================================================================
    # Auxiliary Loss Configuration
    # ===========================================================================
    # Weight for auxiliary FNO prediction loss
    aux_loss_weight: float = 1.0
    # Minimum aux loss weight (after decay)
    aux_loss_weight_min: float = 0.1
    # Fraction of training for warmup (aux_loss_weight = 1.0)
    aux_loss_warmup_frac: float = 0.1
    # Fraction of training for decay (cosine decay from 1.0 to min)
    aux_loss_decay_end_frac: float = 0.8

    def __post_init__(self):
        # Set defaults based on transformer config
        if self.d_spectral is None:
            self.d_spectral = self.d_model // 4
        if self.n_fno_layers is None:
            self.n_fno_layers = self.n_layers
        if self.k_max is None:
            self.k_max = self.seq_len // 8

        # Validations
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_spectral > 0, "d_spectral must be positive"
        assert self.n_fno_layers > 0, "n_fno_layers must be positive"
        assert self.k_max > 0, "k_max must be positive"
        # seq_len should be power of 2 for efficient FFT
        assert self.seq_len & (self.seq_len - 1) == 0, "seq_len should be power of 2"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def get_aux_loss_weight(self, progress: float) -> float:
        """
        Get auxiliary loss weight based on training progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Auxiliary loss weight
        """
        if progress < self.aux_loss_warmup_frac:
            # Warmup: hold at max weight
            return self.aux_loss_weight
        elif progress < self.aux_loss_decay_end_frac:
            # Decay: cosine from max to min
            decay_progress = (progress - self.aux_loss_warmup_frac) / (
                self.aux_loss_decay_end_frac - self.aux_loss_warmup_frac
            )
            import math

            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return (
                self.aux_loss_weight_min
                + (self.aux_loss_weight - self.aux_loss_weight_min) * cosine_decay
            )
        else:
            # Terminal: hold at min weight
            return self.aux_loss_weight_min

    def experiment_summary(self) -> str:
        """Return a summary of configuration."""
        return (
            f"SAT(d={self.d_model}, d_spec={self.d_spectral}, "
            f"layers={self.n_layers}, fno={self.n_fno_layers}, "
            f"k_max={self.k_max}, mode={self.integration_mode})"
        )


@dataclass
class NanoGPTConfig:
    """Configuration for the NanoGPT baseline."""

    # d_model will be adjusted by match_parameter_counts to match ESMT
    d_model: int = 64  # Will be adjusted to match ESMT parameter count
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    seq_len: int = 512  # Maximum sequence length
    mlp_ratio: int = 4  # MLP expansion ratio
    dropout: float = 0.0  # Dropout rate (0 for this experiment)
    eps: float = 1e-6  # LayerNorm epsilon

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    batch_size: int = 128  # Batch size for H100
    lr: float = 3e-4  # Learning rate
    weight_decay: float = 0.1  # AdamW weight decay
    beta1: float = 0.9  # AdamW beta1
    beta2: float = 0.95  # AdamW beta2
    grad_clip: float = 1.0  # Gradient clipping norm
    warmup_steps: int = 100  # Linear warmup steps
    epochs: int = 1  # Number of training epochs

    # Matryoshka training (weighted stochastic - single forward pass per step)
    # Each step samples ONE bandwidth: full (100%) with prob=full_bandwidth_prob,
    # otherwise uniform random from [min_bandwidth, max_bandwidth]
    min_bandwidth: float = 0.25  # Minimum bandwidth ratio (25%)
    max_bandwidth: float = 1.0  # Maximum bandwidth ratio (100%)
    full_bandwidth_prob: float = 0.5  # Probability of training at full bandwidth

    # Evaluation
    eval_interval: int = 2000  # Evaluate every N steps
    eval_bandwidths: tuple = (0.25, 0.50, 0.75, 1.0)  # Bandwidth ratios for spectral sweep

    # Reproducibility
    seed: int = 42  # Random seed

    # Logging
    log_dir: str = "runs"  # TensorBoard log directory
    checkpoint_dir: str = "checkpoints"  # Model checkpoint directory

    # Device
    compile_model: bool = True  # Use torch.compile for speedup
