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
    dct_token_emb: bool = True   # DCT token embeddings
    dct_pos_emb: bool = True     # DCT positional embeddings
    dct_lm_head: bool = True     # DCT output projection (lm_head)
    
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
    n_heads: int = 8   # Number of attention heads
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
    eval_interval: int = 2000                         # Evaluate every N steps
    eval_bandwidths: tuple = (0.25, 0.50, 0.75, 1.0)  # Bandwidth ratios for spectral sweep

    # Reproducibility
    seed: int = 42  # Random seed

    # Logging
    log_dir: str = "runs"  # TensorBoard log directory
    checkpoint_dir: str = "checkpoints"  # Model checkpoint directory

    # Device
    compile_model: bool = True  # Use torch.compile for speedup
