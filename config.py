"""Configuration dataclasses for ESMT and training."""

from dataclasses import dataclass


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
        return ", ".join(features) if features else "Baseline (no experiments)"


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
    eval_interval: int = 500  # Evaluate every N steps
    eval_bandwidths: tuple = (0.25, 0.50, 0.75, 1.0)  # Bandwidth ratios for spectral sweep

    # Reproducibility
    seed: int = 42  # Random seed

    # Logging
    log_dir: str = "runs"  # TensorBoard log directory
    checkpoint_dir: str = "checkpoints"  # Model checkpoint directory

    # Device
    compile_model: bool = True  # Use torch.compile for speedup
