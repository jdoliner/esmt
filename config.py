"""Configuration dataclasses for ESMT and training."""

from dataclasses import dataclass


@dataclass
class ESMTConfig:
    """Configuration for the Elastic Spectral Matryoshka Transformer."""

    # Default config targets ~6-7M parameters to fit within 5-10M budget
    d_model: int = 64  # Hidden dimension (must be even, divisible by n_heads)
    n_layers: int = 6  # Number of SpectralGatedLayers
    n_heads: int = 8  # Number of spectral filter heads
    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    seq_len: int = 512  # Maximum sequence length
    mlp_ratio: int = 4  # MLP expansion ratio (d_model -> mlp_ratio * d_model)
    pool_width: int = 3  # Frequency pooling width for spectral attention (must be odd)
    dropout: float = 0.0  # Dropout rate (0 for this experiment)
    eps: float = 1e-6  # Epsilon for SpectralNorm

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.pool_width % 2 == 1, "pool_width must be odd for symmetric pooling"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


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

    # Matryoshka training
    lambda_trunc: float = 1.0  # Weight for truncated loss
    min_bandwidth: float = 0.25  # Minimum bandwidth ratio (25%)
    max_bandwidth: float = 1.0  # Maximum bandwidth ratio (100%)

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
