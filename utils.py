"""Utility functions for ESMT training and evaluation."""

import os

# Suppress tokenizers parallelism warning when using multiprocessing DataLoaders
# Must be set before importing transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


class TinyStoriesDataset(Dataset):
    """Dataset wrapper for TinyStories."""

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 512,
        tokenizer: GPT2TokenizerFast | None = None,
    ):
        """
        Initialize TinyStories dataset.

        Args:
            split: Dataset split ('train' or 'validation')
            seq_len: Sequence length for chunking
            tokenizer: GPT-2 tokenizer (will load default if None)
        """
        self.seq_len = seq_len

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Suppress the "Token indices sequence length is longer than the specified
        # maximum sequence length" warning by setting model_max_length high
        self.tokenizer.model_max_length = 100000

        # Load and tokenize dataset
        print(f"Loading TinyStories {split} split...")
        dataset = load_dataset("roneneldan/TinyStories", split=split)

        # Tokenize all texts using multiprocessing
        print(f"Tokenizing {len(dataset)} examples with multiprocessing...")

        def tokenize_fn(examples):
            """Batch tokenization function for dataset.map()"""
            tokenized = self.tokenizer(
                examples["text"],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            return {"input_ids": tokenized["input_ids"]}

        # Use all available CPU cores for tokenization
        num_proc = os.cpu_count() or 4
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        # Filter and concatenate tokens
        all_tokens = []
        skipped = 0
        for example in tokenized_dataset:
            tokens = example["input_ids"]
            # Skip examples that are too long
            if len(tokens) > seq_len:
                skipped += 1
                continue
            all_tokens.extend(tokens)

        if skipped > 0:
            print(f"Filtered out {skipped} examples exceeding {seq_len} tokens")

        # Chunk into sequences of seq_len + 1 (for input/target shift)
        self.chunks = []
        chunk_size = seq_len + 1
        for i in range(0, len(all_tokens) - chunk_size + 1, chunk_size):
            chunk = all_tokens[i : i + chunk_size]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Created {len(self.chunks)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        x = chunk[:-1]  # Input: all but last token
        y = chunk[1:]  # Target: all but first token
        return x, y


def create_dataloader(
    split: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    tokenizer: GPT2TokenizerFast | None = None,
) -> DataLoader:
    """Create a DataLoader for TinyStories."""
    dataset = TinyStoriesDataset(split=split, seq_len=seq_len, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class PG19Dataset(Dataset):
    """Dataset wrapper for PG-19 (Project Gutenberg long-form books).

    PG-19 contains full books from Project Gutenberg, providing much longer
    sequences than TinyStories. This is useful for testing long-range
    dependency modeling.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 2048,
        tokenizer: GPT2TokenizerFast | None = None,
        max_books: int | None = None,
    ):
        """
        Initialize PG-19 dataset.

        Args:
            split: Dataset split ('train', 'validation', or 'test')
            seq_len: Sequence length for chunking (can be much longer than TinyStories)
            tokenizer: GPT-2 tokenizer (will load default if None)
            max_books: Maximum number of books to load (None = all). Useful for testing.
        """
        self.seq_len = seq_len

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Suppress the length warning (PG-19 books can be very long)
        self.tokenizer.model_max_length = 10000000

        # Load PG-19 dataset using streaming to avoid downloading everything at once
        # Then take only the books we need
        print(f"Loading PG-19 {split} split (streaming)...")
        dataset = load_dataset("emozilla/pg19", split=split, streaming=True)

        # Collect books up to max_books
        books = []
        for i, example in enumerate(dataset):
            if max_books is not None and i >= max_books:
                break
            books.append(example["text"])
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1} books...")

        print(f"Loaded {len(books)} books")

        # Tokenize books
        print(f"Tokenizing {len(books)} books...")

        all_tokens = []
        for i, text in enumerate(books):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Tokenizing book {i + 1}/{len(books)}...")

            # Tokenize the full book text
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            all_tokens.extend(tokens)
            # Add EOS token between books to mark boundaries
            all_tokens.append(self.tokenizer.eos_token_id)

        print(f"Total tokens: {len(all_tokens):,}")

        # Chunk into sequences of seq_len + 1 (for input/target shift)
        self.chunks = []
        chunk_size = seq_len + 1
        for i in range(0, len(all_tokens) - chunk_size + 1, chunk_size):
            chunk = all_tokens[i : i + chunk_size]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Created {len(self.chunks):,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        x = chunk[:-1]  # Input: all but last token
        y = chunk[1:]  # Target: all but first token
        return x, y


def create_pg19_dataloader(
    split: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    tokenizer: GPT2TokenizerFast | None = None,
    max_books: int | None = None,
) -> DataLoader:
    """Create a DataLoader for PG-19.

    Args:
        split: Dataset split ('train', 'validation', or 'test')
        seq_len: Sequence length (e.g., 2048, 4096, 8192)
        batch_size: Batch size
        num_workers: Number of data loading workers
        tokenizer: GPT-2 tokenizer (will load default if None)
        max_books: Maximum number of books to load (None = all)

    Returns:
        DataLoader for PG-19 dataset
    """
    dataset = PG19Dataset(
        split=split,
        seq_len=seq_len,
        tokenizer=tokenizer,
        max_books=max_books,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of this experiment run
        """
        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        print(f"TensorBoard logging to: {self.log_path}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalars under a common main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_bandwidth_losses(
        self, losses: dict[float, float], step: int, prefix: str = "val"
    ) -> None:
        """
        Log losses at different bandwidths.

        Args:
            losses: Dict mapping bandwidth ratio to loss value
            step: Current training step
            prefix: Prefix for the tag (e.g., 'val', 'train')
        """
        for bandwidth, loss in losses.items():
            tag = f"{prefix}/loss_bw_{int(bandwidth * 100)}pct"
            self.writer.add_scalar(tag, loss, step)

        # Also log as a group for easy comparison
        tag_dict = {f"bw_{int(bw * 100)}pct": loss for bw, loss in losses.items()}
        self.writer.add_scalars(f"{prefix}/loss_by_bandwidth", tag_dict, step)

    def log_speed(self, tokens_per_sec: float, step: int, bandwidth: float = 1.0) -> None:
        """Log inference speed."""
        tag = f"speed/tokens_per_sec_bw_{int(bandwidth * 100)}pct"
        self.writer.add_scalar(tag, tokens_per_sec, step)

    def log_spectral_diagnostics(
        self,
        filter_stats: dict[str, dict[str, float]],
        step: int,
    ) -> None:
        """
        Log spectral gate filter diagnostics.

        Args:
            filter_stats: Nested dict with structure:
                {layer_name: {metric_name: value}}
                e.g., {"layer_0": {"filter_l1_norm": 1.5, "filter_l2_norm": 2.3, ...}}
            step: Current training step
        """
        for layer_name, metrics in filter_stats.items():
            for metric_name, value in metrics.items():
                tag = f"spectral/{layer_name}/{metric_name}"
                self.writer.add_scalar(tag, value, step)

    def log_filter_histograms(
        self,
        filter_weights: dict[str, torch.Tensor],
        step: int,
    ) -> None:
        """
        Log histograms of filter weights for visualization.

        Args:
            filter_weights: Dict mapping layer name to filter tensor
            step: Current training step
        """
        for layer_name, weights in filter_weights.items():
            self.writer.add_histogram(f"spectral/{layer_name}/filter_weights", weights, step)

    def log_gradient_norms(
        self,
        grad_norms: dict[str, float],
        step: int,
    ) -> None:
        """
        Log gradient norms for different parameter groups.

        Args:
            grad_norms: Dict mapping parameter group name to gradient norm
            step: Current training step
        """
        for name, norm in grad_norms.items():
            self.writer.add_scalar(f"gradients/{name}", norm, step)

    def flush(self) -> None:
        """Flush pending writes."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: str,
    model_name: str,
) -> str:
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_path / f"{model_name}_step_{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_file,
    )
    print(f"Saved checkpoint: {checkpoint_file}")
    return str(checkpoint_file)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle state dict from torch.compile() which adds "_orig_mod." prefix
    state_dict = checkpoint["model_state_dict"]
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        # Strip "_orig_mod." prefix if present
        new_key = key.replace("_orig_mod.", "")
        unwrapped_state_dict[new_key] = value

    model.load_state_dict(unwrapped_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint
