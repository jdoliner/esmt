"""Evaluation script for ESMT and NanoGPT models."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from config import ESMTConfig, NanoGPTConfig, TrainConfig
from model import NanoGPT, SpectralGPT, count_parameters, create_matched_models
from utils import (
    TensorBoardLogger,
    Timer,
    create_dataloader,
    format_params,
    get_device,
    load_checkpoint,
    set_seed,
)


@torch.no_grad()
def evaluate_bandwidth(
    model: nn.Module,
    val_loader,
    bandwidth: float,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """
    Evaluate model at a specific bandwidth.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        bandwidth: Bandwidth ratio (0.25 to 1.0)
        device: Device to use
        max_batches: Maximum batches to evaluate (None for full dataset)

    Returns:
        Tuple of (average loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc=f"Eval @ {int(bandwidth * 100)}%")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        batch_tokens = y.numel()

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, bandwidth_ratio=bandwidth)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )

        total_loss += loss.item()
        total_tokens += batch_tokens
        n_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    device: torch.device,
    bandwidth: float,
    batch_size: int = 32,
    seq_len: int = 512,
    n_iterations: int = 100,
    warmup_iterations: int = 10,
) -> float:
    """
    Benchmark inference speed at a specific bandwidth.

    Args:
        model: The model to benchmark
        device: Device to use
        bandwidth: Bandwidth ratio
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        n_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations

    Returns:
        Tokens per second
    """
    model.eval()

    # Create dummy input
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup_iterations):
        with autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, bandwidth_ratio=bandwidth)
    torch.cuda.synchronize()

    # Timed iterations
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        with autocast("cuda", dtype=torch.bfloat16):
            _ = model(x, bandwidth_ratio=bandwidth)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    total_tokens = batch_size * seq_len * n_iterations
    tokens_per_second = total_tokens / elapsed_time

    return tokens_per_second


def evaluate_elasticity(
    model: nn.Module,
    model_name: str,
    val_loader,
    bandwidths: tuple[float, ...],
    device: torch.device,
    logger: TensorBoardLogger | None = None,
    max_batches: int | None = 100,
) -> dict:
    """
    Evaluate model elasticity across multiple bandwidths.

    Args:
        model: The model to evaluate
        model_name: Name for logging
        val_loader: Validation data loader
        bandwidths: Tuple of bandwidth ratios to evaluate
        device: Device to use
        logger: Optional TensorBoard logger
        max_batches: Maximum batches for evaluation

    Returns:
        Dictionary with evaluation results
    """
    results = {
        "model": model_name,
        "bandwidths": {},
        "speed": {},
    }

    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 60}")

    # Evaluate at each bandwidth
    for bandwidth in bandwidths:
        print(f"\nBandwidth: {int(bandwidth * 100)}%")

        # Loss evaluation
        loss, ppl = evaluate_bandwidth(
            model, val_loader, bandwidth, device, max_batches
        )
        results["bandwidths"][bandwidth] = {"loss": loss, "perplexity": ppl}

        print(f"  Loss: {loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")

        if logger:
            logger.log_scalar(f"{model_name}/loss_bw_{int(bandwidth * 100)}", loss, 0)
            logger.log_scalar(f"{model_name}/ppl_bw_{int(bandwidth * 100)}", ppl, 0)

        # Speed benchmark
        tokens_per_sec = benchmark_speed(model, device, bandwidth)
        results["speed"][bandwidth] = tokens_per_sec
        print(f"  Speed: {tokens_per_sec:,.0f} tokens/sec")

        if logger:
            logger.log_scalar(
                f"{model_name}/speed_bw_{int(bandwidth * 100)}", tokens_per_sec, 0
            )

    return results


def compare_models(
    esmt: nn.Module,
    nanogpt: nn.Module,
    val_loader,
    bandwidths: tuple[float, ...],
    device: torch.device,
    log_dir: str = "runs/evaluation",
) -> dict:
    """
    Compare ESMT and NanoGPT across bandwidths.

    Args:
        esmt: Trained ESMT model
        nanogpt: Trained NanoGPT model
        val_loader: Validation data loader
        bandwidths: Bandwidths to evaluate
        device: Device to use
        log_dir: Directory for TensorBoard logs

    Returns:
        Comparison results
    """
    logger = TensorBoardLogger(log_dir, "comparison")

    # Evaluate both models
    esmt_results = evaluate_elasticity(
        esmt, "esmt", val_loader, bandwidths, device, logger
    )
    nanogpt_results = evaluate_elasticity(
        nanogpt, "nanogpt", val_loader, bandwidths, device, logger
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: ESMT vs NanoGPT (Bandwidth Elasticity)")
    print("=" * 80)
    print(f"{'Bandwidth':<12} {'ESMT Loss':<12} {'NanoGPT Loss':<14} {'ESMT PPL':<12} {'NanoGPT PPL':<14}")
    print("-" * 80)

    for bw in bandwidths:
        esmt_loss = esmt_results["bandwidths"][bw]["loss"]
        nano_loss = nanogpt_results["bandwidths"][bw]["loss"]
        esmt_ppl = esmt_results["bandwidths"][bw]["perplexity"]
        nano_ppl = nanogpt_results["bandwidths"][bw]["perplexity"]

        print(f"{int(bw * 100):>3}%{'':<8} {esmt_loss:<12.4f} {nano_loss:<14.4f} {esmt_ppl:<12.2f} {nano_ppl:<14.2f}")

    print("-" * 80)

    # Speed comparison
    print("\n" + "=" * 80)
    print("SPEED COMPARISON (tokens/sec)")
    print("=" * 80)
    print(f"{'Bandwidth':<12} {'ESMT':<20} {'NanoGPT':<20} {'Speedup':<12}")
    print("-" * 80)

    for bw in bandwidths:
        esmt_speed = esmt_results["speed"][bw]
        nano_speed = nanogpt_results["speed"][bw]
        speedup = esmt_speed / nano_speed if nano_speed > 0 else 0

        print(f"{int(bw * 100):>3}%{'':<8} {esmt_speed:>15,.0f}     {nano_speed:>15,.0f}     {speedup:.2f}x")

    print("-" * 80)

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check elasticity hypothesis
    esmt_50_loss = esmt_results["bandwidths"][0.5]["loss"]
    nano_50_loss = nanogpt_results["bandwidths"][0.5]["loss"]

    print(f"\nAt 50% bandwidth:")
    print(f"  ESMT Loss: {esmt_50_loss:.4f}")
    print(f"  NanoGPT Loss: {nano_50_loss:.4f}")

    if nano_50_loss > 10.0:
        print("  -> NanoGPT output is essentially random noise (Loss > 10)")
    if esmt_50_loss < 4.0:
        print("  -> ESMT remains coherent (Loss < 4)")

    # Degradation comparison
    esmt_100_loss = esmt_results["bandwidths"][1.0]["loss"]
    nano_100_loss = nanogpt_results["bandwidths"][1.0]["loss"]

    esmt_degradation = (esmt_50_loss - esmt_100_loss) / esmt_100_loss * 100
    nano_degradation = (nano_50_loss - nano_100_loss) / nano_100_loss * 100

    print(f"\nDegradation from 100% to 50%:")
    print(f"  ESMT: {esmt_degradation:.1f}%")
    print(f"  NanoGPT: {nano_degradation:.1f}%")

    logger.close()

    return {
        "esmt": esmt_results,
        "nanogpt": nanogpt_results,
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ESMT and NanoGPT models")
    parser.add_argument(
        "--esmt_checkpoint",
        type=str,
        default=None,
        help="Path to ESMT checkpoint",
    )
    parser.add_argument(
        "--nanogpt_checkpoint",
        type=str,
        default=None,
        help="Path to NanoGPT checkpoint",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model hidden dimension (default 64 gives ~6.7M params)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of layers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Bandwidths to evaluate",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum batches for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    device = get_device()
    print(f"Evaluating on {device}")

    # Create configs
    esmt_config = ESMTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    nano_config = NanoGPTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )

    # Create matched models
    esmt, nanogpt, esmt_config, nano_config = create_matched_models(esmt_config, nano_config)

    # Load checkpoints if provided
    if args.esmt_checkpoint:
        load_checkpoint(args.esmt_checkpoint, esmt)
    if args.nanogpt_checkpoint:
        load_checkpoint(args.nanogpt_checkpoint, nanogpt)

    # Move to device
    esmt = esmt.to(device)
    nanogpt = nanogpt.to(device)

    # Create validation loader
    val_loader = create_dataloader(
        split="validation",
        seq_len=esmt_config.seq_len,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Run comparison
    bandwidths = tuple(args.bandwidths)
    results = compare_models(
        esmt, nanogpt, val_loader, bandwidths, device
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
