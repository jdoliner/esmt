"""Training script for ESMT and NanoGPT baseline."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import ESMTConfig, NanoGPTConfig, TrainConfig
from model import NanoGPT, SpectralGPT, count_parameters, create_matched_models
from utils import (
    TensorBoardLogger,
    Timer,
    create_dataloader,
    format_params,
    get_device,
    save_checkpoint,
    set_seed,
)


def get_lr(step: int, config: TrainConfig, total_steps: int) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step
        config: Training configuration
        total_steps: Total number of training steps

    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    # Cosine decay
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


import math


def train_model(
    model: nn.Module,
    train_config: TrainConfig,
    model_config: ESMTConfig | NanoGPTConfig,
    model_name: str,
    use_matryoshka: bool = True,
) -> nn.Module:
    """
    Train a model with optional Matryoshka loss.

    Args:
        model: The model to train (SpectralGPT or NanoGPT)
        train_config: Training configuration
        model_config: Model configuration
        model_name: Name for logging/checkpointing
        use_matryoshka: Whether to use Matryoshka dual-loss training

    Returns:
        Trained model
    """
    device = get_device()
    print(f"Training {model_name} on {device}")
    print(f"Parameters: {format_params(count_parameters(model))}")

    # Move model to device
    model = model.to(device)

    # Enable torch.compile for speedup (if supported)
    if train_config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create data loaders
    train_loader = create_dataloader(
        split="train",
        seq_len=model_config.seq_len,
        batch_size=train_config.batch_size,
        num_workers=4,
    )
    val_loader = create_dataloader(
        split="validation",
        seq_len=model_config.seq_len,
        batch_size=train_config.batch_size,
        num_workers=4,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda")

    # Logger
    logger = TensorBoardLogger(train_config.log_dir, model_name)

    # Calculate total steps
    total_steps = len(train_loader) * train_config.epochs
    print(f"Total training steps: {total_steps}")

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_full = 0.0
        epoch_loss_trunc = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{train_config.epochs}",
            leave=True,
        )

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step, train_config, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                # Full bandwidth forward pass
                logits_full = model(x, bandwidth_ratio=1.0)
                loss_full = nn.functional.cross_entropy(
                    logits_full.view(-1, logits_full.size(-1)), y.view(-1)
                )

                if use_matryoshka:
                    # Sample random bandwidth for truncated pass
                    bandwidth = random.uniform(
                        train_config.min_bandwidth, train_config.max_bandwidth
                    )
                    logits_trunc = model(x, bandwidth_ratio=bandwidth)
                    loss_trunc = nn.functional.cross_entropy(
                        logits_trunc.view(-1, logits_trunc.size(-1)), y.view(-1)
                    )

                    # Combined loss
                    loss = loss_full + train_config.lambda_trunc * loss_trunc
                else:
                    loss = loss_full
                    loss_trunc = torch.tensor(0.0)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Logging
            epoch_loss += loss.item()
            epoch_loss_full += loss_full.item()
            if use_matryoshka:
                epoch_loss_trunc += loss_trunc.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "loss_full": f"{loss_full.item():.4f}",
                    "lr": f"{lr:.2e}",
                }
            )

            # Log to TensorBoard
            if global_step % 10 == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_scalar("train/loss_full", loss_full.item(), global_step)
                if use_matryoshka:
                    logger.log_scalar("train/loss_trunc", loss_trunc.item(), global_step)
                    logger.log_scalar("train/bandwidth", bandwidth, global_step)
                logger.log_scalar("train/lr", lr, global_step)

            # Evaluation at intervals
            if (global_step + 1) % train_config.eval_interval == 0:
                val_losses = evaluate_spectral_sweep(
                    model, val_loader, train_config.eval_bandwidths, device
                )
                logger.log_bandwidth_losses(val_losses, global_step, prefix="val")

                # Log individual losses
                for bw, val_loss in val_losses.items():
                    print(f"  Val Loss @ {int(bw * 100)}%: {val_loss:.4f}")

                # Save checkpoint if best
                if val_losses[1.0] < best_val_loss:
                    best_val_loss = val_losses[1.0]
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        best_val_loss,
                        train_config.checkpoint_dir,
                        model_name,
                    )

                model.train()

            global_step += 1

        # End of epoch logging
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_full = epoch_loss_full / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Full Loss: {avg_loss_full:.4f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    val_losses = evaluate_spectral_sweep(
        model, val_loader, train_config.eval_bandwidths, device
    )
    for bw, val_loss in val_losses.items():
        print(f"  Val Loss @ {int(bw * 100)}%: {val_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, global_step, val_losses[1.0], train_config.checkpoint_dir, f"{model_name}_final"
    )

    logger.close()
    return model


@torch.no_grad()
def evaluate_spectral_sweep(
    model: nn.Module,
    val_loader,
    bandwidths: tuple[float, ...],
    device: torch.device,
    max_batches: int = 50,
) -> dict[float, float]:
    """
    Evaluate model at multiple bandwidth levels.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        bandwidths: Tuple of bandwidth ratios to evaluate
        device: Device to use
        max_batches: Maximum batches to evaluate (for speed)

    Returns:
        Dictionary mapping bandwidth to loss
    """
    model.eval()
    losses = {bw: 0.0 for bw in bandwidths}
    n_batches = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        for bw in bandwidths:
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, bandwidth_ratio=bw)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            losses[bw] += loss.item()

        n_batches += 1

    # Average losses
    for bw in bandwidths:
        losses[bw] /= n_batches

    return losses


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ESMT and NanoGPT models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["esmt", "nanogpt", "both"],
        default="both",
        help="Which model(s) to train",
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
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create configs
    esmt_config = ESMTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    nano_config = NanoGPTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        compile_model=not args.no_compile,
        seed=args.seed,
    )

    # Create matched models
    print("Creating models with matched parameter counts...")
    esmt, nanogpt, esmt_config, nano_config = create_matched_models(esmt_config, nano_config)

    if args.model in ["esmt", "both"]:
        print("\n" + "=" * 60)
        print("Training ESMT (with Matryoshka loss)")
        print("=" * 60)
        esmt = train_model(
            esmt,
            train_config,
            esmt_config,
            model_name="esmt",
            use_matryoshka=True,
        )

    if args.model in ["nanogpt", "both"]:
        print("\n" + "=" * 60)
        print("Training NanoGPT (standard training, no Matryoshka)")
        print("=" * 60)
        nanogpt = train_model(
            nanogpt,
            train_config,
            nano_config,
            model_name="nanogpt",
            use_matryoshka=False,
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
