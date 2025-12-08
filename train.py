"""Training script for ESMT and NanoGPT baseline."""

import os

# Suppress tokenizers parallelism warning when using multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
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


def collect_spectral_diagnostics(model: nn.Module) -> dict[str, dict[str, float]]:
    """
    Collect diagnostic statistics from SpectralGate filters.
    
    Args:
        model: A SpectralGPT model
        
    Returns:
        Dict with structure: {layer_name: {metric_name: value}}
    """
    stats = {}
    
    # Check if this is a compiled model and unwrap if needed
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    
    # Only collect for SpectralGPT models
    if not hasattr(model, "layers"):
        return stats
    
    for layer_idx, layer in enumerate(model.layers):
        layer_name = f"layer_{layer_idx}"
        gate = layer.gate
        filter_weights = gate.filter  # [n_heads, head_dim]
        
        # Per-layer aggregate statistics
        stats[layer_name] = {
            # Filter magnitude statistics
            "filter_l1_norm": filter_weights.abs().mean().item(),
            "filter_l2_norm": filter_weights.pow(2).mean().sqrt().item(),
            "filter_min": filter_weights.min().item(),
            "filter_max": filter_weights.max().item(),
            "filter_std": filter_weights.std().item(),
            
            # Deviation from identity (initialized at 1.0)
            "filter_mean_deviation_from_1": (filter_weights - 1.0).abs().mean().item(),
            
            # Sparsity: fraction of filters close to zero (< 0.1)
            "filter_near_zero_frac": (filter_weights.abs() < 0.1).float().mean().item(),
            
            # Saturation: fraction of filters with large magnitude (> 2.0)  
            "filter_saturated_frac": (filter_weights.abs() > 2.0).float().mean().item(),
        }
        
        # Per-head statistics (averaged over head_dim)
        head_l1_norms = filter_weights.abs().mean(dim=1)  # [n_heads]
        for head_idx, norm in enumerate(head_l1_norms):
            stats[layer_name][f"head_{head_idx}_l1_norm"] = norm.item()
    
    return stats


def collect_filter_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Collect filter weight tensors for histogram logging.
    
    Args:
        model: A SpectralGPT model
        
    Returns:
        Dict mapping layer name to filter tensor
    """
    weights = {}
    
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
        
    if not hasattr(model, "layers"):
        return weights
    
    for layer_idx, layer in enumerate(model.layers):
        weights[f"layer_{layer_idx}"] = layer.gate.filter.detach().cpu()
    
    return weights


def collect_gradient_norms(model: nn.Module) -> dict[str, float]:
    """
    Collect gradient norms for different parameter groups.
    
    Args:
        model: The model (SpectralGPT or NanoGPT)
        
    Returns:
        Dict mapping parameter group name to gradient norm
    """
    norms = {}
    
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    
    # Collect by parameter type
    filter_grads = []
    conv_grads = []
    mlp_grads = []
    embedding_grads = []
    other_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.norm().item()
        
        if "filter" in name:
            filter_grads.append(grad_norm)
        elif "conv" in name:
            conv_grads.append(grad_norm)
        elif "mlp" in name or "fc1" in name or "fc2" in name:
            mlp_grads.append(grad_norm)
        elif "emb" in name:
            embedding_grads.append(grad_norm)
        else:
            other_grads.append(grad_norm)
    
    # Aggregate norms
    if filter_grads:
        norms["filter_grad_norm"] = sum(filter_grads) / len(filter_grads)
    if conv_grads:
        norms["conv_grad_norm"] = sum(conv_grads) / len(conv_grads)
    if mlp_grads:
        norms["mlp_grad_norm"] = sum(mlp_grads) / len(mlp_grads)
    if embedding_grads:
        norms["embedding_grad_norm"] = sum(embedding_grads) / len(embedding_grads)
    if other_grads:
        norms["other_grad_norm"] = sum(other_grads) / len(other_grads)
    
    # Total gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    norms["total_grad_norm"] = total_norm ** 0.5
    
    return norms


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
    
    # Stats tracking
    tokens_per_batch = train_config.batch_size * model_config.seq_len
    
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_full = 0.0
        epoch_loss_trunc = 0.0
        epoch_tokens = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{train_config.epochs}",
            leave=True,
        )

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            batch_start_time = time.time()

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

                # Initialize variables that may not be set in all branches
                bandwidth = 1.0
                loss_trunc = torch.tensor(0.0, device=x.device)
                reg_loss = torch.tensor(0.0, device=x.device)

                if use_matryoshka:
                    # Sample random bandwidth for truncated pass
                    bandwidth = random.uniform(
                        train_config.min_bandwidth, train_config.max_bandwidth
                    )
                    logits_trunc = model(x, bandwidth_ratio=bandwidth)
                    loss_trunc = nn.functional.cross_entropy(
                        logits_trunc.view(-1, logits_trunc.size(-1)), y.view(-1)
                    )

                    # Get spectral regularization loss (prevents degeneration to standard attention)
                    # Access the underlying model if compiled
                    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                    if hasattr(base_model, "get_spectral_regularization_loss"):
                        reg_loss = base_model.get_spectral_regularization_loss()

                    # Combined loss: LM loss + truncated loss + regularization
                    loss = loss_full + train_config.lambda_trunc * loss_trunc + reg_loss
                else:
                    loss = loss_full

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
            
            # Track tokens processed
            epoch_tokens += tokens_per_batch
            batch_time = time.time() - batch_start_time
            tokens_per_sec = tokens_per_batch / batch_time if batch_time > 0 else 0
            
            # Calculate running averages and ETA
            elapsed_time = time.time() - epoch_start_time
            avg_tokens_per_sec = epoch_tokens / elapsed_time if elapsed_time > 0 else 0
            batches_remaining = len(train_loader) - (batch_idx + 1)
            eta_seconds = (batches_remaining * elapsed_time / (batch_idx + 1)) if batch_idx > 0 else 0
            
            # Format ETA
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_min}m{eta_sec:02d}s" if eta_min > 0 else f"{eta_sec}s"

            # Update progress bar with high-level stats
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                    "tok/s": f"{avg_tokens_per_sec:.0f}",
                    "lr": f"{lr:.2e}",
                    "eta": eta_str,
                }
            )

            # Log to TensorBoard
            if global_step % 10 == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_scalar("train/loss_full", loss_full.item(), global_step)
                if use_matryoshka:
                    logger.log_scalar("train/loss_trunc", loss_trunc.item(), global_step)
                    logger.log_scalar("train/bandwidth", bandwidth, global_step)
                    logger.log_scalar("train/reg_loss", reg_loss.item(), global_step)
                logger.log_scalar("train/lr", lr, global_step)
                
                # Log gradient norms
                grad_norms = collect_gradient_norms(model)
                logger.log_gradient_norms(grad_norms, global_step)
            
            # Log spectral diagnostics less frequently (every 100 steps)
            # if global_step % 100 == 0 and use_matryoshka:
            #     spectral_stats = collect_spectral_diagnostics(model)
            #     if spectral_stats:
            #         logger.log_spectral_diagnostics(spectral_stats, global_step)
            #         
            #     # Log filter weight histograms even less frequently (every 500 steps)
            #     if global_step % 500 == 0:
            #         filter_weights = collect_filter_weights(model)
            #         if filter_weights:
            #             logger.log_filter_histograms(filter_weights, global_step)

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

        # End of epoch logging with high-level stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_full = epoch_loss_full / len(train_loader)
        epoch_tokens_per_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f} | Avg Full Loss: {avg_loss_full:.4f}")
        print(f"  Tokens processed: {epoch_tokens:,} | Time: {epoch_time:.1f}s | Throughput: {epoch_tokens_per_sec:,.0f} tok/s")

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
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for TensorBoard logging (default: model name)",
    )
    
    # ===========================================================================
    # Experimental Spectral Features
    # ===========================================================================
    parser.add_argument(
        "--spectral_blur",
        action="store_true",
        help="Enable SpectralBlurMLP: local feature mixing via 1D convolution",
    )
    parser.add_argument(
        "--blur_kernel",
        type=int,
        default=3,
        choices=[3, 5, 7],
        help="Kernel size for spectral blur convolution (default: 3)",
    )
    parser.add_argument(
        "--harmonic",
        action="store_true",
        help="Enable HarmonicMixing: octave skip connections (f <-> 2f)",
    )
    parser.add_argument(
        "--n_octaves",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Number of octave levels for harmonic mixing (default: 3)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create configs
    esmt_config = ESMTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        # Experimental features
        use_spectral_blur=args.spectral_blur,
        blur_kernel_size=args.blur_kernel,
        use_harmonic_mixing=args.harmonic,
        n_octaves=args.n_octaves,
    )
    nano_config = NanoGPTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    
    # Print experiment configuration
    print(f"Experimental features: {esmt_config.experiment_summary()}")
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
        esmt_run_name = f"{args.run_name}_esmt" if args.run_name else "esmt"
        esmt = train_model(
            esmt,
            train_config,
            esmt_config,
            model_name=esmt_run_name,
            use_matryoshka=True,
        )

    if args.model in ["nanogpt", "both"]:
        print("\n" + "=" * 60)
        print("Training NanoGPT (standard training, no Matryoshka)")
        print("=" * 60)
        nanogpt_run_name = f"{args.run_name}_nanogpt" if args.run_name else "nanogpt"
        nanogpt = train_model(
            nanogpt,
            train_config,
            nano_config,
            model_name=nanogpt_run_name,
            use_matryoshka=False,
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
