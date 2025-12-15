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

from config import ESMTConfig, NanoGPTConfig, TrainConfig, ComplexESMTConfig, SATConfig
from model import NanoGPT, SpectralGPT, ComplexSpectralGPT, count_parameters, create_matched_models
from sat_model import SpectralAugmentedTransformer, count_parameters as sat_count_parameters
from spectral_init import (
    initialize_esmt_from_nanogpt,
    freeze_embeddings,
    count_frozen_parameters,
    analyze_embedding_spectrum,
    initialize_complex_esmt_from_nanogpt,
    freeze_complex_embeddings,
    analyze_complex_embedding_spectrum,
)
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
        model: The model (SpectralGPT, ComplexSpectralGPT, or NanoGPT)

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
    attention_grads = []
    other_grads = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Handle complex gradients (use magnitude)
        if param.grad.is_complex():
            grad_norm = param.grad.abs().norm().item()
        else:
            grad_norm = param.grad.norm().item()

        if "filter" in name:
            filter_grads.append(grad_norm)
        elif "conv" in name:
            conv_grads.append(grad_norm)
        elif "mlp" in name or "fc1" in name or "fc2" in name or "ffn" in name:
            mlp_grads.append(grad_norm)
        elif "emb" in name:
            embedding_grads.append(grad_norm)
        elif "attn" in name or "qkv" in name or "proj" in name:
            attention_grads.append(grad_norm)
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
    if attention_grads:
        norms["attention_grad_norm"] = sum(attention_grads) / len(attention_grads)
    if other_grads:
        norms["other_grad_norm"] = sum(other_grads) / len(other_grads)

    # Total gradient norm (handle complex)
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            if param.grad.is_complex():
                total_norm += param.grad.abs().norm().item() ** 2
            else:
                total_norm += param.grad.norm().item() ** 2
    norms["total_grad_norm"] = total_norm**0.5

    return norms


def collect_complex_diagnostics(model: nn.Module) -> dict[str, float]:
    """
    Collect diagnostic statistics specific to complex models.

    Args:
        model: A ComplexSpectralGPT model

    Returns:
        Dict with complex-specific metrics
    """
    stats = {}

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    # Check if this is a complex model
    if not hasattr(model, "token_emb") or not hasattr(model.token_emb, "embedding"):
        return stats

    # Token embedding statistics
    emb = model.token_emb.embedding.data
    if emb.is_complex():
        stats["emb_magnitude_mean"] = emb.abs().mean().item()
        stats["emb_magnitude_std"] = emb.abs().std().item()
        stats["emb_phase_std"] = emb.angle().std().item()

    # Collect per-layer statistics
    if hasattr(model, "layers"):
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layer_{layer_idx}"

            # Attention QKV weights
            if hasattr(layer, "attn") and hasattr(layer.attn, "qkv"):
                qkv_weight = layer.attn.qkv.weight.data
                if qkv_weight.is_complex():
                    stats[f"{prefix}_attn_qkv_mag_mean"] = qkv_weight.abs().mean().item()
                    stats[f"{prefix}_attn_qkv_phase_std"] = qkv_weight.angle().std().item()

            # FFN weights
            if hasattr(layer, "ffn") and hasattr(layer.ffn, "fc1"):
                fc1_weight = layer.ffn.fc1.weight.data
                if fc1_weight.is_complex():
                    stats[f"{prefix}_ffn_fc1_mag_mean"] = fc1_weight.abs().mean().item()

    return stats


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
    model_config: ESMTConfig | NanoGPTConfig | ComplexESMTConfig,
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

    # Check if this is a complex model (affects compile and mixed precision)
    is_complex_model = isinstance(model_config, ComplexESMTConfig)

    # Enable torch.compile for speedup (if supported)
    # Note: torch.compile has limited support for complex ops (falls back to eager)
    if train_config.compile_model and hasattr(torch, "compile") and not is_complex_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    elif is_complex_model and train_config.compile_model:
        print("Skipping torch.compile for complex model (limited complex op support)")

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

    # Complex models don't support GradScaler (AMP unscaling not implemented for complex)
    if is_complex_model:
        print("Complex model: disabling GradScaler (not supported for complex tensors)")
        use_grad_scaler = False
    else:
        use_grad_scaler = True

    # Mixed precision scaler (only for real-valued models)
    scaler = GradScaler("cuda") if use_grad_scaler else None

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

            # Matryoshka training: weighted stochastic bandwidth sampling
            # Single forward pass per step for efficiency (~1x vs ~2.5x with dual pass)
            if use_matryoshka:
                # With probability full_bandwidth_prob, use full bandwidth
                # Otherwise, sample uniformly from [min_bandwidth, max_bandwidth]
                if random.random() < train_config.full_bandwidth_prob:
                    bandwidth = 1.0
                else:
                    bandwidth = random.uniform(
                        train_config.min_bandwidth, train_config.max_bandwidth
                    )
            else:
                bandwidth = 1.0

            if use_grad_scaler:
                # Standard mixed precision training with GradScaler
                with autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, bandwidth_ratio=bandwidth)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()  # type: ignore

                # Gradient clipping
                scaler.unscale_(optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

                # Optimizer step
                scaler.step(optimizer)  # type: ignore
                scaler.update()  # type: ignore
            else:
                # Complex model: no GradScaler, but still use autocast for speed
                # Note: autocast with bfloat16 may not work well with complex ops
                # so we run in full precision (float32 for complex64)
                logits = model(x, bandwidth_ratio=bandwidth)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                # Standard backward pass
                loss.backward()

                # Gradient clipping (handles complex gradients via magnitude)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

                # Optimizer step
                optimizer.step()

            # Logging
            epoch_loss += loss.item()

            # Track tokens processed
            epoch_tokens += tokens_per_batch
            batch_time = time.time() - batch_start_time
            tokens_per_sec = tokens_per_batch / batch_time if batch_time > 0 else 0

            # Calculate running averages and ETA
            elapsed_time = time.time() - epoch_start_time
            avg_tokens_per_sec = epoch_tokens / elapsed_time if elapsed_time > 0 else 0
            batches_remaining = len(train_loader) - (batch_idx + 1)
            eta_seconds = (
                (batches_remaining * elapsed_time / (batch_idx + 1)) if batch_idx > 0 else 0
            )

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
                logger.log_scalar("train/bandwidth", bandwidth, global_step)
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
                    model,
                    val_loader,
                    train_config.eval_bandwidths,
                    device,
                    use_autocast=use_grad_scaler,  # No autocast for complex models
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
        epoch_tokens_per_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(
            f"  Tokens processed: {epoch_tokens:,} | Time: {epoch_time:.1f}s | Throughput: {epoch_tokens_per_sec:,.0f} tok/s"
        )

    # Final evaluation
    print("\nFinal Evaluation:")
    val_losses = evaluate_spectral_sweep(
        model,
        val_loader,
        train_config.eval_bandwidths,
        device,
        use_autocast=use_grad_scaler,  # No autocast for complex models
    )
    for bw, val_loss in val_losses.items():
        print(f"  Val Loss @ {int(bw * 100)}%: {val_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model,
        optimizer,
        global_step,
        val_losses[1.0],
        train_config.checkpoint_dir,
        f"{model_name}_final",
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
    use_autocast: bool = True,
) -> dict[float, float]:
    """
    Evaluate model at multiple bandwidth levels.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        bandwidths: Tuple of bandwidth ratios to evaluate
        device: Device to use
        max_batches: Maximum batches to evaluate (for speed)
        use_autocast: Whether to use autocast (False for complex models)

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
            if use_autocast:
                with autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, bandwidth_ratio=bw)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits = model(x, bandwidth_ratio=bw)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[bw] += loss.item()

        n_batches += 1

    # Average losses
    for bw in bandwidths:
        losses[bw] /= n_batches

    return losses


# ==============================================================================
# SAT Training
# ==============================================================================


def train_sat(
    model: SpectralAugmentedTransformer,
    train_config: TrainConfig,
    sat_config: SATConfig,
    model_name: str,
) -> SpectralAugmentedTransformer:
    """
    Train a Spectral-Augmented Transformer with auxiliary FNO loss.

    Args:
        model: SAT model to train
        train_config: Training configuration
        sat_config: SAT model configuration
        model_name: Name for logging/checkpointing

    Returns:
        Trained model
    """
    device = get_device()
    print(f"Training {model_name} on {device}")
    print(f"Parameters: {format_params(sat_count_parameters(model))}")
    print(f"Config: {sat_config.experiment_summary()}")

    # Move model to device
    model = model.to(device)

    # Enable torch.compile for speedup
    if train_config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create data loaders
    train_loader = create_dataloader(
        split="train",
        seq_len=sat_config.seq_len,
        batch_size=train_config.batch_size,
        num_workers=4,
    )
    val_loader = create_dataloader(
        split="validation",
        seq_len=sat_config.seq_len,
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
    tokens_per_batch = train_config.batch_size * sat_config.seq_len

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
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

            # Get auxiliary loss weight based on training progress
            progress = global_step / total_steps
            aux_weight = sat_config.get_aux_loss_weight(progress)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                # Forward pass with spectral output for auxiliary loss
                logits, spectral = model(x, return_spectral=True)

                # Main loss: next-token prediction
                main_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

                # Auxiliary loss: FNO prediction
                aux_loss = model.compute_auxiliary_loss(spectral, x)

                # Combined loss
                loss = main_loss + aux_weight * aux_loss

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
            epoch_main_loss += main_loss.item()
            epoch_aux_loss += aux_loss.item()

            # Track tokens processed
            epoch_tokens += tokens_per_batch
            batch_time = time.time() - batch_start_time

            # Calculate running averages and ETA
            elapsed_time = time.time() - epoch_start_time
            avg_tokens_per_sec = epoch_tokens / elapsed_time if elapsed_time > 0 else 0
            batches_remaining = len(train_loader) - (batch_idx + 1)
            eta_seconds = (
                (batches_remaining * elapsed_time / (batch_idx + 1)) if batch_idx > 0 else 0
            )

            # Format ETA
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_min}m{eta_sec:02d}s" if eta_min > 0 else f"{eta_sec}s"

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "main": f"{main_loss.item():.4f}",
                    "aux": f"{aux_loss.item():.4f}",
                    "aux_w": f"{aux_weight:.2f}",
                    "lr": f"{lr:.2e}",
                    "eta": eta_str,
                }
            )

            # Log to TensorBoard
            if global_step % 10 == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_scalar("train/main_loss", main_loss.item(), global_step)
                logger.log_scalar("train/aux_loss", aux_loss.item(), global_step)
                logger.log_scalar("train/aux_weight", aux_weight, global_step)
                logger.log_scalar("train/lr", lr, global_step)
                logger.log_scalar("train/tokens_per_sec", avg_tokens_per_sec, global_step)

                # Log gradient norms
                grad_norms = collect_gradient_norms(model)
                logger.log_gradient_norms(grad_norms, global_step)

            # Evaluation at intervals
            if (global_step + 1) % train_config.eval_interval == 0:
                val_loss = evaluate_sat(model, val_loader, device, max_batches=50)
                logger.log_scalar("val/loss", val_loss, global_step)
                print(f"\n  Val Loss: {val_loss:.4f}")

                # Save checkpoint if best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_loader)
        avg_main_loss = epoch_main_loss / len(train_loader)
        avg_aux_loss = epoch_aux_loss / len(train_loader)
        epoch_tokens_per_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f} (main: {avg_main_loss:.4f}, aux: {avg_aux_loss:.4f})")
        print(
            f"  Tokens: {epoch_tokens:,} | Time: {epoch_time:.1f}s | Throughput: {epoch_tokens_per_sec:,.0f} tok/s"
        )

    # Final evaluation
    print("\nFinal Evaluation:")
    val_loss = evaluate_sat(model, val_loader, device, max_batches=100)
    print(f"  Val Loss: {val_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, global_step, val_loss, train_config.checkpoint_dir, f"{model_name}_final"
    )

    logger.close()
    return model


@torch.no_grad()
def evaluate_sat(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """
    Evaluate SAT model on validation set.

    Args:
        model: SAT model to evaluate
        val_loader: Validation data loader
        device: Device to use
        max_batches: Maximum batches to evaluate

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, return_spectral=False)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


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

    # ===========================================================================
    # Spectral Initialization from Pretrained NanoGPT
    # ===========================================================================
    parser.add_argument(
        "--spectral_init",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Initialize ESMT embeddings from a pretrained NanoGPT checkpoint with DCT transform",
    )
    parser.add_argument(
        "--no_dct_token",
        action="store_true",
        help="Don't apply DCT to token embeddings (only with --spectral_init)",
    )
    parser.add_argument(
        "--no_dct_pos",
        action="store_true",
        help="Don't apply DCT to positional embeddings (only with --spectral_init)",
    )
    parser.add_argument(
        "--no_dct_lm_head",
        action="store_true",
        help="Don't apply DCT to lm_head weights (only with --spectral_init)",
    )
    parser.add_argument(
        "--freeze_embeddings",
        action="store_true",
        help="Freeze DCT'd embeddings during training (only with --spectral_init)",
    )

    # ===========================================================================
    # Complex Spectral Transformer Options
    # ===========================================================================
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Train ComplexSpectralGPT instead of real-valued ESMT",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        choices=["magnitude", "phase_aware"],
        default="magnitude",
        help="Complex attention mode: 'magnitude' (default) or 'phase_aware'",
    )
    parser.add_argument(
        "--layernorm_mode",
        type=str,
        choices=["magnitude", "split"],
        default="magnitude",
        help="Complex LayerNorm mode: 'magnitude' (default) or 'split' (Trabelsi)",
    )
    parser.add_argument(
        "--residual_mode",
        type=str,
        choices=["multiplicative", "additive"],
        default="multiplicative",
        help="Residual connection mode: 'multiplicative' (default) or 'additive'",
    )
    parser.add_argument(
        "--fft_init",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Initialize ComplexSpectralGPT from NanoGPT checkpoint with FFT transform",
    )
    parser.add_argument(
        "--freeze_complex_embeddings",
        action="store_true",
        help="Freeze FFT'd embeddings in complex model during training",
    )

    # ===========================================================================
    # Spectral-Augmented Transformer (SAT) Options
    # ===========================================================================
    parser.add_argument(
        "--sat",
        action="store_true",
        help="Train Spectral-Augmented Transformer (SAT) model",
    )
    parser.add_argument(
        "--sat_d_spectral",
        type=int,
        default=None,
        help="SAT spectral dimension (default: d_model // 4)",
    )
    parser.add_argument(
        "--sat_n_fno_layers",
        type=int,
        default=None,
        help="Number of FNO layers (default: same as n_layers)",
    )
    parser.add_argument(
        "--sat_k_max",
        type=int,
        default=None,
        help="Number of frequency modes (default: seq_len // 8)",
    )
    parser.add_argument(
        "--sat_integration",
        type=str,
        choices=["adaln", "cross_attention", "both"],
        default="adaln",
        help="How spectral stream conditions transformer: 'adaln', 'cross_attention', or 'both'",
    )
    parser.add_argument(
        "--sat_aux_weight",
        type=float,
        default=1.0,
        help="Initial weight for auxiliary FNO loss",
    )
    parser.add_argument(
        "--sat_aux_weight_min",
        type=float,
        default=0.1,
        help="Minimum auxiliary loss weight after decay",
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
        # Spectral initialization
        spectral_init_checkpoint=args.spectral_init,
        dct_token_emb=not args.no_dct_token,
        dct_pos_emb=not args.no_dct_pos,
        dct_lm_head=not args.no_dct_lm_head,
        freeze_embeddings=args.freeze_embeddings,
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

    # ===========================================================================
    # Complex Spectral Transformer Training
    # ===========================================================================
    if args.complex:
        print("\n" + "=" * 60)
        print("Complex Spectral Transformer Mode")
        print("=" * 60)

        # Create complex config
        complex_config = ComplexESMTConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            attention_mode=args.attention_mode,
            layernorm_mode=args.layernorm_mode,
            residual_mode=args.residual_mode,
            fft_init_checkpoint=args.fft_init,
            freeze_embeddings=args.freeze_complex_embeddings,
        )

        print(f"Complex model config: {complex_config.experiment_summary()}")

        # Create complex model
        complex_model = ComplexSpectralGPT(complex_config)
        print(f"ComplexSpectralGPT parameters: {format_params(count_parameters(complex_model))}")

        # Apply FFT initialization if configured
        if complex_config.fft_init_checkpoint:
            print("\n" + "=" * 60)
            print("Applying FFT Initialization from NanoGPT")
            print("=" * 60)
            complex_model = initialize_complex_esmt_from_nanogpt(
                complex_model,
                complex_config.fft_init_checkpoint,
                verbose=True,
            )

            # Freeze embeddings if configured
            if complex_config.freeze_embeddings:
                print("\nFreezing FFT'd embeddings...")
                freeze_complex_embeddings(complex_model)
                trainable, frozen = count_frozen_parameters(complex_model)
                print(f"  Trainable parameters: {format_params(trainable)}")
                print(f"  Frozen parameters: {format_params(frozen)}")

            # Show initial spectral analysis
            analyze_complex_embedding_spectrum(complex_model, "Initial Complex Embedding Analysis")

        # Train complex model
        print("\n" + "=" * 60)
        print("Training ComplexSpectralGPT (with Matryoshka loss)")
        print("=" * 60)
        complex_run_name = f"{args.run_name}_complex" if args.run_name else "complex_esmt"
        complex_model = train_model(
            complex_model,
            train_config,
            complex_config,
            model_name=complex_run_name,
            use_matryoshka=True,
        )

        print("\nTraining complete!")
        return

    # ===========================================================================
    # Spectral-Augmented Transformer (SAT) Training
    # ===========================================================================
    if args.sat:
        print("\n" + "=" * 60)
        print("Spectral-Augmented Transformer (SAT) Mode")
        print("=" * 60)

        # Create SAT config
        sat_config = SATConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            seq_len=512,  # Match TinyStories default
            d_spectral=args.sat_d_spectral,
            n_fno_layers=args.sat_n_fno_layers,
            k_max=args.sat_k_max,
            integration_mode=args.sat_integration,
            aux_loss_weight=args.sat_aux_weight,
            aux_loss_weight_min=args.sat_aux_weight_min,
        )

        print(f"SAT config: {sat_config.experiment_summary()}")

        # Create SAT model
        sat_model = SpectralAugmentedTransformer(sat_config)
        print(f"SAT parameters: {format_params(sat_count_parameters(sat_model))}")

        # Also create NanoGPT baseline for comparison
        nano_config = NanoGPTConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
        nanogpt_baseline = NanoGPT(nano_config)
        print(f"NanoGPT baseline parameters: {format_params(count_parameters(nanogpt_baseline))}")

        # Train SAT
        print("\n" + "=" * 60)
        print("Training SAT")
        print("=" * 60)
        sat_run_name = f"{args.run_name}_sat" if args.run_name else "sat"
        sat_model = train_sat(
            sat_model,
            train_config,
            sat_config,
            model_name=sat_run_name,
        )

        # Optionally train baseline for comparison
        if args.model in ["nanogpt", "both"]:
            print("\n" + "=" * 60)
            print("Training NanoGPT Baseline (for comparison)")
            print("=" * 60)
            nanogpt_run_name = f"{args.run_name}_nanogpt" if args.run_name else "nanogpt"
            nanogpt_baseline = train_model(
                nanogpt_baseline,
                train_config,
                nano_config,
                model_name=nanogpt_run_name,
                use_matryoshka=False,
            )

        print("\nTraining complete!")
        return

    # ===========================================================================
    # Standard Real-Valued Training
    # ===========================================================================

    # Print experiment configuration
    print(f"Experimental features: {esmt_config.experiment_summary()}")

    # Create matched models
    print("Creating models with matched parameter counts...")
    esmt, nanogpt, esmt_config, nano_config = create_matched_models(esmt_config, nano_config)

    # Apply spectral initialization if configured
    if esmt_config.spectral_init_checkpoint:
        print("\n" + "=" * 60)
        print("Applying Spectral Initialization")
        print("=" * 60)
        esmt = initialize_esmt_from_nanogpt(
            esmt,
            esmt_config.spectral_init_checkpoint,
            dct_token_emb=esmt_config.dct_token_emb,
            dct_pos_emb=esmt_config.dct_pos_emb,
            dct_lm_head=esmt_config.dct_lm_head,
            verbose=True,
        )

        # Freeze embeddings if configured
        if esmt_config.freeze_embeddings:
            print("\nFreezing DCT'd embeddings...")
            freeze_embeddings(
                esmt,
                freeze_token=esmt_config.dct_token_emb,
                freeze_pos=esmt_config.dct_pos_emb,
                freeze_lm_head=esmt_config.dct_lm_head,
            )
            trainable, frozen = count_frozen_parameters(esmt)
            print(f"  Trainable parameters: {format_params(trainable)}")
            print(f"  Frozen parameters: {format_params(frozen)}")

        # Show initial spectral energy distribution
        analyze_embedding_spectrum(esmt, "Initial Embedding Spectrum (after DCT init)")

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
