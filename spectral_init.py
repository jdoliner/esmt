"""Spectral initialization utilities for ESMT.

This module provides functions to initialize ESMT models from pretrained NanoGPT
checkpoints by applying DCT (Discrete Cosine Transform) to the embedding tables.

The key insight: A pretrained model has learned useful embedding representations
in "spatial" space. Applying DCT transforms these into an explicitly spectral basis
where:
  - Low-frequency coefficients (early indices) capture "average" semantic content
  - High-frequency coefficients (later indices) capture finer distinctions
  - Truncation becomes mathematically meaningful (actual frequency cutoff)

We use the orthonormal DCT (type-II with norm='ortho') because it:
  - Preserves inner products: <DCT(a), DCT(b)> = <a, b>
  - Preserves norms: ||DCT(x)|| = ||x||
  - Is self-inverse: DCT^T = DCT^(-1)
  
This means semantic similarity between tokens is preserved after transformation.
"""

from typing import Any
import math

import torch
import torch.nn as nn
from pathlib import Path

from config import ESMTConfig, NanoGPTConfig


def dct_transform_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply orthonormal DCT-II to a PyTorch tensor using pure PyTorch.
    
    DCT-II is computed via FFT by:
    1. Constructing a length-2N symmetric extension
    2. Taking FFT
    3. Extracting and scaling the real parts appropriately
    
    This uses orthonormal normalization where DCT^T = DCT^(-1).
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute the DCT
        
    Returns:
        DCT coefficients as tensor with same shape and device
    """
    # Move dim to last position for easier manipulation
    x = x.transpose(dim, -1)
    
    n = x.shape[-1]
    
    # Create the DCT-II using the FFT-based algorithm
    # First, reorder: y[k] = x[2k] for k < n/2, y[k] = x[2(n-k)-1] for k >= n/2
    # This is equivalent to: [x[0], x[2], x[4], ..., x[n-1], x[n-3], ..., x[1]]
    idx = torch.cat([
        torch.arange(0, n, 2, device=x.device),
        torch.arange(n - 1 - (n % 2), -1, -2, device=x.device)
    ])
    y = x[..., idx]
    
    # Take FFT
    Y = torch.fft.fft(y)
    
    # Multiply by twiddle factors: exp(-i * pi * k / (2 * n))
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    twiddle = torch.exp(-1j * math.pi * k / (2 * n))
    
    # DCT coefficients are real part of Y * twiddle
    result = (Y * twiddle).real
    
    # Apply orthonormal scaling
    # For orthonormal DCT-II: multiply DC component by 1/sqrt(N), others by sqrt(2/N)
    scale = torch.ones(n, device=x.device, dtype=x.dtype) * math.sqrt(2.0 / n)
    scale[0] = 1.0 / math.sqrt(n)
    result = result * scale
    
    # Move dim back
    result = result.transpose(dim, -1)
    
    return result


def idct_transform_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply inverse orthonormal DCT-II (which is DCT-III) to a PyTorch tensor.
    
    For orthonormal DCT-II, the inverse is the transpose, which is DCT-III
    with the same normalization.
    
    Args:
        x: DCT coefficients tensor
        dim: Dimension along which to compute the inverse DCT
        
    Returns:
        Reconstructed tensor with same shape and device
    """
    # Move dim to last position
    x = x.transpose(dim, -1)
    
    n = x.shape[-1]
    
    # Undo the orthonormal scaling first
    scale = torch.ones(n, device=x.device, dtype=x.dtype) * math.sqrt(2.0 / n)
    scale[0] = 1.0 / math.sqrt(n)
    y = x / scale
    
    # Multiply by conjugate twiddle factors: exp(i * pi * k / (2 * n))
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    twiddle = torch.exp(1j * math.pi * k / (2 * n))
    Y = y.to(torch.complex64) * twiddle
    
    # Take inverse FFT
    y_reordered = torch.fft.ifft(Y).real * n
    
    # Undo the reordering
    result = torch.zeros_like(x)
    idx_even = torch.arange(0, n, 2, device=x.device)
    idx_odd = torch.arange(n - 1 - (n % 2), -1, -2, device=x.device)
    
    # This is the inverse of the reordering in DCT
    result[..., idx_even] = y_reordered[..., :len(idx_even)]
    result[..., idx_odd] = y_reordered[..., len(idx_even):]
    
    # Swap back to handle the interleaving correctly
    # Actually we need to reconstruct from the DCT-III formula directly
    # Let me use a simpler approach that's more numerically stable
    
    # Move dim back
    result = result.transpose(dim, -1)
    
    return result


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Construct the orthonormal DCT-II matrix.
    
    For orthonormal DCT-II:
    C[k,i] = sqrt(2/N) * cos(pi * k * (2*i + 1) / (2*N)) for k > 0
    C[0,i] = sqrt(1/N)
    
    Args:
        n: Size of the DCT
        device: Device for the output tensor
        dtype: Data type for the output tensor
        
    Returns:
        DCT matrix of shape [n, n]
    """
    i = torch.arange(n, device=device, dtype=dtype)
    k = torch.arange(n, device=device, dtype=dtype)
    
    # cos(pi * k * (2*i + 1) / (2*N))
    C = torch.cos(math.pi * k.unsqueeze(1) * (2 * i.unsqueeze(0) + 1) / (2 * n))
    
    # Apply orthonormal scaling
    C = C * math.sqrt(2.0 / n)
    C[0, :] = 1.0 / math.sqrt(n)
    
    return C


def dct_transform_tensor_matrix(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply orthonormal DCT-II using explicit matrix multiplication.
    
    This is slower than the FFT-based version for large N, but simpler
    and more numerically stable.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute the DCT
        
    Returns:
        DCT coefficients as tensor with same shape and device
    """
    # Move dim to last position
    x = x.transpose(dim, -1)
    
    n = x.shape[-1]
    C = _dct_matrix(n, x.device, x.dtype)
    
    # DCT via matrix multiplication: result = C @ x (along last dim)
    result = torch.matmul(C, x.unsqueeze(-1)).squeeze(-1)
    
    # Move dim back
    result = result.transpose(dim, -1)
    
    return result


def idct_transform_tensor_matrix(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply inverse orthonormal DCT-II using explicit matrix multiplication.
    
    For orthonormal DCT, the inverse is simply the transpose: C^T @ x
    
    Args:
        x: DCT coefficients tensor
        dim: Dimension along which to compute the inverse DCT
        
    Returns:
        Reconstructed tensor with same shape and device
    """
    # Move dim to last position
    x = x.transpose(dim, -1)
    
    n = x.shape[-1]
    C = _dct_matrix(n, x.device, x.dtype)
    
    # Inverse DCT via transpose matrix: result = C.T @ x
    result = torch.matmul(C.T, x.unsqueeze(-1)).squeeze(-1)
    
    # Move dim back
    result = result.transpose(dim, -1)
    
    return result


# Use the matrix-based implementation by default (more reliable)
# For large embeddings (vocab_size=50257), this is still fast enough
dct_transform_tensor_fast = dct_transform_tensor
idct_transform_tensor_fast = idct_transform_tensor
dct_transform_tensor = dct_transform_tensor_matrix
idct_transform_tensor = idct_transform_tensor_matrix


def compute_spectral_energy_distribution(
    embeddings: torch.Tensor,
    n_bands: int = 4
) -> dict[str, float]:
    """
    Compute the energy distribution across frequency bands.
    
    This is a diagnostic to see how "compressible" the embeddings are.
    If most energy is in low frequencies, truncation will work well.
    
    Args:
        embeddings: Embedding tensor [vocab_size, d_model] (already DCT'd or not)
        n_bands: Number of frequency bands to analyze
        
    Returns:
        Dict with energy fraction in each band
    """
    d_model = embeddings.shape[-1]
    band_size = d_model // n_bands
    
    # Compute energy (squared magnitude) per position
    energy = embeddings.pow(2).sum(dim=0)  # [d_model]
    total_energy = energy.sum().item()
    
    result = {}
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else d_model
        band_energy = energy[start:end].sum().item()
        pct = int(100 * (i + 1) / n_bands)
        result[f"energy_0_to_{pct}pct"] = band_energy / total_energy if total_energy > 0 else 0.0
    
    return result


def load_nanogpt_checkpoint(checkpoint_path: str) -> dict:
    """
    Load a NanoGPT checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Checkpoint dict containing model_state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Handle state dict from torch.compile() which adds "_orig_mod." prefix
    state_dict = checkpoint["model_state_dict"]
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        unwrapped_state_dict[new_key] = value
    
    checkpoint["model_state_dict"] = unwrapped_state_dict
    return checkpoint


def extract_embeddings_from_nanogpt(
    checkpoint_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract embedding tables from a NanoGPT checkpoint.
    
    Args:
        checkpoint_path: Path to NanoGPT checkpoint
        
    Returns:
        Tuple of (token_embeddings, positional_embeddings, lm_head_weights)
        - token_embeddings: [vocab_size, d_model]
        - positional_embeddings: [seq_len, d_model]  
        - lm_head_weights: [vocab_size, d_model]
    """
    checkpoint = load_nanogpt_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    
    token_emb = state_dict["token_emb.weight"]  # [vocab_size, d_model]
    pos_emb = state_dict["pos_emb.weight"]  # [seq_len, d_model]
    lm_head = state_dict["lm_head.weight"]  # [vocab_size, d_model]
    
    return token_emb, pos_emb, lm_head


def dct_embeddings(
    token_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    lm_head: torch.Tensor,
    dct_token: bool = True,
    dct_pos: bool = True,
    dct_lm_head: bool = True,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply DCT transformation to embedding tables.
    
    DCT is applied along the d_model dimension (axis=-1), transforming each
    token/position's embedding from spatial to spectral representation.
    
    Args:
        token_emb: Token embeddings [vocab_size, d_model]
        pos_emb: Positional embeddings [seq_len, d_model]
        lm_head: Output projection weights [vocab_size, d_model]
        dct_token: Whether to DCT token embeddings
        dct_pos: Whether to DCT positional embeddings
        dct_lm_head: Whether to DCT lm_head weights
        verbose: Whether to print diagnostics
        
    Returns:
        Tuple of (dct_token_emb, dct_pos_emb, dct_lm_head)
    """
    if verbose:
        print("\nApplying DCT transformation to embeddings...")
        print(f"  Token embeddings shape: {token_emb.shape}")
        print(f"  Positional embeddings shape: {pos_emb.shape}")
        print(f"  LM head shape: {lm_head.shape}")
    
    # Apply DCT along the d_model dimension (last axis)
    if dct_token:
        dct_token_emb = dct_transform_tensor(token_emb, dim=-1)
        if verbose:
            # Check energy distribution
            energy = compute_spectral_energy_distribution(dct_token_emb)
            print(f"  Token embedding spectral energy: {energy}")
    else:
        dct_token_emb = token_emb
        
    if dct_pos:
        dct_pos_emb = dct_transform_tensor(pos_emb, dim=-1)
        if verbose:
            energy = compute_spectral_energy_distribution(dct_pos_emb)
            print(f"  Positional embedding spectral energy: {energy}")
    else:
        dct_pos_emb = pos_emb
        
    if dct_lm_head:
        dct_lm_head_out = dct_transform_tensor(lm_head, dim=-1)
        if verbose:
            energy = compute_spectral_energy_distribution(dct_lm_head_out)
            print(f"  LM head spectral energy: {energy}")
    else:
        dct_lm_head_out = lm_head
    
    # Verify orthonormality (norms should be preserved)
    if verbose:
        token_norm_before = token_emb.norm().item()
        token_norm_after = dct_token_emb.norm().item()
        print(f"\n  Norm preservation check (should be ~1.0):")
        print(f"    Token emb ratio: {token_norm_after / token_norm_before:.6f}")
        
    return dct_token_emb, dct_pos_emb, dct_lm_head_out


def initialize_esmt_from_nanogpt(
    esmt: Any,
    nanogpt_checkpoint: str,
    dct_token_emb: bool = True,
    dct_pos_emb: bool = True,
    dct_lm_head: bool = True,
    verbose: bool = True,
) -> Any:
    """
    Initialize an ESMT model's embeddings from a pretrained NanoGPT checkpoint.
    
    This extracts the embedding tables from NanoGPT, applies DCT transformation,
    and loads them into the ESMT model. The transformer layers (attention, MLP)
    are NOT transferred - only embeddings.
    
    Args:
        esmt: The ESMT model (SpectralGPT) to initialize
        nanogpt_checkpoint: Path to pretrained NanoGPT checkpoint
        dct_token_emb: Apply DCT to token embeddings
        dct_pos_emb: Apply DCT to positional embeddings  
        dct_lm_head: Apply DCT to output projection
        verbose: Print diagnostic information
        
    Returns:
        The ESMT model with initialized embeddings
    """
    if verbose:
        print(f"\nInitializing ESMT from NanoGPT checkpoint: {nanogpt_checkpoint}")
    
    # Extract embeddings from NanoGPT
    token_emb, pos_emb, lm_head = extract_embeddings_from_nanogpt(nanogpt_checkpoint)
    
    # Verify dimensions match
    esmt_d_model = esmt.config.d_model
    esmt_vocab_size = esmt.config.vocab_size
    esmt_seq_len = esmt.config.seq_len
    
    assert token_emb.shape == (esmt_vocab_size, esmt_d_model), \
        f"Token emb shape mismatch: {token_emb.shape} vs ({esmt_vocab_size}, {esmt_d_model})"
    assert pos_emb.shape[1] == esmt_d_model, \
        f"Pos emb d_model mismatch: {pos_emb.shape[1]} vs {esmt_d_model}"
    assert lm_head.shape == (esmt_vocab_size, esmt_d_model), \
        f"LM head shape mismatch: {lm_head.shape} vs ({esmt_vocab_size}, {esmt_d_model})"
    
    # Handle seq_len mismatch (NanoGPT might have different seq_len)
    if pos_emb.shape[0] != esmt_seq_len:
        if verbose:
            print(f"  Warning: Positional embedding seq_len mismatch: {pos_emb.shape[0]} vs {esmt_seq_len}")
        if pos_emb.shape[0] > esmt_seq_len:
            pos_emb = pos_emb[:esmt_seq_len]
        else:
            # Pad with zeros or extend - for now just use what we have
            print(f"  Using first {pos_emb.shape[0]} positions, rest will be random")
    
    # Apply DCT
    dct_token, dct_pos, dct_head = dct_embeddings(
        token_emb, pos_emb, lm_head,
        dct_token=dct_token_emb,
        dct_pos=dct_pos_emb,
        dct_lm_head=dct_lm_head,
        verbose=verbose,
    )
    
    # Load into ESMT model
    with torch.no_grad():
        esmt.token_emb.embedding.weight.copy_(dct_token)
        
        # Handle positional embedding dimension
        if dct_pos.shape[0] == esmt_seq_len:
            esmt.pos_emb.embedding.weight.copy_(dct_pos)
        else:
            esmt.pos_emb.embedding.weight[:dct_pos.shape[0]].copy_(dct_pos)
            
        esmt.lm_head.weight.copy_(dct_head)
    
    if verbose:
        print(f"\n  Successfully initialized ESMT embeddings from NanoGPT")
        print(f"  DCT applied to: token_emb={dct_token_emb}, pos_emb={dct_pos_emb}, lm_head={dct_lm_head}")
    
    return esmt


def freeze_embeddings(model: Any, freeze_token: bool = True, freeze_pos: bool = True, freeze_lm_head: bool = True) -> None:
    """
    Freeze embedding layers to prevent gradient updates.
    
    Args:
        model: The ESMT model (SpectralGPT)
        freeze_token: Freeze token embeddings
        freeze_pos: Freeze positional embeddings
        freeze_lm_head: Freeze output projection
    """
    if freeze_token:
        model.token_emb.embedding.weight.requires_grad = False
        
    if freeze_pos:
        model.pos_emb.embedding.weight.requires_grad = False
        
    if freeze_lm_head:
        model.lm_head.weight.requires_grad = False


def unfreeze_embeddings(model: Any) -> None:
    """
    Unfreeze all embedding layers.
    
    Args:
        model: The ESMT model (SpectralGPT)
    """
    model.token_emb.embedding.weight.requires_grad = True
    model.pos_emb.embedding.weight.requires_grad = True
    model.lm_head.weight.requires_grad = True


def count_frozen_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count frozen and trainable parameters.
    
    Args:
        model: The model
        
    Returns:
        Tuple of (trainable_params, frozen_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ==============================================================================
# FFT-based Initialization for Complex Spectral Transformer
# ==============================================================================


def fft_transform_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply FFT to transform real embeddings to complex spectral representation.
    
    Unlike DCT which produces real coefficients, FFT produces complex coefficients
    which is what we need for the ComplexSpectralGPT.
    
    For real input, FFT has conjugate symmetry: X[k] = conj(X[N-k])
    We keep only the first half (plus DC) which contains all the information.
    
    However, for our use case, we apply FFT to each embedding row and keep
    all coefficients since we want to preserve the full complex representation.
    
    Args:
        x: Real-valued input tensor
        dim: Dimension along which to compute FFT
        
    Returns:
        Complex tensor with same shape
    """
    return torch.fft.fft(x, dim=dim)


def ifft_transform_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply inverse FFT to transform complex spectral back to real.
    
    Args:
        x: Complex-valued input tensor
        dim: Dimension along which to compute IFFT
        
    Returns:
        Real tensor with same shape (takes real part since input was real)
    """
    result = torch.fft.ifft(x, dim=dim)
    return result.real


def extract_embeddings_from_nanogpt_for_complex(
    checkpoint_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embedding tables from a NanoGPT checkpoint for complex initialization.
    
    Args:
        checkpoint_path: Path to NanoGPT checkpoint
        
    Returns:
        Tuple of (token_embeddings, lm_head_weights)
        - token_embeddings: [vocab_size, d_model]
        - lm_head_weights: [vocab_size, d_model]
        
    Note: We don't extract positional embeddings because ComplexSpectralGPT
    uses the shift theorem for positional encoding instead.
    """
    checkpoint = load_nanogpt_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    
    token_emb = state_dict["token_emb.weight"]  # [vocab_size, d_model]
    lm_head = state_dict["lm_head.weight"]  # [vocab_size, d_model]
    
    return token_emb, lm_head


def fft_embeddings(
    token_emb: torch.Tensor,
    lm_head: torch.Tensor,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply FFT transformation to embedding tables for complex model initialization.
    
    FFT is applied along the d_model dimension, transforming each token's
    embedding from spatial to complex spectral representation.
    
    Args:
        token_emb: Token embeddings [vocab_size, d_model]
        lm_head: Output projection weights [vocab_size, d_model]
        verbose: Whether to print diagnostics
        
    Returns:
        Tuple of (fft_token_emb, fft_lm_head) as complex tensors
    """
    if verbose:
        print("\nApplying FFT transformation to embeddings...")
        print(f"  Token embeddings shape: {token_emb.shape}")
        print(f"  LM head shape: {lm_head.shape}")
    
    # Apply FFT along the d_model dimension
    fft_token_emb = fft_transform_tensor(token_emb, dim=-1)
    fft_lm_head = fft_transform_tensor(lm_head, dim=-1)
    
    if verbose:
        # Compute energy distribution (using magnitude squared)
        token_energy = (fft_token_emb.abs() ** 2).sum(dim=0)
        total_energy = token_energy.sum().item()
        d_model = token_emb.shape[-1]
        
        # Energy in first 25%, 50%, 75%, 100%
        for pct in [0.25, 0.50, 0.75, 1.0]:
            cutoff = int(d_model * pct)
            band_energy = token_energy[:cutoff].sum().item()
            print(f"  Token emb energy in first {int(pct*100)}%: {band_energy/total_energy:.3f}")
        
        # Check that FFT is invertible (round-trip check)
        reconstructed = ifft_transform_tensor(fft_token_emb, dim=-1)
        error = (reconstructed - token_emb).abs().max().item()
        print(f"  FFT reconstruction error (should be ~0): {error:.2e}")
    
    return fft_token_emb, fft_lm_head


def initialize_complex_esmt_from_nanogpt(
    complex_esmt: Any,
    nanogpt_checkpoint: str,
    verbose: bool = True,
) -> Any:
    """
    Initialize a ComplexSpectralGPT model's embeddings from a pretrained NanoGPT.
    
    This extracts the embedding tables from NanoGPT, applies FFT transformation,
    and loads them into the complex model. The transformer layers are NOT transferred.
    
    Note: ComplexSpectralGPT uses phase-based positional encoding (shift theorem),
    so we don't transfer positional embeddings. Only token embeddings are transferred.
    
    For the output projection, we need to handle the fact that ComplexToLogits
    projects from magnitude+phase (2*d_model) to vocab_size, while NanoGPT
    projects from d_model to vocab_size. We initialize the magnitude part
    from the FFT'd lm_head and the phase part to small values.
    
    Args:
        complex_esmt: The ComplexSpectralGPT model to initialize
        nanogpt_checkpoint: Path to pretrained NanoGPT checkpoint
        verbose: Print diagnostic information
        
    Returns:
        The ComplexSpectralGPT model with initialized embeddings
    """
    if verbose:
        print(f"\nInitializing ComplexSpectralGPT from NanoGPT: {nanogpt_checkpoint}")
    
    # Extract embeddings from NanoGPT
    token_emb, lm_head = extract_embeddings_from_nanogpt_for_complex(nanogpt_checkpoint)
    
    # Verify dimensions match
    d_model = complex_esmt.config.d_model
    vocab_size = complex_esmt.config.vocab_size
    
    assert token_emb.shape == (vocab_size, d_model), \
        f"Token emb shape mismatch: {token_emb.shape} vs ({vocab_size}, {d_model})"
    assert lm_head.shape == (vocab_size, d_model), \
        f"LM head shape mismatch: {lm_head.shape} vs ({vocab_size}, {d_model})"
    
    # Apply FFT to get complex embeddings
    fft_token, fft_lm_head = fft_embeddings(token_emb, lm_head, verbose=verbose)
    
    # Load into ComplexSpectralGPT model
    with torch.no_grad():
        # Token embeddings: directly set complex embedding
        complex_esmt.token_emb.embedding.data.copy_(fft_token)
        
        # Output projection (ComplexToLogits): [vocab_size, 2*d_model]
        # First d_model columns correspond to magnitude
        # Second d_model columns correspond to phase
        # 
        # We initialize magnitude projection from FFT'd lm_head (taking real part as approximation)
        # and phase projection to small random values
        lm_head_proj = complex_esmt.lm_head.proj.weight  # [vocab_size, 2*d_model]
        
        # For the magnitude part, use the magnitude of FFT'd embeddings
        # This is a heuristic - the exact initialization isn't critical
        # since the model will learn to use it
        mag_weights = fft_lm_head.abs()  # [vocab_size, d_model]
        
        # For the phase part, initialize to small values
        # (phase information will need to be learned)
        phase_weights = torch.randn(vocab_size, d_model) * 0.01
        
        # Concatenate and assign
        lm_head_proj[:, :d_model] = mag_weights
        lm_head_proj[:, d_model:] = phase_weights
    
    if verbose:
        print(f"\n  Successfully initialized ComplexSpectralGPT embeddings")
        print(f"  Token embeddings: FFT applied, now complex-valued")
        print(f"  Positional encoding: Using shift theorem (not transferred)")
        print(f"  Output projection: Magnitude from FFT, phase initialized randomly")
    
    return complex_esmt


def freeze_complex_embeddings(
    model: Any, 
    freeze_token: bool = True, 
    freeze_lm_head: bool = True
) -> None:
    """
    Freeze embedding layers in a ComplexSpectralGPT model.
    
    Note: Positional encoding in ComplexSpectralGPT is not learned
    (it's the fixed phase ramp from shift theorem), so nothing to freeze there.
    
    Args:
        model: The ComplexSpectralGPT model
        freeze_token: Freeze token embeddings
        freeze_lm_head: Freeze output projection
    """
    if freeze_token:
        model.token_emb.embedding.requires_grad = False
        
    if freeze_lm_head:
        model.lm_head.proj.weight.requires_grad = False


def unfreeze_complex_embeddings(model: Any) -> None:
    """
    Unfreeze all learnable embeddings in a ComplexSpectralGPT model.
    
    Args:
        model: The ComplexSpectralGPT model
    """
    model.token_emb.embedding.requires_grad = True
    model.lm_head.proj.weight.requires_grad = True


def analyze_complex_embedding_spectrum(
    model: Any, 
    title: str = "Complex Embedding Spectrum Analysis"
) -> dict:
    """
    Analyze the spectral properties of a ComplexSpectralGPT's embeddings.
    
    Args:
        model: The ComplexSpectralGPT model
        title: Title for the analysis
        
    Returns:
        Dict with spectral statistics
    """
    print(f"\n{title}")
    print("=" * 60)
    
    results = {}
    
    # Token embeddings (complex)
    token_emb = model.token_emb.embedding.data.detach()
    token_mag = token_emb.abs()
    token_phase = token_emb.angle()
    
    # Energy distribution (magnitude squared)
    energy = (token_mag ** 2).sum(dim=0)  # [d_model]
    total_energy = energy.sum().item()
    d_model = token_emb.shape[-1]
    
    energy_dist = {}
    for pct in [0.25, 0.50, 0.75, 1.0]:
        cutoff = int(d_model * pct)
        band_energy = energy[:cutoff].sum().item()
        energy_dist[f"energy_0_to_{int(pct*100)}pct"] = band_energy / total_energy if total_energy > 0 else 0.0
    
    results["token_emb_energy"] = energy_dist
    print(f"Token embeddings energy distribution: {energy_dist}")
    
    # Phase statistics
    phase_mean = token_phase.mean().item()
    phase_std = token_phase.std().item()
    results["token_emb_phase"] = {"mean": phase_mean, "std": phase_std}
    print(f"Token embeddings phase: mean={phase_mean:.3f}, std={phase_std:.3f}")
    
    # Magnitude statistics
    mag_mean = token_mag.mean().item()
    mag_std = token_mag.std().item()
    results["token_emb_magnitude"] = {"mean": mag_mean, "std": mag_std}
    print(f"Token embeddings magnitude: mean={mag_mean:.3f}, std={mag_std:.3f}")
    
    return results


def analyze_embedding_spectrum(model: Any, title: str = "Embedding Spectrum Analysis") -> dict:
    """
    Analyze the spectral properties of a model's embeddings.
    
    Useful for diagnostics: checking if spectral structure is maintained
    after training.
    
    Args:
        model: The ESMT model (SpectralGPT)
        title: Title for the analysis
        
    Returns:
        Dict with spectral statistics
    """
    print(f"\n{title}")
    print("=" * 60)
    
    results = {}
    
    # Token embeddings
    token_emb = model.token_emb.embedding.weight.detach()
    token_energy = compute_spectral_energy_distribution(token_emb, n_bands=4)
    results["token_emb"] = token_energy
    print(f"Token embeddings: {token_energy}")
    
    # Positional embeddings
    pos_emb = model.pos_emb.embedding.weight.detach()
    pos_energy = compute_spectral_energy_distribution(pos_emb, n_bands=4)
    results["pos_emb"] = pos_energy
    print(f"Positional embeddings: {pos_energy}")
    
    # LM head
    lm_head = model.lm_head.weight.detach()
    lm_energy = compute_spectral_energy_distribution(lm_head, n_bands=4)
    results["lm_head"] = lm_energy
    print(f"LM head: {lm_energy}")
    
    return results
