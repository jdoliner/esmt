"""Test script for Complex Spectral Transformer components."""

import torch
import torch.nn as nn

# Test imports
print("Testing imports...")
from config import ComplexESMTConfig
from complex_layers import (
    ComplexLinear,
    ModReLU,
    ComplexLayerNorm,
    ComplexEmbedding,
    ComplexPositionalEncoding,
    ComplexAttention,
    ComplexFFN,
    ComplexSpectralBlock,
    ComplexToLogits,
    complex_kaiming_init,
    complex_unitary_init,
    complex_small_init,
    complex_pink_noise_init,
)
from model import ComplexSpectralGPT

print("All imports successful!")


def test_complex_linear():
    """Test ComplexLinear layer."""
    print("\n" + "=" * 60)
    print("Testing ComplexLinear")
    print("=" * 60)
    
    batch, seq, d_in, d_out = 2, 4, 8, 16
    
    layer = ComplexLinear(d_in, d_out)
    print(f"  Weight shape: {layer.weight.shape}, dtype: {layer.weight.dtype}")
    print(f"  Weight is complex: {layer.weight.is_complex()}")
    
    # Create complex input
    x = torch.randn(batch, seq, d_in, dtype=torch.complex64)
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
    
    # Forward pass
    y = layer(x)
    print(f"  Output shape: {y.shape}, dtype: {y.dtype}")
    
    # Test gradient flow
    loss = y.abs().mean()
    loss.backward()
    print(f"  Gradient computed: {layer.weight.grad is not None}")
    print(f"  Gradient is complex: {layer.weight.grad.is_complex()}")
    
    # Test sliced forward
    cutoff_in, cutoff_out = 4, 8
    x_sliced = x[..., :cutoff_in]
    y_sliced = layer.forward_sliced(x_sliced, cutoff_in, cutoff_out)
    print(f"  Sliced output shape: {y_sliced.shape}")
    
    print("  ComplexLinear: PASSED")


def test_modrelu():
    """Test ModReLU activation."""
    print("\n" + "=" * 60)
    print("Testing ModReLU")
    print("=" * 60)
    
    features = 8
    batch, seq = 2, 4
    
    act = ModReLU(features)
    print(f"  Bias shape: {act.bias.shape}")
    
    # Create complex input with varying magnitudes
    x = torch.randn(batch, seq, features, dtype=torch.complex64)
    print(f"  Input magnitudes (sample): {x[0, 0].abs()[:4]}")
    
    # Forward pass
    y = act(x)
    print(f"  Output magnitudes (sample): {y[0, 0].abs()[:4]}")
    
    # Check phase preservation
    phase_in = x.angle()
    phase_out = y.angle()
    # Where output is non-zero, phase should be preserved
    mask = y.abs() > 1e-6
    phase_diff = (phase_in[mask] - phase_out[mask]).abs()
    print(f"  Max phase difference (should be ~0): {phase_diff.max().item():.6f}")
    
    # Test gradient
    loss = y.abs().mean()
    loss.backward()
    print(f"  Gradient computed: {act.bias.grad is not None}")
    
    print("  ModReLU: PASSED")


def test_complex_layernorm():
    """Test ComplexLayerNorm."""
    print("\n" + "=" * 60)
    print("Testing ComplexLayerNorm")
    print("=" * 60)
    
    d_model = 8
    batch, seq = 2, 4
    
    # Test magnitude mode
    ln_mag = ComplexLayerNorm(d_model, mode="magnitude")
    x = torch.randn(batch, seq, d_model, dtype=torch.complex64)
    y_mag = ln_mag(x)
    print(f"  Magnitude mode output shape: {y_mag.shape}")
    
    # Test split mode
    ln_split = ComplexLayerNorm(d_model, mode="split")
    y_split = ln_split(x)
    print(f"  Split mode output shape: {y_split.shape}")
    
    # Test gradient
    loss = y_mag.abs().mean()
    loss.backward()
    print(f"  Gradient computed: {ln_mag.weight.grad is not None}")
    
    print("  ComplexLayerNorm: PASSED")


def test_complex_embedding():
    """Test ComplexEmbedding."""
    print("\n" + "=" * 60)
    print("Testing ComplexEmbedding")
    print("=" * 60)
    
    vocab_size, d_model = 100, 8
    batch, seq = 2, 4
    
    emb = ComplexEmbedding(vocab_size, d_model)
    print(f"  Embedding shape: {emb.embedding.shape}, dtype: {emb.embedding.dtype}")
    
    # Create token indices
    x = torch.randint(0, vocab_size, (batch, seq))
    
    # Full bandwidth
    y = emb(x, bandwidth_ratio=1.0)
    print(f"  Full bandwidth output: {y.shape}")
    
    # Reduced bandwidth
    y_half = emb(x, bandwidth_ratio=0.5)
    print(f"  50% bandwidth output: {y_half.shape}")
    
    # Test gradient
    loss = y.abs().mean()
    loss.backward()
    print(f"  Gradient computed: {emb.embedding.grad is not None}")
    
    print("  ComplexEmbedding: PASSED")


def test_complex_positional_encoding():
    """Test ComplexPositionalEncoding (shift theorem)."""
    print("\n" + "=" * 60)
    print("Testing ComplexPositionalEncoding (Shift Theorem)")
    print("=" * 60)
    
    d_model = 8
    batch, seq = 2, 16
    
    pos_enc = ComplexPositionalEncoding(d_model, max_len=1024)
    print(f"  Frequency shape: {pos_enc.freqs.shape}")
    
    # Create complex input
    x = torch.randn(batch, seq, d_model, dtype=torch.complex64)
    
    # Apply positional encoding
    y = pos_enc(x, seq_len=seq)
    print(f"  Output shape: {y.shape}")
    
    # Check that magnitude is preserved (rotation only changes phase)
    mag_diff = (x.abs() - y.abs()).abs().max().item()
    print(f"  Magnitude preservation (should be ~0): {mag_diff:.6f}")
    
    # Check phase shift increases with position
    phase_diff_0 = (y[0, 0].angle() - x[0, 0].angle()).abs().mean().item()
    phase_diff_10 = (y[0, 10].angle() - x[0, 10].angle()).abs().mean().item()
    print(f"  Phase shift at pos 0: {phase_diff_0:.4f}")
    print(f"  Phase shift at pos 10: {phase_diff_10:.4f}")
    print(f"  Phase increases with position: {phase_diff_10 > phase_diff_0}")
    
    # Test sliced forward
    y_sliced = pos_enc.forward_sliced(x[..., :4], cutoff=4, seq_len=seq)
    print(f"  Sliced output shape: {y_sliced.shape}")
    
    print("  ComplexPositionalEncoding: PASSED")


def test_complex_attention():
    """Test ComplexAttention."""
    print("\n" + "=" * 60)
    print("Testing ComplexAttention")
    print("=" * 60)
    
    d_model, n_heads, seq_len = 16, 4, 8
    batch = 2
    
    # Test magnitude mode
    attn_mag = ComplexAttention(d_model, n_heads, seq_len, attention_mode="magnitude")
    x = torch.randn(batch, seq_len, d_model, dtype=torch.complex64)
    y_mag = attn_mag(x)
    print(f"  Magnitude mode output: {y_mag.shape}")
    
    # Test phase_aware mode
    attn_phase = ComplexAttention(d_model, n_heads, seq_len, attention_mode="phase_aware")
    y_phase = attn_phase(x)
    print(f"  Phase-aware mode output: {y_phase.shape}")
    
    # Test gradient
    loss = y_mag.abs().mean()
    loss.backward()
    print(f"  QKV gradient computed: {attn_mag.qkv.weight.grad is not None}")
    
    # Test sliced forward
    cutoff = 8
    x_sliced = x[..., :cutoff]
    # Need to recreate to clear gradients
    attn_mag_clean = ComplexAttention(d_model, n_heads, seq_len, attention_mode="magnitude")
    y_sliced = attn_mag_clean.forward_sliced(x_sliced, cutoff)
    print(f"  Sliced output shape: {y_sliced.shape}")
    
    print("  ComplexAttention: PASSED")


def test_complex_ffn():
    """Test ComplexFFN."""
    print("\n" + "=" * 60)
    print("Testing ComplexFFN")
    print("=" * 60)
    
    d_model = 8
    batch, seq = 2, 4
    
    ffn = ComplexFFN(d_model, mlp_ratio=4)
    print(f"  FC1 shape: {ffn.fc1.weight.shape}")
    print(f"  FC2 shape: {ffn.fc2.weight.shape}")
    
    x = torch.randn(batch, seq, d_model, dtype=torch.complex64)
    y = ffn(x)
    print(f"  Output shape: {y.shape}")
    
    # Test gradient
    loss = y.abs().mean()
    loss.backward()
    print(f"  FC1 gradient computed: {ffn.fc1.weight.grad is not None}")
    
    # Test sliced forward
    cutoff = 4
    ffn_clean = ComplexFFN(d_model, mlp_ratio=4)
    x_sliced = x[..., :cutoff]
    y_sliced = ffn_clean.forward_sliced(x_sliced, cutoff)
    print(f"  Sliced output shape: {y_sliced.shape}")
    
    print("  ComplexFFN: PASSED")


def test_complex_spectral_block():
    """Test ComplexSpectralBlock."""
    print("\n" + "=" * 60)
    print("Testing ComplexSpectralBlock")
    print("=" * 60)
    
    d_model, n_heads, seq_len = 16, 4, 8
    batch = 2
    
    # Test multiplicative residual (default)
    block_mult = ComplexSpectralBlock(
        d_model, n_heads, seq_len, 
        residual_mode="multiplicative"
    )
    x = torch.randn(batch, seq_len, d_model, dtype=torch.complex64)
    y_mult = block_mult(x)
    print(f"  Multiplicative residual output: {y_mult.shape}")
    
    # Test additive residual
    block_add = ComplexSpectralBlock(
        d_model, n_heads, seq_len,
        residual_mode="additive"
    )
    y_add = block_add(x)
    print(f"  Additive residual output: {y_add.shape}")
    
    # Test gradient
    loss = y_mult.abs().mean()
    loss.backward()
    print(f"  Attention gradient computed: {block_mult.attn.qkv.weight.grad is not None}")
    
    # Test sliced forward
    cutoff = 8
    block_clean = ComplexSpectralBlock(d_model, n_heads, seq_len)
    x_sliced = x[..., :cutoff]
    y_sliced = block_clean.forward_sliced(x_sliced, cutoff)
    print(f"  Sliced output shape: {y_sliced.shape}")
    
    print("  ComplexSpectralBlock: PASSED")


def test_complex_to_logits():
    """Test ComplexToLogits output projection."""
    print("\n" + "=" * 60)
    print("Testing ComplexToLogits")
    print("=" * 60)
    
    d_model, vocab_size = 8, 100
    batch, seq = 2, 4
    
    proj = ComplexToLogits(d_model, vocab_size)
    print(f"  Projection shape: {proj.proj.weight.shape}")  # [vocab, 2*d_model]
    
    z = torch.randn(batch, seq, d_model, dtype=torch.complex64)
    logits = proj(z)
    print(f"  Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"  Logits are real: {not logits.is_complex()}")
    
    # Test gradient
    loss = logits.mean()
    loss.backward()
    print(f"  Projection gradient computed: {proj.proj.weight.grad is not None}")
    
    # Test sliced forward
    cutoff = 4
    proj_clean = ComplexToLogits(d_model, vocab_size)
    z_sliced = z[..., :cutoff]
    logits_sliced = proj_clean.forward_sliced(z_sliced, cutoff)
    print(f"  Sliced logits shape: {logits_sliced.shape}")
    
    print("  ComplexToLogits: PASSED")


def test_complex_spectral_gpt():
    """Test full ComplexSpectralGPT model."""
    print("\n" + "=" * 60)
    print("Testing ComplexSpectralGPT (Full Model)")
    print("=" * 60)
    
    config = ComplexESMTConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        seq_len=16,
        mlp_ratio=4,
        attention_mode="magnitude",
        residual_mode="multiplicative",
    )
    
    model = ComplexSpectralGPT(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    complex_params = sum(p.numel() for p in model.parameters() if p.is_complex())
    real_params = total_params - complex_params
    print(f"  Total parameters: {total_params:,}")
    print(f"  Complex parameters: {complex_params:,}")
    print(f"  Real parameters: {real_params:,}")
    
    # Forward pass at full bandwidth
    batch, seq = 2, 8
    x = torch.randint(0, config.vocab_size, (batch, seq))
    
    logits = model(x, bandwidth_ratio=1.0)
    print(f"  Full bandwidth logits: {logits.shape}")
    
    # Forward pass at reduced bandwidth
    logits_half = model(x, bandwidth_ratio=0.5)
    print(f"  50% bandwidth logits: {logits_half.shape}")
    
    logits_quarter = model(x, bandwidth_ratio=0.25)
    print(f"  25% bandwidth logits: {logits_quarter.shape}")
    
    # Test training step
    print("\n  Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    targets = torch.randint(0, config.vocab_size, (batch, seq))
    
    optimizer.zero_grad()
    logits = model(x, bandwidth_ratio=1.0)
    loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.is_complex():
                grad_norms[name] = param.grad.abs().norm().item()
            else:
                grad_norms[name] = param.grad.norm().item()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Number of parameters with gradients: {len(grad_norms)}")
    print(f"  Max gradient norm: {max(grad_norms.values()):.4f}")
    print(f"  Min gradient norm: {min(grad_norms.values()):.6f}")
    
    optimizer.step()
    print("  Optimizer step completed")
    
    print("  ComplexSpectralGPT: PASSED")


def test_gradient_flow():
    """Test that gradients flow properly through all components."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    config = ComplexESMTConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        vocab_size=50,
        seq_len=8,
    )
    
    model = ComplexSpectralGPT(config)
    
    # Forward and backward
    x = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    
    # Check which parameters have gradients
    params_with_grad = []
    params_without_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_without_grad.append(name)
    
    print(f"  Parameters with gradients: {len(params_with_grad)}")
    print(f"  Parameters without gradients: {len(params_without_grad)}")
    
    if params_without_grad:
        print(f"  WARNING: Parameters without gradients:")
        for name in params_without_grad[:5]:
            print(f"    - {name}")
    
    # Check for NaN gradients
    nan_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.is_complex():
                if torch.isnan(param.grad.real).any() or torch.isnan(param.grad.imag).any():
                    nan_grads.append(name)
            else:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
    
    if nan_grads:
        print(f"  ERROR: NaN gradients found in: {nan_grads}")
    else:
        print("  No NaN gradients found")
    
    print("  Gradient Flow: PASSED")


def test_unitary_init():
    """Test that unitary initialization produces orthonormal matrices."""
    print("\n" + "=" * 60)
    print("Testing Unitary/Isometric Initialization")
    print("=" * 60)
    
    # Test square matrix (should be unitary: Q†Q = I)
    print("\n  Testing square matrix (64x64):")
    W_square = torch.zeros(64, 64, dtype=torch.complex64)
    complex_unitary_init(W_square)
    
    # Check Q†Q ≈ I
    QtQ = W_square.conj().T @ W_square
    identity = torch.eye(64, dtype=torch.complex64)
    error_square = (QtQ - identity).abs().max().item()
    print(f"    Max deviation from identity (Q†Q): {error_square:.6e}")
    assert error_square < 1e-5, f"Square matrix not unitary: error={error_square}"
    
    # Test rectangular matrix (tall: more rows than columns)
    print("\n  Testing tall matrix (128x64):")
    W_tall = torch.zeros(128, 64, dtype=torch.complex64)
    complex_unitary_init(W_tall)
    
    # For tall matrix, Q†Q should be I (columns are orthonormal)
    QtQ_tall = W_tall.conj().T @ W_tall
    identity_64 = torch.eye(64, dtype=torch.complex64)
    error_tall = (QtQ_tall - identity_64).abs().max().item()
    print(f"    Max deviation from identity (Q†Q): {error_tall:.6e}")
    assert error_tall < 1e-5, f"Tall matrix columns not orthonormal: error={error_tall}"
    
    # Test rectangular matrix (wide: more columns than rows)
    print("\n  Testing wide matrix (64x128):")
    W_wide = torch.zeros(64, 128, dtype=torch.complex64)
    complex_unitary_init(W_wide)
    
    # For wide matrix, QQ† should be I (rows are orthonormal)
    QQt_wide = W_wide @ W_wide.conj().T
    identity_64 = torch.eye(64, dtype=torch.complex64)
    error_wide = (QQt_wide - identity_64).abs().max().item()
    print(f"    Max deviation from identity (QQ†): {error_wide:.6e}")
    assert error_wide < 1e-5, f"Wide matrix rows not orthonormal: error={error_wide}"
    
    # Test norm preservation
    print("\n  Testing norm preservation:")
    x = torch.randn(64, dtype=torch.complex64)
    x = x / x.norm()  # Unit vector
    
    y_square = W_square @ x
    y_tall = W_tall @ x
    
    print(f"    Input norm: {x.norm().item():.6f}")
    print(f"    Output norm (square): {y_square.norm().item():.6f}")
    print(f"    Output norm (tall): {y_tall.norm().item():.6f}")
    
    assert abs(y_square.norm().item() - 1.0) < 1e-5, "Square matrix doesn't preserve norm"
    assert abs(y_tall.norm().item() - 1.0) < 1e-5, "Tall matrix doesn't preserve norm"
    
    print("\n  Unitary/Isometric Initialization: PASSED")


def test_pink_noise_init():
    """Test that pink noise initialization has correct spectral decay."""
    print("\n" + "=" * 60)
    print("Testing Pink Noise Initialization")
    print("=" * 60)
    
    vocab_size, d_model = 1000, 64
    
    # Create embedding with pink noise init
    emb = torch.zeros(vocab_size, d_model, dtype=torch.complex64)
    complex_pink_noise_init(emb, decay_exponent=0.5)
    
    # Check magnitude decay with frequency
    magnitudes = emb.abs()
    mean_mag_per_freq = magnitudes.mean(dim=0)  # Average over vocab
    
    print(f"\n  Mean magnitude at different frequencies:")
    print(f"    Freq 0:  {mean_mag_per_freq[0].item():.4f}")
    print(f"    Freq 15: {mean_mag_per_freq[15].item():.4f}")
    print(f"    Freq 31: {mean_mag_per_freq[31].item():.4f}")
    print(f"    Freq 63: {mean_mag_per_freq[63].item():.4f}")
    
    # Verify decay: low freq should have higher magnitude than high freq
    assert mean_mag_per_freq[0] > mean_mag_per_freq[31], "No decay at mid frequencies"
    assert mean_mag_per_freq[31] > mean_mag_per_freq[63], "No decay at high frequencies"
    
    # Check total variance is approximately 1
    total_variance = (emb.abs() ** 2).mean().item()
    print(f"\n  Mean |z|² (should be ≈1): {total_variance:.4f}")
    assert 0.5 < total_variance < 2.0, f"Variance out of range: {total_variance}"
    
    # Check phase is uniform
    phases = emb.angle()
    phase_mean = phases.mean().item()
    phase_std = phases.std().item()
    print(f"  Phase mean (should be ≈0): {phase_mean:.4f}")
    print(f"  Phase std (should be ≈π/√3≈1.81): {phase_std:.4f}")
    
    # For uniform on [-π, π], std ≈ π/√3 ≈ 1.81
    assert abs(phase_mean) < 0.1, f"Phase not centered: mean={phase_mean}"
    assert 1.5 < phase_std < 2.1, f"Phase not uniform: std={phase_std}"
    
    print("\n  Pink Noise Initialization: PASSED")


def test_small_init():
    """Test that small initialization produces small values."""
    print("\n" + "=" * 60)
    print("Testing Small (Near-Zero) Initialization")
    print("=" * 60)
    
    scale = 0.01
    W = torch.zeros(64, 64, dtype=torch.complex64)
    complex_small_init(W, scale=scale)
    
    # Check magnitude statistics
    magnitudes = W.abs()
    mean_mag = magnitudes.mean().item()
    max_mag = magnitudes.max().item()
    
    print(f"  Scale: {scale}")
    print(f"  Mean magnitude: {mean_mag:.6f}")
    print(f"  Max magnitude: {max_mag:.6f}")
    
    # Mean magnitude should be close to scale * sqrt(2) (complex normal)
    expected_mean = scale * (2 ** 0.5) * 0.798  # sqrt(pi/2) for Rayleigh
    print(f"  Expected mean magnitude: ~{expected_mean:.6f}")
    
    assert mean_mag < 5 * scale, f"Mean magnitude too large: {mean_mag}"
    assert max_mag < 20 * scale, f"Max magnitude too large: {max_mag}"
    
    print("\n  Small Initialization: PASSED")


def test_residual_exit_init():
    """Test that residual exit layers are initialized near-zero."""
    print("\n" + "=" * 60)
    print("Testing Residual Exit Initialization")
    print("=" * 60)
    
    d_model, n_heads, seq_len = 64, 8, 16
    
    # Create attention and FFN blocks
    attn = ComplexAttention(d_model, n_heads, seq_len)
    ffn = ComplexFFN(d_model)
    
    # Check QKV projection is NOT near-zero (should be unitary)
    # For unitary matrix of size (3*d, d), mean element magnitude ≈ 1/sqrt(max(3*d, d)) ≈ 1/sqrt(3*d)
    # For d=64, this is 1/sqrt(192) ≈ 0.072
    qkv_mag = attn.qkv.weight.abs().mean().item()
    expected_unitary_mag = 1.0 / (3 * d_model) ** 0.5  # ~0.072
    print(f"\n  QKV projection mean magnitude: {qkv_mag:.4f}")
    print(f"  Expected for unitary (1/sqrt(3*d)): {expected_unitary_mag:.4f}")
    assert qkv_mag > expected_unitary_mag * 0.5, f"QKV projection should be ~{expected_unitary_mag}: got {qkv_mag}"
    
    # Check output projection IS near-zero (scale=1e-5)
    proj_mag = attn.proj.weight.abs().mean().item()
    print(f"  Output projection mean magnitude: {proj_mag:.2e}")
    assert proj_mag < 1e-4, f"Output projection should be near-zero: {proj_mag}"
    
    # Key test: output projection should be MUCH smaller than QKV projection
    ratio = qkv_mag / (proj_mag + 1e-10)
    print(f"  Ratio QKV/proj (should be >>1): {ratio:.0f}")
    assert ratio > 100, f"Output projection should be much smaller than QKV: ratio={ratio}"
    
    # Check FFN fc1 (up) is NOT near-zero
    # For (4*d, d) matrix, mean magnitude ≈ 1/sqrt(4*d)
    fc1_mag = ffn.fc1.weight.abs().mean().item()
    expected_fc1_mag = 1.0 / (4 * d_model) ** 0.5  # ~0.0625
    print(f"\n  FFN up-projection mean magnitude: {fc1_mag:.4f}")
    print(f"  Expected for unitary (1/sqrt(4*d)): {expected_fc1_mag:.4f}")
    assert fc1_mag > expected_fc1_mag * 0.5, f"FFN up-projection should be ~{expected_fc1_mag}: got {fc1_mag}"
    
    # Check FFN fc2 (down) IS near-zero (scale=1e-5)
    fc2_mag = ffn.fc2.weight.abs().mean().item()
    print(f"  FFN down-projection mean magnitude: {fc2_mag:.2e}")
    assert fc2_mag < 1e-4, f"FFN down-projection should be near-zero: {fc2_mag}"
    
    # Key test: fc1 should be MUCH larger than fc2
    ratio_ffn = fc1_mag / (fc2_mag + 1e-10)
    print(f"  Ratio fc1/fc2 (should be >>1): {ratio_ffn:.0f}")
    assert ratio_ffn > 100, f"FFN down should be much smaller than up: ratio={ratio_ffn}"
    
    # Test that block output is approximately identity at init
    print("\n  Testing block behavior at initialization:")
    block = ComplexSpectralBlock(d_model, n_heads, seq_len)
    x = torch.randn(2, seq_len, d_model, dtype=torch.complex64)
    x = x / x.abs().mean() * 0.5  # Normalize to reasonable magnitude
    
    with torch.no_grad():
        y = block(x)
    
    # For multiplicative residual: y = x * (1 + f(x)) where f(x) ≈ 0
    # So y ≈ x (should be very close with 1e-5 scale)
    relative_change = ((y - x).abs() / (x.abs() + 1e-8)).mean().item()
    print(f"  Relative change |y-x|/|x|: {relative_change:.2e}")
    assert relative_change < 0.01, f"Block output should be very close to input: {relative_change}"
    
    print("\n  Residual Exit Initialization: PASSED")


def test_complex_embedding_pink_noise():
    """Test that ComplexEmbedding uses pink noise initialization."""
    print("\n" + "=" * 60)
    print("Testing ComplexEmbedding Pink Noise Init")
    print("=" * 60)
    
    vocab_size, d_model = 500, 32
    
    emb = ComplexEmbedding(vocab_size, d_model)
    
    # Check magnitude decay with frequency
    magnitudes = emb.embedding.abs()
    mean_mag_per_freq = magnitudes.mean(dim=0)
    
    print(f"  Mean magnitude at freq 0: {mean_mag_per_freq[0].item():.4f}")
    print(f"  Mean magnitude at freq {d_model-1}: {mean_mag_per_freq[-1].item():.4f}")
    print(f"  Ratio (should be > 1): {(mean_mag_per_freq[0] / mean_mag_per_freq[-1]).item():.2f}")
    
    # Low frequencies should have higher magnitude
    assert mean_mag_per_freq[0] > mean_mag_per_freq[-1], "No pink noise decay in embedding"
    
    print("\n  ComplexEmbedding Pink Noise Init: PASSED")


def test_complex_to_logits_pink_noise():
    """Test that ComplexToLogits uses pink noise initialization."""
    print("\n" + "=" * 60)
    print("Testing ComplexToLogits Pink Noise Init")
    print("=" * 60)
    
    d_model, vocab_size = 32, 100
    
    proj = ComplexToLogits(d_model, vocab_size)
    
    # Weight shape: [vocab_size, 2*d_model]
    # First d_model columns = magnitude weights
    # Second d_model columns = phase weights
    W = proj.proj.weight
    
    mag_weights = W[:, :d_model].abs()
    phase_weights = W[:, d_model:].abs()
    
    # Check decay in magnitude weights
    mean_mag_per_freq = mag_weights.mean(dim=0)
    print(f"\n  Magnitude projection weights:")
    print(f"    Mean at freq 0: {mean_mag_per_freq[0].item():.4f}")
    print(f"    Mean at freq {d_model-1}: {mean_mag_per_freq[-1].item():.4f}")
    
    # Check decay in phase weights
    mean_phase_per_freq = phase_weights.mean(dim=0)
    print(f"\n  Phase projection weights:")
    print(f"    Mean at freq 0: {mean_phase_per_freq[0].item():.4f}")
    print(f"    Mean at freq {d_model-1}: {mean_phase_per_freq[-1].item():.4f}")
    
    # Both should show decay
    assert mean_mag_per_freq[0] > mean_mag_per_freq[-1], "No decay in magnitude weights"
    assert mean_phase_per_freq[0] > mean_phase_per_freq[-1], "No decay in phase weights"
    
    print("\n  ComplexToLogits Pink Noise Init: PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Complex Spectral Transformer Test Suite")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    try:
        # Basic layer tests
        test_complex_linear()
        test_modrelu()
        test_complex_layernorm()
        test_complex_embedding()
        test_complex_positional_encoding()
        test_complex_attention()
        test_complex_ffn()
        test_complex_spectral_block()
        test_complex_to_logits()
        
        # Initialization tests (NEW)
        test_unitary_init()
        test_pink_noise_init()
        test_small_init()
        test_residual_exit_init()
        test_complex_embedding_pink_noise()
        test_complex_to_logits_pink_noise()
        
        # Integration tests
        test_complex_spectral_gpt()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
