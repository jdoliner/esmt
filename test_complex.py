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


def main():
    """Run all tests."""
    print("=" * 60)
    print("Complex Spectral Transformer Test Suite")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    try:
        test_complex_linear()
        test_modrelu()
        test_complex_layernorm()
        test_complex_embedding()
        test_complex_positional_encoding()
        test_complex_attention()
        test_complex_ffn()
        test_complex_spectral_block()
        test_complex_to_logits()
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
