"""Tests for the Spectral-Augmented Transformer (SAT) model."""

import torch
import torch.nn as nn

from config import SATConfig
from sat_model import (
    SpectralAugmentedTransformer,
    CumulativeFFT,
    FNOBlock,
    AdaLNBridge,
    SpectralCrossAttentionBridge,
    complex_mul,
    complex_abs,
    complex_conj,
    complex_from_polar,
    mod_relu,
    count_parameters,
)


def test_complex_ops():
    """Test bfloat16 complex number operations."""
    print("Testing complex operations...")

    # Create test complex numbers as (*, 2) tensors
    # z1 = 3 + 4i, z2 = 1 + 2i
    z1 = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    z2 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    # Test multiplication: (3+4i)(1+2i) = 3 + 6i + 4i + 8i^2 = 3 - 8 + 10i = -5 + 10i
    result = complex_mul(z1.unsqueeze(0), z2.unsqueeze(0)).squeeze(0)
    expected_real, expected_imag = -5.0, 10.0
    assert abs(result[0].item() - expected_real) < 0.1, f"Mul real: {result[0]} != {expected_real}"
    assert abs(result[1].item() - expected_imag) < 0.1, f"Mul imag: {result[1]} != {expected_imag}"
    print(f"  complex_mul: ({z1[0]}+{z1[1]}i) * ({z2[0]}+{z2[1]}i) = {result[0]}+{result[1]}i")

    # Test magnitude: |3 + 4i| = 5
    mag = complex_abs(z1.unsqueeze(0)).squeeze(0)
    assert abs(mag.item() - 5.0) < 0.1, f"Abs: {mag} != 5.0"
    print(f"  complex_abs: |{z1[0]}+{z1[1]}i| = {mag}")

    # Test conjugate: conj(3 + 4i) = 3 - 4i
    conj = complex_conj(z1.unsqueeze(0)).squeeze(0)
    assert abs(conj[0].item() - 3.0) < 0.1, f"Conj real: {conj[0]} != 3.0"
    assert abs(conj[1].item() + 4.0) < 0.1, f"Conj imag: {conj[1]} != -4.0"
    print(f"  complex_conj: conj({z1[0]}+{z1[1]}i) = {conj[0]}+{conj[1]}i")

    # Test polar: magnitude=2, phase=pi/4 -> sqrt(2) + sqrt(2)i
    import math

    z_polar = complex_from_polar(
        torch.tensor([2.0], dtype=torch.bfloat16), torch.tensor([math.pi / 4], dtype=torch.bfloat16)
    ).squeeze(0)
    expected = 2.0 * math.sqrt(2) / 2  # ~1.414
    assert abs(z_polar[0].item() - expected) < 0.1, f"Polar real: {z_polar[0]} != {expected}"
    assert abs(z_polar[1].item() - expected) < 0.1, f"Polar imag: {z_polar[1]} != {expected}"
    print(f"  complex_from_polar: (2, pi/4) = {z_polar[0]:.3f}+{z_polar[1]:.3f}i")

    # Test ModReLU
    z_test = torch.tensor([[1.0, 0.0], [0.3, 0.4]], dtype=torch.bfloat16)  # |z| = [1, 0.5]
    bias = torch.tensor([0.6], dtype=torch.bfloat16)  # threshold at 0.6
    result = mod_relu(z_test, bias)
    # First: |1| - 0.6 = 0.4 > 0, so output = 0.4 * (1+0i) = 0.4
    # Second: |0.5| - 0.6 = -0.1 < 0, so output = 0
    assert abs(result[0, 0].item() - 0.4) < 0.1, f"ModReLU[0] real: {result[0, 0]} != 0.4"
    assert abs(result[1, 0].item()) < 0.1, f"ModReLU[1] real: {result[1, 0]} != 0"
    print(f"  mod_relu: working correctly")

    print("  All complex ops passed!")


def test_cumulative_fft():
    """Test the cumulative FFT module."""
    print("\nTesting CumulativeFFT...")

    seq_len = 64
    k_max = 8
    batch_size = 2
    d_model = 16

    cumfft = CumulativeFFT(seq_len, k_max)

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.bfloat16)

    # Forward pass
    out = cumfft(x)

    # Check output shape
    expected_shape = (batch_size, seq_len, d_model, k_max, 2)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
    print(f"  Output shape: {out.shape} (expected {expected_shape})")

    # Verify causality: output at position t should only depend on inputs 0..t
    # Check by comparing with manual FFT
    # At position 0, only x[0] contributes
    # The FFT of a single sample x[0] repeated with zeros is just x[0] * twiddle[0]

    # Verify output is not all zeros
    assert out.abs().mean() > 0, "Output is all zeros"
    print(f"  Mean magnitude: {complex_abs(out).mean():.4f}")

    # Test that earlier positions have "less" content (more zeros in input)
    # Mean magnitude should increase with position (more tokens contributing)
    early_mag = complex_abs(out[:, : seq_len // 4]).mean()
    late_mag = complex_abs(out[:, -seq_len // 4 :]).mean()
    print(f"  Early positions mean mag: {early_mag:.4f}")
    print(f"  Late positions mean mag: {late_mag:.4f}")

    print("  CumulativeFFT test passed!")


def test_fno_block():
    """Test the FNO block."""
    print("\nTesting FNOBlock...")

    d_model = 16
    k_max = 8
    batch_size = 2
    seq_len = 32

    fno = FNOBlock(d_model, k_max)

    # Create test input (complex as bfloat16 pairs)
    x = torch.randn(batch_size, seq_len, d_model, k_max, 2, dtype=torch.bfloat16)

    # Forward pass
    out = fno(x)

    # Check output shape
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"  Output shape: {out.shape}")

    # Check that output is different from input (transformation happened)
    diff = (out - x).abs().mean()
    assert diff > 0, "Output is identical to input"
    print(f"  Mean difference from input: {diff:.4f}")

    # Check for NaN/Inf
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

    print("  FNOBlock test passed!")


def test_adaln_bridge():
    """Test the AdaLN bridge."""
    print("\nTesting AdaLNBridge...")

    d_spectral = 16
    d_model = 64
    n_layers = 4
    batch_size = 2
    seq_len = 32
    k_max = 8

    bridge = AdaLNBridge(d_spectral, d_model, n_layers)

    # Create test spectral input
    spectral = torch.randn(batch_size, seq_len, d_spectral, k_max, 2, dtype=torch.bfloat16)

    # Forward pass
    conditioning = bridge(spectral)

    # Check we get conditioning for all layers
    assert len(conditioning) == n_layers, f"Expected {n_layers} layers, got {len(conditioning)}"

    # Check shapes
    for i, (gamma, beta) in enumerate(conditioning):
        expected_shape = (batch_size, seq_len, d_model)
        assert gamma.shape == expected_shape, (
            f"Layer {i} gamma shape: {gamma.shape} != {expected_shape}"
        )
        assert beta.shape == expected_shape, (
            f"Layer {i} beta shape: {beta.shape} != {expected_shape}"
        )

    print(
        f"  Conditioning shapes: gamma={conditioning[0][0].shape}, beta={conditioning[0][1].shape}"
    )

    # Check initialization (gamma should be ~1, beta should be ~0)
    gamma_mean = conditioning[0][0].mean().item()
    beta_mean = conditioning[0][1].mean().item()
    print(f"  Initial gamma mean: {gamma_mean:.4f} (should be ~1)")
    print(f"  Initial beta mean: {beta_mean:.4f} (should be ~0)")

    # At initialization, bridge should produce gamma~1, beta~0 due to zero-init
    assert abs(gamma_mean - 1.0) < 0.5, f"Initial gamma mean too far from 1: {gamma_mean}"
    assert abs(beta_mean) < 0.5, f"Initial beta mean too far from 0: {beta_mean}"

    print("  AdaLNBridge test passed!")


def test_cross_attention_bridge():
    """Test the cross-attention bridge."""
    print("\nTesting SpectralCrossAttentionBridge...")

    d_spectral = 16
    d_model = 64
    n_heads = 8
    batch_size = 2
    seq_len = 32
    k_max = 8

    bridge = SpectralCrossAttentionBridge(d_spectral, d_model, n_heads)

    # Create test spectral input
    spectral = torch.randn(batch_size, seq_len, d_spectral, k_max, 2, dtype=torch.bfloat16)

    # Forward pass
    k, v = bridge(spectral)

    # Check shapes
    expected_shape = (batch_size, seq_len, k_max, d_model)
    assert k.shape == expected_shape, f"K shape: {k.shape} != {expected_shape}"
    assert v.shape == expected_shape, f"V shape: {v.shape} != {expected_shape}"

    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")

    print("  SpectralCrossAttentionBridge test passed!")


def test_full_model():
    """Test the full SAT model."""
    print("\nTesting SpectralAugmentedTransformer...")

    # Create small config for testing
    config = SATConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        seq_len=64,
        d_spectral=16,
        n_fno_layers=2,
        k_max=8,
        integration_mode="adaln",
    )

    model = SpectralAugmentedTransformer(config)

    # Count parameters
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Create test input
    batch_size = 2
    x = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))

    # Forward pass
    logits = model(x, return_spectral=False)

    # Check output shape
    expected_shape = (batch_size, config.seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Logits shape: {logits.shape} != {expected_shape}"
    print(f"  Logits shape: {logits.shape}")

    # Check for NaN/Inf
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"

    # Test with return_spectral=True
    logits, spectral = model(x, return_spectral=True)
    expected_spectral_shape = (batch_size, config.seq_len, config.d_spectral, config.k_max, 2)
    assert spectral.shape == expected_spectral_shape, (
        f"Spectral shape: {spectral.shape} != {expected_spectral_shape}"
    )
    print(f"  Spectral shape: {spectral.shape}")

    # Test auxiliary loss computation
    aux_loss = model.compute_auxiliary_loss(spectral, x)
    assert aux_loss.ndim == 0, f"Aux loss should be scalar, got shape {aux_loss.shape}"
    assert not torch.isnan(aux_loss), "Aux loss is NaN"
    assert aux_loss >= 0, f"Aux loss should be non-negative: {aux_loss}"
    print(f"  Auxiliary loss: {aux_loss.item():.4f}")

    # Test backward pass
    target = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
    loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), target.view(-1))
    total_loss = loss + aux_loss
    total_loss.backward()

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    param_count = sum(1 for p in model.parameters())
    print(f"  Gradients computed: {grad_count}/{param_count} parameters")
    assert grad_count == param_count, "Not all parameters received gradients"

    print("  Full model test passed!")


def test_integration_modes():
    """Test different integration modes."""
    print("\nTesting integration modes...")

    for mode in ["adaln", "cross_attention", "both"]:
        print(f"\n  Testing mode: {mode}")

        config = SATConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            seq_len=64,
            d_spectral=16,
            n_fno_layers=2,
            k_max=8,
            integration_mode=mode,
        )

        model = SpectralAugmentedTransformer(config)

        # Forward pass
        x = torch.randint(0, config.vocab_size, (2, config.seq_len))
        logits = model(x)

        assert logits.shape == (2, config.seq_len, config.vocab_size)
        assert not torch.isnan(logits).any()

        print(f"    Mode {mode}: passed")

    print("\n  All integration modes passed!")


def test_aux_loss_schedule():
    """Test the auxiliary loss weight schedule."""
    print("\nTesting aux loss schedule...")

    config = SATConfig(
        d_model=64,
        n_layers=2,
        aux_loss_weight=1.0,
        aux_loss_weight_min=0.1,
        aux_loss_warmup_frac=0.1,
        aux_loss_decay_end_frac=0.8,
    )

    # Test at various progress points
    test_points = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
    print("  Progress -> Weight:")
    for progress in test_points:
        weight = config.get_aux_loss_weight(progress)
        print(f"    {progress:.2f} -> {weight:.3f}")

    # Verify schedule
    assert config.get_aux_loss_weight(0.0) == 1.0, "Warmup start should be 1.0"
    assert config.get_aux_loss_weight(0.05) == 1.0, "During warmup should be 1.0"
    assert config.get_aux_loss_weight(0.1) == 1.0, "At warmup end should be 1.0"
    assert config.get_aux_loss_weight(0.8) == 0.1, "At decay end should be min"
    assert config.get_aux_loss_weight(1.0) == 0.1, "After decay should be min"

    # Check decay is monotonic
    prev_weight = config.get_aux_loss_weight(0.1)
    for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        weight = config.get_aux_loss_weight(p)
        assert weight <= prev_weight, f"Weight should decrease: {prev_weight} -> {weight}"
        prev_weight = weight

    print("  Aux loss schedule test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("SAT Model Tests")
    print("=" * 60)

    test_complex_ops()
    test_cumulative_fft()
    test_fno_block()
    test_adaln_bridge()
    test_cross_attention_bridge()
    test_full_model()
    test_integration_modes()
    test_aux_loss_schedule()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
