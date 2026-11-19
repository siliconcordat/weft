"""Tests for post-quantization division-free FixedRadixCertificate."""

from __future__ import annotations

import numpy as np
import pytest

from weft.certificate import FixedRadixCertificate, GatePolicy, build_certificate
from weft.quant import quantize_bf16, sha256_factor_pair


def test_certificate_exact_singular_values_and_division_free_gate() -> None:
    rng = np.random.default_rng(20261119)
    b_raw = rng.standard_normal((24, 4)) * 0.25
    a_raw = rng.standard_normal((4, 32)) * 0.25
    b_bits, b_q = quantize_bf16(b_raw)
    a_bits, a_q = quantize_bf16(a_raw)
    cov = np.eye(32, dtype=np.float64) * 2.0
    digest = sha256_factor_pair(b_bits.tobytes(), a_bits.tobytes(), b_q.shape, a_q.shape, "bf16")

    cert = build_certificate(
        b_q,
        a_q,
        cov,
        factor_digest=digest,
        quant_dtype="bf16",
        projector_epoch=1,
        radix_scale_exp2=24,
    )

    dense = b_q @ a_q
    expected_frob_sq = float(np.sum(dense * dense))
    expected_spec_sq = float(np.linalg.norm(dense, ord=2) ** 2)
    expected_leak = float(np.trace(dense @ cov @ dense.T))
    scale = float(1 << 24)

    # Ceiling rounding guarantees conservative upper bounds
    assert cert.frobenius_sq_scaled >= expected_frob_sq * scale
    assert cert.frobenius_sq_scaled - expected_frob_sq * scale <= 2.0
    assert cert.spectral_norm_sq_scaled >= expected_spec_sq * scale
    assert cert.leakage_energy_scaled >= expected_leak * scale

    pass_policy = GatePolicy(
        max_frob_sq_num=int(expected_frob_sq * 100) + 10,
        max_frob_sq_den=100,
        max_spectral_sq_num=int(expected_spec_sq * 100) + 10,
        max_spectral_sq_den=100,
        max_leakage_num=int(expected_leak * 100) + 10,
        max_leakage_den=100,
    )
    ok, problems = cert.verify_division_free(pass_policy)
    assert ok and problems == []

    fail_policy = GatePolicy(max_frob_sq_num=1, max_frob_sq_den=1000)
    ok2, problems2 = cert.verify_division_free(fail_policy)
    assert not ok2
    assert "frobenius_norm_exceeds_cap" in problems2


def test_certificate_rejects_negative_and_forged_radix_fields() -> None:
    forged = FixedRadixCertificate(
        radix_scale_exp2=24,
        frobenius_sq_scaled=-1,
        spectral_norm_sq_scaled=0,
        leakage_energy_scaled=0,
        max_safety_dot_sq_scaled=0,
        projector_epoch=1,
        factor_digest="a" * 64,
        quant_dtype="bf16",
    )
    ok, problems = forged.verify_division_free(GatePolicy(max_frob_sq_num=10))
    assert not ok
    assert "negative_frobenius_sq_scaled" in problems

    bad_radix = FixedRadixCertificate(
        radix_scale_exp2=64,
        frobenius_sq_scaled=0,
        spectral_norm_sq_scaled=0,
        leakage_energy_scaled=0,
        max_safety_dot_sq_scaled=0,
        projector_epoch=1,
        factor_digest="a" * 64,
        quant_dtype="bf16",
    )
    ok2, problems2 = bad_radix.verify_division_free(GatePolicy(max_frob_sq_num=10))
    assert not ok2
    assert "invalid_radix_scale_exp2" in problems2


def test_certificate_safety_direction_overlap_blocks_attack() -> None:
    b_safe = np.zeros((16, 1), dtype=np.float64)
    a_safe = np.zeros((1, 16), dtype=np.float64)
    b_safe[0, 0] = 1.0
    a_safe[0, 0] = 1.0

    # Attack aligned directly with safety direction
    b_att = b_safe * 2.0
    a_att = a_safe * 1.0
    cert = build_certificate(
        b_att,
        a_att,
        np.eye(16),
        factor_digest="b" * 64,
        quant_dtype="f32",
        projector_epoch=1,
        safety_directions=[(b_safe, a_safe)],
    )
    policy = GatePolicy(
        max_frob_sq_num=100,
        max_safety_dot_sq_num=1,
        max_safety_dot_sq_den=100,
    )
    ok, problems = cert.verify_division_free(policy)
    assert not ok
    assert "safety_direction_overlap_exceeds_cap" in problems

    # Round-trip JSON digest stability
    parsed = FixedRadixCertificate.from_dict(
        {
            "radix_scale_exp2": cert.radix_scale_exp2,
            "frobenius_sq_scaled": cert.frobenius_sq_scaled,
            "spectral_norm_sq_scaled": cert.spectral_norm_sq_scaled,
            "leakage_energy_scaled": cert.leakage_energy_scaled,
            "max_safety_dot_sq_scaled": cert.max_safety_dot_sq_scaled,
            "projector_epoch": cert.projector_epoch,
            "factor_digest": cert.factor_digest,
            "quant_dtype": cert.quant_dtype,
        }
    )
    assert parsed.sha256() == cert.sha256()


def test_quantize_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        quantize_bf16(np.array([[1.0, np.nan]]))


def test_certificate_rejects_bool_float_impossible_spectral_and_negative_policy() -> None:
    # Reject bool and float in from_dict
    base_dict = {
        "radix_scale_exp2": 24,
        "frobenius_sq_scaled": 100,
        "spectral_norm_sq_scaled": 50,
        "leakage_energy_scaled": 10,
        "max_safety_dot_sq_scaled": 0,
        "projector_epoch": 1,
        "factor_digest": "a" * 64,
        "quant_dtype": "bf16",
    }
    with pytest.raises(TypeError, match="strict int"):
        FixedRadixCertificate.from_dict({**base_dict, "frobenius_sq_scaled": True})
    with pytest.raises(TypeError, match="strict int"):
        FixedRadixCertificate.from_dict({**base_dict, "frobenius_sq_scaled": 100.9})

    # Reject bool/float directly on FixedRadixCertificate in verify_division_free
    bad_bool_cert = FixedRadixCertificate(
        radix_scale_exp2=24,
        frobenius_sq_scaled=True,  # type: ignore[arg-type]
        spectral_norm_sq_scaled=0,
        leakage_energy_scaled=0,
        max_safety_dot_sq_scaled=0,
        projector_epoch=1,
        factor_digest="a" * 64,
        quant_dtype="bf16",
    )
    ok_b, probs_b = bad_bool_cert.verify_division_free(GatePolicy(max_frob_sq_num=10))
    assert not ok_b
    assert "non_integer_frobenius_sq_scaled" in probs_b

    # Reject impossible spectral > frobenius
    impossible = FixedRadixCertificate(
        radix_scale_exp2=24,
        frobenius_sq_scaled=100,
        spectral_norm_sq_scaled=999999,
        leakage_energy_scaled=0,
        max_safety_dot_sq_scaled=0,
        projector_epoch=1,
        factor_digest="a" * 64,
        quant_dtype="bf16",
    )
    ok_imp, probs_imp = impossible.verify_division_free(GatePolicy(max_frob_sq_num=1000))
    assert not ok_imp
    assert "spectral_exceeds_frobenius" in probs_imp

    # Reject negative optional policy caps and bool caps
    with pytest.raises(ValueError, match="non-negative"):
        GatePolicy(max_frob_sq_num=10, max_spectral_sq_num=-1)
    with pytest.raises(ValueError, match="non-negative"):
        GatePolicy(max_frob_sq_num=10, max_leakage_num=-1)
    with pytest.raises(ValueError, match="non-negative"):
        GatePolicy(max_frob_sq_num=10, max_safety_dot_sq_num=-1)
    with pytest.raises(TypeError, match="strict int"):
        GatePolicy(max_frob_sq_num=True)  # type: ignore[arg-type]
