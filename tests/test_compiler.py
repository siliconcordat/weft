"""Tests for deterministic CPU float64 BetaEditNullspaceProjector and SGMV arena."""

from __future__ import annotations

import numpy as np
import pytest

from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.quant import dequantize_int4, dequantize_int8, quantize_int4, quantize_int8


def test_spd_cholesky_reduction_matches_dense_betaedit_eq9() -> None:
    """Verify exact algebraic equivalence to dense (d_in x d_in) BetaEdit Eq. (9)."""
    rng = np.random.default_rng(20261119)
    d_in, d_out, u = 48, 36, 4
    w0 = rng.standard_normal((d_out, d_in)) * 0.2
    k0 = rng.standard_normal((d_in, 16))
    k_hist = rng.standard_normal((d_in, 6)) * 0.3
    k_edit = rng.standard_normal((d_in, u))
    r_edit = rng.standard_normal((d_out, u))

    cfg = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=u,
        lambda_leakage=50.0,
        lambda_ridge=5.0,
        nullspace_threshold=1e-2,
        threshold_mode="relative_max",
        quant_dtype="f32",
    )
    proj = BetaEditNullspaceProjector(w0, cfg)
    proj.initialize_anchor_covariance(k0)
    # Inject intermediate history update between projector refreshes
    hist_patch = proj.stage_edit(k_hist, np.zeros((d_out, 6)), is_residual=True, rank=u)
    proj.commit_patch(hist_patch, k_hist)

    # Dense BetaEdit Eq. (9): Delta = R K^T P (lambda_2 I + (C^(t) + K K^T) P)^{-1}
    p_mat = proj.projection_matrix
    m_mat = proj.cov_accum + (k_edit @ k_edit.T)
    lhs = cfg.lambda_ridge * np.eye(d_in) + (m_mat @ p_mat)
    dense_delta = r_edit @ k_edit.T @ p_mat @ np.linalg.inv(lhs)

    # Compare against unquantized float64 Cholesky solve inside projector
    k_proj = proj.u_small.T @ k_edit
    spd = cfg.lambda_ridge * np.eye(proj.nullspace_dim) + (
        proj.u_small.T @ proj.cov_accum @ proj.u_small
    ) + (k_proj @ k_proj.T)
    solved = np.linalg.solve(spd, k_proj)
    reduced_delta = r_edit @ (proj.u_small @ solved).T

    rel_err = np.linalg.norm(reduced_delta - dense_delta, "fro") / np.linalg.norm(
        dense_delta, "fro"
    )
    assert rel_err < 1e-12


def test_zero_byte_deterministic_rederivation_and_rollback() -> None:
    rng = np.random.default_rng(42)
    d_in, d_out = 64, 32
    w0 = rng.standard_normal((d_out, d_in)) * 0.1
    k0 = rng.standard_normal((d_in, 24))
    keys = rng.standard_normal((d_in, 6))
    targets = rng.standard_normal((d_out, 6)) * 0.5

    cfg = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=8,
        lambda_leakage=200.0,
        lambda_ridge=2.0,
        nullspace_threshold=1e-3,
        threshold_mode="relative_max",
        quant_dtype="bf16",
    )
    proj = BetaEditNullspaceProjector(w0, cfg)
    proj.initialize_anchor_covariance(k0)
    patch = proj.stage_edit(keys, targets, is_residual=True, rank=4, slot_name="corpus")

    # Veto 1: exact 0-byte deterministic re-derivation from scratch
    ok, info = verify_rederivation(
        patch,
        base_weight=w0,
        config=cfg,
        k0=k0,
        keys=keys,
        target_values_or_residuals=targets,
        is_residual=True,
        rank=4,
    )
    assert ok, f"0-byte re-derivation failed: {info}"

    # Verify frozen W0, additive forward pass, and O(1) rollback
    x_probe = rng.standard_normal((10, d_in))
    y_base = proj.forward(x_probe)
    cov_before = proj.cov_accum.copy()

    slot = proj.commit_patch(patch, keys)
    assert proj.active_slot_count == 1
    y_patched = proj.forward(x_probe)
    assert not np.allclose(y_patched, y_base)

    # Dynamic scale=0.0 matches base output without unloading
    y_muted = proj.forward(x_probe, slot_scales={slot: 0.0})
    assert np.array_equal(y_muted, y_base)

    # O(1) rollback restores bit-identical W0 output and covariance
    proj.rollback_slot(slot, revert_covariance=True)
    assert proj.active_slot_count == 0
    assert np.array_equal(proj.forward(x_probe), y_base)
    assert np.allclose(proj.cov_accum, cov_before, atol=1e-12)


def test_tcb_gate_rejects_excessive_norm_before_commit() -> None:
    rng = np.random.default_rng(7)
    w0 = rng.standard_normal((16, 32))
    k0 = rng.standard_normal((32, 8))
    cfg = BetaEditConfig(d_in=32, d_out=16, max_rank=4, quant_dtype="bf16")
    proj = BetaEditNullspaceProjector(w0, cfg)
    proj.initialize_anchor_covariance(k0)
    patch = proj.stage_edit(
        rng.standard_normal((32, 4)),
        rng.standard_normal((16, 4)) * 20.0,
        is_residual=True,
    )
    tight_policy = GatePolicy(max_frob_sq_num=1, max_frob_sq_den=100)
    with pytest.raises(PermissionError, match="TCB gate rejected"):
        proj.commit_patch(patch, rng.standard_normal((32, 4)), gate_policy=tight_policy)
    assert proj.active_slot_count == 0


def test_sequential_betaedit_lowers_k0_leakage_versus_static_unregularized() -> None:
    rng = np.random.default_rng(99)
    d_in, d_out = 48, 24
    w0 = rng.standard_normal((d_out, d_in)) * 0.1
    # Rank-16 preserved space inside 48 dims
    k0 = rng.standard_normal((d_in, 16))

    cfg_beta = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=4,
        max_slots=32,
        lambda_leakage=500.0,
        lambda_ridge=1.0,
        nullspace_threshold=1e-6,
        threshold_mode="absolute",
        refresh_period=5,
        quant_dtype="bf16",
    )
    cfg_unreg = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=4,
        max_slots=32,
        lambda_leakage=0.0,
        lambda_ridge=1.0,
        nullspace_threshold=1e9,
        threshold_mode="absolute",
        refresh_period=1000,
        quant_dtype="bf16",
    )
    p_beta = BetaEditNullspaceProjector(w0, cfg_beta)
    p_beta.initialize_anchor_covariance(k0)
    p_unreg = BetaEditNullspaceProjector(w0, cfg_unreg)
    p_unreg.initialize_anchor_covariance(k0)

    for step in range(12):
        k_i = rng.standard_normal((d_in, 2))
        r_i = rng.standard_normal((d_out, 2)) * 0.5
        pb = p_beta.stage_edit(k_i, r_i, is_residual=True, rank=2)
        p_beta.commit_patch(pb, k_i)
        pu = p_unreg.stage_edit(k_i, r_i, is_residual=True, rank=2)
        p_unreg.commit_patch(pu, k_i)

    leak_beta = p_beta.knowledge_leakage_norm(k0)
    leak_unreg = p_unreg.knowledge_leakage_norm(k0)
    assert leak_beta < leak_unreg * 0.05


def test_int8_and_int4_roundtrip_and_subnormal_safety() -> None:
    rng = np.random.default_rng(1234)
    # Odd number of elements (5 x 7 = 35) to exercise nibble padding plus zero and subnormal rows
    mat = rng.standard_normal((5, 7))
    mat[1, :] = 0.0
    mat[2, :] = 1e-45  # subnormal float32 scale underflow guard

    raw8, deq8 = quantize_int8(mat, axis=1)
    assert np.array_equal(dequantize_int8(raw8, (5, 7), axis=1), deq8)
    assert np.all(np.isfinite(deq8))

    raw8_c, deq8_c = quantize_int8(mat, axis=0)
    assert np.array_equal(dequantize_int8(raw8_c, (5, 7), axis=0), deq8_c)
    assert np.all(np.isfinite(deq8_c))

    raw4, deq4 = quantize_int4(mat, axis=1)
    assert np.array_equal(dequantize_int4(raw4, (5, 7), axis=1), deq4)
    assert np.all(np.isfinite(deq4))
    assert len(raw4) < len(raw8)

    raw4_c, deq4_c = quantize_int4(mat, axis=0)
    assert np.array_equal(dequantize_int4(raw4_c, (5, 7), axis=0), deq4_c)
    assert np.all(np.isfinite(deq4_c))

    with pytest.raises(ValueError, match="Expected"):
        dequantize_int4(raw4[:-1], (5, 7))
    with pytest.raises(ValueError, match="non-empty"):
        quantize_int4(np.zeros((0, 4)))


def test_asymmetric_int4_b_int8_a_preserves_nullspace_and_rederives() -> None:
    """Asymmetric Factor Leakage Theorem: INT4 B + INT8 A beats INT4/INT4 and INT8/INT4."""
    rng = np.random.default_rng(2026)
    d_in, d_out = 48, 48
    w0 = rng.standard_normal((d_out, d_in)) * 0.1
    # Exact rank-12 preserved subspace inside R^48 so A_j K0 = 0 in float64
    k0 = rng.standard_normal((d_in, 12))
    keys = rng.standard_normal((d_in, 4))
    residuals = rng.standard_normal((d_out, 4)) * 0.6

    def compile_and_measure(b_dt: str, a_dt: str) -> tuple[float, int, str, bool]:
        cfg = BetaEditConfig(
            d_in=d_in,
            d_out=d_out,
            max_rank=4,
            lambda_leakage=1000.0,
            lambda_ridge=1.0,
            nullspace_threshold=1e-6,
            threshold_mode="absolute",
            quant_dtype="f16",
            b_quant_dtype=b_dt,  # type: ignore[arg-type]
            a_quant_dtype=a_dt,  # type: ignore[arg-type]
        )
        proj = BetaEditNullspaceProjector(w0, cfg)
        proj.initialize_anchor_covariance(k0)
        patch = proj.stage_edit(keys, residuals, is_residual=True, rank=4, slot_name="asym")
        ok, _ = verify_rederivation(
            patch,
            base_weight=w0,
            config=cfg,
            k0=k0,
            keys=keys,
            target_values_or_residuals=residuals,
            is_residual=True,
            rank=4,
        )
        proj.commit_patch(patch, keys)
        leak = proj.knowledge_leakage_norm(k0)
        payload_bytes = len(patch.b_bytes) + len(patch.a_bytes)
        return leak, payload_bytes, patch.certificate.quant_dtype, ok

    leak_88, bytes_88, lbl_88, ok_88 = compile_and_measure("int8", "int8")
    leak_44, bytes_44, lbl_44, ok_44 = compile_and_measure("int4", "int4")
    leak_48, bytes_48, lbl_48, ok_48 = compile_and_measure("int4", "int8")
    leak_84, bytes_84, lbl_84, ok_84 = compile_and_measure("int8", "int4")

    assert ok_88 and ok_44 and ok_48 and ok_84
    assert lbl_48 == "int4/int8"
    assert lbl_84 == "int8/int4"
    # Matched bit budget: d_in == d_out == 48, so bytes_48 == bytes_84
    assert bytes_48 == bytes_84
    assert bytes_44 < bytes_48 < bytes_88
    # Theorem: INT4 B + INT8 A stays near INT8/INT8 and beats BOTH INT4/INT4 and INT8/INT4 by >8x
    assert leak_48 < leak_88 * 1.25
    assert leak_48 < leak_44 * 0.12
    assert leak_48 < leak_84 * 0.12
