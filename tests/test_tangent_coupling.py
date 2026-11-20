"""Unit and regression tests for weft.tangent and Experiment E8 (python3 -O safe)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.e8_swiglu_rmsnorm_tangent_coupling import run_experiment as run_e8
from weft.compiler import BetaEditNullspaceProjector
from weft.tangent import (
    LayerPackSpec,
    TangentBetaEditConfig,
    compare_subspace_vs_functional_retention_f64,
    compile_multilayer_tangent_suite,
    decompose_rmsnorm_perturbation_f64,
    project_up_residual_rmsnorm_tangent_f64,
    rational_shrinkage_bounds,
    rmsnorm_f64,
    rmsnorm_jacobian_matvec_f64,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_rmsnorm_jacobian_matches_central_finite_differences() -> None:
    rng = np.random.default_rng(101)
    d = 64
    h0 = rng.standard_normal(d) * 1.8
    delta = rng.standard_normal(d) * 0.4
    gamma = 1.0 + 0.2 * rng.standard_normal(d)
    eps = 1e-6
    step = 1e-6

    analytic = rmsnorm_jacobian_matvec_f64(h0, delta, gamma, eps=eps)
    numeric = (
        rmsnorm_f64(h0 + step * delta, gamma, eps=eps)
        - rmsnorm_f64(h0 - step * delta, gamma, eps=eps)
    ) / (2.0 * step)
    _require(float(np.max(np.abs(analytic - numeric))) < 1e-9, "Jacobian finite-diff mismatch")

    h_batch = rng.standard_normal((d, 5))
    d_batch = rng.standard_normal((d, 5))
    j_batch = rmsnorm_jacobian_matvec_f64(h_batch, d_batch, gamma, eps=eps)
    for col in range(5):
        j_col = rmsnorm_jacobian_matvec_f64(h_batch[:, col], d_batch[:, col], gamma, eps=eps)
        _require(float(np.max(np.abs(j_batch[:, col] - j_col))) < 1e-14, "Batch 2D parity mismatch")


def test_rmsnorm_finite_decomposition_and_rational_bounds() -> None:
    rng = np.random.default_rng(202)
    d = 128
    gamma = 1.0 + 0.1 * rng.standard_normal(d)
    for eps in (0.0, 1e-6, 1e-3):
        h0 = rng.standard_normal(d) * 3.0
        delta = rng.standard_normal(d) * 2.5
        dec = decompose_rmsnorm_perturbation_f64(h0, delta, gamma, eps=eps)
        _require(dec.reconstruction_l2_error < 1e-13, "RMSNorm reconstruction error too large")
        _require(
            dec.rational_lower_bound <= dec.pure_tangent_shrinkage_s + 1e-15,
            "Rational lower bound violated",
        )
        _require(
            dec.pure_tangent_shrinkage_s <= dec.rational_upper_bound + 1e-15,
            "Rational upper bound violated",
        )

    for eta in (0.0, 1e-6, 0.01, 0.25, 1.0, 4.0, 25.0):
        low, s_val, high = rational_shrinkage_bounds(eta)
        _require(low <= s_val + 1e-15, f"Lower bound failed at eta={eta}")
        _require(s_val <= high + 1e-15, f"Upper bound failed at eta={eta}")


def test_swiglu_wdown_radial_null_projection_is_exact() -> None:
    rng = np.random.default_rng(303)
    d_in, d_ff, n = 48, 128, 10
    u_raw = rng.standard_normal((d_ff, n))
    gate = rng.standard_normal((d_ff, n))
    w_down = rng.standard_normal((d_in, d_ff)) / np.sqrt(d_ff)
    h_pre = rng.standard_normal((d_in, n)) * 2.0

    u_perp = project_up_residual_rmsnorm_tangent_f64(u_raw, gate, w_down, h_pre)
    dh = w_down @ (gate * u_perp)
    dots = np.sum(dh * h_pre, axis=0)
    _require(float(np.max(np.abs(dots))) < 1e-12, "Radial inner product non-zero")

    u_1d = project_up_residual_rmsnorm_tangent_f64(u_raw[:, 0], gate[:, 0], w_down, h_pre[:, 0])
    _require(float(np.max(np.abs(u_1d - u_perp[:, 0]))) < 1e-14, "1D parity mismatch")


def test_scalar_projection_retention_illusion_counterexample() -> None:
    d_in, d_ff = 16, 32
    b_solo = np.zeros((d_ff, 1), dtype=np.float64)
    b_solo[0, 0] = 1.0
    b_other = np.zeros((d_ff, 1), dtype=np.float64)
    b_other[1, 0] = 10.0
    a_shared = np.zeros((1, d_in), dtype=np.float64)
    a_shared[0, 0] = 1.0

    w_down = np.zeros((d_in, d_ff), dtype=np.float64)
    w_down[1, 0] = 1.0
    w_down[2, 1] = 1.0
    probe_k = np.zeros((d_in, 1), dtype=np.float64)
    probe_k[0, 0] = 1.0
    probe_g = np.ones((d_ff, 1), dtype=np.float64)
    h_pre = np.zeros((d_in, 1), dtype=np.float64)
    h_pre[0, 0] = 5.0
    gamma = np.ones(d_in, dtype=np.float64)

    cmp_res = compare_subspace_vs_functional_retention_f64(
        solo_name="solo",
        scales={"solo": 1.0, "other": 1.0},
        layer_deltas_by_pack={
            "solo": {0: b_solo @ a_shared},
            "other": {0: b_other @ a_shared},
        },
        probe_keys_by_layer={0: probe_k},
        probe_gates_by_layer={0: probe_g},
        w_down_by_layer={0: w_down},
        h_pre=h_pre,
        gamma=gamma,
        eps=1e-6,
    )
    _require(
        abs(cmp_res["linear_scalar_projection_retention"] - 1.0) < 1e-12,
        "Expected 100% scalar projection illusion",
    )
    _require(cmp_res["post_wdown_perturbation_retention"] == 0.0, "Expected 0% post-wdown retention")
    _require(cmp_res["post_rmsnorm_functional_retention"] == 0.0, "Expected 0% post-RMS retention")
    _require(cmp_res["crosstalk_to_solo_post_wdown_ratio"] > 9.9, "Expected >9.9x crosstalk")


def test_e8_reproduces_paradox_and_verifies_multilayer_tangent_cure() -> None:
    res = run_e8()
    pa = res["part_a_rmsnorm_identity_and_multilayer_bounds"]
    _require(pa["general_reconstruction_l2_error"] < 1e-13, "General RMSNorm recon error")
    _require(pa["pure_tangent_reconstruction_l2_error"] < 1e-13, "Tangent RMSNorm recon error")
    _require(abs(pa["pure_tangent_radial_coord_c"]) < 1e-14, "Non-zero radial coord c")
    _require(pa["radial_null_projection_max_abs_dot"] < 1e-12, "Radial null dot non-zero")
    curve = pa["multilayer_shrinkage_curve"]
    _require(
        abs(
            curve["L4"]["cum_quadratic_upper_bound"]
            - 0.25 * curve["L1"]["cum_quadratic_upper_bound"]
        )
        < 1e-14,
        "Expected exact 1/4 quadratic upper bound scaling for L=4",
    )
    _require(
        curve["L4"]["cum_exact_shrinkage"] < 0.30 * curve["L1"]["cum_exact_shrinkage"],
        "Expected >3.3x exact shrinkage reduction at L=4",
    )

    pb = res["part_b_three_pack_composition"]
    _require(pb["sequential_commit_compile_order_fro_diff"] > 0.1, "Expected sequential order diff")
    _require(pb["arm4_zero_byte_rederivation"] is True, "Arm 4 0-byte rederivation failed")
    _require(
        pb["arm4_reversed_pack_order_zero_byte_match"] is True,
        "Arm 4 reversed pack order mismatch",
    )
    _require(pb["arm4_gguf_roundtrip_and_zero_byte_match"] is True, "Arm 4 GGUF roundtrip failed")
    _require(pb["arm4_six_permutation_max_logit_diff"] < 1e-13, "Arm 4 permutation drift")
    _require(pb["arm4_rollback_max_logit_diff"] == 0.0, "Arm 4 rollback non-zero")

    arms = pb["arms"]
    a1 = arms["arm1_single_layer_unsym_betaedit"]
    a3 = arms["arm3_single_layer_sym_tangent_betaedit"]
    a4 = arms["arm4_multilayer_4l_sym_tangent_betaedit"]

    _require(a1["linear_scalar_projection_retention"] > 0.90, "Expected Arm 1 scalar illusion")
    _require(a1["ui_module_02_10_00_step0_retention"] < 0.50, "Expected Arm 1 ui_module collapse")
    _require(a1["research_page_00_10_10_step0_retention"] == 0.0, "Expected Arm 1 research_page 0%")

    _require(a4["ui_module_02_10_00_step0_retention"] >= 0.95, "Arm 4 ui_module < 95%")
    _require(a4["research_page_00_10_10_step0_retention"] >= 0.92, "Arm 4 research_page < 92%")
    _require(a4["anchor_top1_retained_of_24"] == 24, "Arm 4 anchor retention < 24/24")
    _require(
        a4["fv_cumulative_layer_shrinkage"] < a3["fv_cumulative_layer_shrinkage"] * 0.50,
        "Expected >2x cumulative layer shrinkage reduction vs 1-layer",
    )
    _require(
        a4["fv_cumulative_layer_shrinkage"] < a1["fv_cumulative_layer_shrinkage"] * 0.10,
        "Expected >10x cumulative layer shrinkage reduction vs Arm 1",
    )


def test_adversarial_numerical_and_nullspace_edge_cases() -> None:
    # 1. Empty feature dimension rejected cleanly without RuntimeWarning
    with pytest.raises(ValueError):
        rmsnorm_f64(np.array([], dtype=np.float64), np.array([], dtype=np.float64))

    # 2. Subnormal/tiny non-zero h0 succeeds without underflowing ||h0||^2 to zero
    dec_tiny = decompose_rmsnorm_perturbation_f64(
        np.array([1e-200, 1e-200], dtype=np.float64),
        np.array([1.0, -1.0], dtype=np.float64),
        np.ones(2, dtype=np.float64),
        eps=1e-6,
    )
    _require(dec_tiny.reconstruction_l2_error < 1e-13, "Tiny h0 decomposition error")

    # 3. Cancellation-free rational sandwich at tiny eta (1e-12, 1e-24) and huge eta (1e200)
    for eta in (1e-24, 1e-12, 1e200):
        low, ex, high = rational_shrinkage_bounds(eta)
        _require(low <= ex <= high, f"Sandwich failed at eta={eta}: {low}, {ex}, {high}")

    # 4. Small SiLU tails (-1.7e-16) and huge gates (1e160) project accurately without overflow
    g_tail = np.array([-1.7e-16, -1.7e-16, 0.0, 0.0], dtype=np.float64)
    h_tail = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    u_tail = project_up_residual_rmsnorm_tangent_f64(
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
        g_tail,
        np.eye(4, dtype=np.float64),
        h_tail,
    )
    dh_tail = g_tail * u_tail
    _require(abs(float(np.dot(dh_tail, h_tail))) < 1e-30, "Small SiLU tail radial projection failed")

    u_huge = project_up_residual_rmsnorm_tangent_f64(
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([1e160, 1e160], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    )
    _require(float(np.linalg.norm(u_huge)) < 1e-14, "Huge gate radial projection failed")

    # Mixed-scale [1e-200, 1e200] * [1e200, 0.0] exponent-shifted Hadamard direction
    u_mix = project_up_residual_rmsnorm_tangent_f64(
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([1e-200, 1e200], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.array([1e200, 0.0], dtype=np.float64),
    )
    _require(abs(float(u_mix[0])) < 1e-14 and abs(float(u_mix[1]) - 4.0) < 1e-14, "Mixed-scale Hadamard failed")

    # 5. NaN rejection in Jacobian and radial-null projection
    with pytest.raises(ValueError):
        rmsnorm_jacobian_matvec_f64(np.array([np.nan, 1.0]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        project_up_residual_rmsnorm_tangent_f64(
            np.array([np.nan, 1.0]), np.ones(2), np.eye(2), np.ones(2)
        )

    # 6. Hard nullspace_threshold respected when K0 eigenvalue exceeds threshold
    proj = BetaEditNullspaceProjector(
        np.zeros((8, 8), dtype=np.float64),
        TangentBetaEditConfig(d_in=8, d_out=8, rank=2, nullspace_threshold=1e6).to_betaedit_config(),
    )
    nd = proj.initialize_anchor_covariance(np.eye(8, dtype=np.float64)[:, :2] * 1e6)
    _require(nd == 6, f"Expected nullspace dim 6, got {nd}")

    # 7. Subspace retention scales solo delta consistently when solo scale != 1.0
    ret = compare_subspace_vs_functional_retention_f64(
        solo_name="A",
        scales={"A": 0.2, "B": 1.0},
        layer_deltas_by_pack={
            "A": {0: np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)},
            "B": {0: np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)},
        },
        probe_keys_by_layer={0: np.array([[1.0], [0.0]], dtype=np.float64)},
        probe_gates_by_layer={0: np.array([[1.0], [1.0]], dtype=np.float64)},
        w_down_by_layer={0: np.eye(2, dtype=np.float64)},
        h_pre=np.array([0.0, 10.0], dtype=np.float64),
        gamma=np.ones(2, dtype=np.float64),
    )
    _require(
        abs(ret["linear_scalar_projection_retention"] - 1.0) < 1e-12,
        "Expected 1.0 retention at solo scale 0.2",
    )

    # 8. Worst-case per-state eta cap enforced & duplicate layer_ids rejected
    h_het = np.zeros((8, 4), dtype=np.float64)
    h_het[0, 0] = 1.0
    h_het[0, 1:] = 100.0
    u_het = np.zeros((8, 4), dtype=np.float64)
    u_het[1, 0] = 50.0
    spec = LayerPackSpec(
        keys_by_layer={28: np.eye(8)[:, :4], 29: np.eye(8)[:, :4]},
        up_residuals_by_layer={28: u_het, 29: u_het},
        gate_by_layer={28: np.ones((8, 4)), 29: np.ones((8, 4))},
        h_pre=h_het,
    )
    cfg_het = TangentBetaEditConfig(d_in=8, d_out=8, rank=4, max_per_layer_radial_eta=0.01)
    suite = compile_multilayer_tangent_suite(
        layer_ids=[29, 28],
        w_down_by_layer={28: np.eye(8), 29: np.eye(8)},
        k0_noise_by_layer={28: np.zeros((8, 4)), 29: np.zeros((8, 4))},
        packs={"p0": spec},
        config=cfg_het,
    )
    _require(suite["p0"].layer_ids == (28, 29), "Expected sorted layer_ids")
    p28 = suite["p0"].patches_by_layer[28]
    dh28 = p28.factor_b @ (p28.factor_a @ np.eye(8)[:, :4])
    eta_state0 = float(np.dot(dh28[:, 0], dh28[:, 0])) / (1.0 + 8e-6)
    _require(eta_state0 <= 0.01, f"Per-state eta {eta_state0} exceeded cap 0.01")

    with pytest.raises(ValueError):
        compile_multilayer_tangent_suite(
            layer_ids=[28, 28],
            w_down_by_layer={28: np.eye(8)},
            k0_noise_by_layer={28: np.zeros((8, 4))},
            packs={
                "p0": LayerPackSpec(
                    keys_by_layer={28: np.eye(8)[:, :4]},
                    up_residuals_by_layer={28: u_het},
                    gate_by_layer={28: np.ones((8, 4))},
                    h_pre=h_het,
                )
            },
            config=cfg_het,
        )
