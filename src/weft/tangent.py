"""SwiGLU gate-metric, RMSNorm tangent projection, and multi-layer TangentBetaEdit.

Derives and implements the exact finite-perturbation nonlinear readout equations
governing simultaneous multi-LoRA composition on pre-norm SwiGLU Transformers:

1. Exact Finite-Perturbation RMSNorm Identity (for all eps >= 0, h0 != 0, delta):
   Decompose delta = c * h0 + delta_perp with c = <delta, h0> / ||h0||_2^2 and
   <delta_perp, h0> = 0. With R0 = sqrt(||h0||_2^2 + d * eps), R1 = sqrt(||h0 + delta||_2^2 + d * eps),
   and q = R0 / R1:
       RMSNorm(h0 + delta) - RMSNorm(h0)
         = q * T(h0) delta_perp + ((1 + c) * q - 1) * RMSNorm(h0),
   where T(h0) v = gamma * (sqrt(d) / R0) * v.
   Even for a purely tangent perturbation (c = 0, eta = ||delta_perp||_2^2 / R0^2),
   RMSNorm shrinks the entire base hidden direction by s(eta) = 1 - 1 / sqrt(1 + eta)
   = eta / (sqrt(1 + eta) * (1 + sqrt(1 + eta))), satisfying the rational sandwich bounds:
       eta / (2 * (1 + eta)) <= s(eta) <= eta / 2.

2. Exact SwiGLU + W_down Radial-Null Hyperplane Projection:
   For an up-projection residual u in R^{d_ff} gated by g(x) = silu(W_gate x),
       <W_down (g(x) * u), h0>_{R^d} = <u, g(x) * (W_down^T h0)>_{R^{d_ff}}.
   Projecting u orthogonal to the scale-invariant unit direction of a(x) = g(x) * (W_down^T h0)
   eliminates first-order radial leakage into h0 prior to BetaEdit rank-r factorization.

3. Symmetric Joint Cross-Pack Null-Space + 1/L Multi-Layer Splitting (TangentBetaEdit):
   Compiling M specialist packs across L upper layers with symmetric sibling-key
   preservation P_m^(l) = [sqrt(w_anchor) K_anchor^(l), sqrt(w_sib) K_{-m}^(l)],
   worst-case per-state radial eta enforcement, and 1/L per-layer norm splitting
   enforces bidirectional order-invariant cross-pack orthogonality while cutting
   cumulative per-layer quadratic RMSNorm radial shrinkage by 1/L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .certificate import FixedRadixCertificate
from .compiler import (
    BetaEditConfig,
    BetaEditNullspaceProjector,
    QuantDtype,
    StagedPatch,
    ThresholdMode,
    verify_rederivation,
)
from .gguf_lora import write_gguf_lora


def _scaled_vec_norm(v: np.ndarray) -> tuple[float, np.ndarray]:
    """Overflow- and underflow-safe Euclidean norm and unit vector for 1D float64 arrays."""
    vmax = float(np.max(np.abs(v))) if v.size > 0 else 0.0
    if vmax == 0.0:
        return 0.0, np.zeros_like(v)
    v_scaled = v / vmax
    unit_norm = float(np.linalg.norm(v_scaled))
    if unit_norm == 0.0:
        return 0.0, np.zeros_like(v)
    return vmax * unit_norm, v_scaled / unit_norm


def _scaled_frob_norm(mat: np.ndarray) -> float:
    """Overflow- and underflow-safe Frobenius norm for 2D float64 arrays."""
    if mat.size == 0:
        return 0.0
    mmax = float(np.max(np.abs(mat)))
    if mmax == 0.0:
        return 0.0
    return mmax * float(np.linalg.norm(mat / mmax, ord="fro"))


def _unit_hadamard_direction(g_col: np.ndarray, wh_col: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact exponent-shifted unit vector of g_col * wh_col immune to mixed-scale underflow/overflow."""
    nz = (g_col != 0.0) & (wh_col != 0.0)
    if not np.any(nz):
        return 0.0, np.zeros_like(g_col)
    mg, eg = np.frexp(g_col[nz])
    mw, ew = np.frexp(wh_col[nz])
    esum = eg.astype(np.int64) + ew.astype(np.int64)
    emax = int(np.max(esum))
    shift = np.clip(esum - emax, -1075, 0).astype(np.int32)
    scaled_nz = np.ldexp(mg * mw, shift)
    rad_scaled = np.zeros_like(g_col)
    rad_scaled[nz] = scaled_nz
    return _scaled_vec_norm(rad_scaled)


@dataclass(frozen=True)
class RMSNormDecomposition:
    """Exact finite-perturbation decomposition of RMSNorm(h0 + delta) - RMSNorm(h0)."""

    r0: float
    delta_norm: float
    radial_coord_c: float
    delta_par_norm: float
    delta_perp_norm: float
    eta_perp_sq: float
    scale_q: float
    pure_tangent_shrinkage_s: float
    rational_lower_bound: float
    rational_upper_bound: float
    linear_tangent_norm: float
    attenuated_tangent_norm: float
    radial_term_norm: float
    exact_diff_norm: float
    reconstruction_l2_error: float


def rmsnorm_f64(
    h: np.ndarray,
    gamma: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Exact float64 RMSNorm on vector (d,) or column-batch matrix (d, n)."""
    h64 = np.ascontiguousarray(h, dtype=np.float64)
    g64 = np.ascontiguousarray(gamma, dtype=np.float64)
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")
    if h64.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array for h, got ndim={h64.ndim}.")
    if h64.shape[0] == 0:
        raise ValueError("Feature dimension d must be positive.")
    if g64.ndim != 1 or g64.shape[0] != h64.shape[0]:
        raise ValueError(f"Incompatible gamma shape {g64.shape} for h shape {h64.shape}.")
    if not (np.all(np.isfinite(h64)) and np.all(np.isfinite(g64))):
        raise ValueError("Inputs to rmsnorm_f64 must be finite.")
    if h64.ndim == 2 and h64.shape[1] == 0:
        return np.zeros_like(h64)

    d_dim = float(h64.shape[0])
    sqrt_d = math.sqrt(d_dim)
    sqrt_d_eps = math.sqrt(d_dim * eps)
    if h64.ndim == 1:
        h_norm, u0 = _scaled_vec_norm(h64)
        r0 = math.hypot(h_norm, sqrt_d_eps)
        if r0 == 0.0:
            return np.zeros_like(h64)
        return g64 * ((sqrt_d * (h_norm / r0)) * u0)

    col_max = np.max(np.abs(h64), axis=0, keepdims=True)
    safe_max = np.where(col_max > 0.0, col_max, 1.0)
    h_scaled = h64 / safe_max
    scaled_norms = np.linalg.norm(h_scaled, axis=0, keepdims=True)
    h_norms = np.where(col_max > 0.0, col_max * scaled_norms, 0.0)
    r0_vec = np.hypot(h_norms, sqrt_d_eps)
    safe_r0 = np.where(r0_vec > 0.0, r0_vec, 1.0)
    safe_sn = np.where(scaled_norms > 0.0, scaled_norms, 1.0)
    u_cols = np.where(col_max > 0.0, h_scaled / safe_sn, 0.0)
    factors = np.where(r0_vec > 0.0, sqrt_d * (h_norms / safe_r0), 0.0)
    return g64[:, np.newaxis] * (factors * u_cols)


def rmsnorm_jacobian_matvec_f64(
    h0: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Exact Frechet derivative J_RMSNorm(h0) @ delta on vector (d,) or matrix (d, n)."""
    h64 = np.ascontiguousarray(h0, dtype=np.float64)
    d64 = np.ascontiguousarray(delta, dtype=np.float64)
    g64 = np.ascontiguousarray(gamma, dtype=np.float64)
    if h64.shape != d64.shape or h64.ndim not in (1, 2):
        raise ValueError("h0 and delta must share the same 1D or 2D shape.")
    if h64.shape[0] == 0:
        raise ValueError("Feature dimension d must be positive.")
    if g64.ndim != 1 or g64.shape[0] != h64.shape[0]:
        raise ValueError("gamma must be 1D matching feature dimension d.")
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")
    if not (np.all(np.isfinite(h64)) and np.all(np.isfinite(d64)) and np.all(np.isfinite(g64))):
        raise ValueError("Inputs to rmsnorm_jacobian_matvec_f64 must be finite.")
    if h64.ndim == 2 and h64.shape[1] == 0:
        return np.zeros_like(d64)

    d_dim = float(h64.shape[0])
    sqrt_d = math.sqrt(d_dim)
    sqrt_d_eps = math.sqrt(d_dim * eps)
    if h64.ndim == 1:
        h_norm, u0 = _scaled_vec_norm(h64)
        r0 = math.hypot(h_norm, sqrt_d_eps)
        if r0 == 0.0:
            return np.zeros_like(d64)
        alpha = h_norm / r0
        proj = ((alpha * alpha) * float(np.dot(u0, d64))) * u0
        return g64 * ((sqrt_d / r0) * (d64 - proj))

    out = np.zeros_like(d64)
    for col in range(h64.shape[1]):
        out[:, col] = rmsnorm_jacobian_matvec_f64(h64[:, col], d64[:, col], g64, eps=eps)
    return out


def rational_shrinkage_bounds(eta: float) -> tuple[float, float, float]:
    """Return (lower_bound, exact_s, upper_bound) for s(eta) = 1 - 1 / sqrt(1 + eta).

    Uses the cancellation-free identity s(eta) = eta / (sqrt(1 + eta) * (1 + sqrt(1 + eta)))
    so lower_bound <= exact_s <= upper_bound holds across tiny and huge IEEE-754 floats.
    """
    if eta < 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and non-negative.")
    if eta == 0.0:
        return 0.0, 0.0, 0.0
    if eta > 1e150:
        inv_sqrt = 1.0 / math.sqrt(eta)
        lower = 0.5 / (1.0 + (1.0 / eta))
        exact = 1.0 - inv_sqrt
        upper = 0.5 * eta
        return lower, exact, upper
    s1 = math.sqrt(1.0 + eta)
    lower = eta / (2.0 * (1.0 + eta))
    exact = eta / (s1 * (1.0 + s1))
    upper = 0.5 * eta
    return lower, exact, upper


def decompose_rmsnorm_perturbation_f64(
    h0: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    *,
    eps: float = 1e-6,
) -> RMSNormDecomposition:
    """Exact finite-perturbation decomposition of RMSNorm(h0 + delta) - RMSNorm(h0)."""
    h64 = np.ascontiguousarray(h0, dtype=np.float64)
    d64 = np.ascontiguousarray(delta, dtype=np.float64)
    g64 = np.ascontiguousarray(gamma, dtype=np.float64)
    if h64.ndim != 1 or d64.ndim != 1 or g64.ndim != 1 or not (h64.shape == d64.shape == g64.shape):
        raise ValueError("h0, delta, and gamma must be 1D vectors of equal length.")
    if h64.shape[0] == 0:
        raise ValueError("Feature dimension d must be positive.")
    if not (np.all(np.isfinite(h64)) and np.all(np.isfinite(d64)) and np.all(np.isfinite(g64))):
        raise ValueError("Inputs to decompose_rmsnorm_perturbation_f64 must be finite.")
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")

    h_norm, u0 = _scaled_vec_norm(h64)
    if h_norm <= 0.0:
        raise ValueError("h0 must be non-zero for tangent-radial decomposition.")

    d_dim = float(h64.shape[0])
    sqrt_d = math.sqrt(d_dim)
    sqrt_d_eps = math.sqrt(d_dim * eps)
    r0 = math.hypot(h_norm, sqrt_d_eps)
    h1 = h64 + d64
    h1_norm, _ = _scaled_vec_norm(h1)
    r1 = math.hypot(h1_norm, sqrt_d_eps)
    if r1 <= 0.0:
        raise ValueError("Perturbed state h0 + delta has zero regularized RMS radius.")

    par_proj = float(np.dot(d64, u0))
    delta_par = par_proj * u0
    delta_perp = d64 - delta_par
    perp_norm, u_perp = _scaled_vec_norm(delta_perp)
    ratio_perp = perp_norm / r0
    eta = ratio_perp * ratio_perp
    scale_q = r0 / r1

    # Guard coord_c against overflow when ||h0|| is subnormal/tiny and <delta, u0> != 0
    coord_c = 0.0 if par_proj == 0.0 else (par_proj / h_norm if abs(par_proj) < h_norm * 1e300 else math.copysign(float("inf"), par_proj))

    base_rms = rmsnorm_f64(h64, g64, eps=eps)
    full_rms = rmsnorm_f64(h1, g64, eps=eps)
    exact_diff = full_rms - base_rms

    linear_tangent = g64 * ((sqrt_d * ratio_perp) * u_perp)
    attenuated_tangent = scale_q * linear_tangent
    # Compute radial term via h_norm + par_proj to remain exact even when coord_c overflows
    radial_rms_vec = g64 * ((sqrt_d * ((h_norm + par_proj) / r1)) * u0)
    radial_term = radial_rms_vec - base_rms
    reconstructed = attenuated_tangent + radial_term
    recon_err = float(np.linalg.norm(exact_diff - reconstructed))

    low_b, shrink_s, up_b = rational_shrinkage_bounds(eta)
    d_tot_norm, _ = _scaled_vec_norm(d64)
    return RMSNormDecomposition(
        r0=r0,
        delta_norm=d_tot_norm,
        radial_coord_c=coord_c,
        delta_par_norm=abs(par_proj),
        delta_perp_norm=perp_norm,
        eta_perp_sq=eta,
        scale_q=scale_q,
        pure_tangent_shrinkage_s=shrink_s,
        rational_lower_bound=low_b,
        rational_upper_bound=up_b,
        linear_tangent_norm=float(np.linalg.norm(linear_tangent)),
        attenuated_tangent_norm=float(np.linalg.norm(attenuated_tangent)),
        radial_term_norm=float(np.linalg.norm(radial_term)),
        exact_diff_norm=float(np.linalg.norm(exact_diff)),
        reconstruction_l2_error=recon_err,
    )


def project_up_residual_rmsnorm_tangent_f64(
    up_residuals: np.ndarray,
    gate_activations: np.ndarray,
    w_down: np.ndarray,
    h_pre: np.ndarray,
) -> np.ndarray:
    """Project FFN up-projection residuals so W_down(g * u_perp) is strictly orthogonal to h_pre.

    Uses the scale-invariant unit direction of a_i = g_i * (W_down^T h_{0,i}) so small SiLU
    tails (e.g. silu(-40) ~ -1.7e-16) and huge dynamic ranges (e.g. 1e160) never underflow
    or overflow.
    """
    u64 = np.ascontiguousarray(up_residuals, dtype=np.float64)
    g64 = np.ascontiguousarray(gate_activations, dtype=np.float64)
    wd64 = np.ascontiguousarray(w_down, dtype=np.float64)
    h64 = np.ascontiguousarray(h_pre, dtype=np.float64)
    if not (
        np.all(np.isfinite(u64))
        and np.all(np.isfinite(g64))
        and np.all(np.isfinite(wd64))
        and np.all(np.isfinite(h64))
    ):
        raise ValueError("All inputs to project_up_residual_rmsnorm_tangent_f64 must be finite.")

    squeeze = False
    if u64.ndim == 1:
        u64 = u64[:, np.newaxis]
        g64 = g64[:, np.newaxis] if g64.ndim == 1 else g64
        h64 = h64[:, np.newaxis] if h64.ndim == 1 else h64
        squeeze = True

    if u64.ndim != 2 or g64.shape != u64.shape:
        raise ValueError("up_residuals and gate_activations must share shape (d_ff, n).")
    d_ff, n_cols = u64.shape
    if d_ff == 0:
        raise ValueError("FFN dimension d_ff must be positive.")
    if wd64.ndim != 2 or wd64.shape[1] != d_ff:
        raise ValueError(f"Expected w_down shape (d, {d_ff}), got {wd64.shape}.")
    d_in = wd64.shape[0]
    if d_in == 0 or h64.shape != (d_in, n_cols):
        raise ValueError(f"Expected h_pre shape ({d_in}, {n_cols}), got {h64.shape}.")
    if n_cols == 0:
        return u64[:, 0] if squeeze else u64.copy()

    wd_max = float(np.max(np.abs(wd64)))
    if wd_max == 0.0:
        return u64[:, 0] if squeeze else u64.copy()
    wd_scaled = wd64 / wd_max

    u_perp = u64.copy()
    for col in range(n_cols):
        h_col = h64[:, col]
        g_col = g64[:, col]
        u_col = u64[:, col]
        h_max = float(np.max(np.abs(h_col)))
        g_max = float(np.max(np.abs(g_col)))
        u_max = float(np.max(np.abs(u_col)))
        if h_max == 0.0 or g_max == 0.0 or u_max == 0.0:
            continue
        wh = wd_scaled.T @ (h_col / h_max)
        r_norm, q_hat = _unit_hadamard_direction(g_col, wh)
        if r_norm == 0.0:
            continue
        proj_scaled = float(np.dot(u_col / u_max, q_hat))
        u_perp[:, col] = u_col - (proj_scaled * u_max) * q_hat

    u_perp = np.ascontiguousarray(u_perp)
    return u_perp[:, 0] if squeeze else u_perp


def compare_subspace_vs_functional_retention_f64(
    *,
    solo_name: str,
    scales: Mapping[str, float],
    layer_deltas_by_pack: Mapping[str, Mapping[int, np.ndarray]],
    probe_keys_by_layer: Mapping[int, np.ndarray],
    probe_gates_by_layer: Mapping[int, np.ndarray],
    w_down_by_layer: Mapping[int, np.ndarray],
    h_pre: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Contrast weight-space scalar projection retention against post-RMSNorm functional retention."""
    if solo_name not in layer_deltas_by_pack:
        raise KeyError(f"Unknown solo_name: {solo_name!r}")
    solo_layers = layer_deltas_by_pack[solo_name]
    layer_ids = sorted(solo_layers.keys())
    if not layer_ids:
        raise ValueError("At least one layer must be provided.")

    h64 = np.ascontiguousarray(h_pre, dtype=np.float64)
    if h64.ndim == 1:
        h64 = h64[:, np.newaxis]
    g64 = np.ascontiguousarray(gamma, dtype=np.float64)
    n_probes = h64.shape[1]

    solo_scale = float(scales.get(solo_name, 1.0))
    if solo_scale == 0.0:
        solo_scale = 1.0

    # 1. Weight-space right-singular scalar-projection metric (scaled consistently by solo_scale)
    num_scalar = 0.0
    den_scalar = 0.0
    for lid in layer_ids:
        d_solo_raw = np.ascontiguousarray(solo_layers[lid], dtype=np.float64)
        if d_solo_raw.shape[0] > d_solo_raw.shape[1] * 2:
            gram = d_solo_raw.T @ d_solo_raw
            eigvals, eigvecs = np.linalg.eigh(0.5 * (gram + gram.T))
            max_ev = float(eigvals[-1]) if eigvals.size > 0 else 0.0
            keep = eigvals > max(1e-24, max_ev * 1e-20)
            if not np.any(keep):
                continue
            v_solo = eigvecs[:, keep]
        else:
            _, s_vals, vh = np.linalg.svd(d_solo_raw, full_matrices=False)
            nz = int(np.sum(s_vals > 1e-12))
            if nz == 0:
                continue
            v_solo = vh[:nz, :].T  # (d_in, nz)
        d_tot = np.zeros_like(d_solo_raw)
        for p_name, p_layers in layer_deltas_by_pack.items():
            sc = float(scales.get(p_name, 0.0))
            if sc != 0.0 and lid in p_layers:
                d_tot = d_tot + sc * np.ascontiguousarray(p_layers[lid], dtype=np.float64)
        solo_proj = (solo_scale * d_solo_raw) @ v_solo
        tot_proj = d_tot @ v_solo
        num_scalar += float(np.sum(tot_proj * solo_proj))
        den_scalar += float(np.sum(solo_proj * solo_proj))

    scalar_proj_ret = (num_scalar / den_scalar) if den_scalar > 1e-30 else 0.0

    # 2. Post-W_down and post-RMSNorm functional retention on probe states
    delta_h_solo = np.zeros((h64.shape[0], n_probes), dtype=np.float64)
    delta_h_tot = np.zeros((h64.shape[0], n_probes), dtype=np.float64)
    for lid in layer_ids:
        k_l = np.ascontiguousarray(probe_keys_by_layer[lid], dtype=np.float64)
        g_l = np.ascontiguousarray(probe_gates_by_layer[lid], dtype=np.float64)
        if k_l.ndim == 1:
            k_l = k_l[:, np.newaxis]
        if g_l.ndim == 1:
            g_l = g_l[:, np.newaxis]
        wd_l = np.ascontiguousarray(w_down_by_layer[lid], dtype=np.float64)
        up_solo = solo_scale * (np.ascontiguousarray(solo_layers[lid], dtype=np.float64) @ k_l)
        delta_h_solo = delta_h_solo + (wd_l @ (g_l * up_solo))

        up_tot = np.zeros_like(up_solo)
        for p_name, p_layers in layer_deltas_by_pack.items():
            sc = float(scales.get(p_name, 0.0))
            if sc != 0.0 and lid in p_layers:
                up_tot = up_tot + sc * (np.ascontiguousarray(p_layers[lid], dtype=np.float64) @ k_l)
        delta_h_tot = delta_h_tot + (wd_l @ (g_l * up_tot))

    solo_h_norm = float(np.linalg.norm(delta_h_solo, ord="fro"))
    crosstalk_h_norm = float(np.linalg.norm(delta_h_tot - delta_h_solo, ord="fro"))
    post_wdown_ret = max(0.0, 1.0 - (crosstalk_h_norm / max(solo_h_norm, 1e-15)))

    rms_base = rmsnorm_f64(h64, g64, eps=eps)
    rms_solo = rmsnorm_f64(h64 + delta_h_solo, g64, eps=eps)
    rms_tot = rmsnorm_f64(h64 + delta_h_tot, g64, eps=eps)
    solo_rms_shift = float(np.linalg.norm(rms_solo - rms_base, ord="fro"))
    tot_rms_drift = float(np.linalg.norm(rms_tot - rms_solo, ord="fro"))
    post_rmsnorm_ret = max(0.0, 1.0 - (tot_rms_drift / max(solo_rms_shift, 1e-15)))

    return {
        "linear_scalar_projection_retention": scalar_proj_ret,
        "post_wdown_perturbation_retention": post_wdown_ret,
        "post_rmsnorm_functional_retention": post_rmsnorm_ret,
        "crosstalk_to_solo_post_wdown_ratio": crosstalk_h_norm / max(solo_h_norm, 1e-15),
        "solo_post_wdown_fro": solo_h_norm,
        "crosstalk_post_wdown_fro": crosstalk_h_norm,
        "solo_post_rmsnorm_fro": solo_rms_shift,
        "drift_post_rmsnorm_fro": tot_rms_drift,
    }


@dataclass(frozen=True)
class LayerPackSpec:
    """Per-specialist multi-layer activation and target specification for TangentBetaEdit."""

    keys_by_layer: Mapping[int, np.ndarray]  # layer -> (d_in, n_m)
    up_residuals_by_layer: Mapping[int, np.ndarray]  # layer -> (d_out, n_m)
    gate_by_layer: Mapping[int, np.ndarray]  # layer -> (d_out, n_m)
    h_pre: np.ndarray  # (d_in, n_m)
    target_total_post_wdown_rms: Optional[float] = None
    safety_directions_by_layer: Mapping[int, Sequence[tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class TangentBetaEditConfig:
    """Configuration for symmetric multi-layer SwiGLU + RMSNorm-tangent BetaEdit."""

    d_in: int
    d_out: int
    rank: int = 8
    lambda_leakage: float = 40.0
    lambda_ridge: float = 0.5
    nullspace_threshold: float = 1e6
    threshold_mode: ThresholdMode = "absolute"
    min_nullspace_dim: int = 1
    trace_normalize_k0: bool = True
    balance_preservation_trace: bool = True
    anchor_preservation_weight: float = 2500.0
    sibling_preservation_weight: float = 2500.0
    project_radial_null: bool = True
    split_by_num_layers: bool = True
    max_per_layer_radial_eta: float = 0.01
    max_gain_multiplier: float = 8.0
    quant_dtype: QuantDtype = "f16"
    radix_scale_exp2: int = 24

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.d_out <= 0 or self.rank <= 0:
            raise ValueError("d_in, d_out, and rank must be positive.")
        if self.lambda_leakage < 0.0 or self.lambda_ridge <= 0.0 or self.nullspace_threshold <= 0.0:
            raise ValueError("Invalid lambda_leakage, lambda_ridge, or nullspace_threshold.")
        if not (1 <= self.min_nullspace_dim <= self.d_in):
            raise ValueError("min_nullspace_dim must lie in [1, d_in].")
        if self.anchor_preservation_weight < 0.0 or self.sibling_preservation_weight < 0.0:
            raise ValueError("Preservation weights must be non-negative.")
        if self.max_per_layer_radial_eta <= 0.0 or self.max_gain_multiplier <= 0.0:
            raise ValueError("Radial eta cap and max_gain_multiplier must be positive.")

    def to_betaedit_config(self) -> BetaEditConfig:
        return BetaEditConfig(
            d_in=self.d_in,
            d_out=self.d_out,
            max_rank=self.rank,
            max_slots=16,
            lambda_leakage=self.lambda_leakage,
            lambda_ridge=self.lambda_ridge,
            nullspace_threshold=self.nullspace_threshold,
            threshold_mode=self.threshold_mode,
            min_nullspace_dim=self.min_nullspace_dim,
            quant_dtype=self.quant_dtype,
            radix_scale_exp2=self.radix_scale_exp2,
        )


@dataclass(frozen=True)
class CompiledMultiLayerPack:
    """Compiled multi-layer specialist pack with exact replay inputs and certificates."""

    pack_name: str
    layer_ids: tuple[int, ...]
    patches_by_layer: dict[int, StagedPatch]
    padded_factors_by_layer: dict[int, tuple[np.ndarray, np.ndarray]]
    calibrated_residuals_by_layer: dict[int, np.ndarray]
    preservation_keys_by_layer: dict[int, np.ndarray]
    k0_effective_by_layer: dict[int, np.ndarray]
    applied_gains_by_layer: dict[int, float]
    rms_eps: float

    def delta_by_layer(self) -> dict[int, np.ndarray]:
        return {
            lid: b_mat @ a_mat
            for lid, (b_mat, a_mat) in self.padded_factors_by_layer.items()
        }

    def write_gguf(
        self,
        path: Path | str,
        *,
        architecture: str,
        rank: int,
        tensor_template: str = "blk.{layer}.ffn_up.weight",
        extra_metadata: Mapping[str, Any] = (),
    ) -> tuple[Path, str]:
        layer_factors = {
            tensor_template.format(layer=lid): self.padded_factors_by_layer[lid]
            for lid in self.layer_ids
        }
        cert_bundle = "|".join(
            f"{lid}:{self.patches_by_layer[lid].certificate.sha256()}" for lid in self.layer_ids
        )
        meta = dict(extra_metadata)
        meta.setdefault("weft.slot", self.pack_name)
        meta.setdefault("weft.layer_count", len(self.layer_ids))
        meta.setdefault("weft.certificates", cert_bundle)
        return write_gguf_lora(
            path,
            architecture=architecture,
            lora_alpha=float(rank),
            layer_factors=layer_factors,
            extra_metadata=meta,
        )


def _pad_factors_to_rank(
    factor_b: np.ndarray,
    factor_a: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad (B, A) to exact rank r so GGUF global alpha / r == 1.0 across every layer."""
    d_out, r_eff = factor_b.shape
    _, d_in = factor_a.shape
    if r_eff == rank:
        return factor_b, factor_a
    if r_eff > rank:
        raise ValueError(f"Factor rank {r_eff} exceeds target rank {rank}.")
    b_pad = np.zeros((d_out, rank), dtype=np.float64)
    a_pad = np.zeros((rank, d_in), dtype=np.float64)
    b_pad[:, :r_eff] = factor_b
    a_pad[:r_eff, :] = factor_a
    return b_pad, a_pad


def _prepare_k0_matrix(k0_raw: np.ndarray, d_in: int, trace_normalize: bool) -> np.ndarray:
    k64 = np.ascontiguousarray(k0_raw, dtype=np.float64)
    if k64.ndim != 2 or k64.shape[0] != d_in:
        raise ValueError(f"Expected K0 matrix of shape ({d_in}, n_0), got {k64.shape}.")
    if not np.all(np.isfinite(k64)):
        raise ValueError("K0 matrix must be finite.")
    if not trace_normalize or k64.size == 0:
        return k64
    kmax = float(np.max(np.abs(k64)))
    if kmax == 0.0:
        return k64
    k_scaled = k64 / kmax
    fnorm = float(np.linalg.norm(k_scaled, ord="fro"))
    if fnorm == 0.0:
        return k64
    return np.ascontiguousarray(k_scaled * (math.sqrt(float(d_in)) / fnorm))


def _build_joint_preservation_keys(
    *,
    pack_name: str,
    self_keys: np.ndarray,
    layer_id: int,
    k0_anchors: Optional[np.ndarray],
    packs: Mapping[str, LayerPackSpec],
    anchor_weight: float,
    sibling_weight: float,
    balance_trace: bool,
    d_in: int,
) -> np.ndarray:
    self_fnorm = max(_scaled_frob_norm(self_keys), 1e-300)
    blocks: list[np.ndarray] = []
    if k0_anchors is not None and k0_anchors.size > 0 and anchor_weight > 0.0:
        anc = np.ascontiguousarray(k0_anchors, dtype=np.float64)
        if anc.ndim != 2 or anc.shape[0] != d_in:
            raise ValueError(f"Expected anchor keys shape ({d_in}, n_a), got {anc.shape}.")
        anc_fnorm = max(_scaled_frob_norm(anc), 1e-300)
        ratio = (self_fnorm / anc_fnorm) if balance_trace else 1.0
        blocks.append(anc * (math.sqrt(anchor_weight) * ratio))
    if sibling_weight > 0.0:
        for other_name in sorted(packs.keys()):
            if other_name == pack_name:
                continue
            sib_k = np.ascontiguousarray(packs[other_name].keys_by_layer[layer_id], dtype=np.float64)
            if sib_k.ndim != 2 or sib_k.shape[0] != d_in:
                raise ValueError(f"Invalid sibling key shape for {other_name} at layer {layer_id}.")
            sib_fnorm = max(_scaled_frob_norm(sib_k), 1e-300)
            ratio = (self_fnorm / sib_fnorm) if balance_trace else 1.0
            blocks.append(sib_k * (math.sqrt(sibling_weight) * ratio))
    if not blocks:
        return np.zeros((d_in, 0), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate(blocks, axis=1))


def compile_multilayer_tangent_suite(
    *,
    layer_ids: Sequence[int],
    w_down_by_layer: Mapping[int, np.ndarray],
    k0_noise_by_layer: Mapping[int, np.ndarray],
    packs: Mapping[str, LayerPackSpec],
    config: TangentBetaEditConfig,
    k0_anchors_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    base_up_weights: Optional[Mapping[int, np.ndarray]] = None,
    rms_eps: float = 1e-6,
) -> dict[str, CompiledMultiLayerPack]:
    """Compile M specialist packs across L layers with symmetric sibling K0 and per-state eta bounds."""
    raw_layers = [int(lid) for lid in layer_ids]
    if not raw_layers:
        raise ValueError("layer_ids must be non-empty.")
    if len(set(raw_layers)) != len(raw_layers):
        raise ValueError(f"layer_ids must not contain duplicates, got {raw_layers}.")
    sorted_layers = tuple(sorted(raw_layers))
    if not packs:
        raise ValueError("packs must be non-empty.")
    if rms_eps < 0.0 or not math.isfinite(rms_eps):
        raise ValueError("rms_eps must be finite and non-negative.")

    n_layers = len(sorted_layers)
    layer_scale = (1.0 / float(n_layers)) if config.split_by_num_layers else 1.0
    be_cfg = config.to_betaedit_config()
    zero_w0 = np.zeros((config.d_out, config.d_in), dtype=np.float64)

    k0_eff_by_layer: dict[int, np.ndarray] = {}
    w_down_64: dict[int, np.ndarray] = {}
    w0_by_layer: dict[int, np.ndarray] = {}
    for lid in sorted_layers:
        k0_eff_by_layer[lid] = _prepare_k0_matrix(
            k0_noise_by_layer[lid], config.d_in, config.trace_normalize_k0
        )
        wd = np.ascontiguousarray(w_down_by_layer[lid], dtype=np.float64)
        if wd.shape != (config.d_in, config.d_out):
            raise ValueError(
                f"Expected w_down shape ({config.d_in}, {config.d_out}) at layer {lid}, got {wd.shape}."
            )
        w_down_64[lid] = wd
        if base_up_weights is not None and lid in base_up_weights:
            w0_by_layer[lid] = np.ascontiguousarray(base_up_weights[lid], dtype=np.float64)
        else:
            w0_by_layer[lid] = zero_w0

    compiled_suite: dict[str, CompiledMultiLayerPack] = {}
    for pack_name in sorted(packs.keys()):
        spec = packs[pack_name]
        h_pre = np.ascontiguousarray(spec.h_pre, dtype=np.float64)
        n_states = int(h_pre.shape[1]) if h_pre.ndim == 2 else 0
        if h_pre.ndim != 2 or h_pre.shape[0] != config.d_in or n_states == 0:
            raise ValueError(f"Invalid h_pre shape {h_pre.shape} for pack {pack_name!r}.")

        r0_sq_vec = np.sum(h_pre * h_pre, axis=0) + float(config.d_in) * rms_eps
        if np.any(r0_sq_vec <= 0.0):
            raise ValueError(f"h_pre column has zero regularized RMS radius in pack {pack_name!r}.")

        patches_by_layer: dict[int, StagedPatch] = {}
        padded_by_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cal_res_by_layer: dict[int, np.ndarray] = {}
        pres_by_layer: dict[int, np.ndarray] = {}
        gains_by_layer: dict[int, float] = {}

        for lid in sorted_layers:
            k_m = np.ascontiguousarray(spec.keys_by_layer[lid], dtype=np.float64)
            u_raw = np.ascontiguousarray(spec.up_residuals_by_layer[lid], dtype=np.float64)
            g_m = np.ascontiguousarray(spec.gate_by_layer[lid], dtype=np.float64)
            if k_m.shape != (config.d_in, n_states):
                raise ValueError(f"Key shape mismatch for {pack_name!r} at layer {lid}.")
            if u_raw.shape != (config.d_out, n_states) or g_m.shape != (config.d_out, n_states):
                raise ValueError(f"Residual/gate shape mismatch for {pack_name!r} at layer {lid}.")

            if config.project_radial_null:
                u_base = project_up_residual_rmsnorm_tangent_f64(u_raw, g_m, w_down_64[lid], h_pre)
            else:
                u_base = u_raw
            u_split = np.ascontiguousarray(u_base * layer_scale)

            anc_l = None if k0_anchors_by_layer is None else k0_anchors_by_layer.get(lid)
            pres_k = _build_joint_preservation_keys(
                pack_name=pack_name,
                self_keys=k_m,
                layer_id=lid,
                k0_anchors=anc_l,
                packs=packs,
                anchor_weight=config.anchor_preservation_weight,
                sibling_weight=config.sibling_preservation_weight,
                balance_trace=config.balance_preservation_trace,
                d_in=config.d_in,
            )

            proj = BetaEditNullspaceProjector(w0_by_layer[lid], be_cfg)
            proj.initialize_anchor_covariance(k0_eff_by_layer[lid])
            for b_s, a_s in spec.safety_directions_by_layer.get(lid, ()):
                proj.register_safety_direction(b_s, a_s)

            pilot = proj.stage_edit(
                k_m,
                u_split,
                is_residual=True,
                rank=config.rank,
                preservation_keys=pres_k if pres_k.size > 0 else None,
                preservation_weight=1.0,
                slot_name=f"{pack_name}.blk{lid}",
            )
            dh_pilot = w_down_64[lid] @ (g_m * (pilot.factor_b @ (pilot.factor_a @ k_m)))
            pilot_sq_vec = np.sum(dh_pilot * dh_pilot, axis=0)
            pilot_rms = math.sqrt(float(np.mean(pilot_sq_vec)))
            max_pilot_eta = float(np.max(pilot_sq_vec / r0_sq_vec))

            gain = 1.0
            if spec.target_total_post_wdown_rms is not None and pilot_rms > 1e-12:
                desired_layer_rms = float(spec.target_total_post_wdown_rms) * layer_scale
                gain = min(config.max_gain_multiplier, desired_layer_rms / pilot_rms)

            # Enforce worst-case per-state radial eta bound across all calibration states
            if max_pilot_eta > 0.0 and (gain * gain * max_pilot_eta) > config.max_per_layer_radial_eta:
                gain = math.sqrt((0.99 * config.max_per_layer_radial_eta) / max_pilot_eta)

            u_calibrated = np.ascontiguousarray(u_split * gain)
            final_patch = (
                pilot
                if abs(gain - 1.0) <= 1e-15
                else proj.stage_edit(
                    k_m,
                    u_calibrated,
                    is_residual=True,
                    rank=config.rank,
                    preservation_keys=pres_k if pres_k.size > 0 else None,
                    preservation_weight=1.0,
                    slot_name=f"{pack_name}.blk{lid}",
                )
            )

            # Post-quantization per-state eta check (catches F16 quantization overshoot)
            dh_final = w_down_64[lid] @ (g_m * (final_patch.factor_b @ (final_patch.factor_a @ k_m)))
            max_final_eta = float(np.max(np.sum(dh_final * dh_final, axis=0) / r0_sq_vec))
            if max_final_eta > config.max_per_layer_radial_eta and max_final_eta > 0.0:
                shrink_fix = math.sqrt((0.98 * config.max_per_layer_radial_eta) / max_final_eta)
                gain = gain * shrink_fix
                u_calibrated = np.ascontiguousarray(u_split * gain)
                final_patch = proj.stage_edit(
                    k_m,
                    u_calibrated,
                    is_residual=True,
                    rank=config.rank,
                    preservation_keys=pres_k if pres_k.size > 0 else None,
                    preservation_weight=1.0,
                    slot_name=f"{pack_name}.blk{lid}",
                )

            patches_by_layer[lid] = final_patch
            padded_by_layer[lid] = _pad_factors_to_rank(
                final_patch.factor_b, final_patch.factor_a, config.rank
            )
            cal_res_by_layer[lid] = u_calibrated
            pres_by_layer[lid] = pres_k
            gains_by_layer[lid] = gain

        compiled_suite[pack_name] = CompiledMultiLayerPack(
            pack_name=pack_name,
            layer_ids=sorted_layers,
            patches_by_layer=patches_by_layer,
            padded_factors_by_layer=padded_by_layer,
            calibrated_residuals_by_layer=cal_res_by_layer,
            preservation_keys_by_layer=pres_by_layer,
            k0_effective_by_layer=k0_eff_by_layer,
            applied_gains_by_layer=gains_by_layer,
            rms_eps=rms_eps,
        )

    return compiled_suite


def verify_multilayer_suite_rederivation(
    expected_suite: Mapping[str, CompiledMultiLayerPack],
    *,
    layer_ids: Sequence[int],
    w_down_by_layer: Mapping[int, np.ndarray],
    k0_noise_by_layer: Mapping[int, np.ndarray],
    packs: Mapping[str, LayerPackSpec],
    config: TangentBetaEditConfig,
    k0_anchors_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    base_up_weights: Optional[Mapping[int, np.ndarray]] = None,
    rms_eps: float = 1e-6,
) -> tuple[bool, dict[str, Any]]:
    """Verify both per-layer BetaEdit 0-byte rederivation and end-to-end suite replay (Veto 1)."""
    be_cfg = config.to_betaedit_config()
    zero_w0 = np.zeros((config.d_out, config.d_in), dtype=np.float64)

    per_layer_checks: dict[str, bool] = {}
    for pack_name, comp_pack in sorted(expected_suite.items()):
        spec = packs[pack_name]
        for lid in comp_pack.layer_ids:
            w0 = (
                np.ascontiguousarray(base_up_weights[lid], dtype=np.float64)
                if (base_up_weights is not None and lid in base_up_weights)
                else zero_w0
            )
            pres = comp_pack.preservation_keys_by_layer[lid]
            ok_l, _ = verify_rederivation(
                comp_pack.patches_by_layer[lid],
                base_weight=w0,
                config=be_cfg,
                k0=comp_pack.k0_effective_by_layer[lid],
                keys=spec.keys_by_layer[lid],
                target_values_or_residuals=comp_pack.calibrated_residuals_by_layer[lid],
                is_residual=True,
                rank=config.rank,
                preservation_keys=pres if pres.size > 0 else None,
                preservation_weight=1.0,
                safety_directions=spec.safety_directions_by_layer.get(lid, ()),
            )
            per_layer_checks[f"{pack_name}.blk{lid}"] = ok_l

    reversed_packs = {k: packs[k] for k in reversed(list(packs.keys()))}
    rebuilt_suite = compile_multilayer_tangent_suite(
        layer_ids=list(reversed(list(layer_ids))),
        w_down_by_layer=w_down_by_layer,
        k0_noise_by_layer=k0_noise_by_layer,
        packs=reversed_packs,
        config=config,
        k0_anchors_by_layer=k0_anchors_by_layer,
        base_up_weights=base_up_weights,
        rms_eps=rms_eps,
    )

    suite_match = True
    digest_report: dict[str, dict[str, str]] = {}
    for pack_name, exp_pack in sorted(expected_suite.items()):
        reb_pack = rebuilt_suite[pack_name]
        for lid in exp_pack.layer_ids:
            ep: StagedPatch = exp_pack.patches_by_layer[lid]
            rp: StagedPatch = reb_pack.patches_by_layer[lid]
            ec: FixedRadixCertificate = ep.certificate
            rc: FixedRadixCertificate = rp.certificate
            same = (
                ep.b_bytes == rp.b_bytes
                and ep.a_bytes == rp.a_bytes
                and ep.factor_digest == rp.factor_digest
                and ec.sha256() == rc.sha256()
            )
            if not same:
                suite_match = False
            digest_report[f"{pack_name}.blk{lid}"] = {
                "expected_factor_digest": ep.factor_digest,
                "rebuilt_factor_digest": rp.factor_digest,
                "expected_cert_sha256": ec.sha256(),
                "rebuilt_cert_sha256": rc.sha256(),
            }

    all_ok = suite_match and all(per_layer_checks.values())
    return all_ok, {
        "all_per_layer_rederivations_ok": all(per_layer_checks.values()),
        "reversed_pack_order_zero_byte_match": suite_match,
        "per_layer_checks": per_layer_checks,
        "digests": digest_report,
    }
