"""Cross-ISA bit-exact deterministic CPU float64 linear algebra for BetaEdit.

Eliminates every source of cross-architecture (arm64 Apple Accelerate vs
x86_64 OpenBLAS/MKL/CBLAS) floating-point and basis divergence in Weft's
BetaEdit compiler while preserving mathematical equivalence (< 1e-12 relative
distance to vendor LAPACK) and full 0-byte SHA-256 identity across ISAs on
every dtype: raw float64, f32, bf16, f16, int8, int4, and FixedRadixCertificate.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np

ThresholdMode = Literal["absolute", "relative_max", "relative_mean"]


def ieee754_binary_sum(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Canonical power-of-2 pairwise IEEE-754 binary64 addition (`roundTiesToEven`)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[axis] == 0:
        out_shape = list(a.shape)
        out_shape.pop(axis)
        return np.zeros(out_shape, dtype=np.float64)
    cur = np.moveaxis(a, axis, -1)
    n = cur.shape[-1]
    p2 = 1 << (n - 1).bit_length() if n > 1 else 1
    if p2 > n:
        pad_shape = (*cur.shape[:-1], p2 - n)
        cur = np.concatenate([cur, np.zeros(pad_shape, dtype=np.float64)], axis=-1)
    while cur.shape[-1] > 1:
        cur = np.add(cur[..., 0::2], cur[..., 1::2])
    res = cur[..., 0]
    if res.ndim == 0:
        return res
    return np.ascontiguousarray(res, dtype=np.float64)


def _ozaki_chunk(a: np.ndarray, b: np.ndarray, *, limb_bits: int = 17, n_limbs: int = 4) -> np.ndarray:
    """Exact-integer limb matmul for inner dimension k <= 16384 (`k * 2^34 < 2^48 < 2^53`)."""
    m, k = a.shape
    _, n = b.shape
    if k == 0 or m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    max_a = np.max(np.abs(a), axis=1)
    _, exp_a = np.frexp(max_a)
    exp_a = np.where(max_a > 0.0, exp_a, 0).astype(np.int32)
    a_norm = np.ldexp(a, -exp_a[:, None])

    max_b = np.max(np.abs(b), axis=0)
    _, exp_b = np.frexp(max_b)
    exp_b = np.where(max_b > 0.0, exp_b, 0).astype(np.int32)
    b_norm = np.ldexp(b, -exp_b[None, :])

    a_limbs = []
    rem_a = a_norm
    for _ in range(n_limbs):
        sc = np.ldexp(rem_a, limb_bits)
        limb = np.trunc(sc)
        rem_a = np.subtract(sc, limb)
        a_limbs.append(np.ascontiguousarray(limb, dtype=np.float64))

    b_limbs = []
    rem_b = b_norm
    for _ in range(n_limbs):
        sc = np.ldexp(rem_b, limb_bits)
        limb = np.trunc(sc)
        rem_b = np.subtract(sc, limb)
        b_limbs.append(np.ascontiguousarray(limb, dtype=np.float64))

    weights = [np.zeros((m, n), dtype=np.int64) for _ in range(n_limbs)]
    for p in range(n_limbs):
        for q in range(n_limbs - p):
            prod_exact = a_limbs[p] @ b_limbs[q]
            weights[p + q] += prod_exact.astype(np.int64)

    mask = (1 << limb_bits) - 1
    w3 = weights[3]
    c3 = w3 >> limb_bits
    r3 = w3 & mask

    t2 = weights[2] + c3
    c2 = t2 >> limb_bits
    r2 = t2 & mask

    t1 = weights[1] + c2
    c1 = t1 >> limb_bits
    r1 = t1 & mask

    t0 = weights[0] + c1
    mid_int = (r1 << limb_bits) + r2

    hi = np.ldexp(t0.astype(np.float64), -2 * limb_bits)
    mid = np.ldexp(mid_int.astype(np.float64), -4 * limb_bits)
    lo = np.ldexp(r3.astype(np.float64), -5 * limb_bits)
    combined = np.add(hi, np.add(mid, lo))
    out = np.ldexp(combined, exp_a[:, None] + exp_b[None, :])
    zero_mask = (max_a[:, None] == 0.0) | (max_b[None, :] == 0.0)
    out = np.where(zero_mask, 0.0, out)
    return np.ascontiguousarray(out, dtype=np.float64)


def ozaki_exact_matmul(a: np.ndarray, b: np.ndarray, *, k_block: int = 8192) -> np.ndarray:
    """Cross-ISA bit-identical float64 matrix multiplication via Ozaki integer-limb splitting."""
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    if a64.ndim != 2 or b64.ndim != 2 or a64.shape[1] != b64.shape[0]:
        raise ValueError(f"Incompatible shapes for ozaki_exact_matmul: {a64.shape} vs {b64.shape}")
    k = a64.shape[1]
    if k <= k_block:
        return _ozaki_chunk(a64, b64)
    blocks = []
    for start in range(0, k, k_block):
        end = min(start + k_block, k)
        blocks.append(_ozaki_chunk(a64[:, start:end], b64[start:end, :]))
    return ieee754_binary_sum(np.stack(blocks, axis=0), axis=0)


def _cholesky_unblocked(a_panel: np.ndarray) -> np.ndarray:
    b = a_panel.shape[0]
    l_blk = np.zeros((b, b), dtype=np.float64)
    for j in range(b):
        if j > 0:
            row_j = l_blk[j, :j]
            dot_jj = float(ieee754_binary_sum(np.multiply(row_j, row_j)))
        else:
            dot_jj = 0.0
        diag_val = float(np.subtract(a_panel[j, j], dot_jj))
        if diag_val <= 0.0:
            raise np.linalg.LinAlgError(f"Matrix not positive definite at local index {j}: {diag_val}")
        l_jj = math.sqrt(diag_val)
        l_blk[j, j] = l_jj
        if j + 1 < b:
            if j > 0:
                dots = ieee754_binary_sum(
                    np.multiply(l_blk[j + 1 :, :j], row_j[None, :]), axis=1
                )
                num = np.subtract(a_panel[j + 1 :, j], dots)
            else:
                num = a_panel[j + 1 :, j].copy()
            l_blk[j + 1 :, j] = np.divide(num, l_jj)
    return l_blk


def _solve_tril_left(l_blk: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    b = l_blk.shape[0]
    x = np.zeros_like(rhs, dtype=np.float64)
    for i in range(b):
        if i > 0:
            corr = ieee754_binary_sum(
                np.multiply(l_blk[i, :i][:, None], x[:i, :]), axis=0
            )
            num = np.subtract(rhs[i, :], corr)
        else:
            num = rhs[i, :]
        x[i, :] = np.divide(num, l_blk[i, i])
    return x


def _solve_triu_left(u_blk: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    b = u_blk.shape[0]
    x = np.zeros_like(rhs, dtype=np.float64)
    for i in range(b - 1, -1, -1):
        if i + 1 < b:
            corr = ieee754_binary_sum(
                np.multiply(u_blk[i, i + 1 :][:, None], x[i + 1 :, :]), axis=0
            )
            num = np.subtract(rhs[i, :], corr)
        else:
            num = rhs[i, :]
        x[i, :] = np.divide(num, u_blk[i, i])
    return x


def det_cholesky_solve(spd: np.ndarray, rhs: np.ndarray, *, block_size: int = 128) -> np.ndarray:
    """Cross-ISA bit-exact blocked Cholesky factorization and triangular solve `(spd) X = rhs`."""
    n = spd.shape[0]
    work = np.ascontiguousarray(spd, dtype=np.float64).copy()
    l_mat = np.zeros((n, n), dtype=np.float64)

    for k0 in range(0, n, block_size):
        k1 = min(k0 + block_size, n)
        l_diag = _cholesky_unblocked(work[k0:k1, k0:k1])
        l_mat[k0:k1, k0:k1] = l_diag
        if k1 < n:
            l_sub_t = _solve_tril_left(l_diag, work[k1:n, k0:k1].T)
            l_sub = np.ascontiguousarray(l_sub_t.T, dtype=np.float64)
            l_mat[k1:n, k0:k1] = l_sub
            work[k1:n, k1:n] = np.subtract(
                work[k1:n, k1:n], ozaki_exact_matmul(l_sub, l_sub_t)
            )

    y = np.ascontiguousarray(rhs, dtype=np.float64).copy()
    for k0 in range(0, n, block_size):
        k1 = min(k0 + block_size, n)
        y[k0:k1, :] = _solve_tril_left(l_mat[k0:k1, k0:k1], y[k0:k1, :])
        if k1 < n:
            y[k1:n, :] = np.subtract(
                y[k1:n, :], ozaki_exact_matmul(l_mat[k1:n, k0:k1], y[k0:k1, :])
            )

    x = y
    starts = list(range(0, n, block_size))
    for k0 in reversed(starts):
        k1 = min(k0 + block_size, n)
        if k1 < n:
            x[k0:k1, :] = np.subtract(
                x[k0:k1, :], ozaki_exact_matmul(l_mat[k1:n, k0:k1].T, x[k1:n, :])
            )
        x[k0:k1, :] = _solve_triu_left(l_mat[k0:k1, k0:k1].T, x[k0:k1, :])
    return np.ascontiguousarray(x, dtype=np.float64)


def det_householder_qr(mat: np.ndarray, *, panel_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Cross-ISA bit-exact WY-blocked Householder reduced QR with non-negative R diagonal."""
    a = np.ascontiguousarray(mat, dtype=np.float64).copy()
    m, n = a.shape
    k = min(m, n)
    v_store = np.zeros((m, k), dtype=np.float64)
    taus = np.zeros(k, dtype=np.float64)

    for j0 in range(0, k, panel_size):
        j1 = min(j0 + panel_size, k)
        b = j1 - j0
        t_wy = np.zeros((b, b), dtype=np.float64)

        for jj in range(b):
            j = j0 + jj
            x = a[j:, j]
            tail_sq = float(ieee754_binary_sum(np.multiply(x[1:], x[1:]))) if x.size > 1 else 0.0
            if tail_sq == 0.0 and x[0] >= 0.0:
                v_store[j, j] = 1.0
                taus[j] = 0.0
                continue
            norm_x = math.sqrt(float(np.add(np.multiply(x[0], x[0]), tail_sq)))
            if norm_x == 0.0:
                v_store[j, j] = 1.0
                taus[j] = 0.0
                continue

            x0 = float(x[0])
            if x0 <= 0.0:
                v0 = float(np.subtract(x0, norm_x))
            else:
                v0 = -float(np.divide(tail_sq, np.add(x0, norm_x)))

            tau = float(np.divide(2.0 * v0 * v0, np.add(tail_sq, v0 * v0)))
            v = np.zeros(m - j, dtype=np.float64)
            v[0] = 1.0
            if x.size > 1:
                v[1:] = np.divide(x[1:], v0)

            v_store[j:, j] = v
            taus[j] = tau
            a[j, j] = norm_x
            if j + 1 < m:
                a[j + 1 :, j] = 0.0

            if j + 1 < j1:
                w = ieee754_binary_sum(
                    np.multiply(v[:, None], a[j:, j + 1 : j1]), axis=0
                )
                a[j:, j + 1 : j1] = np.subtract(
                    a[j:, j + 1 : j1], np.multiply(tau * v[:, None], w[None, :])
                )

            t_wy[jj, jj] = tau
            if jj > 0 and tau != 0.0:
                vt_v = ieee754_binary_sum(
                    np.multiply(v_store[j:, j0 : j0 + jj], v[:, None]), axis=0
                )
                z = -tau * ieee754_binary_sum(
                    np.multiply(t_wy[:jj, :jj], vt_v[None, :]), axis=1
                )
                t_wy[:jj, jj] = z

        if j1 < n:
            v_panel = v_store[j0:, j0:j1]
            vt_a = ozaki_exact_matmul(v_panel.T, a[j0:, j1:])
            t_vt_a = ozaki_exact_matmul(t_wy.T, vt_a)
            a[j0:, j1:] = np.subtract(a[j0:, j1:], ozaki_exact_matmul(v_panel, t_vt_a))

    r_mat = np.triu(a[:k, :]).copy()

    q_mat = np.zeros((m, k), dtype=np.float64)
    for i in range(k):
        q_mat[i, i] = 1.0

    panel_starts = list(range(0, k, panel_size))
    for j0 in reversed(panel_starts):
        j1 = min(j0 + panel_size, k)
        b = j1 - j0
        t_wy = np.zeros((b, b), dtype=np.float64)
        for jj in range(b):
            j = j0 + jj
            tau = taus[j]
            t_wy[jj, jj] = tau
            if jj > 0 and tau != 0.0:
                vt_v = ieee754_binary_sum(
                    np.multiply(v_store[j:, j0 : j0 + jj], v_store[j:, j][:, None]), axis=0
                )
                t_wy[:jj, jj] = -tau * ieee754_binary_sum(
                    np.multiply(t_wy[:jj, :jj], vt_v[None, :]), axis=1
                )
        v_panel = v_store[j0:, j0:j1]
        vt_q = ozaki_exact_matmul(v_panel.T, q_mat[j0:, j0:])
        t_vt_q = ozaki_exact_matmul(t_wy, vt_q)
        q_mat[j0:, j0:] = np.subtract(q_mat[j0:, j0:], ozaki_exact_matmul(v_panel, t_vt_q))

    return np.ascontiguousarray(q_mat, dtype=np.float64), np.ascontiguousarray(r_mat, dtype=np.float64)


def _round_robin_pairs(p_even: int) -> list[tuple[np.ndarray, np.ndarray]]:
    ring = list(range(p_even))
    schedule = []
    half = p_even // 2
    for _ in range(p_even - 1):
        left = np.array(ring[:half], dtype=np.int64)
        right = np.array(ring[half:][::-1], dtype=np.int64)
        lo = np.minimum(left, right)
        hi = np.maximum(left, right)
        schedule.append((lo, hi))
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return schedule


def det_jacobi_svd(
    mat: np.ndarray,
    *,
    max_sweeps: int = 15,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-ISA bit-exact SVD via algebraic sqrt-only Round-Robin One-Sided Jacobi."""
    a = np.ascontiguousarray(mat, dtype=np.float64)
    m, n = a.shape
    transposed = False
    if m < n:
        a = np.ascontiguousarray(a.T, dtype=np.float64)
        m, n = a.shape
        transposed = True

    if n == 0:
        return (
            np.zeros((mat.shape[0], 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, mat.shape[1]), dtype=np.float64),
        )
    if n == 1:
        col = a[:, 0]
        s0 = math.sqrt(float(ieee754_binary_sum(np.multiply(col, col))))
        u = np.divide(col, s0)[:, None] if s0 > 0.0 else np.eye(m, 1, dtype=np.float64)
        vh = np.ones((1, 1), dtype=np.float64)
        if transposed:
            return vh.T, np.array([s0], dtype=np.float64), u.T
        return u, np.array([s0], dtype=np.float64), vh

    pad = n % 2
    p = n + pad
    w = np.zeros((m, p), dtype=np.float64)
    w[:, :n] = a
    v = np.eye(p, dtype=np.float64)
    schedule = _round_robin_pairs(p)

    for _ in range(max_sweeps):
        max_rel_off = 0.0
        for idx_i, idx_j in schedule:
            wi = w[:, idx_i]
            wj = w[:, idx_j]
            aii = ieee754_binary_sum(np.multiply(wi, wi), axis=0)
            ajj = ieee754_binary_sum(np.multiply(wj, wj), axis=0)
            aij = ieee754_binary_sum(np.multiply(wi, wj), axis=0)

            denom_norm = np.sqrt(np.multiply(aii, ajj))
            safe_denom = np.where(denom_norm > 0.0, denom_norm, 1.0)
            active = (denom_norm > 0.0) & (np.abs(aij) > tol * denom_norm)
            if not np.any(active):
                continue

            rel = np.where(denom_norm > 0.0, np.divide(np.abs(aij), safe_denom), 0.0)
            step_max = float(np.max(rel))
            if step_max > max_rel_off:
                max_rel_off = step_max

            safe_aij = np.where(active, aij, 1.0)
            tau = np.divide(np.subtract(ajj, aii), np.multiply(2.0, safe_aij))
            sign_tau = np.where(tau >= 0.0, 1.0, -1.0)
            t = np.divide(
                sign_tau,
                np.add(np.abs(tau), np.sqrt(np.add(1.0, np.multiply(tau, tau)))),
            )
            t = np.where(active, t, 0.0)
            c = np.divide(1.0, np.sqrt(np.add(1.0, np.multiply(t, t))))
            s = np.multiply(t, c)

            w[:, idx_i] = np.subtract(np.multiply(wi, c[None, :]), np.multiply(wj, s[None, :]))
            w[:, idx_j] = np.add(np.multiply(wi, s[None, :]), np.multiply(wj, c[None, :]))

            vi = v[:, idx_i]
            vj = v[:, idx_j]
            v[:, idx_i] = np.subtract(np.multiply(vi, c[None, :]), np.multiply(vj, s[None, :]))
            v[:, idx_j] = np.add(np.multiply(vi, s[None, :]), np.multiply(vj, c[None, :]))

        if max_rel_off <= tol:
            break

    w_real = w[:, :n]
    v_real = v[:n, :n]
    s_raw = np.sqrt(ieee754_binary_sum(np.multiply(w_real, w_real), axis=0))

    order = np.lexsort((np.arange(n, dtype=np.int64), -s_raw))
    s_vals = np.ascontiguousarray(s_raw[order], dtype=np.float64)
    w_sorted = w_real[:, order]
    v_sorted = v_real[:, order]

    u_mat = np.zeros((m, n), dtype=np.float64)
    for k in range(n):
        sk = float(s_vals[k])
        if sk > 0.0:
            u_mat[:, k] = np.divide(w_sorted[:, k], sk)
        elif k < m:
            u_mat[k, k] = 1.0
    vh_mat = np.ascontiguousarray(v_sorted.T, dtype=np.float64)

    if transposed:
        return (
            np.ascontiguousarray(vh_mat.T, dtype=np.float64),
            s_vals,
            np.ascontiguousarray(u_mat.T, dtype=np.float64),
        )
    return np.ascontiguousarray(u_mat, dtype=np.float64), s_vals, vh_mat


def canonicalize_output_factors(b_mat: np.ndarray, a_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize rank-component signs directly in physical `(d_out, r)` / `(r, d_in)` space."""
    b = np.ascontiguousarray(b_mat, dtype=np.float64).copy()
    a = np.ascontiguousarray(a_mat, dtype=np.float64).copy()
    for k in range(b.shape[1]):
        idx = int(np.argmax(np.abs(b[:, k])))
        if b[idx, k] < 0.0:
            b[:, k] = -b[:, k]
            a[k, :] = -a[k, :]
    return b, a


def det_decompose_rank_r(
    left: np.ndarray,
    right: np.ndarray,
    max_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cross-ISA bit-exact skinny-QR + Jacobi SVD rank-r decomposition."""
    l64 = np.ascontiguousarray(left, dtype=np.float64)
    r64 = np.ascontiguousarray(right, dtype=np.float64)
    q_l, r_l = det_householder_qr(l64)
    q_r, r_r = det_householder_qr(r64.T)
    core = ozaki_exact_matmul(r_l, r_r.T)
    u_c, s_vals, vh_c = det_jacobi_svd(core)

    tot = float(ieee754_binary_sum(np.multiply(s_vals, s_vals)))
    if tot == 0.0:
        return (
            np.zeros((l64.shape[0], 1), dtype=np.float64),
            np.zeros((1, r64.shape[1]), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            0.0,
        )

    r_eff = min(max_rank, int(s_vals.size))
    s_kept = np.ascontiguousarray(s_vals[:r_eff], dtype=np.float64)
    disc = (
        float(ieee754_binary_sum(np.multiply(s_vals[r_eff:], s_vals[r_eff:])))
        if r_eff < s_vals.size
        else 0.0
    )
    sqrt_s = np.sqrt(s_kept)
    b_raw = np.multiply(ozaki_exact_matmul(q_l, u_c[:, :r_eff]), sqrt_s[None, :])
    a_raw = np.multiply(sqrt_s[:, None], ozaki_exact_matmul(vh_c[:r_eff, :], q_r.T))
    b_canon, a_canon = canonicalize_output_factors(b_raw, a_raw)
    trunc_err = math.sqrt(max(0.0, disc) / tot)
    return b_canon, a_canon, s_kept, trunc_err


def _resolve_cutoff(
    eigvals_desc: np.ndarray,
    *,
    d_in: int,
    eps: float,
    threshold_mode: ThresholdMode,
) -> float:
    if eigvals_desc.size == 0:
        return eps
    if threshold_mode == "relative_max":
        m_ev = float(eigvals_desc[0])
        return float(np.multiply(eps, m_ev)) if m_ev > 0.0 else eps
    if threshold_mode == "relative_mean":
        tot_ev = float(ieee754_binary_sum(eigvals_desc))
        m_ev = float(np.divide(tot_ev, float(d_in)))
        return float(np.multiply(eps, m_ev)) if m_ev > 0.0 else eps
    return eps


def det_active_basis(
    k0_scaled: np.ndarray,
    *,
    d_in: int,
    cutoff: float,
    threshold_mode: ThresholdMode = "absolute",
    min_nullspace_dim: int = 1,
) -> tuple[Optional[np.ndarray], int]:
    """Extract orthonormal active range `U_active` (`s_k^2 >= eff_cutoff`) without dense `d_in x d_in` eigh."""
    k0 = np.ascontiguousarray(k0_scaled, dtype=np.float64)
    n0 = k0.shape[1]
    if n0 == 0:
        return None, d_in

    frob_sq_k0 = float(ieee754_binary_sum(np.multiply(k0, k0).reshape(-1)))
    if threshold_mode == "absolute" and frob_sq_k0 < cutoff:
        return None, d_in

    if n0 <= d_in:
        q_k, r_k = det_householder_qr(k0)
        u_r, s_k, _ = det_jacobi_svd(r_k)
        eigvals_desc = np.multiply(s_k, s_k)
        eff_cutoff = _resolve_cutoff(
            eigvals_desc, d_in=d_in, eps=cutoff, threshold_mode=threshold_mode
        )
        if float(eigvals_desc[0]) < eff_cutoff:
            return None, d_in
        max_active = max(0, d_in - min_nullspace_dim)
        n_active = min(int(np.sum(eigvals_desc >= eff_cutoff)), max_active)
        if n_active == 0:
            return None, d_in
        u_active = ozaki_exact_matmul(q_k, u_r[:, :n_active])
    else:
        q_k, r_k = det_householder_qr(k0.T)
        _, s_k, vh_r = det_jacobi_svd(r_k)
        eigvals_desc = np.multiply(s_k, s_k)
        eff_cutoff = _resolve_cutoff(
            eigvals_desc, d_in=d_in, eps=cutoff, threshold_mode=threshold_mode
        )
        if float(eigvals_desc[0]) < eff_cutoff:
            return None, d_in
        max_active = max(0, d_in - min_nullspace_dim)
        n_active = min(int(np.sum(eigvals_desc >= eff_cutoff)), max_active)
        if n_active == 0:
            return None, d_in
        u_active = np.ascontiguousarray(vh_r[:n_active, :].T, dtype=np.float64)

    return np.ascontiguousarray(u_active, dtype=np.float64), d_in - n_active


def apply_p_null(u_active: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Matrix-free application `P_null(X) = X - U_active @ (U_active.T @ X)`."""
    if u_active is None:
        return x
    ut_x = ozaki_exact_matmul(u_active.T, x)
    return np.subtract(x, ozaki_exact_matmul(u_active, ut_x))


def det_build_certificate_values(
    b_quant_f64: np.ndarray,
    a_quant_f64: np.ndarray,
    k0_scaled: np.ndarray,
    *,
    radix_scale_exp2: int = 24,
    safety_directions: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> tuple[int, int, int, int]:
    """Cross-ISA bit-exact integer certificate quantities `(frob, spec, leak, safety)`."""
    b64 = np.ascontiguousarray(b_quant_f64, dtype=np.float64)
    a64 = np.ascontiguousarray(a_quant_f64, dtype=np.float64)
    k0_s = np.ascontiguousarray(k0_scaled, dtype=np.float64)

    _, r_b = det_householder_qr(b64)
    _, r_a = det_householder_qr(a64.T)
    _, s_vals, _ = det_jacobi_svd(ozaki_exact_matmul(r_b, r_a.T))

    frob_sq = float(ieee754_binary_sum(np.multiply(s_vals, s_vals)))
    spec_sq = float(np.multiply(s_vals[0], s_vals[0])) if s_vals.size > 0 else 0.0

    bt_b = ozaki_exact_matmul(b64.T, b64)
    if k0_s.shape[1] > 0:
        a_k0 = ozaki_exact_matmul(a64, k0_s)
        ac_at = ozaki_exact_matmul(a_k0, a_k0.T)
        leak = max(0.0, float(ieee754_binary_sum(np.multiply(bt_b, ac_at).reshape(-1))))
    else:
        leak = 0.0

    max_dot_sq = 0.0
    for b_s, a_s in safety_directions:
        bs_b = ozaki_exact_matmul(b_s.T, b64)
        as_a = ozaki_exact_matmul(a_s, a64.T)
        dot = float(ieee754_binary_sum(np.multiply(bs_b, as_a).reshape(-1)))
        dot_sq = float(np.multiply(dot, dot))
        if dot_sq > max_dot_sq:
            max_dot_sq = dot_sq

    scale = float(1 << radix_scale_exp2)
    return (
        int(math.ceil(frob_sq * scale)),
        int(math.ceil(spec_sq * scale)),
        int(math.ceil(leak * scale)),
        int(math.ceil(max_dot_sq * scale)),
    )


def solve_betaedit_with_basis(
    *,
    u_active: Optional[np.ndarray],
    cov_factors: np.ndarray,
    k_solve: np.ndarray,
    r_solve: np.ndarray,
    rank: int,
    lambda_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve projected BetaEdit Eq. (9) given precomputed active-range basis `u_active`."""
    d_in = k_solve.shape[0]
    pk0 = apply_p_null(u_active, cov_factors) if cov_factors.shape[1] > 0 else cov_factors
    pk_solve = apply_p_null(u_active, k_solve)
    m_all = np.concatenate([pk0, pk_solve], axis=1) if pk0.shape[1] > 0 else pk_solve
    n0_cols = pk0.shape[1]
    m_total = m_all.shape[1]
    lr = float(lambda_ridge)

    if m_total <= d_in:
        gram = ozaki_exact_matmul(m_all.T, m_all)
        gram_sym = np.multiply(0.5, np.add(gram, gram.T))
        spd_dual = np.add(np.multiply(lr, np.eye(m_total, dtype=np.float64)), gram_sym)
        rhs_dual = np.ascontiguousarray(gram[:, n0_cols:], dtype=np.float64)
        w_dual = det_cholesky_solve(spd_dual, rhs_dual)
        z_solved = np.divide(np.subtract(pk_solve, ozaki_exact_matmul(m_all, w_dual)), lr)
    else:
        mm_t = ozaki_exact_matmul(m_all, m_all.T)
        mm_sym = np.multiply(0.5, np.add(mm_t, mm_t.T))
        spd_primal = np.add(np.multiply(lr, np.eye(d_in, dtype=np.float64)), mm_sym)
        z_solved = det_cholesky_solve(spd_primal, pk_solve)

    x_solved = apply_p_null(u_active, z_solved)
    right64 = np.ascontiguousarray(x_solved.T, dtype=np.float64)
    return det_decompose_rank_r(r_solve, right64, rank)


def compile_betaedit_cross_isa(
    *,
    k0: np.ndarray,
    keys: np.ndarray,
    residuals: np.ndarray,
    rank: int,
    lambda_leakage: float,
    lambda_ridge: float,
    nullspace_threshold: float = 1e9,
    threshold_mode: ThresholdMode = "absolute",
    min_nullspace_dim: int = 1,
    preservation_keys: Optional[np.ndarray] = None,
    preservation_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int]:
    """Compile `(b64, a64, s_vals, trunc_err, k0_scaled, nullspace_dim)` with 0-ULP cross-ISA parity."""
    k0_64 = np.ascontiguousarray(k0, dtype=np.float64)
    k_64 = np.ascontiguousarray(keys, dtype=np.float64)
    r_64 = np.ascontiguousarray(residuals, dtype=np.float64)
    d_in = k_64.shape[0]
    d_out = r_64.shape[0]

    if preservation_keys is not None and preservation_keys.size > 0:
        pk = np.ascontiguousarray(preservation_keys, dtype=np.float64)
        w_sqrt = math.sqrt(max(0.0, float(preservation_weight)))
        k_solve = np.concatenate([k_64, np.multiply(pk, w_sqrt)], axis=1)
        r_solve = np.concatenate([r_64, np.zeros((d_out, pk.shape[1]), dtype=np.float64)], axis=1)
    else:
        k_solve = k_64
        r_solve = r_64

    k0_scaled = np.multiply(math.sqrt(max(0.0, float(lambda_leakage))), k0_64)
    u_active, null_dim = det_active_basis(
        k0_scaled,
        d_in=d_in,
        cutoff=float(nullspace_threshold),
        threshold_mode=threshold_mode,
        min_nullspace_dim=min_nullspace_dim,
    )
    b64, a64, s_vals, trunc_err = solve_betaedit_with_basis(
        u_active=u_active,
        cov_factors=k0_scaled,
        k_solve=k_solve,
        r_solve=r_solve,
        rank=rank,
        lambda_ridge=lambda_ridge,
    )
    return b64, a64, s_vals, trunc_err, k0_scaled, null_dim
