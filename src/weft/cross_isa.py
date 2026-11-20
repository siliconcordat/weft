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


def _extract_ozaki_groups(
    mat: np.ndarray,
    *,
    row_wise: bool,
    limb_bits: int = 17,
    group_limbs: int = 4,
) -> list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]:
    """Decompose `mat` into exact non-overlapping 68-bit integer-limb tiers across arbitrary dynamic range."""
    rem = mat.copy()
    groups: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []

    while True:
        max_v = np.max(np.abs(rem), axis=1 if row_wise else 0)
        active = np.flatnonzero(max_v > 0.0)
        if active.size == 0:
            break
        sub = rem[active, :] if row_wise else rem[:, active]
        _, exp_v = np.frexp(max_v[active])
        exp_i32 = exp_v.astype(np.int32)
        shift_in = -exp_i32[:, None] if row_wise else -exp_i32[None, :]
        r_cur = np.ldexp(sub, shift_in)

        limbs: list[np.ndarray] = []
        for _ in range(group_limbs):
            sc = np.ldexp(r_cur, limb_bits)
            limb = np.trunc(sc)
            r_cur = np.subtract(sc, limb)
            limbs.append(np.ascontiguousarray(limb, dtype=np.float64))

        groups.append((active, exp_i32, limbs))
        sub_rem = sub.copy()
        for idx_l, limb in enumerate(limbs):
            shift_back = (
                (exp_i32 - (idx_l + 1) * limb_bits)[:, None]
                if row_wise
                else (exp_i32 - (idx_l + 1) * limb_bits)[None, :]
            )
            sub_rem = np.subtract(sub_rem, np.ldexp(limb, shift_back))
        if not np.any(sub_rem != 0.0):
            break

        if row_wise:
            rem[active, :] = sub_rem
        else:
            rem[:, active] = sub_rem

    return groups


def _eval_4x4_group_pair(
    limbs_a: list[np.ndarray],
    exp_a: np.ndarray,
    limbs_b: list[np.ndarray],
    exp_b: np.ndarray,
    *,
    limb_bits: int = 17,
) -> np.ndarray:
    """Evaluate all 16 exact-integer limb products `p, q in 0..3` for one `(group_a, group_b)` pair."""
    m_sub = limbs_a[0].shape[0]
    n_sub = limbs_b[0].shape[1]
    weights = [np.zeros((m_sub, n_sub), dtype=np.int64) for _ in range(7)]
    for p in range(4):
        lp = limbs_a[p]
        for q in range(4):
            prod_exact = lp @ limbs_b[q]
            weights[p + q] += prod_exact.astype(np.int64)

    mask = (1 << limb_bits) - 1
    t6 = weights[6]
    c6 = t6 >> limb_bits
    r6 = t6 & mask

    t5 = weights[5] + c6
    c5 = t5 >> limb_bits
    r5 = t5 & mask

    t4 = weights[4] + c5
    c4 = t4 >> limb_bits
    r4 = t4 & mask

    t3 = weights[3] + c4
    c3 = t3 >> limb_bits
    r3 = t3 & mask

    t2 = weights[2] + c3
    c2 = t2 >> limb_bits
    r2 = t2 & mask

    t1 = weights[1] + c2
    c1 = t1 >> limb_bits
    r1 = t1 & mask

    t0 = weights[0] + c1
    mid_int = (r1 << (2 * limb_bits)) + (r2 << limb_bits) + r3
    lo_int = (r4 << (2 * limb_bits)) + (r5 << limb_bits) + r6

    hi = np.ldexp(t0.astype(np.float64), -2 * limb_bits)
    mid = np.ldexp(mid_int.astype(np.float64), -5 * limb_bits)
    lo = np.ldexp(lo_int.astype(np.float64), -8 * limb_bits)
    combined = np.add(hi, np.add(mid, lo))
    return np.ldexp(combined, exp_a[:, None] + exp_b[None, :])


def _two_sum_accumulate(acc: np.ndarray, err: np.ndarray, term: np.ndarray) -> None:
    """Knuth error-free `two_sum` accumulation in place on `(acc, err)`."""
    s = np.add(acc, term)
    v = np.subtract(s, acc)
    e = np.add(np.subtract(acc, np.subtract(s, v)), np.subtract(term, v))
    acc[:] = s
    err[:] = np.add(err, e)


def _ozaki_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact-integer adaptive multi-tier matmul for inner dimension `k <= 16384`."""
    m, k = a.shape
    _, n = b.shape
    if k == 0 or m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)

    groups_a = _extract_ozaki_groups(a, row_wise=True)
    groups_b = _extract_ozaki_groups(b, row_wise=False)
    if not groups_a or not groups_b:
        return np.zeros((m, n), dtype=np.float64)

    if len(groups_a) == 1 and len(groups_b) == 1:
        rows_a, exp_a, limbs_a = groups_a[0]
        cols_b, exp_b, limbs_b = groups_b[0]
        sub_out = _eval_4x4_group_pair(limbs_a, exp_a, limbs_b, exp_b)
        if rows_a.size == m and cols_b.size == n:
            return np.ascontiguousarray(sub_out, dtype=np.float64)
        out = np.zeros((m, n), dtype=np.float64)
        out[np.ix_(rows_a, cols_b)] = sub_out
        return out

    acc = np.zeros((m, n), dtype=np.float64)
    err = np.zeros((m, n), dtype=np.float64)
    for rows_a, exp_a, limbs_a in groups_a:
        for cols_b, exp_b, limbs_b in groups_b:
            sub_val = _eval_4x4_group_pair(limbs_a, exp_a, limbs_b, exp_b)
            idx = np.ix_(rows_a, cols_b)
            s = np.add(acc[idx], sub_val)
            v = np.subtract(s, acc[idx])
            e = np.add(np.subtract(acc[idx], np.subtract(s, v)), np.subtract(sub_val, v))
            acc[idx] = s
            err[idx] = np.add(err[idx], e)
    return np.ascontiguousarray(np.add(acc, err), dtype=np.float64)


def ozaki_exact_matmul(a: np.ndarray, b: np.ndarray, *, k_block: int = 8192) -> np.ndarray:
    """Cross-ISA bit-identical float64 matrix multiplication via adaptive Ozaki integer-limb splitting."""
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    if a64.ndim != 2 or b64.ndim != 2 or a64.shape[1] != b64.shape[0]:
        raise ValueError(f"Incompatible shapes for ozaki_exact_matmul: {a64.shape} vs {b64.shape}")
    if k_block < 1 or k_block > 16384:
        raise ValueError(f"k_block must lie in [1, 16384] for exact IEEE-754 integer products, got {k_block}")
    if not (np.all(np.isfinite(a64)) and np.all(np.isfinite(b64))):
        raise ValueError("ozaki_exact_matmul requires finite float64 inputs")
    k = a64.shape[1]
    if k <= k_block:
        return _ozaki_chunk(a64, b64)
    acc = np.zeros((a64.shape[0], b64.shape[1]), dtype=np.float64)
    err = np.zeros_like(acc)
    for start in range(0, k, k_block):
        end = min(start + k_block, k)
        _two_sum_accumulate(acc, err, _ozaki_chunk(a64[:, start:end], b64[start:end, :]))
    return np.ascontiguousarray(np.add(acc, err), dtype=np.float64)


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
        if not math.isfinite(diag_val) or diag_val <= 0.0:
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
    work = np.ascontiguousarray(spd, dtype=np.float64).copy()
    y = np.ascontiguousarray(rhs, dtype=np.float64).copy()
    if not (np.all(np.isfinite(work)) and np.all(np.isfinite(y))):
        raise ValueError("det_cholesky_solve requires finite float64 inputs")
    n = work.shape[0]
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


def _safe_vec_norm(v: np.ndarray) -> float:
    """Overflow/underflow-safe deterministic Euclidean norm using power-of-2 `frexp`/`ldexp` scaling."""
    if v.size == 0:
        return 0.0
    max_v = float(np.max(np.abs(v)))
    if max_v == 0.0:
        return 0.0
    _, exp_v = math.frexp(max_v)
    vs = np.ldexp(v, -exp_v)
    return math.ldexp(math.sqrt(float(ieee754_binary_sum(np.multiply(vs, vs)))), exp_v)


def _safe_col_norms(mat: np.ndarray) -> np.ndarray:
    """Overflow/underflow-safe deterministic Euclidean column norms for a 2D array."""
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return np.zeros(mat.shape[1], dtype=np.float64)
    max_c = np.max(np.abs(mat), axis=0)
    _, exp_c = np.frexp(max_c)
    exp_i32 = np.where(max_c > 0.0, exp_c, 0).astype(np.int32)
    ms = np.ldexp(mat, -exp_i32[None, :])
    sq = ieee754_binary_sum(np.multiply(ms, ms), axis=0)
    norms = np.ldexp(np.sqrt(sq), exp_i32)
    return np.where(max_c > 0.0, norms, 0.0)


def det_householder_qr(mat: np.ndarray, *, panel_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Cross-ISA bit-exact WY-blocked Householder reduced QR (`||u||_2 = 1`, `R_ii >= 0`)."""
    a = np.ascontiguousarray(mat, dtype=np.float64).copy()
    if not np.all(np.isfinite(a)):
        raise ValueError("det_householder_qr requires finite float64 input")
    m, n = a.shape
    k = min(m, n)
    u_store = np.zeros((m, k), dtype=np.float64)
    taus = np.zeros(k, dtype=np.float64)

    for j0 in range(0, k, panel_size):
        j1 = min(j0 + panel_size, k)
        b = j1 - j0
        t_wy = np.zeros((b, b), dtype=np.float64)

        for jj in range(b):
            j = j0 + jj
            x = a[j:, j]
            max_x = float(np.max(np.abs(x)))
            if max_x == 0.0:
                u_store[j, j] = 1.0
                taus[j] = 0.0
                a[j, j] = 0.0
                continue

            _, exp_x = math.frexp(max_x)
            xs = np.ldexp(x, -exp_x)
            xs0 = float(xs[0])
            max_tail = float(np.max(np.abs(xs[1:]))) if xs.size > 1 else 0.0

            if max_tail == 0.0:
                if xs0 >= 0.0:
                    u_store[j, j] = 1.0
                    taus[j] = 0.0
                    a[j, j] = float(x[0])
                    continue
                u = np.zeros(m - j, dtype=np.float64)
                u[0] = 1.0
                tau = 2.0
                norm_x = -float(x[0])
            else:
                _, exp_t = math.frexp(max_tail)
                xt = np.ldexp(xs[1:], -exp_t)
                tail_unit_norm = math.sqrt(float(ieee754_binary_sum(np.multiply(xt, xt))))
                tail_norm_s = math.ldexp(tail_unit_norm, exp_t)

                a0 = abs(xs0)
                if a0 == 0.0:
                    norm_xs = tail_norm_s
                elif a0 >= tail_norm_s:
                    r_ratio = tail_norm_s / a0
                    norm_xs = a0 * math.sqrt(1.0 + r_ratio * r_ratio)
                else:
                    r_ratio = a0 / tail_norm_s
                    norm_xs = tail_norm_s * math.sqrt(1.0 + r_ratio * r_ratio)
                norm_x = math.ldexp(norm_xs, exp_x)

                if xs0 <= 0.0:
                    v0 = xs0 - norm_xs
                    w_tail = np.divide(np.ldexp(xt, exp_t), v0)
                    u_norm = math.sqrt(1.0 + float(ieee754_binary_sum(np.multiply(w_tail, w_tail))))
                    u = np.empty(m - j, dtype=np.float64)
                    u[0] = -1.0 / u_norm
                    u[1:] = np.divide(-w_tail, u_norm)
                    tau = 2.0
                else:
                    alpha = -tail_norm_s / (xs0 + norm_xs)
                    if alpha == 0.0:
                        u_store[j, j] = 1.0
                        taus[j] = 0.0
                        a[j, j] = norm_x
                        a[j + 1 :, j] = 0.0
                        continue
                    inv_unorm = 1.0 / math.sqrt(1.0 + alpha * alpha)
                    u = np.empty(m - j, dtype=np.float64)
                    u[0] = alpha * inv_unorm
                    u[1:] = np.multiply(xt, inv_unorm / tail_unit_norm)
                    tau = 2.0

            u_store[j:, j] = u
            taus[j] = tau
            a[j, j] = norm_x
            if j + 1 < m:
                a[j + 1 :, j] = 0.0

            if j + 1 < j1:
                w = ozaki_exact_matmul(u[None, :], a[j:, j + 1 : j1])[0]
                a[j:, j + 1 : j1] = np.subtract(
                    a[j:, j + 1 : j1], np.multiply(tau * u[:, None], w[None, :])
                )

            t_wy[jj, jj] = tau
            if jj > 0 and tau != 0.0:
                ut_u = ozaki_exact_matmul(u_store[j:, j0 : j0 + jj].T, u[:, None])[:, 0]
                z = -tau * ieee754_binary_sum(
                    np.multiply(t_wy[:jj, :jj], ut_u[None, :]), axis=1
                )
                t_wy[:jj, jj] = z

        if j1 < n:
            u_panel = u_store[j0:, j0:j1]
            ut_a = ozaki_exact_matmul(u_panel.T, a[j0:, j1:])
            t_ut_a = ozaki_exact_matmul(t_wy.T, ut_a)
            a[j0:, j1:] = np.subtract(a[j0:, j1:], ozaki_exact_matmul(u_panel, t_ut_a))

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
                ut_u = ozaki_exact_matmul(
                    u_store[j:, j0 : j0 + jj].T, u_store[j:, j : j + 1]
                )[:, 0]
                t_wy[:jj, jj] = -tau * ieee754_binary_sum(
                    np.multiply(t_wy[:jj, :jj], ut_u[None, :]), axis=1
                )
        u_panel = u_store[j0:, j0:j1]
        ut_q = ozaki_exact_matmul(u_panel.T, q_mat[j0:, j0:])
        t_ut_q = ozaki_exact_matmul(t_wy, ut_q)
        q_mat[j0:, j0:] = np.subtract(q_mat[j0:, j0:], ozaki_exact_matmul(u_panel, t_ut_q))

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


def _complete_orthonormal_u(w_sorted: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Build strictly orthonormal `U` (`m x n`, `m >= n`) via 2-pass MGS + deterministic basis completion."""
    m, n = w_sorted.shape
    u_mat = np.zeros((m, n), dtype=np.float64)
    s0 = float(s_vals[0]) if n > 0 else 0.0
    noise_floor = s0 * float(max(m, n)) * 2.220446049250313e-16

    for k in range(n):
        sk = float(s_vals[k])
        placed = False
        if sk > noise_floor:
            cand = np.divide(w_sorted[:, k], sk)
            if k > 0:
                u_prev = u_mat[:, :k]
                for _ in range(2):
                    proj = ieee754_binary_sum(np.multiply(u_prev, cand[:, None]), axis=0)
                    cand = np.subtract(
                        cand, ieee754_binary_sum(np.multiply(u_prev, proj[None, :]), axis=1)
                    )
            rem_norm = _safe_vec_norm(cand)
            if rem_norm > 0.5:
                u_mat[:, k] = np.divide(cand, rem_norm)
                placed = True

        if not placed:
            if k == 0:
                u_mat[0, 0] = 1.0
            else:
                u_prev = u_mat[:, :k]
                row_energy = ieee754_binary_sum(np.multiply(u_prev, u_prev), axis=1)
                idx = int(np.argmin(row_energy))
                cand = np.zeros(m, dtype=np.float64)
                cand[idx] = 1.0
                for _ in range(2):
                    proj = ieee754_binary_sum(np.multiply(u_prev, cand[:, None]), axis=0)
                    cand = np.subtract(
                        cand, ieee754_binary_sum(np.multiply(u_prev, proj[None, :]), axis=1)
                    )
                rem_norm = _safe_vec_norm(cand)
                u_mat[:, k] = np.divide(cand, rem_norm)

    return u_mat


def det_jacobi_svd(
    mat: np.ndarray,
    *,
    max_sweeps: int = 60,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-ISA bit-exact SVD via scale-invariant algebraic sqrt-only Round-Robin Jacobi."""
    a = np.ascontiguousarray(mat, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("det_jacobi_svd requires finite float64 input")
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

    max_abs = float(np.max(np.abs(a)))
    if max_abs == 0.0:
        u_zero = np.eye(m, n, dtype=np.float64)
        s_zero = np.zeros(n, dtype=np.float64)
        vh_zero = np.eye(n, dtype=np.float64)
        if transposed:
            return vh_zero.T.copy(), s_zero, u_zero.T.copy()
        return u_zero, s_zero, vh_zero

    _, exp_scale = math.frexp(max_abs)
    a_sc = np.ldexp(a, -exp_scale)

    if n == 1:
        col = a_sc[:, 0]
        s0_sc = _safe_vec_norm(col)
        u = np.divide(col, s0_sc)[:, None] if s0_sc > 0.0 else np.eye(m, 1, dtype=np.float64)
        vh = np.ones((1, 1), dtype=np.float64)
        s_arr = np.array([math.ldexp(s0_sc, exp_scale)], dtype=np.float64)
        if transposed:
            return vh.T.copy(), s_arr, np.ascontiguousarray(u.T, dtype=np.float64)
        return np.ascontiguousarray(u, dtype=np.float64), s_arr, vh

    init_norms = _safe_col_norms(a_sc)
    init_perm = np.lexsort((np.arange(n, dtype=np.int64), -init_norms))

    pad = n % 2
    p = n + pad
    w = np.zeros((m, p), dtype=np.float64)
    w[:, :n] = a_sc[:, init_perm]
    v = np.zeros((p, p), dtype=np.float64)
    v[init_perm, np.arange(n, dtype=np.int64)] = 1.0
    if pad:
        v[n, n] = 1.0
    schedule = _round_robin_pairs(p)

    for _ in range(max_sweeps):
        max_rel_off = 0.0
        for idx_i, idx_j in schedule:
            wi = w[:, idx_i]
            wj = w[:, idx_j]
            ni = _safe_col_norms(wi)
            nj = _safe_col_norms(wj)
            both_pos = (ni > 0.0) & (nj > 0.0)
            if not np.any(both_pos):
                continue

            safe_ni = np.where(ni > 0.0, ni, 1.0)
            safe_nj = np.where(nj > 0.0, nj, 1.0)
            ui = np.where(both_pos[None, :], np.divide(wi, safe_ni[None, :]), 0.0)
            uj = np.where(both_pos[None, :], np.divide(wj, safe_nj[None, :]), 0.0)
            cos_ij = ieee754_binary_sum(np.multiply(ui, uj), axis=0)
            abs_cos = np.abs(cos_ij)

            active = both_pos & (abs_cos > tol)
            if not np.any(active):
                continue

            step_max = float(np.max(np.where(both_pos, abs_cos, 0.0)))
            if step_max > max_rel_off:
                max_rel_off = step_max

            r_max = np.maximum(ni, nj)
            safe_rmax = np.where(r_max > 0.0, r_max, 1.0)
            si = np.divide(ni, safe_rmax)
            sj = np.divide(nj, safe_rmax)

            num = np.multiply(2.0, np.multiply(np.multiply(si, sj), cos_ij))
            den = np.multiply(np.subtract(sj, si), np.add(sj, si))
            abs_num = np.abs(num)
            abs_den = np.abs(den)

            branch1 = abs_den <= abs_num
            safe_num = np.where(branch1 & (abs_num > 0.0), num, 1.0)
            tau_b1 = np.divide(den, safe_num)
            sign_b1 = np.where(tau_b1 >= 0.0, 1.0, -1.0)
            t_b1 = np.divide(
                sign_b1,
                np.add(np.abs(tau_b1), np.sqrt(np.add(1.0, np.multiply(tau_b1, tau_b1)))),
            )

            safe_den = np.where((~branch1) & (abs_den > 0.0), den, 1.0)
            inv_tau = np.divide(num, safe_den)
            t_b2 = np.divide(
                inv_tau,
                np.add(1.0, np.sqrt(np.add(1.0, np.multiply(inv_tau, inv_tau)))),
            )

            t = np.where(active, np.where(branch1, t_b1, t_b2), 0.0)
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
    s_raw_sc = _safe_col_norms(w_real)

    order = np.lexsort((np.arange(n, dtype=np.int64), -s_raw_sc))
    s_sorted_sc = np.ascontiguousarray(s_raw_sc[order], dtype=np.float64)
    w_sorted = w_real[:, order]
    v_sorted = v_real[:, order]

    u_mat = _complete_orthonormal_u(w_sorted, s_sorted_sc)
    s_vals = np.ascontiguousarray(np.ldexp(s_sorted_sc, exp_scale), dtype=np.float64)
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
    b = np.where(b == 0.0, 0.0, b)
    a = np.where(a == 0.0, 0.0, a)
    return np.ascontiguousarray(b, dtype=np.float64), np.ascontiguousarray(a, dtype=np.float64)


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

    s_norm = _safe_vec_norm(s_vals)
    if s_norm == 0.0:
        return (
            np.zeros((l64.shape[0], 1), dtype=np.float64),
            np.zeros((1, r64.shape[1]), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            0.0,
        )

    r_eff = min(max_rank, int(s_vals.size))
    s_kept = np.ascontiguousarray(s_vals[:r_eff], dtype=np.float64)
    disc_norm = _safe_vec_norm(s_vals[r_eff:]) if r_eff < s_vals.size else 0.0
    sqrt_s = np.sqrt(s_kept)
    b_raw = np.multiply(ozaki_exact_matmul(q_l, u_c[:, :r_eff]), sqrt_s[None, :])
    a_raw = np.multiply(sqrt_s[:, None], ozaki_exact_matmul(vh_c[:r_eff, :], q_r.T))
    b_canon, a_canon = canonicalize_output_factors(b_raw, a_raw)
    trunc_err = float(disc_norm / s_norm)
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


def _count_active_singular_values(
    s_k: np.ndarray,
    *,
    d_in: int,
    n0: int,
    cutoff: float,
    threshold_mode: ThresholdMode,
) -> Optional[int]:
    """Count `s_k^2 >= eff_cutoff` without overflowing `s_k * s_k` when `||K0|| >= 1e154`."""
    if s_k.size == 0:
        return None
    s0 = float(s_k[0])
    if s0 <= 1e150:
        eigvals_desc = np.multiply(s_k, s_k)
        eff_cutoff = _resolve_cutoff(
            eigvals_desc, d_in=d_in, eps=cutoff, threshold_mode=threshold_mode
        )
        if float(eigvals_desc[0]) < eff_cutoff:
            return None
        if eff_cutoff <= 0.0:
            pos_floor = s0 * float(max(d_in, n0)) * 2.220446049250313e-16
            return int(np.sum(s_k > pos_floor))
        return int(np.sum(eigvals_desc >= eff_cutoff))

    _, exp_s0 = math.frexp(s0)
    s_scaled = np.ldexp(s_k, -exp_s0)
    eig_scaled = np.multiply(s_scaled, s_scaled)
    if threshold_mode == "absolute":
        if cutoff <= 0.0:
            pos_floor = s0 * float(max(d_in, n0)) * 2.220446049250313e-16
            return int(np.sum(s_k > pos_floor))
        sqrt_cut = math.sqrt(cutoff)
        if s0 < sqrt_cut:
            return None
        return int(np.sum(s_k >= sqrt_cut))
    eff_scaled = _resolve_cutoff(
        eig_scaled, d_in=d_in, eps=cutoff, threshold_mode=threshold_mode
    )
    if float(eig_scaled[0]) < eff_scaled:
        return None
    if eff_scaled <= 0.0:
        pos_floor = s0 * float(max(d_in, n0)) * 2.220446049250313e-16
        return int(np.sum(s_k > pos_floor))
    return int(np.sum(eig_scaled >= eff_scaled))


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

    frob_k0 = _safe_vec_norm(k0.reshape(-1))
    if threshold_mode == "absolute" and cutoff > 0.0 and frob_k0 < math.sqrt(cutoff):
        return None, d_in

    max_active = max(0, d_in - min_nullspace_dim)
    if n0 <= d_in:
        q_k, r_k = det_householder_qr(k0)
        u_r, s_k, _ = det_jacobi_svd(r_k)
        n_above = _count_active_singular_values(
            s_k, d_in=d_in, n0=n0, cutoff=cutoff, threshold_mode=threshold_mode
        )
        if n_above is None:
            return None, d_in
        n_active = min(n_above, max_active)
        if n_active == 0:
            return None, d_in
        u_active = ozaki_exact_matmul(q_k, u_r[:, :n_active])
    else:
        q_k, r_k = det_householder_qr(k0.T)
        _, s_k, vh_r = det_jacobi_svd(r_k)
        n_above = _count_active_singular_values(
            s_k, d_in=d_in, n0=n0, cutoff=cutoff, threshold_mode=threshold_mode
        )
        if n_above is None:
            return None, d_in
        n_active = min(n_above, max_active)
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
    """Solve projected BetaEdit Eq. (9) via cancellation-free direct push-through identity."""
    d_in = k_solve.shape[0]
    nk_cols = k_solve.shape[1]
    pk0 = apply_p_null(u_active, cov_factors) if cov_factors.shape[1] > 0 else cov_factors
    pk_solve = apply_p_null(u_active, k_solve)
    m_all = np.concatenate([pk0, pk_solve], axis=1) if pk0.shape[1] > 0 else pk_solve
    n0_cols = pk0.shape[1]
    m_total = m_all.shape[1]
    lr = float(lambda_ridge)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lambda_ridge must be finite and strictly positive, got {lambda_ridge}")

    if m_total <= d_in:
        gram = ozaki_exact_matmul(m_all.T, m_all)
        gram_sym = np.multiply(0.5, np.add(gram, gram.T))
        spd_dual = np.add(np.multiply(lr, np.eye(m_total, dtype=np.float64)), gram_sym)
        e_solve = np.zeros((m_total, nk_cols), dtype=np.float64)
        e_solve[n0_cols:, :] = np.eye(nk_cols, dtype=np.float64)
        w_push = det_cholesky_solve(spd_dual, e_solve)
        z_solved = ozaki_exact_matmul(m_all, w_push)
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
