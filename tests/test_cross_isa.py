"""Tests for cross-ISA bit-exact deterministic BetaEdit linear algebra (`weft.cross_isa`)."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.cross_isa import (
    apply_p_null,
    canonicalize_output_factors,
    compile_betaedit_cross_isa,
    det_active_basis,
    det_cholesky_solve,
    det_householder_qr,
    det_jacobi_svd,
    ozaki_exact_matmul,
    solve_betaedit_with_basis,
)


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class TestCrossISABetaEdit(unittest.TestCase):
    def test_ozaki_exact_matmul_slice_identity_and_accuracy(self) -> None:
        rng = np.random.default_rng(20261119)
        a = rng.standard_normal((37, 519)) * 12.5
        b = rng.standard_normal((519, 29)) * 0.37
        c_full = ozaki_exact_matmul(a, b, k_block=8192)
        c_ref = a @ b
        rel_err = float(np.linalg.norm(c_full - c_ref) / np.linalg.norm(c_ref))
        self.assertLess(rel_err, 1e-14)
        self.assertTrue(np.array_equal(c_full[:11, :13], ozaki_exact_matmul(a[:11, :], b[:, :13])))

    def test_householder_qr_and_jacobi_svd_reconstruction(self) -> None:
        rng = np.random.default_rng(7)
        m = rng.standard_normal((83, 27))
        q, r = det_householder_qr(m, panel_size=8)
        self.assertTrue(bool(np.all(np.diag(r) >= 0.0)))
        self.assertLess(float(np.max(np.abs((q.T @ q) - np.eye(27)))), 1e-13)
        self.assertLess(float(np.linalg.norm((q @ r) - m) / np.linalg.norm(m)), 1e-13)

        u, s, vh = det_jacobi_svd(r)
        self.assertTrue(bool(np.all(np.diff(s) <= 0.0)))
        self.assertLess(float(np.linalg.norm(((u * s[None, :]) @ vh) - r) / np.linalg.norm(r)), 1e-13)

    def test_output_space_sign_canonicalization_fixes_reflector_flip(self) -> None:
        rng = np.random.default_rng(42)
        q_l = np.linalg.qr(rng.standard_normal((64, 6)), mode="reduced")[0]
        q_r = np.linalg.qr(rng.standard_normal((48, 6)), mode="reduced")[0]
        u_c = np.eye(6, dtype=np.float64)
        vh_c = np.eye(6, dtype=np.float64)
        sqrt_s = np.sqrt(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float64))

        q_l_flip = q_l.copy()
        q_l_flip[:, 0] *= -1.0
        u_c_flip = u_c.copy()
        u_c_flip[0, :] *= -1.0

        b1, a1 = canonicalize_output_factors(
            (q_l @ u_c) * sqrt_s[None, :], sqrt_s[:, None] * (vh_c @ q_r.T)
        )
        b2, a2 = canonicalize_output_factors(
            (q_l_flip @ u_c_flip) * sqrt_s[None, :], sqrt_s[:, None] * (vh_c @ q_r.T)
        )
        self.assertEqual(_sha(b1), _sha(b2))
        self.assertEqual(_sha(a1), _sha(a2))

        b_neg0, _ = canonicalize_output_factors(
            np.array([[-1.0, 0.0], [1.0, -0.0]], dtype=np.float64),
            np.eye(2, dtype=np.float64),
        )
        self.assertFalse(bool(np.signbit(b_neg0[1, 1])))

    def test_projector_cross_isa_mode_matches_dense_and_rederives(self) -> None:
        rng = np.random.default_rng(101)
        d_in, d_out = 48, 36
        w0 = rng.standard_normal((d_out, d_in)) * 0.2
        k0 = rng.standard_normal((d_in, 16)) * 0.3
        keys = rng.standard_normal((d_in, 6)) * 0.5
        targets = rng.standard_normal((d_out, 6))

        for thresh, mode in ((1e9, "absolute"), (5e-2, "relative_max"), (5e-2, "relative_mean")):
            cfg_det = BetaEditConfig(
                d_in=d_in,
                d_out=d_out,
                max_rank=6,
                lambda_leakage=80.0,
                lambda_ridge=3.0,
                nullspace_threshold=thresh,
                threshold_mode=mode,
                quant_dtype="f32",
                cross_isa_deterministic=True,
            )
            cfg_leg = BetaEditConfig(
                d_in=d_in,
                d_out=d_out,
                max_rank=6,
                lambda_leakage=80.0,
                lambda_ridge=3.0,
                nullspace_threshold=thresh,
                threshold_mode=mode,
                quant_dtype="f32",
                cross_isa_deterministic=False,
            )
            p_det = BetaEditNullspaceProjector(w0, cfg_det)
            p_leg = BetaEditNullspaceProjector(w0, cfg_leg)
            self.assertEqual(
                p_det.initialize_anchor_covariance(k0),
                p_leg.initialize_anchor_covariance(k0),
            )
            p_det.register_safety_direction(targets[:, :1], keys[:, :1].T)
            p_leg.register_safety_direction(targets[:, :1], keys[:, :1].T)

            patch_det = p_det.stage_edit(keys, targets, is_residual=False, rank=6, slot_name="det")
            patch_leg = p_leg.stage_edit(keys, targets, is_residual=False, rank=6, slot_name="det")

            op_det = patch_det.factor_b @ patch_det.factor_a
            op_leg = patch_leg.factor_b @ patch_leg.factor_a
            self.assertLess(float(np.linalg.norm(op_det - op_leg) / np.linalg.norm(op_leg)), 1e-6)

            ok, info = verify_rederivation(
                patch_det,
                base_weight=w0,
                config=cfg_det,
                k0=k0,
                keys=keys,
                target_values_or_residuals=targets,
                is_residual=False,
                rank=6,
                safety_directions=[(targets[:, :1], keys[:, :1].T)],
            )
            self.assertTrue(ok, msg=str(info))

    def test_woodbury_vs_primal_and_must_fail_controls(self) -> None:
        rng = np.random.default_rng(99)
        d_in, d_out = 40, 56
        k0_small = rng.standard_normal((d_in, 12)) * 0.2
        keys = rng.standard_normal((d_in, 8)) * 0.5
        res = rng.standard_normal((d_out, 8))

        b_d, a_d, _, _, _, nd_d = compile_betaedit_cross_isa(
            k0=k0_small,
            keys=keys,
            residuals=res,
            rank=4,
            lambda_leakage=50.0,
            lambda_ridge=2.0,
            nullspace_threshold=0.5,
        )
        k0_dup = np.concatenate([k0_small] * 4, axis=1) / 2.0
        b_p, a_p, _, _, _, nd_p = compile_betaedit_cross_isa(
            k0=k0_dup,
            keys=keys,
            residuals=res,
            rank=4,
            lambda_leakage=50.0,
            lambda_ridge=2.0,
            nullspace_threshold=0.5,
        )
        self.assertEqual(nd_p, nd_d)
        self.assertLess(float(np.linalg.norm((b_d @ a_d) - (b_p @ a_p)) / np.linalg.norm(b_d @ a_d)), 1e-12)

        bad_spd = np.eye(8, dtype=np.float64)
        bad_spd[3, 3] = -0.5
        with self.assertRaises(np.linalg.LinAlgError):
            det_cholesky_solve(bad_spd, np.ones((8, 2), dtype=np.float64), block_size=4)

        with self.assertRaises(ValueError):
            ozaki_exact_matmul(np.zeros((4, 5)), np.zeros((6, 3)))

    def test_neon_a1_direct_pushthrough_avoids_cancellation(self) -> None:
        rng = np.random.default_rng(42)
        q_m, _ = np.linalg.qr(rng.standard_normal((32, 12)))
        r_solve = rng.standard_normal((8, 4))
        m_all = q_m * 1e8
        lr = 1e-4
        k0 = m_all[:, :8]
        k_solve = m_all[:, 8:]
        delta_exact = r_solve @ (k_solve / (1e16 + lr)).T
        b_ci, a_ci, _, _ = solve_betaedit_with_basis(
            u_active=None,
            cov_factors=k0,
            k_solve=k_solve,
            r_solve=r_solve,
            rank=4,
            lambda_ridge=lr,
        )
        rel_err = float(np.linalg.norm((b_ci @ a_ci) - delta_exact) / np.linalg.norm(delta_exact))
        self.assertLess(rel_err, 1e-13)

    def test_neon_a2_rank_deficient_orthonormal_u_and_active_basis(self) -> None:
        for mat in (
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float64),
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64),
        ):
            u, s, vh = det_jacobi_svd(mat)
            self.assertLess(float(np.linalg.norm((u.T @ u) - np.eye(u.shape[1]))), 1e-14)
            self.assertLess(float(np.linalg.norm((vh @ vh.T) - np.eye(vh.shape[0]))), 1e-14)
            self.assertLess(float(np.linalg.norm((u @ np.diag(s) @ vh) - mat)), 1e-14)

        k0_rank1 = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        u_act, null_dim = det_active_basis(
            k0_rank1, d_in=3, cutoff=0.0, threshold_mode="absolute", min_nullspace_dim=1
        )
        self.assertIsNotNone(u_act)
        if u_act is None:
            raise AssertionError("u_act must not be None")
        self.assertEqual(null_dim, 2)
        self.assertLess(float(np.linalg.norm((u_act.T @ u_act) - np.eye(u_act.shape[1]))), 1e-14)
        p_null = apply_p_null(u_act, np.eye(3, dtype=np.float64))
        self.assertLess(float(np.linalg.norm((p_null @ p_null) - p_null)), 1e-14)

    def test_neon_a3_jacobi_scale_invariance_and_steep_convergence(self) -> None:
        base = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        for scale in (1e80, 1e-82):
            u, s, _ = det_jacobi_svd(base * scale)
            self.assertLess(float(np.linalg.norm((u.T @ u) - np.eye(2))), 1e-14)
            self.assertGreater(float(s[0]), 0.0)

        u_tau, _, vh_tau = det_jacobi_svd(np.array([[1e73, 1e-86], [0.0, 1e-73]], dtype=np.float64))
        self.assertLess(float(np.linalg.norm((u_tau.T @ u_tau) - np.eye(2))), 1e-14)
        self.assertLess(float(np.linalg.norm((vh_tau @ vh_tau.T) - np.eye(2))), 1e-14)

        rng = np.random.default_rng(7)
        q1, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        q2, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        sig = np.geomspace(1.0, 1e-8, 64)
        sig[1] = sig[0] * (1.0 - 1e-12)
        sig[3] = sig[2]
        mat_c = q1 @ np.diag(sig) @ q2.T
        u64, s64, vh64 = det_jacobi_svd(mat_c)
        self.assertLess(float(np.linalg.norm((u64.T @ u64) - np.eye(64))), 1e-13)
        self.assertLess(float(np.linalg.norm((u64 @ np.diag(s64) @ vh64) - mat_c) / np.linalg.norm(mat_c)), 1e-13)

    def test_neon_a4_householder_qr_tiny_tail_and_subscale(self) -> None:
        m_sub = np.array([[1.0, 0.0], [2.2e-162, 1.0]], dtype=np.float64)
        q_s, r_s = det_householder_qr(m_sub)
        self.assertTrue(bool(np.all(np.isfinite(q_s)) and np.all(np.isfinite(r_s))))
        self.assertLess(float(np.linalg.norm((q_s @ r_s) - m_sub)), 1e-15)

        m_tiny = np.array([[1e-170, 0.0], [1e-170, 1e-170]], dtype=np.float64)
        q_t, r_t = det_householder_qr(m_tiny)
        self.assertLess(float(np.linalg.norm(((q_t @ r_t) - m_tiny) * 1e170)), 1e-15)

    def test_neon_a5_ozaki_disparate_scales_and_guards(self) -> None:
        a_ext = np.array([[1e150, 1e-150]], dtype=np.float64)
        b_ext = np.array([[1e-150], [1e150]], dtype=np.float64)
        self.assertEqual(float(ozaki_exact_matmul(a_ext, b_ext)[0, 0]), 2.0)

        for big in (1e160, 1e165, 1e300):
            inv = 1.0 / big
            a_span = np.array([[big, inv]], dtype=np.float64)
            b_span = np.array([[inv], [big]], dtype=np.float64)
            self.assertEqual(float(ozaki_exact_matmul(a_span, b_span)[0, 0]), 2.0)

        min_sub = np.nextafter(0.0, 1.0)
        a_sub = np.array([[1e100, min_sub]], dtype=np.float64)
        b_sub = np.array([[0.0], [1e300]], dtype=np.float64)
        self.assertEqual(float(ozaki_exact_matmul(a_sub, b_sub)[0, 0]), float(min_sub * 1e300))

        for mode in ("absolute", "relative_max", "relative_mean"):
            u_huge, nd_huge = det_active_basis(
                np.array([[1e200, 0.0], [0.0, 1e200]], dtype=np.float64),
                d_in=2,
                cutoff=0.5,
                threshold_mode=mode,
                min_nullspace_dim=1,
            )
            self.assertIsNotNone(u_huge)
            self.assertEqual(nd_huge, 1)

        s = 2.0 ** -36
        a_mild = np.array([[1.0, s, 0.0]], dtype=np.float64)
        b_mild = np.array([[0.0], [s], [1.0]], dtype=np.float64)
        self.assertEqual(float(ozaki_exact_matmul(a_mild, b_mild)[0, 0]), 2.0 ** -72)

        with self.assertRaises(ValueError):
            ozaki_exact_matmul(np.array([[np.inf, 1.0]]), np.array([[1.0], [1.0]]))
        with self.assertRaises(ValueError):
            ozaki_exact_matmul(np.ones((1, 4)), np.ones((4, 1)), k_block=32768)


if __name__ == "__main__":
    unittest.main()
