"""Tests for cross-ISA bit-exact deterministic BetaEdit linear algebra (`weft.cross_isa`)."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.cross_isa import (
    canonicalize_output_factors,
    compile_betaedit_cross_isa,
    det_cholesky_solve,
    det_householder_qr,
    det_jacobi_svd,
    ozaki_exact_matmul,
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


if __name__ == "__main__":
    unittest.main()
