"""Deterministic CPU float64 BetaEdit null-space compiler and SGMV slab arena.

Implements history-aware null-space projection (BetaEdit) with exact m x m
block-triangular SPD Cholesky reduction, matrix-free skinny-QR Eckart-Young
rank-r factorization, canonical BF16/F16 quantization, frozen-base additive
multi-slot composition, O(1) rollback, and 0-byte re-derivation verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from .certificate import FixedRadixCertificate, GatePolicy, build_certificate
from .cross_isa import (
    apply_p_null,
    det_active_basis,
    det_build_certificate_values,
    ieee754_binary_sum,
    ozaki_exact_matmul,
    solve_betaedit_with_basis,
)
from .quant import SUPPORTED_QUANT_DTYPES, quantize_array, sha256_factor_pair

ThresholdMode = Literal["absolute", "relative_max", "relative_mean"]
QuantDtype = Literal["bf16", "f16", "f32", "int8", "int4"]


@dataclass(frozen=True)
class BetaEditConfig:
    d_in: int
    d_out: int
    max_rank: int = 16
    max_slots: int = 64
    lambda_leakage: float = 1000.0
    lambda_ridge: float = 10.0
    nullspace_threshold: float = 2e-2
    threshold_mode: ThresholdMode = "absolute"
    min_nullspace_dim: int = 1
    refresh_period: int = 100
    radix_scale_exp2: int = 24
    quant_dtype: QuantDtype = "bf16"
    b_quant_dtype: Optional[QuantDtype] = None
    a_quant_dtype: Optional[QuantDtype] = None
    # Reproducibility contract:
    # - True: every float64 product goes through `ozaki_exact_matmul` (exact integer limbs),
    #   so raw factors, f32/f16 bytes, digests and certificates are bit-identical across BLAS
    #   kernels and thread counts (pinned by tests/test_cross_isa_golden.py). Cost is a
    #   Python-level O(d^3) with a large constant (roughly 13x legacy at d=512).
    # - False (legacy): uses BLAS `@`/LAPACK, so float64 results and f32 bytes vary with the
    #   BLAS kernel and thread count. Only reproducible with pinned BLAS and threads
    #   (e.g. OPENBLAS_NUM_THREADS=1) on a single host.
    cross_isa_deterministic: bool = False

    @property
    def effective_b_dtype(self) -> QuantDtype:
        return self.quant_dtype if self.b_quant_dtype is None else self.b_quant_dtype

    @property
    def effective_a_dtype(self) -> QuantDtype:
        return self.quant_dtype if self.a_quant_dtype is None else self.a_quant_dtype

    @property
    def quant_label(self) -> str:
        eb, ea = self.effective_b_dtype, self.effective_a_dtype
        return eb if eb == ea else f"{eb}/{ea}"

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.d_out <= 0 or self.max_rank <= 0 or self.max_slots <= 0:
            raise ValueError("Dimensions, max_rank, and max_slots must be positive.")
        if self.lambda_leakage < 0.0 or self.lambda_ridge <= 0.0 or self.nullspace_threshold <= 0.0:
            raise ValueError("Invalid regularization or threshold parameters.")
        if not (1 <= self.min_nullspace_dim <= self.d_in):
            raise ValueError("min_nullspace_dim must lie in [1, d_in].")
        if self.refresh_period <= 0 or not (8 <= self.radix_scale_exp2 <= 48):
            raise ValueError("Invalid refresh_period or radix_scale_exp2.")
        for dt in (self.quant_dtype, self.effective_b_dtype, self.effective_a_dtype):
            if dt not in SUPPORTED_QUANT_DTYPES:
                raise ValueError(f"Unsupported quant_dtype: {dt!r}")


@dataclass(frozen=True)
class StagedPatch:
    """Post-quantized rank-r adapter patch paired with its exact integer certificate."""

    factor_b: np.ndarray  # float64 exact view of quantized B_q, shape (d_out, r_eff)
    factor_a: np.ndarray  # float64 exact view of quantized A_q, shape (r_eff, d_in)
    b_bytes: bytes
    a_bytes: bytes
    singular_values: np.ndarray
    effective_rank: int
    truncation_rel_error: float
    factor_digest: str
    certificate: FixedRadixCertificate
    slot_name: str = ""


class BetaEditNullspaceProjector:
    """History-aware null-space compiler backed by an immutable multi-slot slab arena."""

    def __init__(self, base_weight: np.ndarray, config: BetaEditConfig) -> None:
        w0 = np.ascontiguousarray(base_weight, dtype=np.float64)
        if w0.ndim != 2 or w0.shape != (config.d_out, config.d_in):
            raise ValueError(
                f"Expected base_weight shape ({config.d_out}, {config.d_in}), got {w0.shape}."
            )
        if not np.all(np.isfinite(w0)):
            raise ValueError("base_weight must be finite.")
        self.config = config
        self._w0 = w0.copy()
        self._w0.setflags(write=False)

        self._cov_accum = np.zeros((config.d_in, config.d_in), dtype=np.float64)
        self._u_small = np.eye(config.d_in, dtype=np.float64)
        self._cached_u_cov_u = np.zeros((config.d_in, config.d_in), dtype=np.float64)

        self._arena_b = np.zeros((config.max_slots, config.d_out, config.max_rank), dtype=np.float64)
        self._arena_a = np.zeros((config.max_slots, config.max_rank, config.d_in), dtype=np.float64)
        self._slot_active = np.zeros(config.max_slots, dtype=bool)
        self._slot_ranks = np.zeros(config.max_slots, dtype=np.int32)
        self._slot_generation = np.zeros(config.max_slots, dtype=np.int64)
        self._slot_names: dict[int, str] = {}
        self._slot_key_grams: dict[int, np.ndarray] = {}
        self._slot_keys_det: dict[int, np.ndarray] = {}
        self._det_k0_scaled = np.zeros((config.d_in, 0), dtype=np.float64)
        self._det_u_active: Optional[np.ndarray] = None
        self._det_nullspace_dim: int = config.d_in
        self._safety_directions: list[tuple[np.ndarray, np.ndarray]] = []

        self._projector_epoch = 0
        self._commits_since_refresh = 0
        self._total_commits = 0

    @property
    def w0(self) -> np.ndarray:
        return self._w0

    @property
    def cov_accum(self) -> np.ndarray:
        return self._cov_accum

    @property
    def u_small(self) -> np.ndarray:
        return self._u_small

    @property
    def nullspace_dim(self) -> int:
        if self.config.cross_isa_deterministic:
            return int(self._det_nullspace_dim)
        return int(self._u_small.shape[1])

    @property
    def projector_epoch(self) -> int:
        return self._projector_epoch

    @property
    def active_slot_count(self) -> int:
        return int(np.sum(self._slot_active))

    @property
    def projection_matrix(self) -> np.ndarray:
        if self.config.cross_isa_deterministic:
            eye = np.eye(self.config.d_in, dtype=np.float64)
            return apply_p_null(self._det_u_active, eye)
        return self._u_small @ self._u_small.T

    def _det_cov_factors(self) -> np.ndarray:
        parts = [self._det_k0_scaled] + [self._slot_keys_det[k] for k in sorted(self._slot_keys_det)]
        non_empty = [p for p in parts if p.shape[1] > 0]
        if not non_empty:
            return np.zeros((self.config.d_in, 0), dtype=np.float64)
        return np.concatenate(non_empty, axis=1)

    def initialize_anchor_covariance(
        self, k0_or_gram: np.ndarray, *, is_gram: bool = False
    ) -> int:
        """Initialize base preserved-knowledge covariance C^(0) = lambda_1 * K0 @ K0.T."""
        mat = np.ascontiguousarray(k0_or_gram, dtype=np.float64)
        if not np.all(np.isfinite(mat)):
            raise ValueError("Anchor keys or Gram matrix must be finite.")
        if self.config.cross_isa_deterministic:
            if is_gram:
                raise ValueError("cross_isa_deterministic requires anchor key matrix (is_gram=False).")
            if mat.ndim != 2 or mat.shape[0] != self.config.d_in:
                raise ValueError(f"Expected ({self.config.d_in}, n_0) anchor key matrix.")
            self._det_k0_scaled = np.multiply(math.sqrt(self.config.lambda_leakage), mat)
            return self.refresh_projector()

        if is_gram:
            if mat.shape != (self.config.d_in, self.config.d_in):
                raise ValueError(f"Expected ({self.config.d_in}, {self.config.d_in}) Gram matrix.")
            gram = 0.5 * (mat + mat.T)
        else:
            if mat.ndim != 2 or mat.shape[0] != self.config.d_in:
                raise ValueError(f"Expected ({self.config.d_in}, n_0) anchor key matrix.")
            gram = mat @ mat.T

        self._cov_accum = self.config.lambda_leakage * gram
        for slot_gram in self._slot_key_grams.values():
            self._cov_accum = self._cov_accum + slot_gram
        return self.refresh_projector()

    def register_safety_direction(self, b_safe: np.ndarray, a_safe: np.ndarray) -> None:
        """Register a unit-normalized protected safety subspace factor pair (B_s, A_s)."""
        b64 = np.ascontiguousarray(b_safe, dtype=np.float64)
        a64 = np.ascontiguousarray(a_safe, dtype=np.float64)
        if (
            b64.ndim != 2
            or a64.ndim != 2
            or b64.shape[0] != self.config.d_out
            or a64.shape[1] != self.config.d_in
            or b64.shape[1] != a64.shape[0]
        ):
            raise ValueError("Incompatible safety direction factor shapes.")
        if self.config.cross_isa_deterministic:
            bt_b = ozaki_exact_matmul(b64.T, b64)
            aa_t = ozaki_exact_matmul(a64, a64.T)
            norm_sq = float(ieee754_binary_sum(np.multiply(bt_b, aa_t).reshape(-1)))
            if norm_sq <= 0.0:
                raise ValueError("Safety direction must have positive norm.")
            scale = 1.0 / math.sqrt(math.sqrt(norm_sq))
            self._safety_directions.append((np.multiply(b64, scale), np.multiply(a64, scale)))
            return
        norm_sq = float(np.sum((b64.T @ b64) * (a64 @ a64.T)))
        if norm_sq <= 0.0:
            raise ValueError("Safety direction must have positive norm.")
        scale = norm_sq ** -0.25
        self._safety_directions.append((b64 * scale, a64 * scale))

    def refresh_projector(self) -> int:
        """Recompute pseudo-nullspace basis U_small from symmetric C^(t)."""
        if self.config.cross_isa_deterministic:
            cov_f = self._det_cov_factors()
            self._det_u_active, self._det_nullspace_dim = det_active_basis(
                cov_f,
                d_in=self.config.d_in,
                cutoff=self.config.nullspace_threshold,
                threshold_mode=self.config.threshold_mode,
                min_nullspace_dim=self.config.min_nullspace_dim,
            )
            self._projector_epoch += 1
            self._commits_since_refresh = 0
            return int(self._det_nullspace_dim)

        sym = 0.5 * (self._cov_accum + self._cov_accum.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, 0.0)

        eps = self.config.nullspace_threshold
        if self.config.threshold_mode == "relative_max":
            m_ev = float(eigvals[-1])
            cutoff = eps * m_ev if m_ev > 0.0 else eps
        elif self.config.threshold_mode == "relative_mean":
            m_ev = float(np.mean(eigvals))
            cutoff = eps * m_ev if m_ev > 0.0 else eps
        else:
            cutoff = eps

        mask = eigvals < cutoff
        if int(np.sum(mask)) < self.config.min_nullspace_dim:
            u_sel = np.ascontiguousarray(eigvecs[:, : self.config.min_nullspace_dim])
            ev_sel = eigvals[: self.config.min_nullspace_dim]
        else:
            u_sel = np.ascontiguousarray(eigvecs[:, mask])
            ev_sel = eigvals[mask]

        # Canonicalize eigenvector signs (largest-magnitude entry positive) for
        # deterministic cross-platform CPU replay.
        for col in range(u_sel.shape[1]):
            idx = int(np.argmax(np.abs(u_sel[:, col])))
            if u_sel[idx, col] < 0.0:
                u_sel[:, col] = -u_sel[:, col]

        self._u_small = u_sel
        self._cached_u_cov_u = np.diag(ev_sel)
        self._projector_epoch += 1
        self._commits_since_refresh = 0
        return int(u_sel.shape[1])

    @staticmethod
    def decompose_rank_r(
        left: np.ndarray,
        right: np.ndarray,
        max_rank: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Matrix-free Eckart-Young rank-r decomposition of left @ right via skinny QR."""
        l64 = np.ascontiguousarray(left, dtype=np.float64)
        r64 = np.ascontiguousarray(right, dtype=np.float64)
        if l64.ndim != 2 or r64.ndim != 2 or l64.shape[1] != r64.shape[0]:
            raise ValueError("Incompatible left/right shapes for rank-r decomposition.")

        q_l, r_l = np.linalg.qr(l64, mode="reduced")
        q_r, r_r = np.linalg.qr(r64.T, mode="reduced")
        u_c, s_vals, vh_c = np.linalg.svd(r_l @ r_r.T, full_matrices=False)

        # Canonicalize singular vector signs so B and A bytes are deterministic.
        for k in range(u_c.shape[1]):
            idx = int(np.argmax(np.abs(u_c[:, k])))
            if u_c[idx, k] < 0.0:
                u_c[:, k] = -u_c[:, k]
                vh_c[k, :] = -vh_c[k, :]

        tot = float(np.sum(s_vals * s_vals))
        if tot == 0.0:
            return (
                np.zeros((l64.shape[0], 1), dtype=np.float64),
                np.zeros((1, r64.shape[1]), dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                0.0,
            )

        r_eff = min(max_rank, int(s_vals.size))
        s_kept = s_vals[:r_eff]
        disc = float(np.sum(s_vals[r_eff:] * s_vals[r_eff:]))
        sqrt_s = np.sqrt(s_kept)
        b_mat = (q_l @ u_c[:, :r_eff]) * sqrt_s[np.newaxis, :]
        a_mat = sqrt_s[:, np.newaxis] * (vh_c[:r_eff, :] @ q_r.T)
        return (
            np.ascontiguousarray(b_mat),
            np.ascontiguousarray(a_mat),
            np.ascontiguousarray(s_kept),
            math.sqrt(max(0.0, disc) / tot),
        )

    def stage_edit(
        self,
        keys: np.ndarray,
        target_values_or_residuals: np.ndarray,
        *,
        is_residual: bool = False,
        rank: Optional[int] = None,
        preservation_keys: Optional[np.ndarray] = None,
        preservation_weight: float = 1.0,
        slot_name: str = "",
    ) -> StagedPatch:
        """Solve BetaEdit Eq. (9) via exact m x m SPD Cholesky inside U_small.

        When preservation_keys is provided (register-scoped y=0 preservation),
        those off-register keys are appended with zero residual target so the
        compiled patch drives zero logit shift on off-register states.
        """
        k64 = np.ascontiguousarray(keys, dtype=np.float64)
        v64 = np.ascontiguousarray(target_values_or_residuals, dtype=np.float64)
        if k64.ndim != 2 or k64.shape[0] != self.config.d_in:
            raise ValueError(f"Expected keys shape ({self.config.d_in}, u), got {k64.shape}.")
        if v64.ndim != 2 or v64.shape != (self.config.d_out, k64.shape[1]):
            raise ValueError(
                f"Expected target shape ({self.config.d_out}, {k64.shape[1]}), got {v64.shape}."
            )

        if is_residual:
            r64 = v64
        else:
            r64 = v64 - self.effective_matvec(k64)

        if preservation_keys is not None and preservation_keys.size > 0:
            pk = np.ascontiguousarray(preservation_keys, dtype=np.float64)
            if pk.ndim != 2 or pk.shape[0] != self.config.d_in:
                raise ValueError(
                    f"Expected preservation_keys shape ({self.config.d_in}, n_p), got {pk.shape}."
                )
            w_sqrt = math.sqrt(max(0.0, float(preservation_weight)))
            k_solve = np.concatenate([k64, pk * w_sqrt], axis=1)
            r_solve = np.concatenate(
                [r64, np.zeros((self.config.d_out, pk.shape[1]), dtype=np.float64)], axis=1
            )
        else:
            k_solve = k64
            r_solve = r64

        rank_cap = self.config.max_rank if rank is None else int(rank)
        if not (1 <= rank_cap <= self.config.max_rank):
            raise ValueError(f"rank must lie in [1, {self.config.max_rank}].")

        eb = self.config.effective_b_dtype
        ea = self.config.effective_a_dtype

        if self.config.cross_isa_deterministic:
            cov_f = self._det_cov_factors()
            b64, a64, s_vals, trunc_err = solve_betaedit_with_basis(
                u_active=self._det_u_active,
                cov_factors=cov_f,
                k_solve=k_solve,
                r_solve=r_solve,
                rank=rank_cap,
                lambda_ridge=self.config.lambda_ridge,
            )
            b_bytes, b_q = quantize_array(b64, eb, axis=0)
            a_bytes, a_q = quantize_array(a64, ea, axis=1)
            digest = sha256_factor_pair(
                b_bytes,
                a_bytes,
                (int(b_q.shape[0]), int(b_q.shape[1])),
                (int(a_q.shape[0]), int(a_q.shape[1])),
                self.config.quant_dtype,
                b_dtype=eb,
                a_dtype=ea,
            )
            fi, si, li, sfi = det_build_certificate_values(
                b_q,
                a_q,
                cov_f,
                radix_scale_exp2=self.config.radix_scale_exp2,
                safety_directions=self._safety_directions,
            )
            cert = FixedRadixCertificate(
                radix_scale_exp2=self.config.radix_scale_exp2,
                frobenius_sq_scaled=fi,
                spectral_norm_sq_scaled=si,
                leakage_energy_scaled=li,
                max_safety_dot_sq_scaled=sfi,
                projector_epoch=self._projector_epoch,
                factor_digest=digest,
                quant_dtype=self.config.quant_label,
            )
            return StagedPatch(
                factor_b=b_q,
                factor_a=a_q,
                b_bytes=b_bytes,
                a_bytes=a_bytes,
                singular_values=s_vals,
                effective_rank=int(b_q.shape[1]),
                truncation_rel_error=trunc_err,
                factor_digest=digest,
                certificate=cert,
                slot_name=slot_name,
            )

        # Exact block-triangular SPD subspace reduction:
        # Solve (lambda_2 I_m + U^T C^(t) U + U^T K K^T U) X = U^T K
        k_proj = self._u_small.T @ k_solve
        m_dim = self._u_small.shape[1]
        spd = (
            self.config.lambda_ridge * np.eye(m_dim, dtype=np.float64)
            + self._cached_u_cov_u
            + (k_proj @ k_proj.T)
        )
        spd = 0.5 * (spd + spd.T)
        try:
            chol = np.linalg.cholesky(spd)
            y_tmp = np.linalg.solve(chol, k_proj)
            solved = np.linalg.solve(chol.T, y_tmp)
        except np.linalg.LinAlgError:
            solved = np.linalg.solve(spd, k_proj)

        right64 = (self._u_small @ solved).T  # (u_total, d_in)
        b64, a64, s_vals, trunc_err = self.decompose_rank_r(r_solve, right64, rank_cap)

        # Per-rank-vector integer scales: axis=0 on B (d_out x r -> r scales),
        # axis=1 on A (r x d_in -> r scales) matches SVD factor geometry without d_out scale bloat.
        b_bytes, b_q = quantize_array(b64, eb, axis=0)
        a_bytes, a_q = quantize_array(a64, ea, axis=1)
        digest = sha256_factor_pair(
            b_bytes,
            a_bytes,
            (int(b_q.shape[0]), int(b_q.shape[1])),
            (int(a_q.shape[0]), int(a_q.shape[1])),
            self.config.quant_dtype,
            b_dtype=eb,
            a_dtype=ea,
        )
        cert = build_certificate(
            b_q,
            a_q,
            self._cov_accum,
            factor_digest=digest,
            quant_dtype=self.config.quant_label,
            projector_epoch=self._projector_epoch,
            radix_scale_exp2=self.config.radix_scale_exp2,
            safety_directions=self._safety_directions,
        )
        return StagedPatch(
            factor_b=b_q,
            factor_a=a_q,
            b_bytes=b_bytes,
            a_bytes=a_bytes,
            singular_values=s_vals,
            effective_rank=int(b_q.shape[1]),
            truncation_rel_error=trunc_err,
            factor_digest=digest,
            certificate=cert,
            slot_name=slot_name,
        )

    def commit_patch(
        self,
        patch: StagedPatch,
        keys: np.ndarray,
        *,
        gate_policy: Optional[GatePolicy] = None,
        slot_id: Optional[int] = None,
    ) -> int:
        """Verify integer certificate (if provided) and mount patch into an arena slot."""
        if gate_policy is not None:
            ok, problems = patch.certificate.verify_division_free(gate_policy)
            if not ok:
                raise PermissionError(f"TCB gate rejected patch certificate: {problems}")

        r_eff = patch.effective_rank
        if r_eff > self.config.max_rank:
            raise ValueError("Patch rank exceeds arena max_rank.")

        if slot_id is None:
            inactive = np.flatnonzero(~self._slot_active)
            if inactive.size == 0:
                raise RuntimeError("All SGMV arena slots are occupied.")
            target = int(inactive[0])
        else:
            if not (0 <= slot_id < self.config.max_slots):
                raise IndexError("slot_id out of range.")
            if bool(self._slot_active[slot_id]):
                self.rollback_slot(slot_id, revert_covariance=True)
            target = int(slot_id)

        self._arena_b[target].fill(0.0)
        self._arena_a[target].fill(0.0)
        self._arena_b[target, :, :r_eff] = patch.factor_b
        self._arena_a[target, :r_eff, :] = patch.factor_a
        self._slot_ranks[target] = r_eff
        self._slot_generation[target] += 1
        self._slot_active[target] = True
        self._slot_names[target] = patch.slot_name or f"slot_{target}"

        k64 = np.ascontiguousarray(keys, dtype=np.float64)
        if self.config.cross_isa_deterministic:
            self._slot_keys_det[target] = k64
            self._commits_since_refresh += 1
            self._total_commits += 1
            if self._commits_since_refresh >= self.config.refresh_period:
                self.refresh_projector()
            return target
        gram = k64 @ k64.T
        self._cov_accum = self._cov_accum + gram
        k_proj = self._u_small.T @ k64
        self._cached_u_cov_u = self._cached_u_cov_u + (k_proj @ k_proj.T)
        self._slot_key_grams[target] = gram

        self._commits_since_refresh += 1
        self._total_commits += 1
        if self._commits_since_refresh >= self.config.refresh_period:
            self.refresh_projector()
        return target

    def rollback_slot(self, slot_id: int, *, revert_covariance: bool = True) -> None:
        """O(1) slot kill switch restoring frozen base weights and optionally covariance."""
        if not (0 <= slot_id < self.config.max_slots):
            raise IndexError("slot_id out of range.")
        self._slot_active[slot_id] = False
        self._slot_ranks[slot_id] = 0
        self._arena_b[slot_id].fill(0.0)
        self._arena_a[slot_id].fill(0.0)
        self._slot_generation[slot_id] += 1
        self._slot_names.pop(slot_id, None)
        if revert_covariance:
            self._slot_keys_det.pop(slot_id, None)
        gram = self._slot_key_grams.pop(slot_id, None)
        if revert_covariance and gram is not None:
            self._cov_accum = self._cov_accum - gram
            diag = np.diag(self._cov_accum).copy()
            np.fill_diagonal(self._cov_accum, np.maximum(diag, 0.0))
            self._cached_u_cov_u = self._u_small.T @ self._cov_accum @ self._u_small

    def effective_matvec(
        self,
        keys: np.ndarray,
        *,
        slot_scales: Optional[dict[int, float]] = None,
    ) -> np.ndarray:
        """Compute (W0 + sum_i scale_i * B_i A_i) @ keys for columns of shape (d_in, u)."""
        k64 = np.ascontiguousarray(keys, dtype=np.float64)
        if self.config.cross_isa_deterministic:
            out = ozaki_exact_matmul(self._w0, k64)
            active_idx = np.flatnonzero(self._slot_active)
            for idx in active_idx:
                s = 1.0 if slot_scales is None else float(slot_scales.get(int(idx), 0.0))
                if s == 0.0:
                    continue
                r_eff = int(self._slot_ranks[idx])
                ak = ozaki_exact_matmul(self._arena_a[idx, :r_eff, :], k64)
                bak = ozaki_exact_matmul(self._arena_b[idx, :, :r_eff], ak)
                out = np.add(out, np.multiply(s, bak))
            return out
        out = self._w0 @ k64
        active_idx = np.flatnonzero(self._slot_active)
        for idx in active_idx:
            s = 1.0 if slot_scales is None else float(slot_scales.get(int(idx), 0.0))
            if s == 0.0:
                continue
            r_eff = int(self._slot_ranks[idx])
            out = out + s * (self._arena_b[idx, :, :r_eff] @ (self._arena_a[idx, :r_eff, :] @ k64))
        return out

    def forward(
        self,
        x: np.ndarray,
        *,
        slot_scales: Optional[dict[int, float]] = None,
    ) -> np.ndarray:
        """Row-major forward pass: y = x @ W0^T + sum_i scale_i * (x @ A_i^T) @ B_i^T."""
        x64 = np.ascontiguousarray(x, dtype=np.float64)
        if x64.shape[-1] != self.config.d_in:
            raise ValueError(f"Expected trailing dimension {self.config.d_in}, got {x64.shape[-1]}.")
        out = x64 @ self._w0.T
        active_idx = np.flatnonzero(self._slot_active)
        for idx in active_idx:
            s = 1.0 if slot_scales is None else float(slot_scales.get(int(idx), 0.0))
            if s == 0.0:
                continue
            r_eff = int(self._slot_ranks[idx])
            a_i = self._arena_a[idx, :r_eff, :]
            b_i = self._arena_b[idx, :, :r_eff]
            out = out + s * ((x64 @ a_i.T) @ b_i.T)
        return out

    def knowledge_leakage_norm(
        self,
        k0: np.ndarray,
        *,
        slot_scales: Optional[dict[int, float]] = None,
    ) -> float:
        """Measure ||Delta_cum @ K0||_F across active (or scaled) arena slots."""
        k64 = np.ascontiguousarray(k0, dtype=np.float64)
        if self.config.cross_isa_deterministic:
            drift = np.subtract(
                self.effective_matvec(k64, slot_scales=slot_scales),
                ozaki_exact_matmul(self._w0, k64),
            )
            return math.sqrt(float(ieee754_binary_sum(np.multiply(drift, drift).reshape(-1))))
        drift = self.effective_matvec(k64, slot_scales=slot_scales) - (self._w0 @ k64)
        return float(np.linalg.norm(drift, ord="fro"))


def verify_rederivation(
    expected_patch: StagedPatch,
    *,
    base_weight: np.ndarray,
    config: BetaEditConfig,
    k0: np.ndarray,
    keys: np.ndarray,
    target_values_or_residuals: np.ndarray,
    is_residual: bool = False,
    rank: Optional[int] = None,
    preservation_keys: Optional[np.ndarray] = None,
    preservation_weight: float = 1.0,
    safety_directions: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> tuple[bool, dict[str, str]]:
    """Re-run the boring CPU float64 compiler from scratch and assert 0-byte match."""
    fresh = BetaEditNullspaceProjector(base_weight, config)
    fresh.initialize_anchor_covariance(k0)
    for b_s, a_s in safety_directions:
        fresh.register_safety_direction(b_s, a_s)
    rebuilt = fresh.stage_edit(
        keys,
        target_values_or_residuals,
        is_residual=is_residual,
        rank=rank,
        preservation_keys=preservation_keys,
        preservation_weight=preservation_weight,
        slot_name=expected_patch.slot_name,
    )
    ok = (
        rebuilt.b_bytes == expected_patch.b_bytes
        and rebuilt.a_bytes == expected_patch.a_bytes
        and rebuilt.factor_digest == expected_patch.factor_digest
        and rebuilt.certificate.sha256() == expected_patch.certificate.sha256()
    )
    return ok, {
        "expected_factor_digest": expected_patch.factor_digest,
        "rebuilt_factor_digest": rebuilt.factor_digest,
        "expected_cert_sha256": expected_patch.certificate.sha256(),
        "rebuilt_cert_sha256": rebuilt.certificate.sha256(),
    }
