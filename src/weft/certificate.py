"""Post-quantization division-free integer certificates for weight adapters.

Every bound is computed directly on the post-quantized factors (B_q, A_q) so
BF16 or F16 rounding noise is captured before GPU load. Verification uses
exact integer cross-multiplication with zero floating-point division.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np


def _require_strict_int(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict int (got {type(value).__name__}).")
    return value


@dataclass(frozen=True)
class GatePolicy:
    """Exact rational policy caps (numerator / denominator) checked without division."""

    max_frob_sq_num: int
    max_frob_sq_den: int = 1
    max_spectral_sq_num: Optional[int] = None
    max_spectral_sq_den: int = 1
    max_leakage_num: Optional[int] = None
    max_leakage_den: int = 1
    max_safety_dot_sq_num: Optional[int] = None
    max_safety_dot_sq_den: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_frob_sq_den",
            "max_spectral_sq_den",
            "max_leakage_den",
            "max_safety_dot_sq_den",
        ):
            val = _require_strict_int(name, getattr(self, name))
            if val <= 0:
                raise ValueError(f"{name} must be strictly positive.")
        if _require_strict_int("max_frob_sq_num", self.max_frob_sq_num) < 0:
            raise ValueError("max_frob_sq_num must be non-negative.")
        for opt_name in ("max_spectral_sq_num", "max_leakage_num", "max_safety_dot_sq_num"):
            opt_val = getattr(self, opt_name)
            if opt_val is not None:
                if _require_strict_int(opt_name, opt_val) < 0:
                    raise ValueError(f"{opt_name} must be non-negative.")


@dataclass(frozen=True)
class FixedRadixCertificate:
    """Division-free fixed-radix integer witness computed on post-quantized (B_q, A_q)."""

    radix_scale_exp2: int
    frobenius_sq_scaled: int
    spectral_norm_sq_scaled: int
    leakage_energy_scaled: int
    max_safety_dot_sq_scaled: int
    projector_epoch: int
    factor_digest: str
    quant_dtype: str

    def verify_division_free(self, policy: GatePolicy) -> tuple[bool, list[str]]:
        """Check all configured policy limits via exact integer cross-multiplication."""
        problems: list[str] = []
        int_fields = (
            "radix_scale_exp2",
            "frobenius_sq_scaled",
            "spectral_norm_sq_scaled",
            "leakage_energy_scaled",
            "max_safety_dot_sq_scaled",
            "projector_epoch",
        )
        for field_name in int_fields:
            if type(getattr(self, field_name)) is not int:
                problems.append(f"non_integer_{field_name}")
        if problems:
            return False, problems

        if not (8 <= self.radix_scale_exp2 <= 48):
            problems.append("invalid_radix_scale_exp2")
            return False, problems
        for field_name in (
            "frobenius_sq_scaled",
            "spectral_norm_sq_scaled",
            "leakage_energy_scaled",
            "max_safety_dot_sq_scaled",
        ):
            if getattr(self, field_name) < 0:
                problems.append(f"negative_{field_name}")
        if self.projector_epoch < 0:
            problems.append("negative_projector_epoch")
        if self.spectral_norm_sq_scaled > self.frobenius_sq_scaled:
            problems.append("spectral_exceeds_frobenius")
        if problems:
            return False, problems

        scale = 1 << self.radix_scale_exp2
        if self.frobenius_sq_scaled * policy.max_frob_sq_den > policy.max_frob_sq_num * scale:
            problems.append("frobenius_norm_exceeds_cap")
        if policy.max_spectral_sq_num is not None and (
            self.spectral_norm_sq_scaled * policy.max_spectral_sq_den
            > policy.max_spectral_sq_num * scale
        ):
            problems.append("spectral_norm_exceeds_cap")
        if policy.max_leakage_num is not None and (
            self.leakage_energy_scaled * policy.max_leakage_den
            > policy.max_leakage_num * scale
        ):
            problems.append("leakage_energy_exceeds_cap")
        if policy.max_safety_dot_sq_num is not None and (
            self.max_safety_dot_sq_scaled * policy.max_safety_dot_sq_den
            > policy.max_safety_dot_sq_num * scale
        ):
            problems.append("safety_direction_overlap_exceeds_cap")
        return not problems, problems

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixedRadixCertificate":
        if not isinstance(data.get("factor_digest"), str) or not isinstance(
            data.get("quant_dtype"), str
        ):
            raise TypeError("factor_digest and quant_dtype must be strings.")
        return cls(
            radix_scale_exp2=_require_strict_int("radix_scale_exp2", data["radix_scale_exp2"]),
            frobenius_sq_scaled=_require_strict_int(
                "frobenius_sq_scaled", data["frobenius_sq_scaled"]
            ),
            spectral_norm_sq_scaled=_require_strict_int(
                "spectral_norm_sq_scaled", data["spectral_norm_sq_scaled"]
            ),
            leakage_energy_scaled=_require_strict_int(
                "leakage_energy_scaled", data["leakage_energy_scaled"]
            ),
            max_safety_dot_sq_scaled=_require_strict_int(
                "max_safety_dot_sq_scaled", data["max_safety_dot_sq_scaled"]
            ),
            projector_epoch=_require_strict_int("projector_epoch", data["projector_epoch"]),
            factor_digest=data["factor_digest"],
            quant_dtype=data["quant_dtype"],
        )


def build_certificate(
    b_quant_f64: np.ndarray,
    a_quant_f64: np.ndarray,
    cov_accum: np.ndarray,
    *,
    factor_digest: str,
    quant_dtype: str,
    projector_epoch: int,
    radix_scale_exp2: int = 24,
    safety_directions: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> FixedRadixCertificate:
    """Compute exact post-quantization FixedRadixCertificate in CPU float64.

    Uses skinny QR on B_q (d_out x r) and A_q^T (d_in x r) so singular values
    of B_q @ A_q are computed in O(r^2 * (d_out + d_in)) without forming the
    dense (d_out x d_in) matrix. Ceiling rounding guarantees conservative
    upper bounds at the integer TCB gate.
    """
    if not (8 <= radix_scale_exp2 <= 48):
        raise ValueError("radix_scale_exp2 must lie in [8, 48].")
    b64 = np.ascontiguousarray(b_quant_f64, dtype=np.float64)
    a64 = np.ascontiguousarray(a_quant_f64, dtype=np.float64)
    if b64.ndim != 2 or a64.ndim != 2 or b64.shape[1] != a64.shape[0]:
        raise ValueError("Incompatible factor shapes for certificate.")
    if not (np.all(np.isfinite(b64)) and np.all(np.isfinite(a64))):
        raise ValueError("Non-finite factor entries cannot be certified.")

    # Exact singular values of B_q @ A_q via reduced QR:
    _, r_b = np.linalg.qr(b64, mode="reduced")
    _, r_a = np.linalg.qr(a64.T, mode="reduced")
    s_vals = np.linalg.svd(r_b @ r_a.T, compute_uv=False)

    frob_sq = float(np.sum(s_vals * s_vals))
    spec_sq = float(s_vals[0] * s_vals[0]) if s_vals.size > 0 else 0.0

    # Leakage energy: Tr((B A) C (B A)^T) = Tr((B^T B) (A C A^T))
    bt_b = b64.T @ b64
    ac_at = (a64 @ cov_accum) @ a64.T
    leak = max(0.0, float(np.sum(bt_b * ac_at)))

    # Max squared Frobenius inner product against unit-normalized safety directions
    max_dot_sq = 0.0
    for b_s, a_s in safety_directions:
        dot = float(np.sum((b_s.T @ b64) * (a_s @ a64.T)))
        max_dot_sq = max(max_dot_sq, dot * dot)

    scale = float(1 << radix_scale_exp2)
    return FixedRadixCertificate(
        radix_scale_exp2=radix_scale_exp2,
        frobenius_sq_scaled=int(math.ceil(frob_sq * scale)),
        spectral_norm_sq_scaled=int(math.ceil(spec_sq * scale)),
        leakage_energy_scaled=int(math.ceil(leak * scale)),
        max_safety_dot_sq_scaled=int(math.ceil(max_dot_sq * scale)),
        projector_epoch=projector_epoch,
        factor_digest=factor_digest,
        quant_dtype=quant_dtype,
    )
