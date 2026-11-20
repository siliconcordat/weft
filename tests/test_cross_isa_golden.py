"""Golden-digest test for the cross-ISA deterministic BetaEdit path (`cross_isa_deterministic=True`).

The fixture is built without BLAS, libm or numpy's RNG: a splitmix64 integer stream gives
dyadic rationals, and the rank-deficient anchor matrix is an exact int64 product. Every input
is therefore bit-identical on any IEEE-754 host. The expected digests below were recorded on
x86_64 (numpy 2.5.5, scipy-openblas 0.3.34) and reproduced across OpenBLAS thread counts and
core types. Running this test on another ISA (e.g. arm64 with Accelerate or OpenBLAS) *is* the
cross-ISA bit-exactness check: any change in a raw float64 factor, the quantized f32/f16 bytes,
the factor digest or the integer certificate fails it.

The legacy path (`cross_isa_deterministic=False`) is deliberately not covered: its float64
results, and hence its f32 bytes, depend on the BLAS kernel and thread count, so it is only
reproducible with pinned BLAS and threads (e.g. `OPENBLAS_NUM_THREADS=1`) on one host.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest

import numpy as np

from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector
from weft.cross_isa import compile_betaedit_cross_isa

_MASK64 = (1 << 64) - 1
_D_IN, _D_OUT, _N_ANCHOR, _ANCHOR_RANK, _N_KEYS, _RANK = 32, 24, 48, 16, 8, 4


def _splitmix64(state: int):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        yield z ^ (z >> 31)


def _int_matrix(gen, rows: int, cols: int, bits: int = 12) -> np.ndarray:
    """Signed integers in [-2**bits, 2**bits) from the top bits of the stream (exact, portable)."""
    half = 1 << bits
    vals = [int(next(gen) >> (64 - bits - 1)) - half for _ in range(rows * cols)]
    return np.array(vals, dtype=np.int64).reshape(rows, cols)


def golden_fixture() -> dict[str, np.ndarray]:
    gen = _splitmix64(20261119)
    w0 = np.ldexp(_int_matrix(gen, _D_OUT, _D_IN).astype(np.float64), -16)
    basis = _int_matrix(gen, _D_IN, _ANCHOR_RANK)
    coeff = _int_matrix(gen, _ANCHOR_RANK, _N_ANCHOR)
    k0 = np.ldexp((basis @ coeff).astype(np.float64), -24)  # exact int64 product, rank 16 < d_in
    keys = np.ldexp(_int_matrix(gen, _D_IN, _N_KEYS).astype(np.float64), -13)
    targets = np.ldexp(_int_matrix(gen, _D_OUT, _N_KEYS).astype(np.float64), -12)
    return {"w0": w0, "k0": k0, "keys": keys, "targets": targets}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_f64(arr: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(arr, dtype="<f8").tobytes())


def compute_golden_digests() -> dict[str, object]:
    fx = golden_fixture()
    out: dict[str, object] = {
        "fixture": _sha(b"".join(_sha_f64(fx[k]).encode() for k in ("w0", "k0", "keys", "targets"))),
    }

    b, a, s_vals, trunc_err, _, null_dim = compile_betaedit_cross_isa(
        k0=fx["k0"],
        keys=fx["keys"],
        residuals=fx["targets"],
        rank=_RANK,
        lambda_leakage=80.0,
        lambda_ridge=3.0,
        nullspace_threshold=5e-2,
        threshold_mode="relative_max",
    )
    out["raw"] = {
        "b_f64": _sha_f64(b),
        "a_f64": _sha_f64(a),
        "s_f64": _sha_f64(s_vals),
        "truncation_rel_error": float(trunc_err).hex(),
        "null_dim": int(null_dim),
    }

    for qd in ("f32", "f16"):
        cfg = BetaEditConfig(
            d_in=_D_IN,
            d_out=_D_OUT,
            max_rank=_RANK,
            lambda_leakage=80.0,
            lambda_ridge=3.0,
            nullspace_threshold=5e-2,
            threshold_mode="relative_max",
            quant_dtype=qd,
            cross_isa_deterministic=True,
        )
        proj = BetaEditNullspaceProjector(fx["w0"], cfg)
        nd = proj.initialize_anchor_covariance(fx["k0"])
        proj.register_safety_direction(fx["targets"][:, :1], fx["keys"][:, :1].T)
        patch = proj.stage_edit(fx["keys"], fx["targets"], is_residual=False, rank=_RANK, slot_name="golden")
        cert = json.dumps(dataclasses.asdict(patch.certificate), sort_keys=True, separators=(",", ":"))
        out[qd] = {
            "null_dim": int(nd),
            "b_bytes": _sha(patch.b_bytes),
            "a_bytes": _sha(patch.a_bytes),
            "singular_values_f64": _sha_f64(patch.singular_values),
            "factor_digest": patch.factor_digest,
            "certificate": _sha(cert.encode("utf-8")),
        }
    return out


# Recorded with cross_isa_deterministic=True; see module docstring.
GOLDEN: dict[str, object] = {
    "f16": {
        "a_bytes": "fa95032f8e3ce9a0800f4feff113db9803ab060275e1097f4f58f8521fa615bd",
        "b_bytes": "9371848fa6df43ebd145f7510fb8c684067b6c7b4058e0dcd0cda3dd9736bec6",
        "certificate": "a67d2ff3bef0ce6a3bedeb7d9f3f79a1db2fbb790aa7db67efd2b78a18132f8c",
        "factor_digest": "8ffee8c3b3f4720cc2d9a351dc303b190074d6d72e5f37049a6e2f333f5e135a",
        "null_dim": 19,
        "singular_values_f64": "6389b49a4f3e29a3115492ecb718282b1b9e8bb15d26e57797c7c7331b66d89d",
    },
    "f32": {
        "a_bytes": "b940f9284f1a18b73108b898b96409d8236c8ee5a1baca9b124a5a5f3540d022",
        "b_bytes": "7e799b7a64c5df3fec58b7aa5bc5f9be2f5432c65e6e647d2f7b13f551d1b657",
        "certificate": "ac0db4738807ca100f98cf05cb28df17f41e31a159973ba7d15f65993c53248e",
        "factor_digest": "f48e3c4b60dd6afbb60da033faa2a865ea1c5f92f94a287fe645d43187677934",
        "null_dim": 19,
        "singular_values_f64": "6389b49a4f3e29a3115492ecb718282b1b9e8bb15d26e57797c7c7331b66d89d",
    },
    "fixture": "e18987cd80917905b46a6aa0860f7d977c51ad27dcd4615a3a9454e06acf7672",
    "raw": {
        "a_f64": "f73e876f973e06971af8516a0aa3e22d73ea90bffec19f758e920dcb3fc23f55",
        "b_f64": "0e2b3e6db821a66870f72014d926ff77055a14b630e2b3778aff9439fbd31364",
        "null_dim": 19,
        "s_f64": "f2680fb1b0b0090c8e47540e65e824c1d90a2355e04565aac84427e808b29f05",
        "truncation_rel_error": "0x1.d1a8ea988db37p-2",
    },
}


class TestCrossISAGoldenDigest(unittest.TestCase):
    def test_fixture_is_exact_and_rank_deficient(self) -> None:
        fx = golden_fixture()
        self.assertEqual(int(np.linalg.matrix_rank(fx["k0"])), _ANCHOR_RANK)
        self.assertEqual(
            _sha(b"".join(_sha_f64(fx[k]).encode() for k in ("w0", "k0", "keys", "targets"))),
            GOLDEN["fixture"],
        )

    def test_deterministic_path_matches_golden_digests(self) -> None:
        got = compute_golden_digests()
        self.maxDiff = None
        self.assertEqual(got, GOLDEN)


if __name__ == "__main__":
    print(json.dumps(compute_golden_digests(), indent=4, sort_keys=True))
