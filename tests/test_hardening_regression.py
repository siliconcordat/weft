"""Regression & adversarial tests for fail-closed verification, GGUF bounds, and determinism.

Assertion-free (unittest.TestCase only) so the suite also runs cleanly under
`python3 -I -O -B` and `python3 -I -B -W error`.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from weft.anchors import (
    AnchorCase,
    AnchorSuite,
    GateVerdict,
    evaluate_adapter_gate,
    lexicase_select,
)
from weft.certificate import FixedRadixCertificate
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector
from weft.cross_isa import ieee754_binary_sum, ozaki_exact_matmul
from weft.gguf_lora import (
    GGUF_MAGIC,
    GGUF_TYPE_STRING,
    MAX_GGUF_ARRAY_DEPTH,
    _pack_string,
    read_gguf_model_geometry,
)
from weft.slab_wire import (
    WEFT_SLAB_FORMAT,
    WEFT_SLAB_MAGIC,
    WEFT_SLAB_VERSION,
    verify_slab_wire,
)


def _build_forged_slab(certificate_obj: object) -> bytes:
    header = {
        "format": WEFT_SLAB_FORMAT,
        "b_dtype": "f16",
        "a_dtype": "f16",
        "b_scale_axis": 0,
        "a_scale_axis": 1,
        "b_shape": [2, 1],
        "a_shape": [1, 2],
        "b_byte_len": 4,
        "a_byte_len": 4,
        "slot_name": "crafted",
        "target_tensor": "blk.0.ffn_down.weight",
        "base_weights_sha256": "a" * 64,
        "zone_manifest_id": "",
        "factor_digest": "b" * 64,
        "certificate_sha256": "c" * 64,
        "certificate": certificate_obj,
    }
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"\x00" * 8
    body = (
        struct.pack("<4sIII", WEFT_SLAB_MAGIC, WEFT_SLAB_VERSION, len(hb), len(payload))
        + hb
        + payload
    )
    return body + hashlib.sha256(body).digest()


def _make_suite(
    *,
    safety_prompt: object = "p_s",
    safety_expected: object = "SAFE",
    target_prompt: object = "p_t",
    target_expected: object = "OK",
    holdout_prompt: object = "p_h",
    holdout_expected: object = "HOLD",
    min_holdout_rate: float = 0.80,
) -> AnchorSuite:
    return AnchorSuite(
        suite_id="hardening-v1",
        safety_canaries=(AnchorCase("s0", "safety", safety_prompt, safety_expected, 0),),
        target_canaries=(AnchorCase("t0", "target", target_prompt, target_expected, 1),),
        private_holdouts=(AnchorCase("h0", "target", holdout_prompt, holdout_expected, 2),),
        min_holdout_rate=min_holdout_rate,
        flake_verified=True,
    )


class TestWeftHardeningRegressions(unittest.TestCase):
    # ------------------------------------------------------------------
    # #2: Fail-closed FixedRadixCertificate.from_dict & verify_slab_wire
    # ------------------------------------------------------------------
    def test_slab_wire_and_certificate_missing_fields_fail_closed(self) -> None:
        full_cert = {
            "radix_scale_exp2": 24,
            "frobenius_sq_scaled": 100,
            "spectral_norm_sq_scaled": 50,
            "leakage_energy_scaled": 10,
            "max_safety_dot_sq_scaled": 0,
            "projector_epoch": 1,
            "factor_digest": "b" * 64,
            "quant_dtype": "f16",
        }
        # Removing any single required field must raise ValueError (not KeyError)
        for key in full_cert:
            partial = {k: v for k, v in full_cert.items() if k != key}
            with self.assertRaises(ValueError):
                FixedRadixCertificate.from_dict(partial)

            slab = _build_forged_slab(partial)
            ok, probs, cont = verify_slab_wire(slab)
            self.assertFalse(ok)
            self.assertIsNone(cont)
            self.assertTrue(any(p.startswith("container_invalid:") for p in probs))

        # Minimal header with only factor_digest + quant_dtype (exact PoC from audit)
        minimal_cert = {"factor_digest": "b" * 64, "quant_dtype": "f16"}
        ok_min, probs_min, cont_min = verify_slab_wire(_build_forged_slab(minimal_cert))
        self.assertFalse(ok_min)
        self.assertIsNone(cont_min)
        self.assertTrue(any("Missing required certificate field" in p for p in probs_min))

        # Non-dict certificate input fails with TypeError
        with self.assertRaises(TypeError):
            FixedRadixCertificate.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

        # Defense-in-depth: even if read_slab_wire raises KeyError, verify_slab_wire fails closed
        with mock.patch("weft.slab_wire.read_slab_wire", side_effect=KeyError("synthetic")):
            ok_k, probs_k, cont_k = verify_slab_wire(b"ignored")
        self.assertFalse(ok_k)
        self.assertIsNone(cont_k)
        self.assertTrue(any("container_invalid:" in p for p in probs_k))

    # ------------------------------------------------------------------
    # #3: Fail-closed default certificate_ok=False in evaluate_adapter_gate
    # ------------------------------------------------------------------
    def test_evaluate_adapter_gate_defaults_certificate_ok_false(self) -> None:
        suite = _make_suite()
        perfect_eval = lambda x: {"p_s": "SAFE", "p_t": "OK", "p_h": "HOLD"}[x]

        # Omitting certificate_ok MUST fail closed (never promote uncertified adapter)
        r_default = evaluate_adapter_gate(
            committed_adapter_digest="a" * 64,
            suite=suite,
            evaluator=perfect_eval,
        )
        self.assertEqual(r_default.verdict, GateVerdict.REJECT)
        self.assertIn("integer_certificate_failed", r_default.reasons)

        # Explicit False and non-bool truthy spoof values MUST fail closed
        for bad_flag in (False, 1, "yes", [True]):
            r_bad = evaluate_adapter_gate(
                committed_adapter_digest="a" * 64,
                suite=suite,
                evaluator=perfect_eval,
                certificate_ok=bad_flag,  # type: ignore[arg-type]
            )
            self.assertEqual(r_bad.verdict, GateVerdict.REJECT)
            self.assertIn("integer_certificate_failed", r_bad.reasons)

        # Explicit strict True promotes cleanly
        r_ok = evaluate_adapter_gate(
            committed_adapter_digest="a" * 64,
            suite=suite,
            evaluator=perfect_eval,
            certificate_ok=True,
        )
        self.assertEqual(r_ok.verdict, GateVerdict.TIER1_PROMOTE)
        self.assertEqual(r_ok.reasons, [])

    # ------------------------------------------------------------------
    # #6: AnchorSuite.suite_digest commits to prompts, answers, groups, & types
    # ------------------------------------------------------------------
    def test_suite_digest_commits_to_prompts_expected_outputs_and_types(self) -> None:
        base = _make_suite()
        d_base = base.suite_digest()
        self.assertEqual(len(d_base), 64)
        self.assertEqual(d_base, _make_suite().suite_digest())

        # Mutating expected_output or prompt_or_input in ANY group changes digest
        self.assertNotEqual(d_base, _make_suite(safety_expected="UNSAFE").suite_digest())
        self.assertNotEqual(d_base, _make_suite(target_expected="WRONG").suite_digest())
        self.assertNotEqual(d_base, _make_suite(holdout_expected="DRIFT").suite_digest())
        self.assertNotEqual(d_base, _make_suite(safety_prompt="other_prompt").suite_digest())
        self.assertNotEqual(d_base, _make_suite(target_prompt="other_target").suite_digest())
        self.assertNotEqual(d_base, _make_suite(holdout_prompt="other_hold").suite_digest())
        self.assertNotEqual(d_base, _make_suite(min_holdout_rate=0.90).suite_digest())

        # Type separation: int 1 vs bool True, list vs tuple
        d_int = _make_suite(target_expected=1).suite_digest()
        d_bool = _make_suite(target_expected=True).suite_digest()
        self.assertNotEqual(d_int, d_bool)
        d_list = _make_suite(target_expected=[1, 2]).suite_digest()
        d_tuple = _make_suite(target_expected=(1, 2)).suite_digest()
        self.assertNotEqual(d_list, d_tuple)

        # Deterministic support for ndarray and dict inputs (used in E2 experiments)
        arr1 = np.array([1.0, -2.5, 3.25], dtype=np.float64)
        arr2 = np.array([1.0, -2.5, 3.50], dtype=np.float64)
        d_arr1 = _make_suite(safety_prompt=arr1, target_prompt={"b": 2, "a": 1}).suite_digest()
        d_arr1_reordered = _make_suite(
            safety_prompt=arr1.copy(), target_prompt={"a": 1, "b": 2}
        ).suite_digest()
        d_arr2 = _make_suite(safety_prompt=arr2, target_prompt={"a": 1, "b": 2}).suite_digest()
        self.assertEqual(d_arr1, d_arr1_reordered)
        self.assertNotEqual(d_arr1, d_arr2)

    # ------------------------------------------------------------------
    # #5 & #9: Strict UTF-8 decoding & bounded nested array depth in GGUF
    # ------------------------------------------------------------------
    def test_gguf_rejects_invalid_utf8_names_keys_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            # Invalid UTF-8 tensor name must raise UnicodeDecodeError (ValueError)
            bad_tname = b"blk.0.\xff\xfe.weight"
            tinfo = (
                struct.pack("<Q", len(bad_tname))
                + bad_tname
                + struct.pack("<IQQIQ", 2, 4, 4, 1, 0)
            )
            p_tname = tdp / "bad_tensor_name.gguf"
            p_tname.write_bytes(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 1, 0) + tinfo)
            with self.assertRaises(UnicodeDecodeError):
                read_gguf_model_geometry(p_tname)

            # Invalid UTF-8 metadata key must raise UnicodeDecodeError
            bad_key = b"general.\x80name"
            kv_bad_key = (
                struct.pack("<Q", len(bad_key))
                + bad_key
                + struct.pack("<I", GGUF_TYPE_STRING)
                + _pack_string("ok")
            )
            p_key = tdp / "bad_key.gguf"
            p_key.write_bytes(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + kv_bad_key)
            with self.assertRaises(UnicodeDecodeError):
                read_gguf_model_geometry(p_key)

            # Invalid UTF-8 kept metadata string value must raise UnicodeDecodeError
            bad_val = b"model_\xc3\x28"
            kv_bad_val = (
                _pack_string("general.name")
                + struct.pack("<I", GGUF_TYPE_STRING)
                + struct.pack("<Q", len(bad_val))
                + bad_val
            )
            p_val = tdp / "bad_val.gguf"
            p_val.write_bytes(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + kv_bad_val)
            with self.assertRaises(UnicodeDecodeError):
                read_gguf_model_geometry(p_val)

    def test_gguf_nested_array_depth_cap_blocks_recursion_dos(self) -> None:
        def _build_nested_gguf(nesting_levels: int) -> bytes:
            # nesting_levels=1 means 1 GGUF_TYPE_ARRAY header (depth 0)
            # nesting_levels=N means N-1 nested (9, 1) headers followed by (4, 0)
            kv = (
                _pack_string("custom.nested")
                + struct.pack("<I", 9)
                + (struct.pack("<IQ", 9, 1) * (nesting_levels - 1))
                + struct.pack("<IQ", 4, 0)
            )
            return struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + kv

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            # Exact cap boundary (MAX_GGUF_ARRAY_DEPTH levels: depth 0..31) succeeds
            p_ok = tdp / "depth_ok.gguf"
            p_ok.write_bytes(_build_nested_gguf(MAX_GGUF_ARRAY_DEPTH))
            geom = read_gguf_model_geometry(p_ok)
            self.assertEqual(geom["n_tensors"], 0)

            # One level past cap (MAX_GGUF_ARRAY_DEPTH + 1) fails closed with ValueError
            p_over = tdp / "depth_over.gguf"
            p_over.write_bytes(_build_nested_gguf(MAX_GGUF_ARRAY_DEPTH + 1))
            with self.assertRaisesRegex(ValueError, "nested array depth"):
                read_gguf_model_geometry(p_over)

            # ~14 KiB 1200-level crafted payload fails with ValueError (never RecursionError)
            p_dos = tdp / "depth_dos.gguf"
            p_dos.write_bytes(_build_nested_gguf(1200))
            with self.assertRaisesRegex(ValueError, "nested array depth"):
                read_gguf_model_geometry(p_dos)

    # ------------------------------------------------------------------
    # #10: Cross-ISA deterministic knowledge_leakage_norm (zero BLAS mix)
    # ------------------------------------------------------------------
    def test_deterministic_knowledge_leakage_norm_exact_zero_and_blas_free(self) -> None:
        rng = np.random.default_rng(20261125)
        d_in, d_out = 32, 24
        w0 = rng.standard_normal((d_out, d_in)) * 0.5
        k0 = rng.standard_normal((d_in, 16))
        keys = rng.standard_normal((d_in, 4))
        residuals = rng.standard_normal((d_out, 4)) * 0.25

        cfg = BetaEditConfig(
            d_in=d_in,
            d_out=d_out,
            max_rank=4,
            lambda_leakage=50.0,
            lambda_ridge=2.0,
            nullspace_threshold=1e-2,
            threshold_mode="relative_max",
            quant_dtype="f32",
            cross_isa_deterministic=True,
        )
        proj = BetaEditNullspaceProjector(w0, cfg)
        proj.initialize_anchor_covariance(k0)

        # With zero committed slots, deterministic leakage MUST be identically 0.0
        self.assertEqual(proj.knowledge_leakage_norm(k0), 0.0)

        # After committing a patch, muting slot_scales={slot: 0.0} MUST return exact 0.0
        patch = proj.stage_edit(keys, residuals, is_residual=True, rank=4, slot_name="det")
        slot = proj.commit_patch(patch, keys)
        self.assertEqual(proj.knowledge_leakage_norm(k0, slot_scales={slot: 0.0}), 0.0)

        # Active slot leakage MUST match BLAS-free Ozaki + ieee754_binary_sum bit-for-bit
        got_leak = proj.knowledge_leakage_norm(k0)
        eff = proj.effective_matvec(k0)
        base_exact = ozaki_exact_matmul(w0, k0)
        drift_exact = np.subtract(eff, base_exact)
        want_leak = math.sqrt(
            float(ieee754_binary_sum(np.multiply(drift_exact, drift_exact).reshape(-1)))
        )
        self.assertEqual(got_leak.hex(), want_leak.hex())
        self.assertGreater(got_leak, 0.0)

    # ------------------------------------------------------------------
    # #11: lexicase_select rejects NaN, Inf, non-real, & invalid epsilon
    # ------------------------------------------------------------------
    def test_lexicase_select_rejects_non_finite_errors_and_bad_epsilon(self) -> None:
        bad_values = [float("nan"), float("inf"), float("-inf"), True, False, None, "0.1"]
        for bad in bad_values:
            with self.assertRaises(ValueError):
                lexicase_select({"bad": [bad, 0.0], "good": [0.1, 0.1]}, seed_material="s")  # type: ignore[list-item]
            with self.assertRaises(ValueError):
                lexicase_select({"bad": [0.0, bad], "good": [0.1, 0.1]}, seed_material="s")  # type: ignore[list-item]

        for bad_eps in (float("nan"), float("inf"), float("-inf"), -1e-6, True):
            with self.assertRaises(ValueError):
                lexicase_select({"a": [0.1], "b": [0.2]}, seed_material="s", epsilon=bad_eps)  # type: ignore[arg-type]

        # Finite NumPy scalar/array error vectors remain fully supported and deterministic
        np_errors = {
            "c1": np.array([0.0, 1.0], dtype=np.float32),
            "c2": np.array([0.5, 0.0], dtype=np.float64),
        }
        w1 = lexicase_select(np_errors, seed_material="np-seed")
        w2 = lexicase_select(np_errors, seed_material="np-seed")
        self.assertEqual(w1, w2)
        self.assertIn(w1, ("c1", "c2"))


if __name__ == "__main__":
    unittest.main()
