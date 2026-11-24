"""Fail-closed real-weight provenance regression tests for siliconcordat/weft#6.

Zero bare ``assert`` statements; runs cleanly with stdlib ``unittest`` across
``-I -B``, ``-I -O -B``, and ``-I -B -W error`` as well as under ``pytest``.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments import checkpoint_provenance as cp  # noqa: E402
from experiments import e3_real_checkpoint_smoke as e3  # noqa: E402
from experiments import e4_asymmetric_quant_packing as e4  # noqa: E402
from experiments import e5_noise_k0_scaling_and_readout as e5  # noqa: E402


class _EvilPath(type(Path("."))):
    """Adversarial Path subclass rejected by coerce_exact_path."""


def _write_minimal_non_synthetic_ckpt(dest: Path, st_payload: bytes = b"REAL_WEIGHTS_PLACEHOLDER") -> str:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.safetensors").write_bytes(st_payload)
    (dest / "config.json").write_text('{"num_layers": 8}\n', encoding="utf-8")
    (dest / "vocab.json").write_text('{"a": 0}\n', encoding="utf-8")
    (dest / "tokenizer.json").write_text('{"version": "1.0"}\n', encoding="utf-8")
    return hashlib.sha256(st_payload).hexdigest()


class TestCheckpointProvenanceUnit(unittest.TestCase):
    def test_constants_and_zero_bare_asserts(self) -> None:
        self.assertEqual(
            cp.TINYSTORIES_1M_SAFETENSORS_SHA256,
            "eb9c5893cb5dc6f78b9b5703ab77b9e94f0b056b7bd920c8fa0146c78af5b55b",
        )
        self.assertEqual(
            cp.SYNTHETIC_SAFETENSORS_SHA256,
            "cd62ff6e11c703e627bfd8a362aba9fe3a027f1a8dfb08f4ceb976ca8c46fe92",
        )
        for rel in ("experiments/checkpoint_provenance.py", "tests/test_real_weights_provenance.py"):
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
            assert_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
            self.assertEqual(assert_lines, [], f"unexpected bare assert in {rel}: {assert_lines}")

    def test_validate_sha256_and_env_flag_and_coerce_path(self) -> None:
        good_sha = "a" * 64
        self.assertEqual(cp.validate_sha256_hex(good_sha), good_sha)
        for bad_sha in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 123, None):
            with self.assertRaises(cp.RealWeightsRequiredError) as cm:
                cp.validate_sha256_hex(bad_sha)
            self.assertEqual(cm.exception.code, "INVALID_SHA256")

        for val in ("", "0", "false", "FALSE", " no "):
            self.assertFalse(cp.env_requires_real_weights({cp.ENV_REQUIRE_REAL_WEIGHTS: val}))
        self.assertFalse(cp.env_requires_real_weights({}))
        for val in ("1", "true", "TRUE", " yes "):
            self.assertTrue(cp.env_requires_real_weights({cp.ENV_REQUIRE_REAL_WEIGHTS: val}))
        for bad_env in ("maybe", "2", "ture", 1):
            with self.assertRaises(cp.RealWeightsRequiredError) as cm:
                cp.env_requires_real_weights({cp.ENV_REQUIRE_REAL_WEIGHTS: bad_env})  # type: ignore[dict-item]
            self.assertEqual(cm.exception.code, "INVALID_ENV_FLAG")

        p_obj = Path("some/dir")
        self.assertEqual(cp.coerce_exact_path("some/dir"), p_obj)
        self.assertIs(cp.coerce_exact_path(p_obj), p_obj)
        for bad_p in ("", "a\x00b", Path("a\x00b"), _EvilPath("some/dir"), 42, None):
            with self.assertRaises(cp.RealWeightsRequiredError) as cm:
                cp.coerce_exact_path(bad_p)
            self.assertEqual(cm.exception.code, "INVALID_PATH")

    def test_sha256_regular_file_and_tokenizer_synthetic_detector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f = root / "blob.bin"
            f.write_bytes(b"hello")
            self.assertEqual(cp._sha256_regular_file(f, "blob"), hashlib.sha256(b"hello").hexdigest())

            # Follows valid HF-style blob symlinks cleanly
            link = root / "model.safetensors"
            link.symlink_to(f)
            self.assertEqual(cp._sha256_regular_file(link, "link"), hashlib.sha256(b"hello").hexdigest())

            # Rejects missing file, directory, FIFO, and unreadable file
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_miss:
                cp._sha256_regular_file(root / "missing", "missing")
            self.assertEqual(cm_miss.exception.code, "MISSING_CHECKPOINT_FILE")

            with self.assertRaises(cp.RealWeightsRequiredError) as cm_dir:
                cp._sha256_regular_file(root, "dir")
            self.assertEqual(cm_dir.exception.code, "NOT_A_REGULAR_FILE")

            fifo = root / "pipe"
            os.mkfifo(fifo)
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_fifo:
                cp._sha256_regular_file(fifo, "fifo")
            self.assertEqual(cm_fifo.exception.code, "NOT_A_REGULAR_FILE")

            if os.geteuid() != 0:
                locked = root / "locked.bin"
                locked.write_bytes(b"secret")
                locked.chmod(0)
                try:
                    with self.assertRaises(cp.RealWeightsRequiredError) as cm_rd:
                        cp._sha256_regular_file(locked, "locked")
                    self.assertEqual(cm_rd.exception.code, "CHECKPOINT_READ_ERROR")
                finally:
                    locked.chmod(stat.S_IRUSR | stat.S_IWUSR)

            # Synthetic tokenizer detector: exact boundary and negative controls
            tok_missing = root / "nope.json"
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_missing))
            self.assertFalse(cp._tokenizer_declares_synthetic(root))
            self.assertFalse(cp._tokenizer_declares_synthetic(fifo))

            tok_list = root / "tok_list.json"
            tok_list.write_text("[1, 2, 3]\n", encoding="utf-8")
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_list))

            tok_syn = root / "tok_syn.json"
            tok_syn.write_text('{"version":"synthetic"}\n', encoding="utf-8")
            self.assertTrue(cp._tokenizer_declares_synthetic(tok_syn))

            tok_syn_case = root / "tok_syn_case.json"
            tok_syn_case.write_text('{"version": " Synthetic "}\n', encoding="utf-8")
            self.assertTrue(cp._tokenizer_declares_synthetic(tok_syn_case))

            tok_int_ver = root / "tok_int_ver.json"
            tok_int_ver.write_text('{"version": 123}\n', encoding="utf-8")
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_int_ver))

            tok_real = root / "tok_real.json"
            tok_real.write_text('{"version":"1.0"}\n', encoding="utf-8")
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_real))

            tok_bad = root / "tok_bad.json"
            tok_bad.write_text("{broken", encoding="utf-8")
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_bad))

            tok_exact_limit = root / "tok_limit.json"
            pad = " " * (cp._MAX_SYNTHETIC_TOKENIZER_BYTES - len('{"version":"synthetic"}'))
            tok_exact_limit.write_text('{"version":"synthetic"}' + pad, encoding="utf-8")
            self.assertEqual(tok_exact_limit.stat().st_size, cp._MAX_SYNTHETIC_TOKENIZER_BYTES)
            self.assertTrue(cp._tokenizer_declares_synthetic(tok_exact_limit))

            tok_over = root / "tok_over.json"
            tok_over.write_text('{"version":"synthetic"}' + pad + " ", encoding="utf-8")
            self.assertEqual(tok_over.stat().st_size, cp._MAX_SYNTHETIC_TOKENIZER_BYTES + 1)
            self.assertFalse(cp._tokenizer_declares_synthetic(tok_over))

            if os.geteuid() != 0:
                tok_locked = root / "tok_locked.json"
                tok_locked.write_text('{"version":"synthetic"}\n', encoding="utf-8")
                tok_locked.chmod(0)
                try:
                    self.assertFalse(cp._tokenizer_declares_synthetic(tok_locked))
                finally:
                    tok_locked.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_discover_tinystories_cache_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hf = Path(td)
            # 1. Empty HF_HOME
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertIsNone(st_f)
            self.assertIsNone(cfg_d)
            self.assertEqual(reason, "no_tinystories_cache")

            # 2. Stock main download: pytorch_model.bin without model.safetensors
            snaps = hf / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots"
            main_snap = snaps / "main_rev"
            main_snap.mkdir(parents=True)
            (snaps / "stray_file.txt").write_text("ignore", encoding="utf-8")
            (main_snap / "pytorch_model.bin").write_bytes(b"bin")
            (main_snap / "config.json").write_text("{}\n", encoding="utf-8")
            (main_snap / "vocab.json").write_text("{}\n", encoding="utf-8")
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertIsNone(st_f)
            self.assertIsNone(cfg_d)
            self.assertEqual(reason, "tinystories_cache_missing_safetensors_use_refs_pr_12")

            # 3. Incomplete snapshots: safetensors-only, config-only, vocab-only, empty
            (main_snap / "pytorch_model.bin").unlink()
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertEqual(reason, "incomplete_tinystories_cache")

            (main_snap / "model.safetensors").write_bytes(b"st")
            (main_snap / "vocab.json").unlink()
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertIsNone(st_f)
            self.assertEqual(reason, "incomplete_tinystories_cache")

            (main_snap / "vocab.json").write_text("{}\n", encoding="utf-8")
            (main_snap / "config.json").unlink()
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertIsNone(st_f)
            self.assertEqual(reason, "incomplete_tinystories_cache")

            # 4. Present safetensors + config + vocab
            (main_snap / "config.json").write_text("{}\n", encoding="utf-8")
            st_f, cfg_d, reason = cp.discover_tinystories_cache_detailed({"HF_HOME": str(hf)})
            self.assertEqual(st_f, (main_snap / "model.safetensors").resolve())
            self.assertEqual(cfg_d, main_snap.resolve())
            self.assertIsNone(reason)

    def test_resolve_checkpoint_fail_closed_and_spoof_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_hf = {"HF_HOME": str(root / "hf_empty")}

            # Invalid require_real_weights type rejected
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_flag:
                cp.resolve_experiment_checkpoint(
                    root / "d0",
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=1,  # type: ignore[arg-type]
                    env=empty_hf,
                )
            self.assertEqual(cm_flag.exception.code, "INVALID_REQUIRE_FLAG")

            # Offline default without require_real_weights clearly labels synthetic fallback
            d_off = root / "d_off"
            d_off.mkdir()
            prov = cp.resolve_experiment_checkpoint(
                d_off,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                env=empty_hf,
            )
            self.assertTrue(prov.is_synthetic_fallback)
            self.assertFalse(prov.real_weights_verified)
            self.assertEqual(prov.checkpoint_source, cp.SYNTHETIC_FALLBACK_LABEL)
            self.assertEqual(prov.safetensors_sha256, cp.SYNTHETIC_SAFETENSORS_SHA256)
            self.assertEqual(prov.cache_miss_reason, "no_tinystories_cache")
            self.assertFalse(prov.to_dict()["real_weights_verified"])

            # Empty HF_HOME with require_real_weights=True or env=1 or expected_sha fails closed
            for kw in (
                {"require_real_weights": True, "env": empty_hf},
                {"env": {**empty_hf, cp.ENV_REQUIRE_REAL_WEIGHTS: "1"}},
                {"require_real_weights": False, "expected_safetensors_sha256": cp.TINYSTORIES_1M_SAFETENSORS_SHA256, "env": empty_hf},
            ):
                d_req = root / f"d_req_{len(list(root.iterdir()))}"
                d_req.mkdir()
                with self.assertRaises(cp.RealWeightsRequiredError) as cm_req:
                    cp.resolve_experiment_checkpoint(
                        d_req,
                        build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                        **kw,
                    )
                self.assertEqual(cm_req.exception.code, "REAL_WEIGHTS_REQUIRED")
                # Must NEVER silently build synthetic weights before raising
                self.assertFalse((d_req / "model.safetensors").exists())

            # checkpoint_dir missing / missing required file / non-regular file rejected
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_nd:
                cp.resolve_experiment_checkpoint(
                    root / "d_nd",
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=root / "does_not_exist",
                )
            self.assertEqual(cm_nd.exception.code, "CHECKPOINT_DIR_NOT_FOUND")

            bad_ck = root / "bad_ck"
            bad_ck.mkdir()
            (bad_ck / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_mf:
                cp.resolve_experiment_checkpoint(
                    root / "d_mf",
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=bad_ck,
                )
            self.assertEqual(cm_mf.exception.code, "MISSING_CHECKPOINT_FILE")

            dir_as_st = root / "dir_as_st"
            dir_as_st.mkdir()
            (dir_as_st / "model.safetensors").mkdir()
            (dir_as_st / "config.json").write_text("{}\n", encoding="utf-8")
            (dir_as_st / "vocab.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_nr:
                cp.resolve_experiment_checkpoint(
                    root / "d_nr",
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=dir_as_st,
                )
            self.assertEqual(cm_nr.exception.code, "NOT_A_REGULAR_FILE")

            # Spoof attack A: synthetic fixture with real-looking tokenizer.json passed via checkpoint_dir
            syn_sha_spoof = root / "syn_sha_spoof"
            syn_sha_spoof.mkdir()
            e3._build_synthetic_gptneo_dir(syn_sha_spoof)
            (syn_sha_spoof / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
            d_s1 = root / "d_s1"
            d_s1.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_s1:
                cp.resolve_experiment_checkpoint(
                    d_s1,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=syn_sha_spoof,
                    require_real_weights=True,
                )
            self.assertEqual(cm_s1.exception.code, "SYNTHETIC_FIXTURE_REJECTED")

            # Spoof attack B: altered bytes but tokenizer.json declares synthetic
            syn_tok_spoof = root / "syn_tok_spoof"
            _write_minimal_non_synthetic_ckpt(syn_tok_spoof, b"ALTERED_SYNTHETIC_BYTES")
            (syn_tok_spoof / "tokenizer.json").write_text('{"version":"synthetic"}\n', encoding="utf-8")
            d_s2 = root / "d_s2"
            d_s2.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_s2:
                cp.resolve_experiment_checkpoint(
                    d_s2,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=syn_tok_spoof,
                    require_real_weights=True,
                )
            self.assertEqual(cm_s2.exception.code, "SYNTHETIC_FIXTURE_REJECTED")

            # Even without require_real_weights, synthetic checkpoint_dir cannot claim real weights
            d_s3 = root / "d_s3"
            d_s3.mkdir()
            prov_s3 = cp.resolve_experiment_checkpoint(
                d_s3,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                checkpoint_dir=syn_sha_spoof,
                require_real_weights=False,
            )
            self.assertTrue(prov_s3.is_synthetic_fallback)
            self.assertFalse(prov_s3.real_weights_verified)
            # And matching vs mismatched expected_sha on synthetic checkpoint_dir
            d_s3b = root / "d_s3b"
            d_s3b.mkdir()
            prov_s3b = cp.resolve_experiment_checkpoint(
                d_s3b,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                checkpoint_dir=syn_sha_spoof,
                require_real_weights=False,
                expected_safetensors_sha256=cp.SYNTHETIC_SAFETENSORS_SHA256,
            )
            self.assertTrue(prov_s3b.is_synthetic_fallback)
            self.assertFalse(prov_s3b.real_weights_verified)

            d_s4 = root / "d_s4"
            d_s4.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_s4:
                cp.resolve_experiment_checkpoint(
                    d_s4,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=syn_sha_spoof,
                    require_real_weights=False,
                    expected_safetensors_sha256="0" * 64,
                )
            self.assertEqual(cm_s4.exception.code, "SHA256_MISMATCH")

            # Valid non-synthetic user checkpoint_dir:
            # 1) require_real_weights=True WITHOUT expected_sha MUST FAIL CLOSED (D1!)
            user_ck = root / "user_ck"
            user_sha = _write_minimal_non_synthetic_ckpt(user_ck, b"REAL_USER_MODEL_WEIGHTS")
            d_u1 = root / "d_u1"
            d_u1.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_u1:
                cp.resolve_experiment_checkpoint(
                    d_u1,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=user_ck,
                    require_real_weights=True,
                )
            self.assertEqual(cm_u1.exception.code, "UNVERIFIED_CHECKPOINT_SHA256")

            # Pre-existing file in dest_dir wraps FileExistsError into RealWeightsRequiredError (D2!)
            d_u_exists = root / "d_u_exists"
            d_u_exists.mkdir()
            (d_u_exists / "config.json").write_text("already_here", encoding="utf-8")
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_u_ex:
                cp.resolve_experiment_checkpoint(
                    d_u_exists,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=user_ck,
                    require_real_weights=False,
                )
            self.assertEqual(cm_u_ex.exception.code, "STAGING_DIR_ERROR")

            # 2) require_real_weights=False without pin succeeds as unpinned_user_checkpoint
            d_u1_off = root / "d_u1_off"
            d_u1_off.mkdir()
            prov_u1_off = cp.resolve_experiment_checkpoint(
                d_u1_off,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                checkpoint_dir=user_ck,
                require_real_weights=False,
            )
            self.assertFalse(prov_u1_off.is_synthetic_fallback)
            self.assertFalse(prov_u1_off.real_weights_verified)
            self.assertEqual(prov_u1_off.pin_source, "none")
            self.assertEqual(prov_u1_off.provenance_tier, "unpinned_user_checkpoint")
            self.assertEqual(prov_u1_off.weights_evidence_claim, "unpinned_user_checkpoint")

            # 3) Known TinyStories-1M SHA match verifies as known_pretrained_pin
            d_u1b = root / "d_u1b"
            d_u1b.mkdir()
            with mock.patch.object(cp, "TINYSTORIES_1M_SAFETENSORS_SHA256", user_sha):
                prov_u1b = cp.resolve_experiment_checkpoint(
                    d_u1b,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=user_ck,
                    require_real_weights=True,
                )
            self.assertTrue(prov_u1b.real_weights_verified)
            self.assertTrue(prov_u1b.matches_known_tinystories_1m)
            self.assertEqual(prov_u1b.pin_source, "known_pretrained_pin")
            self.assertEqual(prov_u1b.provenance_tier, "verified_pretrained_safetensors")
            self.assertEqual(prov_u1b.weights_evidence_claim, "verified_tinystories_1m_refs_pr_12")

            # 4) Caller-supplied SHA pin (not TinyStories-1M): pin_source="caller_supplied_pin",
            #    real_weights_verified=False, tier != verified_pretrained_safetensors (D2!)
            d_u2 = root / "d_u2"
            d_u2.mkdir()
            prov_u2 = cp.resolve_experiment_checkpoint(
                d_u2,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                checkpoint_dir=user_ck,
                require_real_weights=True,
                expected_safetensors_sha256=user_sha,
            )
            self.assertFalse(prov_u2.real_weights_verified)
            self.assertFalse(prov_u2.matches_known_tinystories_1m)
            self.assertEqual(prov_u2.pin_source, "caller_supplied_pin")
            self.assertEqual(prov_u2.provenance_tier, "caller_pinned_checkpoint")
            self.assertEqual(prov_u2.weights_evidence_claim, "caller_supplied_sha256_pin_not_known_pretrained")

            d_u3 = root / "d_u3"
            d_u3.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_u3:
                cp.resolve_experiment_checkpoint(
                    d_u3,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    checkpoint_dir=user_ck,
                    expected_safetensors_sha256="1" * 64,
                )
            self.assertEqual(cm_u3.exception.code, "SHA256_MISMATCH")

            # HF cache paths: synthetic spoof in cache, unverified cache SHA, pinned cache SHA match
            hf_syn = root / "hf_syn"
            snap_syn = hf_syn / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots" / "rev1"
            snap_syn.mkdir(parents=True)
            e3._build_synthetic_gptneo_dir(snap_syn)
            d_h1 = root / "d_h1"
            d_h1.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h1:
                cp.resolve_experiment_checkpoint(
                    d_h1,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=True,
                    env={"HF_HOME": str(hf_syn)},
                )
            self.assertEqual(cm_h1.exception.code, "SYNTHETIC_FIXTURE_REJECTED")

            d_h2 = root / "d_h2"
            d_h2.mkdir()
            prov_h2 = cp.resolve_experiment_checkpoint(
                d_h2,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                require_real_weights=False,
                env={"HF_HOME": str(hf_syn)},
            )
            self.assertTrue(prov_h2.is_synthetic_fallback)
            self.assertFalse(prov_h2.real_weights_verified)
            self.assertEqual(prov_h2.pin_source, "none")

            # Synthetic cache with matching vs mismatched expected_sha when require_real_weights=False
            d_h2_match = root / "d_h2_match"
            d_h2_match.mkdir()
            prov_h2_match = cp.resolve_experiment_checkpoint(
                d_h2_match,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                require_real_weights=False,
                expected_safetensors_sha256=cp.SYNTHETIC_SAFETENSORS_SHA256,
                env={"HF_HOME": str(hf_syn)},
            )
            self.assertTrue(prov_h2_match.is_synthetic_fallback)
            self.assertFalse(prov_h2_match.real_weights_verified)

            d_h2_mis = root / "d_h2_mis"
            d_h2_mis.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h2_mis:
                cp.resolve_experiment_checkpoint(
                    d_h2_mis,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=False,
                    expected_safetensors_sha256="0" * 64,
                    env={"HF_HOME": str(hf_syn)},
                )
            self.assertEqual(cm_h2_mis.exception.code, "SHA256_MISMATCH")

            # Synthetic cache with non-synthetic tokenizer.json (tests SHA half of cache BoolOp)
            (snap_syn / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
            d_h2b = root / "d_h2b"
            d_h2b.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h2b:
                cp.resolve_experiment_checkpoint(
                    d_h2b,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=True,
                    env={"HF_HOME": str(hf_syn)},
                )
            self.assertEqual(cm_h2b.exception.code, "SYNTHETIC_FIXTURE_REJECTED")

            # Synthetic cache with altered bytes but synthetic tokenizer.json (tests tokenizer half of cache BoolOp)
            hf_syn_tok = root / "hf_syn_tok"
            snap_syn_tok = hf_syn_tok / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots" / "rev1"
            _write_minimal_non_synthetic_ckpt(snap_syn_tok, b"ALTERED_CACHE_BYTES")
            (snap_syn_tok / "tokenizer.json").write_text('{"version":"Synthetic"}\n', encoding="utf-8")
            d_h2c = root / "d_h2c"
            d_h2c.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h2c:
                cp.resolve_experiment_checkpoint(
                    d_h2c,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=True,
                    env={"HF_HOME": str(hf_syn_tok)},
                )
            self.assertEqual(cm_h2c.exception.code, "SYNTHETIC_FIXTURE_REJECTED")

            # Non-synthetic HF cache: unverified SHA, caller-pinned SHA (NOT verified_pretrained!), known TinyStories SHA
            hf_unv = root / "hf_unv"
            snap_unv = hf_unv / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots" / "rev1"
            unv_sha = _write_minimal_non_synthetic_ckpt(snap_unv, b"UNVERIFIED_CACHE_WEIGHTS")
            d_h3 = root / "d_h3"
            d_h3.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h3:
                cp.resolve_experiment_checkpoint(
                    d_h3,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=True,
                    env={"HF_HOME": str(hf_unv)},
                )
            self.assertEqual(cm_h3.exception.code, "UNVERIFIED_CACHE_SHA256")

            d_h4 = root / "d_h4"
            d_h4.mkdir()
            prov_h4 = cp.resolve_experiment_checkpoint(
                d_h4,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                require_real_weights=False,
                env={"HF_HOME": str(hf_unv)},
            )
            self.assertFalse(prov_h4.is_synthetic_fallback)
            self.assertFalse(prov_h4.real_weights_verified)
            self.assertEqual(prov_h4.pin_source, "none")
            self.assertEqual(prov_h4.provenance_tier, "unverified_cache_safetensors")

            # Self-pinning an altered/unknown cache blob MUST NOT emit real_weights_verified=True
            # or verified_pretrained_safetensors (D2!)
            d_h5 = root / "d_h5"
            d_h5.mkdir()
            prov_h5 = cp.resolve_experiment_checkpoint(
                d_h5,
                build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                require_real_weights=True,
                expected_safetensors_sha256=unv_sha,
                env={"HF_HOME": str(hf_unv)},
            )
            self.assertFalse(prov_h5.real_weights_verified)
            self.assertFalse(prov_h5.matches_known_tinystories_1m)
            self.assertEqual(prov_h5.pin_source, "caller_supplied_pin")
            self.assertEqual(prov_h5.provenance_tier, "caller_pinned_cache_safetensors")
            self.assertEqual(prov_h5.weights_evidence_claim, "caller_supplied_sha256_pin_not_known_pretrained")

            # Known TinyStories-1M SHA in HF cache DOES emit verified_pretrained_safetensors + known_pretrained_pin
            d_h5_ts = root / "d_h5_ts"
            d_h5_ts.mkdir()
            with mock.patch.object(cp, "TINYSTORIES_1M_SAFETENSORS_SHA256", unv_sha):
                prov_h5_ts = cp.resolve_experiment_checkpoint(
                    d_h5_ts,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=True,
                    env={"HF_HOME": str(hf_unv)},
                )
            self.assertTrue(prov_h5_ts.real_weights_verified)
            self.assertTrue(prov_h5_ts.matches_known_tinystories_1m)
            self.assertEqual(prov_h5_ts.pin_source, "known_pretrained_pin")
            self.assertEqual(prov_h5_ts.provenance_tier, "verified_pretrained_safetensors")
            self.assertEqual(prov_h5_ts.weights_evidence_claim, "verified_tinystories_1m_refs_pr_12")

            d_h6 = root / "d_h6"
            d_h6.mkdir()
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h6:
                cp.resolve_experiment_checkpoint(
                    d_h6,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=False,
                    expected_safetensors_sha256="2" * 64,
                    env={"HF_HOME": str(hf_unv)},
                )
            self.assertEqual(cm_h6.exception.code, "SHA256_MISMATCH")

            # Pre-existing dest_dir collision in HF cache path wraps cleanly into STAGING_DIR_ERROR
            d_h_exists = root / "d_h_exists"
            d_h_exists.mkdir()
            (d_h_exists / "model.safetensors").write_bytes(b"preexisting")
            with self.assertRaises(cp.RealWeightsRequiredError) as cm_h_ex:
                cp.resolve_experiment_checkpoint(
                    d_h_exists,
                    build_synthetic_fn=e3._build_synthetic_gptneo_dir,
                    require_real_weights=False,
                    env={"HF_HOME": str(hf_unv)},
                )
            self.assertEqual(cm_h_ex.exception.code, "STAGING_DIR_ERROR")


class TestExperimentsIssue6Integration(unittest.TestCase):
    def test_e3_e4_e5_fail_closed_under_require_real_weights_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty_env = {"HF_HOME": td}
            with mock.patch.dict(os.environ, empty_env, clear=False):
                os.environ.pop(cp.ENV_REQUIRE_REAL_WEIGHTS, None)
                # All three experiments fail closed when require_real_weights=True and no cache exists
                for mod in (e3, e4, e5):
                    with self.assertRaises(cp.RealWeightsRequiredError) as cm:
                        mod.run_experiment(require_real_weights=True)
                    self.assertEqual(cm.exception.code, "REAL_WEIGHTS_REQUIRED")

                    err_buf = io.StringIO()
                    with redirect_stderr(err_buf):
                        rc = mod.main(["--require-real-weights"])
                    self.assertEqual(rc, 2)
                    self.assertIn("REAL_WEIGHTS_REQUIRED", err_buf.getvalue())

                # WEFT_REQUIRE_REAL_WEIGHTS=1 also fails closed with default run_experiment()
                with mock.patch.dict(os.environ, {cp.ENV_REQUIRE_REAL_WEIGHTS: "1"}, clear=False):
                    for mod in (e3, e4, e5):
                        with self.assertRaises(cp.RealWeightsRequiredError) as cm:
                            mod.run_experiment()
                        self.assertEqual(cm.exception.code, "REAL_WEIGHTS_REQUIRED")

                # --require-real-weights with unpinned --checkpoint-dir fails closed via CLI (rc=2)
                unpinned_cli = Path(td) / "unpinned_cli"
                _write_minimal_non_synthetic_ckpt(unpinned_cli, b"UNPINNED_CLI_WEIGHTS")
                err_unp = io.StringIO()
                with redirect_stderr(err_unp):
                    rc_unp = e3.main(["--require-real-weights", "--checkpoint-dir", str(unpinned_cli)])
                self.assertEqual(rc_unp, 2)
                self.assertIn("UNVERIFIED_CHECKPOINT_SHA256", err_unp.getvalue())

                # Offline default E3 succeeds AND explicitly marks synthetic fallback provenance
                out_buf = io.StringIO()
                with redirect_stdout(out_buf):
                    rc_ok = e3.main([])
                self.assertEqual(rc_ok, 0)
                parsed = json.loads(out_buf.getvalue())
                self.assertTrue(parsed["is_synthetic_fallback"])
                self.assertFalse(parsed["real_weights_verified"])
                self.assertEqual(parsed["checkpoint_source"], cp.SYNTHETIC_FALLBACK_LABEL)
                self.assertEqual(parsed["weights_provenance"]["pin_source"], "none")
                self.assertEqual(parsed["weights_provenance"]["safetensors_sha256"], cp.SYNTHETIC_SAFETENSORS_SHA256)
                self.assertEqual(parsed["weights_provenance"]["cache_miss_reason"], "no_tinystories_cache")


if __name__ == "__main__":
    unittest.main()
