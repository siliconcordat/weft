"""Tests for canonical `weft-slab/v0.1` packed asymmetric slab wire container."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector
from weft.gguf_lora import write_gguf_lora
from weft.slab_wire import (
    WEFT_SLAB_FORMAT,
    WEFT_SLAB_MAGIC,
    WEFT_SLAB_VERSION,
    read_slab_wire,
    verify_slab_wire,
    write_slab_wire,
)

FAKE_BASE_SHA = "a" * 64
FAKE_ZONE_ID = "sha256:" + ("b" * 64)


def _make_patch(b_dtype: str, a_dtype: str, d_out: int = 64, d_in: int = 48, rank: int = 4):
    rng = np.random.default_rng(20261119)
    w0 = rng.standard_normal((d_out, d_in)) * 0.1
    k0 = rng.standard_normal((d_in, 12))
    keys = rng.standard_normal((d_in, rank))
    res = rng.standard_normal((d_out, rank)) * 0.5
    cfg = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=rank,
        lambda_leakage=500.0,
        lambda_ridge=1.0,
        nullspace_threshold=1e-6,
        threshold_mode="absolute",
        quant_dtype="f16",
        b_quant_dtype=b_dtype,  # type: ignore[arg-type]
        a_quant_dtype=a_dtype,  # type: ignore[arg-type]
    )
    proj = BetaEditNullspaceProjector(w0, cfg)
    proj.initialize_anchor_covariance(k0)
    return proj.stage_edit(keys, res, is_residual=True, rank=rank, slot_name="corpus")


@pytest.mark.parametrize(
    ("b_dt", "a_dt"),
    [
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("int8", "int8"),
        ("int4", "int8"),
        ("int4", "f16"),
        ("int4", "int4"),
    ],
)
def test_slab_wire_roundtrip_and_zero_byte_determinism(
    tmp_path: Path, b_dt: str, a_dt: str
) -> None:
    patch = _make_patch(b_dt, a_dt)
    out_file = tmp_path / f"adapter_{b_dt}_{a_dt}.wfsl"

    blob_a = write_slab_wire(
        patch,
        target_tensor="transformer.h.7.mlp.c_proj.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
        output_path=out_file,
    )
    blob_b = write_slab_wire(
        patch,
        target_tensor="transformer.h.7.mlp.c_proj.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
    )
    assert blob_a == blob_b
    assert out_file.read_bytes() == blob_a

    decoded = read_slab_wire(out_file)
    assert decoded.format_version == WEFT_SLAB_FORMAT
    assert decoded.b_dtype == b_dt
    assert decoded.a_dtype == a_dt
    assert decoded.b_scale_axis == 0
    assert decoded.a_scale_axis == 1
    assert decoded.slot_name == "corpus"
    assert decoded.zone_manifest_id == FAKE_ZONE_ID
    assert decoded.base_weights_sha256 == FAKE_BASE_SHA
    assert decoded.certificate == patch.certificate
    assert decoded.factor_digest == patch.factor_digest
    assert np.array_equal(decoded.factor_b, patch.factor_b)
    assert np.array_equal(decoded.factor_a, patch.factor_a)


def test_slab_wire_rejects_tamper_truncation_trailing_and_duplicate_keys() -> None:
    patch = _make_patch("int4", "int8")
    blob = write_slab_wire(
        patch,
        target_tensor="transformer.h.7.mlp.c_proj.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
    )

    # 1-bit flip in header, payload, and trailer all fail closed
    for offset in (20, len(blob) - 40, len(blob) - 1):
        mut = bytearray(blob)
        mut[offset] ^= 0x01
        with pytest.raises(ValueError, match="SHA-256 trailer mismatch"):
            read_slab_wire(bytes(mut))

    # Truncation and trailing garbage
    with pytest.raises(ValueError, match="length mismatch"):
        read_slab_wire(blob[:-1])
    with pytest.raises(ValueError, match="length mismatch"):
        read_slab_wire(blob + b"\x00")
    with pytest.raises(ValueError, match="truncated"):
        read_slab_wire(b"WFSL")

    # Bad magic and version
    bad_magic = b"NOPE" + blob[4:]
    with pytest.raises(ValueError, match="Bad slab magic"):
        read_slab_wire(bad_magic)
    bad_ver = WEFT_SLAB_MAGIC + struct.pack("<I", 99) + blob[8:]
    with pytest.raises(ValueError, match="Unsupported slab version"):
        read_slab_wire(bad_ver)

    # Forged valid-trailer container with duplicate JSON key in header
    dup_hdr = b'{"format":"weft-slab/v0.1","format":"weft-slab/v0.1"}'
    forged_payload = b"\x00" * 16
    forged_body = (
        struct.pack("<4sIII", WEFT_SLAB_MAGIC, WEFT_SLAB_VERSION, len(dup_hdr), len(forged_payload))
        + dup_hdr
        + forged_payload
    )
    forged = forged_body + hashlib.sha256(forged_body).digest()
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_slab_wire(forged)


def test_verify_slab_wire_enforces_zone_pin_base_hash_and_policy() -> None:
    patch = _make_patch("int4", "int8")
    blob = write_slab_wire(
        patch,
        target_tensor="transformer.h.7.mlp.c_proj.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
    )

    ok, probs, cont = verify_slab_wire(
        blob,
        gate_policy=GatePolicy(max_frob_sq_num=100, max_frob_sq_den=1),
        expected_zone_manifest_id=FAKE_ZONE_ID,
        expected_base_weights_sha256=FAKE_BASE_SHA,
    )
    assert ok and probs == [] and cont is not None

    # 1-char zone manifest rotation triggers zone_manifest_drift
    ok_z, probs_z, _ = verify_slab_wire(
        blob,
        expected_zone_manifest_id="sha256:" + ("c" * 64),
    )
    assert not ok_z and "zone_manifest_drift" in probs_z

    # Base checkpoint upgrade triggers base_weights_sha256_mismatch
    ok_b, probs_b, _ = verify_slab_wire(
        blob,
        expected_base_weights_sha256="d" * 64,
    )
    assert not ok_b and "base_weights_sha256_mismatch" in probs_b

    # Tight GatePolicy rejects before mount
    ok_p, probs_p, _ = verify_slab_wire(
        blob,
        gate_policy=GatePolicy(max_frob_sq_num=1, max_frob_sq_den=10000),
    )
    assert not ok_p and "frobenius_norm_exceeds_cap" in probs_p


def test_slab_wire_size_beats_f16_gguf_lora_on_wide_output_layer(tmp_path: Path) -> None:
    # Wide-output layer matching TinyStories mlp.c_fc (256 x 64, rank 4)
    patch = _make_patch("int4", "int8", d_out=256, d_in=64, rank=4)
    slab_bytes = write_slab_wire(
        patch,
        target_tensor="blk.7.ffn_up.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
    )
    gguf_path = tmp_path / "adapter.gguf"
    write_gguf_lora(
        gguf_path,
        layer_factors={"blk.7.ffn_up.weight": (patch.factor_b, patch.factor_a)},
        architecture="gpt_neo",
        lora_alpha=4.0,
    )
    gguf_size = gguf_path.stat().st_size
    # Even at tiny 256x64 rank-4 with embedded certificate & SHA-256 trailer, slab saves >40%
    assert len(slab_bytes) < gguf_size * 0.60

    # At larger wide-output dimension (1024 x 256, rank 8), slab_wire saves >64% vs F16 GGUF LoRA
    patch_lg = _make_patch("int4", "int8", d_out=1024, d_in=256, rank=8)
    slab_lg = write_slab_wire(
        patch_lg,
        target_tensor="blk.7.ffn_up.weight",
        base_weights_sha256=FAKE_BASE_SHA,
        zone_manifest_id=FAKE_ZONE_ID,
    )
    gguf_lg = tmp_path / "adapter_lg.gguf"
    write_gguf_lora(
        gguf_lg,
        layer_factors={"blk.7.ffn_up.weight": (patch_lg.factor_b, patch_lg.factor_a)},
        architecture="qwen2",
        lora_alpha=8.0,
    )
    assert len(slab_lg) < gguf_lg.stat().st_size * 0.36
