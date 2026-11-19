"""Tests for deterministic GGUF v3 F16 LoRA binary codec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weft.gguf_lora import (
    GGUF_MAGIC,
    build_llama_server_lora_flags,
    decode_gguf_lora,
    read_gguf_lora,
    read_gguf_model_geometry,
    write_gguf_lora,
)


def test_gguf_lora_roundtrip_and_zero_byte_determinism(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261119)
    b0 = rng.standard_normal((64, 4)) * 0.1
    a0 = rng.standard_normal((4, 128)) * 0.1
    b1 = rng.standard_normal((32, 2)) * 0.2
    a1 = rng.standard_normal((2, 64)) * 0.2

    factors = {
        "blk.7.ffn_down.weight": (b0, a0),
        "blk.3.ffn_down.weight": (b1, a1),
    }
    meta = {
        "weft.adapter_class": "class1_rederivable",
        "weft.binding_sha256": "a" * 64,
        "weft.certificate_sha256": "b" * 64,
    }

    p1 = tmp_path / "run1.gguf"
    p2 = tmp_path / "run2.gguf"
    _, h1 = write_gguf_lora(
        p1,
        architecture="gptneo",
        lora_alpha=8.0,
        layer_factors=factors,
        extra_metadata=meta,
    )
    # Reverse insertion order to prove canonical key/tensor sorting
    _, h2 = write_gguf_lora(
        p2,
        architecture="gptneo",
        lora_alpha=8.0,
        layer_factors=dict(reversed(list(factors.items()))),
        extra_metadata=dict(reversed(list(meta.items()))),
    )

    assert h1 == h2
    raw = p1.read_bytes()
    assert raw == p2.read_bytes()
    assert raw[:4] == GGUF_MAGIC

    loaded = read_gguf_lora(p1)
    assert loaded.architecture == "gptneo"
    assert loaded.lora_alpha == pytest.approx(8.0)
    assert loaded.metadata["general.type"] == "adapter"
    assert loaded.metadata["adapter.type"] == "lora"
    assert loaded.metadata["weft.binding_sha256"] == "a" * 64
    assert loaded.tensors["blk.7.ffn_down.weight.lora_a"].shape == (4, 128)
    assert loaded.tensors["blk.7.ffn_down.weight.lora_b"].shape == (64, 4)
    assert loaded.tensors["blk.7.ffn_down.weight.lora_a"].dtype == np.float16
    assert np.array_equal(
        loaded.tensors["blk.7.ffn_down.weight.lora_a"], a0.astype(np.float16)
    )
    assert np.array_equal(
        loaded.tensors["blk.7.ffn_down.weight.lora_b"], b0.astype(np.float16)
    )


def test_gguf_lora_rejects_bad_magic_and_truncation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid GGUF magic"):
        decode_gguf_lora(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="Truncated"):
        decode_gguf_lora(b"GGUF\x03\x00\x00\x00")


def test_gguf_lora_rejects_crafted_huge_ndim_and_truncated_kv() -> None:
    import struct

    from weft.gguf_lora import GGUF_TYPE_STRING, GGUF_TYPE_UINT32, _pack_kv, _pack_string

    # Crafted ndim = 0xFFFFFFFF must fail cleanly with ValueError before allocating
    header = struct.pack("<4sIQQ", GGUF_MAGIC, 3, 1, 2)
    kvs = _pack_kv("general.type", "adapter") + _pack_kv("adapter.type", "lora")
    bad_tensor = _pack_string("blk.0.lora_a") + struct.pack("<I", 0xFFFFFFFF)
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        decode_gguf_lora(header + kvs + bad_tensor)

    # Truncated UINT32 KV payload must raise ValueError (not struct.error)
    hdr_kv = struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1)
    trunc_kv = _pack_string("general.alignment") + struct.pack("<I", GGUF_TYPE_UINT32) + b"\x20"
    with pytest.raises(ValueError, match="Truncated GGUF UINT32"):
        decode_gguf_lora(hdr_kv + trunc_kv)

    # Crafted n_kv > buffer size fails closed immediately
    huge_kv_hdr = struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 10_000_000)
    with pytest.raises(ValueError, match="exceeds buffer size"):
        decode_gguf_lora(huge_kv_hdr + b"\x00" * 16)
    assert GGUF_TYPE_STRING == 8


def test_read_gguf_model_geometry_and_llama_server_flags(tmp_path: Path) -> None:
    import struct

    from weft.gguf_lora import _pack_kv, _pack_string

    # Build minimal synthetic GGUF v3 model header with tokenizer array + non-square FFN tensors
    tok_arr = _pack_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
    tok_arr += _pack_string("a") + _pack_string("b")
    kvs = (
        _pack_kv("general.architecture", "qwen2")
        + _pack_kv("general.name", "Synthetic-1.5B")
        + _pack_kv("qwen2.block_count", 28)
        + _pack_kv("qwen2.embedding_length", 1536)
        + _pack_kv("qwen2.feed_forward_length", 8960)
        + tok_arr
    )
    tinfo = (
        _pack_string("blk.27.ffn_up.weight")
        + struct.pack("<IQQIQ", 2, 1536, 8960, 12, 0)
        + _pack_string("blk.27.ffn_down.weight")
        + struct.pack("<IQQIQ", 2, 8960, 1536, 14, 1024)
    )
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 2, 6) + kvs + tinfo)

    geom = read_gguf_model_geometry(model_path, tensor_names=["blk.27.ffn_up.weight"])
    assert geom["architecture"] == "qwen2"
    assert geom["model_name"] == "Synthetic-1.5B"
    assert geom["block_count"] == 28
    assert geom["embedding_length"] == 1536
    assert geom["feed_forward_length"] == 8960
    assert list(geom["tensors"].keys()) == ["blk.27.ffn_up.weight"]
    assert geom["tensors"]["blk.27.ffn_up.weight"]["ggml_ne"] == (1536, 8960)
    assert geom["tensors"]["blk.27.ffn_up.weight"]["numpy_shape"] == (8960, 1536)

    # Crafted huge array count fails closed immediately
    bad_path = tmp_path / "bad_arr.gguf"
    bad_arr = _pack_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 99_999_999)
    bad_path.write_bytes(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 0, 1) + bad_arr)
    with pytest.raises(ValueError, match="safety cap"):
        read_gguf_model_geometry(bad_path)

    # Verified fail-closed llama-server CLI flags
    flags = build_llama_server_lora_flags(
        [tmp_path / "a.gguf", tmp_path / "b.gguf"],
        initial_scale=0.0,
        slot_save_path=tmp_path / "slots",
    )
    assert "--lora-scaled" in flags
    scaled_val = flags[flags.index("--lora-scaled") + 1]
    assert scaled_val.endswith("b.gguf:0.0") and "," in scaled_val
    assert "--lora-init-without-apply" in flags
    assert "--slot-save-path" in flags
    assert "--no-cache-idle-slots" in flags
    with pytest.raises(ValueError, match="non-empty"):
        build_llama_server_lora_flags([])
