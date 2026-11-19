"""Tests for deterministic GGUF v3 F16 LoRA binary codec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weft.gguf_lora import GGUF_MAGIC, decode_gguf_lora, read_gguf_lora, write_gguf_lora


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
