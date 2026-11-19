"""Tests for 5-layer fail-closed checkpoint, Plane 1 lineage, and runtime-flag binding."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from weft.binding import (
    ADAPTER_CLASS_1,
    ADAPTER_CLASS_2,
    RuntimeFlags,
    build_binding_manifest,
    verify_binding,
)


def _write_fake_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in sorted(tensors.items()):
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for blob in blobs:
            f.write(blob)


def _make_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_fake_safetensors(
        ck / "model.safetensors",
        {
            "layers.0.down_proj.weight": np.arange(32, dtype=np.float32).reshape(4, 8),
            "embed_tokens.weight": np.ones((10, 8), dtype=np.float32),
        },
    )
    (ck / "config.json").write_text('{"hidden_size": 8}\n', encoding="utf-8")
    (ck / "tokenizer.json").write_text('{"version": "1.0"}\n', encoding="utf-8")
    (ck / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    p1 = tmp_path / "plane1"
    (p1 / "principles").mkdir(parents=True)
    (p1 / "principles" / "writing.md").write_text("# Writing\nPlain language.\n", encoding="utf-8")
    return ck, p1


def test_five_layer_binding_accepts_clean_and_rejects_every_tamper(tmp_path: Path) -> None:
    ck, p1 = _make_checkpoint(tmp_path)
    flags = RuntimeFlags(
        quantization_mode="bf16",
        mamba_cache_mode="align",
        enable_thinking=False,
        tool_call_parser="qwen3_xml",
        ctx_size=32768,
    )
    manifest = build_binding_manifest(
        adapter_name="corpus_writing",
        adapter_class=ADAPTER_CLASS_1,
        checkpoint_dir=ck,
        plane1_root=p1,
        plane1_files=["principles/writing.md"],
        derivation_recipe={"compiler": "weft/0.1.0", "seed": 20261119},
        runtime_flags=flags,
        factor_digest="1" * 64,
        certificate_sha256="2" * 64,
    )

    # Clean verify passes
    res = verify_binding(
        manifest,
        checkpoint_dir=ck,
        runtime_flags=flags,
        plane1_root=p1,
        expected_factor_digest="1" * 64,
        expected_certificate_sha256="2" * 64,
    )
    assert res.ok and res.problems == []

    # 1. Runtime flag drift (Layer 5)
    bad_flags = RuntimeFlags(
        quantization_mode="bf16",
        mamba_cache_mode="none",
        enable_thinking=True,
        tool_call_parser="qwen3_xml",
        ctx_size=32768,
    )
    res_flags = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=bad_flags, plane1_root=p1)
    assert not res_flags.ok
    assert "runtime_flag_drift" in res_flags.problems

    # 2. 1-byte Plane 1 source edit -> stale_lineage (Layer 4)
    (p1 / "principles" / "writing.md").write_text("# Writing\nPlain language!\n", encoding="utf-8")
    res_p1 = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=flags, plane1_root=p1)
    assert not res_p1.ok
    assert "stale_lineage" in res_p1.problems
    (p1 / "principles" / "writing.md").write_text("# Writing\nPlain language.\n", encoding="utf-8")

    # 3. 1-byte chat_template.jinja edit (Layer 3)
    (ck / "chat_template.jinja").write_text("{{ messages }} ", encoding="utf-8")
    res_tpl = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=flags, plane1_root=p1)
    assert not res_tpl.ok
    assert "tokenizer_or_chat_template_drift" in res_tpl.problems
    (ck / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    # 4. 1-value weight tamper in base checkpoint (Layer 2)
    tampered = np.arange(32, dtype=np.float32).reshape(4, 8)
    tampered[0, 0] = 999.0
    _write_fake_safetensors(
        ck / "model.safetensors",
        {
            "layers.0.down_proj.weight": tampered,
            "embed_tokens.weight": np.ones((10, 8), dtype=np.float32),
        },
    )
    res_w = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=flags, plane1_root=p1)
    assert not res_w.ok
    assert "checkpoint_fingerprint_mismatch" in res_w.problems


def test_class1_rejects_opaque_missing_derivation_and_path_escape(tmp_path: Path) -> None:
    ck, p1 = _make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Hard Veto 1"):
        build_binding_manifest(
            adapter_name="opaque",
            adapter_class=ADAPTER_CLASS_1,
            checkpoint_dir=ck,
            plane1_root=p1,
            plane1_files=["principles/writing.md"],
            derivation_recipe=None,
            factor_digest="1" * 64,
            certificate_sha256="2" * 64,
        )

    # Path confinement blocks traversal
    with pytest.raises(ValueError, match="Invalid non-canonical"):
        build_binding_manifest(
            adapter_name="escape",
            adapter_class=ADAPTER_CLASS_1,
            checkpoint_dir=ck,
            plane1_root=p1,
            plane1_files=["../ckpt/config.json"],
            derivation_recipe={"compiler": "weft/0.1.0"},
            factor_digest="1" * 64,
            certificate_sha256="2" * 64,
        )

    # Class 2 base checkpoint may omit deterministic Class 1 Plane 1 compilation recipe
    c2 = build_binding_manifest(
        adapter_name="pi_base_lora",
        adapter_class=ADAPTER_CLASS_2,
        checkpoint_dir=ck,
        derivation_recipe={"recipe": "open_async_grpo"},
        factor_digest="3" * 64,
        certificate_sha256="4" * 64,
    )
    ok_res = verify_binding(c2, checkpoint_dir=ck, runtime_flags=RuntimeFlags())
    assert ok_res.ok

    # Internal manifest.runtime_flags tampered without updating runtime_flags_sha256
    from dataclasses import replace

    tampered_manifest = replace(
        c2,
        runtime_flags={**c2.runtime_flags, "enable_thinking": True},
    )
    bad_internal = verify_binding(
        tampered_manifest, checkpoint_dir=ck, runtime_flags=RuntimeFlags()
    )
    assert not bad_internal.ok
    assert "manifest_runtime_flags_digest_mismatch" in bad_internal.problems
