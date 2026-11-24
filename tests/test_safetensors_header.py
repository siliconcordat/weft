"""Strict safetensors header reader: one test per defect, each isolated.

Every defect test builds a file that is valid except for that one defect, and
first checks that the matching defect-free file is accepted. Checks use
``pytest.raises`` and explicit ``raise`` so they still run under ``python -O``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from weft.binding import (
    BINDING_SCHEMA,
    RuntimeFlags,
    SafetensorsHeaderError,
    build_binding_manifest,
    identify_checkpoint,
    read_safetensors_header,
    verify_binding,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _entry(shape: object, lo: object, hi: object, dtype: object = "U8") -> str:
    return json.dumps({"dtype": dtype, "shape": shape, "data_offsets": [lo, hi]}, separators=(",", ":"))


def _write(path: Path, header_text: str, data_len: int) -> Path:
    raw = header_text.encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(range(256))[:data_len].ljust(data_len, b"\0"))
    return path


def _accepts(tmp_path: Path, header_text: str, data_len: int) -> dict:
    header, offset = read_safetensors_header(_write(tmp_path / "ok.safetensors", header_text, data_len))
    _require(offset == 8 + len(header_text.encode("utf-8")), "data offset must follow the header")
    return header


def _rejects(tmp_path: Path, header_text: str, data_len: int, code: str) -> None:
    path = _write(tmp_path / "bad.safetensors", header_text, data_len)
    with pytest.raises(SafetensorsHeaderError) as info:
        read_safetensors_header(path)
    _require(info.value.code == code, f"expected {code}, got {info.value.code}")
    _require(str(info.value).startswith(code + ":"), "message must start with the code")
    _require(isinstance(info.value, ValueError), "error must stay a ValueError")


TWO = '{"a":' + _entry([8], 0, 8) + ',"b":' + _entry([8], 8, 16) + "}"


def test_valid_file_known_answer(tmp_path: Path) -> None:
    header = _accepts(tmp_path, TWO, 16)
    _require(
        header
        == {
            "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
            "b": {"dtype": "U8", "shape": [8], "data_offsets": [8, 16]},
        },
        "valid header must come back unchanged",
    )
    ck = tmp_path / "ck"
    ck.mkdir()
    _write(ck / "m.safetensors", TWO, 16)
    ident = identify_checkpoint(ck)
    by_hand = hashlib.sha256(b"a\tU8\t8\nb\tU8\t8\n").hexdigest()
    _require(ident.layout_sha256 == by_hand, "layout hash must match the hand-built text")
    _require(
        ident.layout_sha256 == "06446c38a5c67c988c166f12a42bb234e12190fb5cde6663cbd13d5984627277",
        "layout hash must match the pinned known answer",
    )
    _require(ident.n_tensors == 2, "two tensors")


def test_empty_tensors_allowed(tmp_path: Path) -> None:
    text = (
        '{"e0":' + _entry([0], 0, 0) + ',"a":' + _entry([8], 0, 8)
        + ',"e1":' + _entry([0, 3], 8, 8) + ',"b":' + _entry([8], 8, 16) + "}"
    )
    _require(len(_accepts(tmp_path, text, 16)) == 4, "empty tensors must be accepted")
    only_empty = '{"e":' + _entry([0], 0, 0) + "}"
    _require("e" in _accepts(tmp_path, only_empty, 0), "a file of one empty tensor is valid")


def test_metadata_string_values_allowed(tmp_path: Path) -> None:
    text = '{"__metadata__":{"format":"pt","x":"1"},"a":' + _entry([8], 0, 8) + "}"
    header = _accepts(tmp_path, text, 8)
    _require(header["__metadata__"] == {"format": "pt", "x": "1"}, "metadata kept as is")


def test_metadata_non_string_value_rejected(tmp_path: Path) -> None:
    ok = '{"__metadata__":{"x":"1"},"a":' + _entry([8], 0, 8) + "}"
    bad = '{"__metadata__":{"x":1},"a":' + _entry([8], 0, 8) + "}"
    _accepts(tmp_path, ok, 8)
    _rejects(tmp_path, bad, 8, "BAD_METADATA")


def test_duplicate_tensor_key_rejected(tmp_path: Path) -> None:
    # Both copies are identical, so a last-wins reader finds a valid file.
    one = '{"a":' + _entry([8], 0, 8) + ',"b":' + _entry([8], 8, 16) + "}"
    dup = '{"a":' + _entry([8], 0, 8) + ',"a":' + _entry([8], 0, 8) + ',"b":' + _entry([8], 8, 16) + "}"
    _require(json.loads(dup) == json.loads(one), "last-wins view must equal the clean file")
    _accepts(tmp_path, one, 16)
    _rejects(tmp_path, dup, 16, "DUPLICATE_KEY")


def test_duplicate_key_spelled_with_escape_rejected(tmp_path: Path) -> None:
    clean = '{"a":' + _entry([8], 0, 8) + "}"
    dup = '{"a":' + _entry([8], 0, 8) + ',"\\u0061":' + _entry([8], 0, 8) + "}"
    _accepts(tmp_path, clean, 8)
    _rejects(tmp_path, dup, 8, "DUPLICATE_KEY")


def test_duplicate_key_inside_entry_rejected(tmp_path: Path) -> None:
    clean = '{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}'
    dup = '{"a":{"dtype":"U8","shape":[8],"shape":[8],"data_offsets":[0,8]}}'
    _accepts(tmp_path, clean, 8)
    _rejects(tmp_path, dup, 8, "DUPLICATE_KEY")


@pytest.mark.parametrize(
    "lo_b, hi_b, data_len",
    [(4, 12, 12), (0, 8, 8), (7, 15, 15)],
    ids=["partial", "identical", "one_byte"],
)
def test_overlapping_spans_rejected(tmp_path: Path, lo_b: int, hi_b: int, data_len: int) -> None:
    _accepts(tmp_path, '{"a":' + _entry([8], 0, 8) + ',"b":' + _entry([8], 8, 16) + "}", 16)
    text = '{"a":' + _entry([8], 0, 8) + ',"b":' + _entry([hi_b - lo_b], lo_b, hi_b) + "}"
    _rejects(tmp_path, text, data_len, "OVERLAPPING_SPANS")


def test_gap_between_spans_rejected(tmp_path: Path) -> None:
    _accepts(tmp_path, TWO, 16)
    text = '{"a":' + _entry([8], 0, 8) + ',"b":' + _entry([8], 12, 20) + "}"
    _rejects(tmp_path, text, 20, "GAP_BETWEEN_SPANS")


def test_leading_gap_rejected(tmp_path: Path) -> None:
    _accepts(tmp_path, '{"a":' + _entry([8], 0, 8) + "}", 8)
    _rejects(tmp_path, '{"a":' + _entry([8], 4, 12) + "}", 12, "GAP_BETWEEN_SPANS")


def test_trailing_bytes_rejected(tmp_path: Path) -> None:
    text = '{"a":' + _entry([8], 0, 8) + "}"
    _accepts(tmp_path, text, 8)
    _rejects(tmp_path, text, 16, "GAP_BETWEEN_SPANS")


def test_spans_past_end_of_file_rejected(tmp_path: Path) -> None:
    _accepts(tmp_path, TWO, 16)
    _rejects(tmp_path, TWO, 12, "TRUNCATED_DATA")


@pytest.mark.parametrize(
    "shape",
    [[2.5], [2.0], ["2"], [True], [-1], [None], "8", 8],
    ids=["float", "whole_float", "string", "bool", "negative", "null", "str_shape", "int_shape"],
)
def test_bad_shape_rejected(tmp_path: Path, shape: object) -> None:
    _accepts(tmp_path, '{"a":' + _entry([8], 0, 8) + "}", 8)
    _rejects(tmp_path, '{"a":' + _entry(shape, 0, 8) + "}", 8, "BAD_SHAPE")


@pytest.mark.parametrize(
    "offsets",
    [[0.0, 8], [8, 0], [0], [0, 8, 8], "08", [-8, 0]],
    ids=["float", "reversed", "short", "long", "string", "negative"],
)
def test_bad_offsets_rejected(tmp_path: Path, offsets: object) -> None:
    _accepts(tmp_path, '{"a":' + _entry([8], 0, 8) + "}", 8)
    text = '{"a":{"dtype":"U8","shape":[8],"data_offsets":' + json.dumps(offsets) + "}}"
    _rejects(tmp_path, text, 8, "BAD_OFFSETS")


def test_bad_entry_rejected(tmp_path: Path) -> None:
    _accepts(tmp_path, '{"a":' + _entry([8], 0, 8) + "}", 8)
    _rejects(tmp_path, '{"a":' + _entry([8], 0, 8, dtype=7) + "}", 8, "BAD_ENTRY")
    _rejects(tmp_path, '{"a":[1,2]}', 8, "BAD_ENTRY")


def test_dtype_and_shape_must_fill_the_span(tmp_path: Path) -> None:
    # F32 is 4 bytes. Two elements fill 8 bytes. One element in those 8 does not.
    _accepts(tmp_path, '{"a":' + _entry([2], 0, 8, dtype="F32") + "}", 8)
    _rejects(tmp_path, '{"a":' + _entry([1], 0, 8, dtype="F32") + "}", 8, "BYTE_LENGTH")
    # A scalar is shape [] and still has a width.
    _accepts(tmp_path, '{"a":' + _entry([], 0, 4, dtype="F32") + "}", 4)
    # F4 is 4 bits. Two elements fill one byte. One element does not fill a byte.
    _accepts(tmp_path, '{"a":' + _entry([2], 0, 1, dtype="F4") + "}", 1)
    _rejects(tmp_path, '{"a":' + _entry([1], 0, 1, dtype="F4") + "}", 1, "MISALIGNED")
    _rejects(tmp_path, '{"a":' + _entry([8], 0, 8, dtype="F33") + "}", 8, "BAD_DTYPE")
    # A zero dimension is empty even when another dimension is large.
    _accepts(tmp_path, '{"a":' + _entry([0, 10**9], 0, 0, dtype="F64") + "}", 0)


@pytest.mark.parametrize(
    "shape",
    [
        [4294967296, 4294967296, 0],
        [4294967296, 0, 4294967296],
        [0, 4294967296, 4294967296],
        [4294967296, 4294967296, 1, 0],
    ],
)
def test_overflowing_product_with_zero_dimension_rejected(tmp_path: Path, shape: list[int]) -> None:
    # The overflow in the non-zero dimensions is not erased by a zero elsewhere.
    _rejects(tmp_path, '{"a":' + _entry(shape, 0, 0, dtype="U8") + "}", 0, "BYTE_SIZE_OVERFLOW")
    _accepts(tmp_path, '{"a":' + _entry([65536, 65536, 0], 0, 0, dtype="U8") + "}", 0)


@pytest.mark.parametrize("shape", [[0, 2**64], [2**64, 0], [0, 2**64 + 5]])
def test_dimension_above_u64_with_zero_dimension_rejected(tmp_path: Path, shape: list[int]) -> None:
    _rejects(tmp_path, '{"a":' + _entry(shape, 0, 0) + "}", 0, "BYTE_SIZE_OVERFLOW")
    _accepts(tmp_path, '{"a":' + _entry([0, 2**32], 0, 0) + "}", 0)


def _with_note(note: str, key: str = "a") -> str:
    return '{"%s":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"note":%s}}' % (key, note)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_constant_in_ignored_field_rejected(tmp_path: Path, constant: str) -> None:
    _accepts(tmp_path, _with_note("1.5"), 8)
    _rejects(tmp_path, _with_note(constant), 8, "BAD_JSON")


@pytest.mark.parametrize(
    "text",
    [
        _with_note("1", key="\\ud800"),
        _with_note('"\\udc00"'),
        _with_note('["\\ud83d"]'),
        _with_note('{"k":"x\\ud800"}'),
        '{"__metadata__":{"k":"\\ud800x"},"a":' + _entry([8], 0, 8) + "}",
    ],
)
def test_unpaired_surrogate_rejected(tmp_path: Path, text: str) -> None:
    _rejects(tmp_path, text, 8, "BAD_JSON")
    # A valid pair is one character and stays accepted.
    _accepts(tmp_path, _with_note('"\\ud83d\\ude00"'), 8)


def test_identify_checkpoint_and_verify_binding_fail_closed(tmp_path: Path) -> None:
    ck = tmp_path / "ck"
    ck.mkdir()
    shard = ck / "m.safetensors"
    _write(shard, '{"a":' + _entry([8], 0, 8) + "}", 8)
    manifest = build_binding_manifest(
        adapter_name="t",
        adapter_class="class2_open_recipe_base",
        checkpoint_dir=ck,
        factor_digest="f" * 64,
        certificate_sha256="c" * 64,
    )
    flags = RuntimeFlags()
    good = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=flags)
    _require(good.ok and manifest.schema == BINDING_SCHEMA, f"control must verify: {good.problems}")
    # Same tensor table, plus 8 bytes that no tensor owns.
    _write(shard, '{"a":' + _entry([8], 0, 8) + "}", 16)
    with pytest.raises(SafetensorsHeaderError):
        identify_checkpoint(ck)
    bad = verify_binding(manifest, checkpoint_dir=ck, runtime_flags=flags)
    _require(not bad.ok, "padded file must not verify")
    _require(
        any(p.startswith("checkpoint_read_error:GAP_BETWEEN_SPANS") for p in bad.problems),
        f"problem must carry the code: {bad.problems}",
    )
