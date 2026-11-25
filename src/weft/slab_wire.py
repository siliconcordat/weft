"""Canonical packed asymmetric slab wire container (`weft-slab/v0.1`).

Binary layout (little-endian, deterministic 0-byte re-derivable):
  [0..4)    magic:            b"WFSL"
  [4..8)    version:          uint32 = 1
  [8..12)   header_json_len:  uint32 (<= 1 MiB)
  [12..16)  payload_len:      uint32 (<= 256 MiB, exact b_byte_len + a_byte_len)
  [16..H)   header_json:      canonical UTF-8 JSON (sort_keys=True, compact)
                              carrying dtypes, shapes, per-rank scale axes,
                              embedded FixedRadixCertificate, factor digest,
                              base_weights_sha256, and L5 zone_manifest_id pin
  [H..P)    payload:          b_bytes (axis=0 scales) || a_bytes (axis=1 scales)
  [P..P+32) trailer_sha256:   raw 32-byte SHA-256 over bytes [0..P)
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .certificate import FixedRadixCertificate, GatePolicy
from .compiler import StagedPatch
from .quant import (
    SUPPORTED_QUANT_DTYPES,
    dequantize_array,
    sha256_factor_pair,
)

WEFT_SLAB_MAGIC = b"WFSL"
WEFT_SLAB_VERSION = 1
WEFT_SLAB_FORMAT = "weft-slab/v0.1"
FIXED_HEADER_SIZE = 16
TRAILER_SIZE = 32
MAX_HEADER_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ZONE_MANIFEST_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _strict_json_loads(raw_text: str) -> Any:
    """Parse JSON while rejecting duplicate keys inside any object."""

    def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key in slab header: {key!r}")
            out[key] = val
        return out

    return json.loads(raw_text, object_pairs_hook=_no_dupes)


def _require_strict_pos_int(name: str, val: Any) -> int:
    if type(val) is not int or val <= 0:
        raise ValueError(f"{name} must be a positive strict int (got {val!r}).")
    return val


def _require_strict_nonneg_int(name: str, val: Any) -> int:
    if type(val) is not int or val < 0:
        raise ValueError(f"{name} must be a non-negative strict int (got {val!r}).")
    return val


def _validate_hex64(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise ValueError(f"{name} must be 64 lowercase hex characters.")
    return value


def _validate_zone_manifest_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("zone_manifest_id must be a string.")
    if value != "" and not _ZONE_MANIFEST_ID_RE.match(value):
        raise ValueError("zone_manifest_id must be empty or 'sha256:<64-hex>'.")
    return value


def _parse_quant_label(quant_label: str) -> tuple[str, str]:
    if "/" in quant_label:
        parts = quant_label.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid asymmetric quant label: {quant_label!r}")
        b_dt, a_dt = parts[0], parts[1]
    else:
        b_dt, a_dt = quant_label, quant_label
    if b_dt not in SUPPORTED_QUANT_DTYPES or a_dt not in SUPPORTED_QUANT_DTYPES:
        raise ValueError(f"Unsupported quant dtypes in label: {quant_label!r}")
    return b_dt, a_dt


@dataclass(frozen=True)
class SlabWireContainer:
    """Decoded and cryptographically verified `weft-slab/v0.1` adapter container."""

    format_version: str
    slot_name: str
    target_tensor: str
    b_dtype: str
    a_dtype: str
    b_shape: tuple[int, int]
    a_shape: tuple[int, int]
    b_scale_axis: int
    a_scale_axis: int
    b_bytes: bytes
    a_bytes: bytes
    factor_b: np.ndarray
    factor_a: np.ndarray
    factor_digest: str
    certificate: FixedRadixCertificate
    certificate_sha256: str
    base_weights_sha256: str
    zone_manifest_id: str
    container_sha256: str
    total_bytes: int


def write_slab_wire(
    patch: StagedPatch,
    *,
    target_tensor: str,
    base_weights_sha256: str,
    zone_manifest_id: str = "",
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """Serialize a StagedPatch into canonical deterministic `weft-slab/v0.1` bytes."""
    if not isinstance(target_tensor, str) or not target_tensor.strip():
        raise ValueError("target_tensor must be a non-empty string.")
    base_sha = _validate_hex64("base_weights_sha256", base_weights_sha256)
    zm_id = _validate_zone_manifest_id(zone_manifest_id)
    b_dtype, a_dtype = _parse_quant_label(patch.certificate.quant_dtype)

    b_shape = (int(patch.factor_b.shape[0]), int(patch.factor_b.shape[1]))
    a_shape = (int(patch.factor_a.shape[0]), int(patch.factor_a.shape[1]))
    if len(patch.factor_b.shape) != 2 or len(patch.factor_a.shape) != 2 or b_shape[1] != a_shape[0]:
        raise ValueError("Incompatible factor shapes in StagedPatch.")

    recomputed_digest = sha256_factor_pair(
        patch.b_bytes,
        patch.a_bytes,
        b_shape,
        a_shape,
        b_dtype,
        b_dtype=b_dtype,
        a_dtype=a_dtype,
    )
    if recomputed_digest != patch.factor_digest or recomputed_digest != patch.certificate.factor_digest:
        raise ValueError("StagedPatch factor_digest does not match raw factor bytes or certificate.")

    header_obj = {
        "a_byte_len": len(patch.a_bytes),
        "a_dtype": a_dtype,
        "a_scale_axis": 1,
        "a_shape": [a_shape[0], a_shape[1]],
        "b_byte_len": len(patch.b_bytes),
        "b_dtype": b_dtype,
        "b_scale_axis": 0,
        "b_shape": [b_shape[0], b_shape[1]],
        "base_weights_sha256": base_sha,
        "certificate": asdict(patch.certificate),
        "certificate_sha256": patch.certificate.sha256(),
        "factor_digest": patch.factor_digest,
        "format": WEFT_SLAB_FORMAT,
        "slot_name": patch.slot_name,
        "target_tensor": target_tensor,
        "zone_manifest_id": zm_id,
    }
    header_bytes = json.dumps(header_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"Slab header exceeds maximum size ({MAX_HEADER_BYTES} bytes).")
    payload = patch.b_bytes + patch.a_bytes
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Slab payload exceeds maximum size ({MAX_PAYLOAD_BYTES} bytes).")

    prefix = struct.pack(
        "<4sIII",
        WEFT_SLAB_MAGIC,
        WEFT_SLAB_VERSION,
        len(header_bytes),
        len(payload),
    )
    body = prefix + header_bytes + payload
    trailer = hashlib.sha256(body).digest()
    blob = body + trailer

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
    return blob


def read_slab_wire(source: Union[bytes, bytearray, memoryview, str, Path]) -> SlabWireContainer:
    """Strictly verify and decode a `weft-slab/v0.1` binary container."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = bytes(source)

    min_len = FIXED_HEADER_SIZE + TRAILER_SIZE
    if len(raw) < min_len:
        raise ValueError(f"Slab container truncated: need at least {min_len} bytes, got {len(raw)}.")

    magic, version, header_len, payload_len = struct.unpack_from("<4sIII", raw, 0)
    if magic != WEFT_SLAB_MAGIC:
        raise ValueError(f"Bad slab magic: expected {WEFT_SLAB_MAGIC!r}, got {magic!r}.")
    if version != WEFT_SLAB_VERSION:
        raise ValueError(f"Unsupported slab version: {version}.")
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Invalid slab header length: {header_len}.")
    if payload_len == 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid slab payload length: {payload_len}.")

    expected_total = FIXED_HEADER_SIZE + header_len + payload_len + TRAILER_SIZE
    if len(raw) != expected_total:
        raise ValueError(
            f"Slab container length mismatch: expected {expected_total} bytes, got {len(raw)}."
        )

    body_end = FIXED_HEADER_SIZE + header_len + payload_len
    expected_trailer = raw[body_end:]
    computed_trailer = hashlib.sha256(raw[:body_end]).digest()
    if expected_trailer != computed_trailer:
        raise ValueError("Slab container SHA-256 trailer mismatch (corrupted or tampered bytes).")

    header_slice = raw[FIXED_HEADER_SIZE : FIXED_HEADER_SIZE + header_len]
    try:
        header = _strict_json_loads(header_slice.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed slab header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Slab header JSON must be an object.")
    if header.get("format") != WEFT_SLAB_FORMAT:
        raise ValueError(f"Unsupported slab format: {header.get('format')!r}.")

    b_dtype = header.get("b_dtype")
    a_dtype = header.get("a_dtype")
    if b_dtype not in SUPPORTED_QUANT_DTYPES or a_dtype not in SUPPORTED_QUANT_DTYPES:
        raise ValueError(f"Unsupported factor dtypes: b={b_dtype!r}, a={a_dtype!r}.")

    b_scale_axis = header.get("b_scale_axis")
    a_scale_axis = header.get("a_scale_axis")
    if type(b_scale_axis) is not int or b_scale_axis != 0:
        raise ValueError(f"Expected b_scale_axis == 0, got {b_scale_axis!r}.")
    if type(a_scale_axis) is not int or a_scale_axis != 1:
        raise ValueError(f"Expected a_scale_axis == 1, got {a_scale_axis!r}.")

    b_shape_raw = header.get("b_shape")
    a_shape_raw = header.get("a_shape")
    if (
        not isinstance(b_shape_raw, list)
        or not isinstance(a_shape_raw, list)
        or len(b_shape_raw) != 2
        or len(a_shape_raw) != 2
    ):
        raise ValueError("b_shape and a_shape must be 2-element integer lists.")
    b_shape = (
        _require_strict_pos_int("b_shape[0]", b_shape_raw[0]),
        _require_strict_pos_int("b_shape[1]", b_shape_raw[1]),
    )
    a_shape = (
        _require_strict_pos_int("a_shape[0]", a_shape_raw[0]),
        _require_strict_pos_int("a_shape[1]", a_shape_raw[1]),
    )
    if b_shape[1] != a_shape[0]:
        raise ValueError(f"Rank mismatch between b_shape={b_shape} and a_shape={a_shape}.")

    b_byte_len = _require_strict_nonneg_int("b_byte_len", header.get("b_byte_len"))
    a_byte_len = _require_strict_nonneg_int("a_byte_len", header.get("a_byte_len"))
    if b_byte_len + a_byte_len != payload_len:
        raise ValueError(
            f"Factor byte lengths ({b_byte_len} + {a_byte_len}) do not match payload_len ({payload_len})."
        )

    slot_name = header.get("slot_name")
    target_tensor = header.get("target_tensor")
    if not isinstance(slot_name, str) or not isinstance(target_tensor, str) or not target_tensor.strip():
        raise ValueError("slot_name must be a string and target_tensor must be non-empty.")

    base_sha = _validate_hex64("base_weights_sha256", header.get("base_weights_sha256"))
    zm_id = _validate_zone_manifest_id(header.get("zone_manifest_id"))
    hdr_factor_digest = _validate_hex64("factor_digest", header.get("factor_digest"))
    hdr_cert_sha = _validate_hex64("certificate_sha256", header.get("certificate_sha256"))

    cert_raw = header.get("certificate")
    if not isinstance(cert_raw, dict):
        raise ValueError("Embedded certificate must be a JSON object.")
    cert = FixedRadixCertificate.from_dict(cert_raw)
    if cert.sha256() != hdr_cert_sha:
        raise ValueError("Embedded certificate SHA-256 mismatch.")
    expected_label = b_dtype if b_dtype == a_dtype else f"{b_dtype}/{a_dtype}"
    if cert.quant_dtype != expected_label:
        raise ValueError(
            f"Certificate quant_dtype ({cert.quant_dtype!r}) disagrees with header ({expected_label!r})."
        )

    payload_start = FIXED_HEADER_SIZE + header_len
    b_bytes = raw[payload_start : payload_start + b_byte_len]
    a_bytes = raw[payload_start + b_byte_len : body_end]

    recomputed_digest = sha256_factor_pair(
        b_bytes,
        a_bytes,
        b_shape,
        a_shape,
        b_dtype,
        b_dtype=b_dtype,
        a_dtype=a_dtype,
    )
    if recomputed_digest != hdr_factor_digest or recomputed_digest != cert.factor_digest:
        raise ValueError("Factor pair SHA-256 digest mismatch between payload, header, and certificate.")

    factor_b = dequantize_array(b_bytes, b_shape, b_dtype, axis=0)
    factor_a = dequantize_array(a_bytes, a_shape, a_dtype, axis=1)

    return SlabWireContainer(
        format_version=WEFT_SLAB_FORMAT,
        slot_name=slot_name,
        target_tensor=target_tensor,
        b_dtype=b_dtype,
        a_dtype=a_dtype,
        b_shape=b_shape,
        a_shape=a_shape,
        b_scale_axis=0,
        a_scale_axis=1,
        b_bytes=b_bytes,
        a_bytes=a_bytes,
        factor_b=factor_b,
        factor_a=factor_a,
        factor_digest=hdr_factor_digest,
        certificate=cert,
        certificate_sha256=hdr_cert_sha,
        base_weights_sha256=base_sha,
        zone_manifest_id=zm_id,
        container_sha256=computed_trailer.hex(),
        total_bytes=len(raw),
    )


def verify_slab_wire(
    source: Union[bytes, bytearray, memoryview, str, Path],
    *,
    gate_policy: Optional[GatePolicy] = None,
    expected_zone_manifest_id: Optional[str] = None,
    expected_base_weights_sha256: Optional[str] = None,
) -> tuple[bool, list[str], Optional[SlabWireContainer]]:
    """Fail-closed verification of container integrity, pins, and integer certificate policy."""
    try:
        container = read_slab_wire(source)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, [f"container_invalid:{exc}"], None

    problems: list[str] = []
    if expected_base_weights_sha256 is not None:
        if container.base_weights_sha256 != expected_base_weights_sha256:
            problems.append("base_weights_sha256_mismatch")
    if expected_zone_manifest_id is not None:
        if container.zone_manifest_id != expected_zone_manifest_id:
            problems.append("zone_manifest_drift")
    if gate_policy is not None:
        cert_ok, cert_probs = container.certificate.verify_division_free(gate_policy)
        if not cert_ok:
            problems.extend(cert_probs)
    return not problems, problems, container
