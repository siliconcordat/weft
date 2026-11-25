"""Deterministic GGUF v3 F16 LoRA binary writer and reader (pure stdlib + NumPy).

Produces 32-byte aligned GGUF v3 adapter binaries compatible with llama.cpp
multi-LoRA conventions:
- general.architecture
- general.type = "adapter"
- adapter.type = "lora"
- adapter.lora.alpha (float32)
- weft.schema, weft.adapter_class, weft.binding_sha256, weft.certificate_sha256
- Tensor pairs: `<tensor_name>.lora_a` (shape [r, d_in]) and `<tensor_name>.lora_b`
  (shape [d_out, r]) stored in little-endian IEEE-754 F16 (GGML_TYPE_F16 = 1).

No timestamps or random fields are emitted: identical inputs produce byte-identical
GGUF files across machines (Zone A deterministic replay).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

import numpy as np

from .quant import quantize_f16

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
MAX_GGUF_ARRAY_DEPTH = 32

# GGUF metadata value type IDs (per ggml/GGUF v3 specification)
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

# ggml_type enum values
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


@dataclass(frozen=True)
class GGUFLoraAdapter:
    """In-memory representation of a Weft GGUF v3 F16 LoRA binary."""

    architecture: str
    lora_alpha: float
    tensors: dict[str, np.ndarray]  # tensor_name -> float16 2D array
    metadata: dict[str, Any] = field(default_factory=dict)

    def sha256_bytes(self) -> str:
        return hashlib.sha256(encode_gguf_lora(self)).hexdigest()


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _unpack_string(buf: memoryview, pos: int) -> tuple[str, int]:
    if pos + 8 > len(buf):
        raise ValueError("Truncated GGUF string length.")
    (slen,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    if pos + slen > len(buf):
        raise ValueError("Truncated GGUF string bytes.")
    s = bytes(buf[pos : pos + slen]).decode("utf-8")
    return s, pos + slen


def _pack_kv(key: str, val: Any) -> bytes:
    out = _pack_string(key)
    if isinstance(val, bool):
        return out + struct.pack("<I?", GGUF_TYPE_BOOL, val)
    if isinstance(val, int):
        if 0 <= val <= 0xFFFFFFFF:
            return out + struct.pack("<II", GGUF_TYPE_UINT32, val)
        if 0 <= val <= 0xFFFFFFFFFFFFFFFF:
            return out + struct.pack("<IQ", GGUF_TYPE_UINT64, val)
        raise ValueError(f"Unsupported integer metadata value for {key}: {val}")
    if isinstance(val, float):
        return out + struct.pack("<If", GGUF_TYPE_FLOAT32, val)
    if isinstance(val, str):
        return out + struct.pack("<I", GGUF_TYPE_STRING) + _pack_string(val)
    raise TypeError(f"Unsupported GGUF metadata type for {key}: {type(val).__name__}")


def _unpack_kv(buf: memoryview, pos: int) -> tuple[str, Any, int]:
    key, pos = _unpack_string(buf, pos)
    if pos + 4 > len(buf):
        raise ValueError(f"Truncated GGUF metadata value type for {key}.")
    (vtype,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    if vtype == GGUF_TYPE_UINT32:
        if pos + 4 > len(buf):
            raise ValueError(f"Truncated GGUF UINT32 metadata value for {key}.")
        (val,) = struct.unpack_from("<I", buf, pos)
        return key, int(val), pos + 4
    if vtype == GGUF_TYPE_UINT64:
        if pos + 8 > len(buf):
            raise ValueError(f"Truncated GGUF UINT64 metadata value for {key}.")
        (val,) = struct.unpack_from("<Q", buf, pos)
        return key, int(val), pos + 8
    if vtype == GGUF_TYPE_FLOAT32:
        if pos + 4 > len(buf):
            raise ValueError(f"Truncated GGUF FLOAT32 metadata value for {key}.")
        (val,) = struct.unpack_from("<f", buf, pos)
        return key, float(val), pos + 4
    if vtype == GGUF_TYPE_BOOL:
        if pos + 1 > len(buf):
            raise ValueError(f"Truncated GGUF BOOL metadata value for {key}.")
        (val,) = struct.unpack_from("<?", buf, pos)
        return key, bool(val), pos + 1
    if vtype == GGUF_TYPE_STRING:
        val, pos = _unpack_string(buf, pos)
        return key, val, pos
    raise ValueError(f"Unsupported GGUF value type {vtype} for key {key!r}.")


def _align_up(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def encode_gguf_lora(adapter: GGUFLoraAdapter) -> bytes:
    """Serialize a GGUFLoraAdapter into deterministic GGUF v3 binary bytes."""
    if not adapter.tensors:
        raise ValueError("GGUFLoraAdapter must contain at least one tensor.")

    meta: dict[str, Any] = {
        "adapter.lora.alpha": float(adapter.lora_alpha),
        "adapter.type": "lora",
        "general.alignment": int(GGUF_DEFAULT_ALIGNMENT),
        "general.architecture": str(adapter.architecture),
        "general.type": "adapter",
    }
    for k, v in adapter.metadata.items():
        meta[str(k)] = v

    sorted_meta_keys = sorted(meta.keys())
    sorted_tensor_names = sorted(adapter.tensors.keys())

    # Prepare canonical F16 payloads and relative offsets
    tensor_blobs: list[tuple[str, tuple[int, ...], bytes, int]] = []
    rel_offset = 0
    for name in sorted_tensor_names:
        arr = adapter.tensors[name]
        f16, _ = quantize_f16(np.asarray(arr, dtype=np.float64))
        if f16.ndim != 2:
            raise ValueError(f"LoRA tensor {name!r} must be 2D, got shape {f16.shape}.")
        raw = f16.tobytes(order="C")
        tensor_blobs.append((name, tuple(int(d) for d in f16.shape), raw, rel_offset))
        rel_offset = _align_up(rel_offset + len(raw), GGUF_DEFAULT_ALIGNMENT)

    parts: list[bytes] = []
    # Header: magic, version, tensor_count, metadata_kv_count
    parts.append(
        struct.pack(
            "<4sIQQ",
            GGUF_MAGIC,
            GGUF_VERSION,
            len(tensor_blobs),
            len(sorted_meta_keys),
        )
    )
    # Metadata KV section (sorted keys)
    for k in sorted_meta_keys:
        parts.append(_pack_kv(k, meta[k]))

    # Tensor info section (ggml dimension order: ne[0] = inner dim = shape[1], ne[1] = shape[0])
    for name, shape, _raw, t_off in tensor_blobs:
        ne0, ne1 = shape[1], shape[0]
        parts.append(_pack_string(name))
        parts.append(struct.pack("<IQQIQ", 2, ne0, ne1, GGML_TYPE_F16, t_off))

    header_bytes = b"".join(parts)
    data_start = _align_up(len(header_bytes), GGUF_DEFAULT_ALIGNMENT)
    pad_len = data_start - len(header_bytes)

    out_parts = [header_bytes, b"\x00" * pad_len]
    written_data = 0
    for _name, _shape, raw, t_off in tensor_blobs:
        if t_off > written_data:
            out_parts.append(b"\x00" * (t_off - written_data))
            written_data = t_off
        out_parts.append(raw)
        written_data += len(raw)

    return b"".join(out_parts)


def decode_gguf_lora(data: bytes) -> GGUFLoraAdapter:
    """Parse and validate a GGUF v3 F16 LoRA binary back into a GGUFLoraAdapter."""
    buf = memoryview(data)
    if len(buf) < 24:
        raise ValueError("Truncated GGUF header.")
    magic, version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    if magic != GGUF_MAGIC:
        raise ValueError(f"Invalid GGUF magic: {magic!r}")
    if version != GGUF_VERSION:
        raise ValueError(f"Unsupported GGUF version: {version}")

    if n_kv > len(buf) or n_tensors > len(buf):
        raise ValueError("Crafted GGUF header count exceeds buffer size.")

    pos = 24
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        k, v, pos = _unpack_kv(buf, pos)
        meta[k] = v

    if meta.get("general.type") != "adapter" or meta.get("adapter.type") != "lora":
        raise ValueError("GGUF binary is not a LoRA adapter.")

    alignment = int(meta.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"Invalid GGUF alignment: {alignment}")

    tensor_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(n_tensors):
        tname, pos = _unpack_string(buf, pos)
        if pos + 4 > len(buf):
            raise ValueError(f"Truncated GGUF tensor ndim for {tname!r}.")
        (ndim,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if ndim != 2:
            raise ValueError(f"Expected 2D tensor for {tname!r}, got ndim={ndim}.")
        if pos + 28 > len(buf):  # 2 * uint64 dims + uint32 ggml_type + uint64 rel_off
            raise ValueError(f"Truncated GGUF tensor info for {tname!r}.")
        dim0, dim1, ggml_type, rel_off = struct.unpack_from("<QQIQ", buf, pos)
        pos += 28
        if ggml_type != GGML_TYPE_F16:
            raise ValueError(f"Unsupported ggml_type {ggml_type} for {tname!r}; expected F16.")
        if dim0 == 0 or dim1 == 0:
            raise ValueError(f"Zero-sized GGUF tensor dimension for {tname!r}.")
        # Reverse ggml ne[0], ne[1] back to NumPy (rows, cols) = (ne1, ne0)
        np_shape = (int(dim1), int(dim0))
        tensor_specs.append((tname, np_shape, ggml_type, int(rel_off)))

    data_start = _align_up(pos, alignment)
    tensors: dict[str, np.ndarray] = {}
    for tname, np_shape, _gtype, rel_off in tensor_specs:
        abs_off = data_start + rel_off
        nbytes = np_shape[0] * np_shape[1] * 2
        if abs_off + nbytes > len(buf):
            raise ValueError(f"Truncated tensor payload for {tname!r}.")
        arr = np.frombuffer(buf[abs_off : abs_off + nbytes], dtype="<f2").reshape(np_shape).copy()
        tensors[tname] = arr

    arch = str(meta.get("general.architecture", "unknown"))
    alpha = float(meta.get("adapter.lora.alpha", 1.0))
    return GGUFLoraAdapter(
        architecture=arch,
        lora_alpha=alpha,
        tensors=tensors,
        metadata=meta,
    )


def write_gguf_lora(
    path: Path | str,
    *,
    architecture: str,
    lora_alpha: float,
    layer_factors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    extra_metadata: Mapping[str, Any] = (),
) -> tuple[Path, str]:
    """Write a deterministic GGUF v3 F16 LoRA file from {base_tensor: (B, A)} pairs.

    In llama.cpp LoRA convention, each layer stores `<base>.lora_a` of shape (r, d_in)
    and `<base>.lora_b` of shape (d_out, r). Returns (output_path, sha256_hex).
    """
    tensors: dict[str, np.ndarray] = {}
    for base_name, (b_mat, a_mat) in sorted(layer_factors.items()):
        b_f16, _ = quantize_f16(np.asarray(b_mat, dtype=np.float64))
        a_f16, _ = quantize_f16(np.asarray(a_mat, dtype=np.float64))
        if b_f16.ndim != 2 or a_f16.ndim != 2 or b_f16.shape[1] != a_f16.shape[0]:
            raise ValueError(f"Incompatible LoRA (B, A) shapes for {base_name!r}.")
        tensors[f"{base_name}.lora_a"] = a_f16
        tensors[f"{base_name}.lora_b"] = b_f16

    meta = dict(extra_metadata)
    meta.setdefault("weft.schema", "weft-gguf-lora/v0.1")
    adapter = GGUFLoraAdapter(
        architecture=architecture,
        lora_alpha=float(lora_alpha),
        tensors=tensors,
        metadata=meta,
    )
    raw = encode_gguf_lora(adapter)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return out_path, hashlib.sha256(raw).hexdigest()


def read_gguf_lora(path: Path | str) -> GGUFLoraAdapter:
    """Read and validate a GGUF v3 F16 LoRA binary from disk."""
    return decode_gguf_lora(Path(path).read_bytes())


_GGUF_SCALAR_SIZES: dict[int, int] = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def _read_exact(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"Unexpected EOF reading GGUF header (wanted {n} bytes, got {len(data)}).")
    return data


def _read_gguf_str(fh: BinaryIO, *, max_len: int = 16 * 1024 * 1024) -> str:
    (slen,) = struct.unpack("<Q", _read_exact(fh, 8))
    if slen > max_len:
        raise ValueError(f"GGUF string length exceeds limit: {slen}")
    return _read_exact(fh, slen).decode("utf-8")


def _skip_or_read_gguf_val(
    fh: BinaryIO,
    vtype: int,
    *,
    keep: bool,
    file_size: int,
    depth: int = 0,
) -> Any:
    if vtype in _GGUF_SCALAR_SIZES:
        raw = _read_exact(fh, _GGUF_SCALAR_SIZES[vtype])
        if not keep:
            return None
        fmt_map = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<?",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        return struct.unpack(fmt_map[vtype], raw)[0]
    if vtype == GGUF_TYPE_STRING:
        if keep:
            return _read_gguf_str(fh)
        (slen,) = struct.unpack("<Q", _read_exact(fh, 8))
        if fh.tell() + slen > file_size:
            raise ValueError("GGUF string extends past end of file.")
        fh.seek(slen, 1)
        return None
    if vtype == 9:  # GGUF_TYPE_ARRAY
        if depth >= MAX_GGUF_ARRAY_DEPTH:
            raise ValueError(
                f"GGUF nested array depth exceeds safety cap ({MAX_GGUF_ARRAY_DEPTH})."
            )
        elem_type, count = struct.unpack("<IQ", _read_exact(fh, 12))
        if count > 1_000_000:
            raise ValueError(f"GGUF array length exceeds safety cap: {count}")
        if elem_type in _GGUF_SCALAR_SIZES:
            skip = _GGUF_SCALAR_SIZES[elem_type] * count
            if fh.tell() + skip > file_size:
                raise ValueError("GGUF scalar array extends past end of file.")
            fh.seek(skip, 1)
            return f"<array:{count}>" if keep else None
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(count):
                (slen,) = struct.unpack("<Q", _read_exact(fh, 8))
                if fh.tell() + slen > file_size:
                    raise ValueError("GGUF string array element extends past end of file.")
                fh.seek(slen, 1)
            return f"<str_array:{count}>" if keep else None
        for _ in range(count):
            _skip_or_read_gguf_val(
                fh, elem_type, keep=False, file_size=file_size, depth=depth + 1
            )
        return f"<nested_array:{count}>" if keep else None
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def read_gguf_model_geometry(
    path: Path | str,
    *,
    tensor_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Read GGUF v3 model metadata and 2D tensor shapes without loading weight bytes.

    Tensor shapes are returned in NumPy row-major order `(d_out, d_in)` (reversed
    from ggml column-major `(ne0, ne1)`), ready for `BetaEditConfig(d_in=..., d_out=...)`.
    """
    wanted = set(tensor_names) if tensor_names is not None else None
    gguf_path = Path(path)
    file_size = gguf_path.stat().st_size
    with open(gguf_path, "rb") as fh:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", _read_exact(fh, 24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        if n_kv > 100_000 or n_tensors > 100_000:
            raise ValueError(f"GGUF header counts exceed safety cap: tensors={n_tensors}, kv={n_kv}")

        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_gguf_str(fh)
            (vtype,) = struct.unpack("<I", _read_exact(fh, 4))
            keep = not key.startswith("tokenizer.")
            val = _skip_or_read_gguf_val(fh, vtype, keep=keep, file_size=file_size)
            if keep:
                meta[key] = val

        tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            tname = _read_gguf_str(fh)
            (ndim,) = struct.unpack("<I", _read_exact(fh, 4))
            if ndim > 8:
                raise ValueError(f"Invalid GGUF tensor ndim={ndim} for {tname!r}.")
            ggml_ne = struct.unpack(f"<{ndim}Q", _read_exact(fh, 8 * ndim))
            ggml_type, rel_off = struct.unpack("<IQ", _read_exact(fh, 12))
            if wanted is None or tname in wanted:
                np_shape = tuple(int(d) for d in reversed(ggml_ne))
                tensors[tname] = {
                    "ggml_ne": tuple(int(d) for d in ggml_ne),
                    "numpy_shape": np_shape,
                    "ggml_type": int(ggml_type),
                    "rel_offset": int(rel_off),
                }

    arch = str(meta.get("general.architecture", "unknown"))
    return {
        "architecture": arch,
        "model_name": str(meta.get("general.name", "")),
        "n_tensors": int(n_tensors),
        "block_count": int(meta.get(f"{arch}.block_count", 0)),
        "embedding_length": int(meta.get(f"{arch}.embedding_length", 0)),
        "feed_forward_length": int(meta.get(f"{arch}.feed_forward_length", 0)),
        "tensors": tensors,
    }


def build_llama_server_lora_flags(
    adapter_paths: Sequence[Path | str],
    *,
    initial_scale: float = 0.0,
    slot_save_path: Optional[Path | str] = None,
) -> list[str]:
    """Build fail-closed llama-server CLI flags for simultaneous multi-LoRA serving.

    Verified against llama-server 0.5.0 (build 11146):
    - Uses comma-separated `--lora-scaled <path>:<scale>,...` alongside
      `--lora-init-without-apply` so adapters start strictly at `scale = 0.0`
      (multiple `--lora` flags are deprecated and default to scale 1.0).
    - Disables idle-slot caching, prompt similarity reuse, and context checkpoints
      so O(1) scale rollback cannot leak prior KV states.
    - Passes `--slot-save-path` when supplied so `POST /slots/{id}?action=erase`
      succeeds with HTTP 200 instead of HTTP 501.
    """
    if not adapter_paths:
        raise ValueError("adapter_paths must be non-empty.")
    if not np.isfinite(initial_scale):
        raise ValueError("initial_scale must be finite.")
    scaled_spec = ",".join(f"{Path(p)}:{float(initial_scale)}" for p in adapter_paths)
    flags = [
        "--lora-scaled",
        scaled_spec,
        "--lora-init-without-apply",
        "--slots",
        "--no-cache-idle-slots",
        "--slot-prompt-similarity",
        "0.0",
        "--ctx-checkpoints",
        "0",
    ]
    if slot_save_path is not None:
        flags.extend(["--slot-save-path", str(Path(slot_save_path))])
    return flags
