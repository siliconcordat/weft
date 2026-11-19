"""Deterministic IEEE-754 and symmetric integer quantization for BF16, F16, INT8, and INT4.

All conversions run purely on CPU bit representations in NumPy so Zone A
re-derivation produces identical bytes across machines and runtimes. Supports
asymmetric (B_dtype, A_dtype) factor quantization (for example, INT4 B paired
with INT8 or F16 A to preserve null-space orthogonality at half the B bandwidth).
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

SUPPORTED_QUANT_DTYPES = ("bf16", "f16", "f32", "int8", "int4")


def _require_finite(arr: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("Non-finite values (NaN or Inf) are forbidden in weight adapters.")
    return a


def float64_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Convert a finite float64 array to little-endian uint16 BF16 bit patterns (RNE)."""
    a = _require_finite(arr)
    f32 = a.astype(np.float32)
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even from IEEE-754 binary32 to bfloat16:
    # rounding bias = 0x7FFF + ((u32 >> 16) & 1)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    rounded = u32 + np.uint32(0x7FFF) + lsb
    bf16_bits = (rounded >> np.uint32(16)).astype("<u2")
    return np.ascontiguousarray(bf16_bits)


def bf16_bits_to_float64(bits: np.ndarray) -> np.ndarray:
    """Dequantize uint16 BF16 bit patterns back to exact float64 values."""
    u16 = np.ascontiguousarray(bits, dtype="<u2")
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).astype(np.float64)


def quantize_bf16(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bf16_uint16_bits, dequantized_float64) with canonical RNE rounding."""
    bits = float64_to_bf16_bits(arr)
    dequant = bf16_bits_to_float64(bits)
    if not np.all(np.isfinite(dequant)):
        raise ValueError("BF16 quantization overflowed to Inf.")
    return bits, dequant


def quantize_f16(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (f16_array, dequantized_float64) using little-endian IEEE-754 binary16."""
    a = _require_finite(arr)
    f16 = np.ascontiguousarray(a.astype("<f2"))
    if not np.all(np.isfinite(f16)):
        raise ValueError("F16 quantization overflowed to Inf.")
    return f16, f16.astype(np.float64)


def quantize_int8(arr: np.ndarray, *, axis: int = 1) -> tuple[bytes, np.ndarray]:
    """Deterministic symmetric per-slice INT8 quantization ([-127, 127], RNE).

    axis=1 quantizes per row (ideal for A of shape (r, d_in): r scales).
    axis=0 quantizes per column (ideal for B of shape (d_out, r): r scales).
    Wire payload: little-endian float32 scales followed by signed int8 values.
    """
    a = _require_finite(arr)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"INT8 factor quantization requires non-empty 2D array, got shape {a.shape}.")
    if axis not in (0, 1):
        raise ValueError(f"INT8 quantization axis must be 0 or 1, got {axis}.")
    max_abs = np.max(np.abs(a), axis=axis)
    raw_scales = (max_abs / 127.0).astype("<f4")
    if not np.all(np.isfinite(raw_scales)):
        raise ValueError("INT8 scale overflowed float32.")
    scales_f32 = np.where(raw_scales > np.float32(0.0), raw_scales, np.float32(1.0)).astype("<f4")
    scales_bcast = (
        scales_f32.astype(np.float64)[:, np.newaxis]
        if axis == 1
        else scales_f32.astype(np.float64)[np.newaxis, :]
    )
    q = np.clip(np.rint(a / scales_bcast), -127, 127).astype(np.int8)
    dequant = q.astype(np.float64) * scales_bcast
    raw = scales_f32.tobytes(order="C") + np.ascontiguousarray(q).tobytes(order="C")
    return raw, np.ascontiguousarray(dequant, dtype=np.float64)


def dequantize_int8(raw: bytes, shape: tuple[int, int], *, axis: int = 1) -> np.ndarray:
    """Decode canonical per-slice INT8 factor bytes back to exact float64 values."""
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid shape {shape} for INT8 dequantization.")
    if axis not in (0, 1):
        raise ValueError(f"INT8 dequantization axis must be 0 or 1, got {axis}.")
    n_scales = rows if axis == 1 else cols
    scale_bytes = n_scales * 4
    expected = scale_bytes + rows * cols
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} INT8 bytes for shape {shape}, got {len(raw)}.")
    scales = np.frombuffer(raw[:scale_bytes], dtype="<f4").astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise ValueError("Non-finite INT8 scale in wire payload.")
    q = np.frombuffer(raw[scale_bytes:], dtype=np.int8).reshape(rows, cols)
    if np.any(q < -127):
        raise ValueError("Out-of-range INT8 value (-128) decoded from symmetric payload.")
    scales_bcast = scales[:, np.newaxis] if axis == 1 else scales[np.newaxis, :]
    return np.ascontiguousarray(q.astype(np.float64) * scales_bcast, dtype=np.float64)


def quantize_int4(arr: np.ndarray, *, axis: int = 1) -> tuple[bytes, np.ndarray]:
    """Deterministic symmetric per-slice INT4 quantization ([-7, 7], RNE, 2 nibbles/byte).

    axis=1 quantizes per row (ideal for A of shape (r, d_in): r scales).
    axis=0 quantizes per column (ideal for B of shape (d_out, r): r scales).
    Wire payload: little-endian float32 scales followed by packed uint8
    nibbles (offset-encoded as q + 8 in [1, 15], two values per byte).
    """
    a = _require_finite(arr)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"INT4 factor quantization requires non-empty 2D array, got shape {a.shape}.")
    if axis not in (0, 1):
        raise ValueError(f"INT4 quantization axis must be 0 or 1, got {axis}.")
    max_abs = np.max(np.abs(a), axis=axis)
    raw_scales = (max_abs / 7.0).astype("<f4")
    if not np.all(np.isfinite(raw_scales)):
        raise ValueError("INT4 scale overflowed float32.")
    scales_f32 = np.where(raw_scales > np.float32(0.0), raw_scales, np.float32(1.0)).astype("<f4")
    scales_bcast = (
        scales_f32.astype(np.float64)[:, np.newaxis]
        if axis == 1
        else scales_f32.astype(np.float64)[np.newaxis, :]
    )
    q = np.clip(np.rint(a / scales_bcast), -7, 7).astype(np.int8)
    dequant = q.astype(np.float64) * scales_bcast

    # Pack 2 signed 4-bit nibbles per byte deterministically
    flat_u = (q.reshape(-1).astype(np.int16) + 8).astype(np.uint8)
    if flat_u.size % 2 == 1:
        flat_u = np.concatenate([flat_u, np.full(1, 8, dtype=np.uint8)])
    packed = ((flat_u[0::2] & np.uint8(0x0F)) | ((flat_u[1::2] & np.uint8(0x0F)) << np.uint8(4)))
    raw = scales_f32.tobytes(order="C") + np.ascontiguousarray(packed).tobytes(order="C")
    return raw, np.ascontiguousarray(dequant, dtype=np.float64)


def dequantize_int4(raw: bytes, shape: tuple[int, int], *, axis: int = 1) -> np.ndarray:
    """Decode canonical packed per-slice INT4 factor bytes back to exact float64 values."""
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid shape {shape} for INT4 dequantization.")
    if axis not in (0, 1):
        raise ValueError(f"INT4 dequantization axis must be 0 or 1, got {axis}.")
    n_elems = rows * cols
    n_scales = rows if axis == 1 else cols
    scale_bytes = n_scales * 4
    packed_bytes = (n_elems + 1) // 2
    expected = scale_bytes + packed_bytes
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} INT4 bytes for shape {shape}, got {len(raw)}.")
    scales = np.frombuffer(raw[:scale_bytes], dtype="<f4").astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise ValueError("Non-finite INT4 scale in wire payload.")
    packed = np.frombuffer(raw[scale_bytes:], dtype=np.uint8)
    low = (packed & np.uint8(0x0F)).astype(np.int16) - 8
    high = ((packed >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int16) - 8
    unpacked = np.empty(packed.size * 2, dtype=np.int16)
    unpacked[0::2] = low
    unpacked[1::2] = high
    q = unpacked[:n_elems].reshape(rows, cols)
    if np.any(q < -7) or np.any(q > 7):
        raise ValueError("Out-of-range INT4 nibble decoded from wire payload.")
    scales_bcast = scales[:, np.newaxis] if axis == 1 else scales[np.newaxis, :]
    return np.ascontiguousarray(q.astype(np.float64) * scales_bcast, dtype=np.float64)


def quantize_array(arr: np.ndarray, dtype: str, *, axis: int = 1) -> tuple[bytes, np.ndarray]:
    """Quantize arr into canonical raw little-endian bytes and exact float64 view."""
    if dtype == "bf16":
        bits, dequant = quantize_bf16(arr)
        return bits.tobytes(order="C"), dequant
    if dtype == "f16":
        f16, dequant = quantize_f16(arr)
        return f16.tobytes(order="C"), dequant
    if dtype == "f32":
        a = _require_finite(arr)
        f32 = np.ascontiguousarray(a.astype("<f4"))
        if not np.all(np.isfinite(f32)):
            raise ValueError("F32 quantization overflowed to Inf.")
        return f32.tobytes(order="C"), f32.astype(np.float64)
    if dtype == "int8":
        return quantize_int8(arr, axis=axis)
    if dtype == "int4":
        return quantize_int4(arr, axis=axis)
    raise ValueError(f"Unsupported adapter quantization dtype: {dtype!r}")


def dequantize_array(
    raw: bytes, shape: tuple[int, int], dtype: str, *, axis: int = 1
) -> np.ndarray:
    """Decode canonical quantized bytes back to exact float64 values for any supported dtype."""
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid shape {shape} for dequantization.")
    n_elems = rows * cols
    if dtype == "bf16":
        expected = n_elems * 2
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} BF16 bytes for shape {shape}, got {len(raw)}.")
        bits = np.frombuffer(raw, dtype="<u2").reshape(rows, cols)
        out = bf16_bits_to_float64(bits)
        return _require_finite(out)
    if dtype == "f16":
        expected = n_elems * 2
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} F16 bytes for shape {shape}, got {len(raw)}.")
        arr = np.frombuffer(raw, dtype="<f2").reshape(rows, cols).astype(np.float64)
        return _require_finite(arr)
    if dtype == "f32":
        expected = n_elems * 4
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} F32 bytes for shape {shape}, got {len(raw)}.")
        arr = np.frombuffer(raw, dtype="<f4").reshape(rows, cols).astype(np.float64)
        return _require_finite(arr)
    if dtype == "int8":
        return dequantize_int8(raw, (rows, cols), axis=axis)
    if dtype == "int4":
        return dequantize_int4(raw, (rows, cols), axis=axis)
    raise ValueError(f"Unsupported adapter quantization dtype: {dtype!r}")


def sha256_factor_pair(
    b_bytes: bytes,
    a_bytes: bytes,
    b_shape: tuple[int, int],
    a_shape: tuple[int, int],
    dtype: str,
    *,
    b_dtype: Optional[str] = None,
    a_dtype: Optional[str] = None,
) -> str:
    """Compute canonical SHA-256 digest over quantized (B_q, A_q) factor bytes and shapes."""
    h = hashlib.sha256()
    eff_b = dtype if b_dtype is None else b_dtype
    eff_a = dtype if a_dtype is None else a_dtype
    label = eff_b if eff_b == eff_a else f"{eff_b}/{eff_a}"
    header = f"weft-factor-pair/v1\ndtype={label}\nb_shape={b_shape}\na_shape={a_shape}\n"
    h.update(header.encode("utf-8"))
    h.update(b_bytes)
    h.update(a_bytes)
    return h.hexdigest()
