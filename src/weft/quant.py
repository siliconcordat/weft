"""Deterministic IEEE-754 round-to-nearest-even quantization for BF16 and F16.

All conversions run purely on CPU bit representations in NumPy so Zone A
re-derivation produces identical bytes across machines and runtimes.
"""

from __future__ import annotations

import hashlib
import numpy as np


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


def quantize_array(arr: np.ndarray, dtype: str) -> tuple[bytes, np.ndarray]:
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
    raise ValueError(f"Unsupported adapter quantization dtype: {dtype!r}")


def sha256_factor_pair(
    b_bytes: bytes,
    a_bytes: bytes,
    b_shape: tuple[int, int],
    a_shape: tuple[int, int],
    dtype: str,
) -> str:
    """Compute canonical SHA-256 digest over quantized (B_q, A_q) factor bytes and shapes."""
    h = hashlib.sha256()
    header = f"weft-factor-pair/v1\ndtype={dtype}\nb_shape={b_shape}\na_shape={a_shape}\n"
    h.update(header.encode("utf-8"))
    h.update(b_bytes)
    h.update(a_bytes)
    return h.hexdigest()
