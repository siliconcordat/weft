"""Five-layer fail-closed checkpoint, Plane 1 lineage, and runtime-flag binding.

Binds a Weft adapter to:
1. Base tensor layout hash (sorted name, dtype, shape across Safetensors shards).
2. Full checkpoint tensor fingerprint (exact raw bytes of every base tensor).
3. Config and tokenizer hashes (config.json, tokenizer.json, chat_template.jinja, etc.).
4. Plane 1 source document hashes, adapter class, and derivation recipe digest.
5. Runtime serving flags (quantization, hybrid cache mode, thinking, tool parser, RoPE, ctx).

Any one-byte mismatch in any layer fails closed before weights reach a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

BINDING_SCHEMA = "weft-binding/v0.1"
ADAPTER_CLASS_1 = "class1_rederivable"
ADAPTER_CLASS_2 = "class2_open_recipe_base"

CONFIG_FILES = (
    "config.json",
    "generation_config.json",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
)


@dataclass(frozen=True)
class RuntimeFlags:
    """Serving-state parameters outside checkpoint files that affect adapter behavior."""

    quantization_mode: str = "bf16"
    mamba_cache_mode: str = "none"
    enable_thinking: bool = False
    tool_call_parser: str = "none"
    rope_scaling: str = "default"
    ctx_size: int = 4096

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointIdentity:
    """Deterministic content identity of a Safetensors checkpoint directory."""

    layout_sha256: str
    fingerprint: str
    n_tensors: int
    config: dict[str, str]
    tokenizer: dict[str, str]


@dataclass(frozen=True)
class BindingManifest:
    """Five-layer binding envelope stamped onto every Weft adapter."""

    schema: str
    adapter_name: str
    adapter_class: str
    layout_sha256: str
    checkpoint_fingerprint: str
    n_tensors: int
    config_hashes: dict[str, str]
    tokenizer_hashes: dict[str, str]
    plane1_hashes: dict[str, str]
    derivation_recipe: Optional[dict[str, Any]]
    runtime_flags: dict[str, Any]
    runtime_flags_sha256: str
    factor_digest: str
    certificate_sha256: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BindingVerification:
    ok: bool
    problems: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class SafetensorsHeaderError(ValueError):
    """A safetensors header the format does not allow.

    ``code`` is a stable machine-readable reason. The message starts with it.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SafetensorsHeaderError("DUPLICATE_KEY", f"key {key!r} appears twice")
        out[key] = value
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_tensor_table(header: dict[str, Any], data_len: int) -> None:
    """Check shapes, offsets and the byte layout the format requires.

    Tensor spans, taken in offset order, must start at 0, follow each other
    with no gap and no overlap, and end exactly at the end of the data area.
    Empty tensors (zero-length spans) are allowed.
    """
    spans: list[tuple[int, int, str]] = []
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
            ):
                raise SafetensorsHeaderError(
                    "BAD_METADATA", "__metadata__ must map strings to strings"
                )
            continue
        if not isinstance(meta, dict):
            raise SafetensorsHeaderError("BAD_ENTRY", f"entry {name!r} must be an object")
        if not isinstance(meta.get("dtype"), str):
            raise SafetensorsHeaderError("BAD_ENTRY", f"dtype of {name!r} must be a string")
        shape = meta.get("shape")
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise SafetensorsHeaderError(
                "BAD_SHAPE", f"shape of {name!r} must be a list of non-negative integers"
            )
        offsets = meta.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(o) and o >= 0 for o in offsets)
            or offsets[1] < offsets[0]
        ):
            raise SafetensorsHeaderError(
                "BAD_OFFSETS", f"data_offsets of {name!r} must be two ordered non-negative integers"
            )
        spans.append((offsets[0], offsets[1], name))

    cursor = 0
    for lo, hi, name in sorted(spans):
        if lo < cursor:
            raise SafetensorsHeaderError("OVERLAPPING_SPANS", f"{name!r} starts at {lo}, before {cursor}")
        if lo > cursor:
            raise SafetensorsHeaderError("GAP_BETWEEN_SPANS", f"bytes {cursor}..{lo} belong to no tensor")
        cursor = hi
    if cursor < data_len:
        raise SafetensorsHeaderError("GAP_BETWEEN_SPANS", f"bytes {cursor}..{data_len} belong to no tensor")
    if cursor > data_len:
        raise SafetensorsHeaderError("TRUNCATED_DATA", f"spans end at {cursor}, data area is {data_len}")


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Read Safetensors JSON header and return (header_dict, data_start_offset).

    Rejects headers the safetensors format disallows, with a
    ``SafetensorsHeaderError`` (a ``ValueError``): duplicate keys, non-integer
    shapes, overlapping spans, and bytes that belong to no tensor.
    """
    with open(path, "rb") as f:
        len_bytes = f.read(8)
        if len(len_bytes) != 8:
            raise ValueError(f"Truncated safetensors header length: {path.name}")
        header_len = struct.unpack("<Q", len_bytes)[0]
        if header_len == 0 or header_len > 100_000_000:
            raise ValueError(f"Unreasonable safetensors header length: {header_len}")
        hdr_raw = f.read(header_len)
        if len(hdr_raw) != header_len:
            raise ValueError(f"Truncated safetensors JSON header: {path.name}")
        file_len = os.fstat(f.fileno()).st_size
    header = json.loads(hdr_raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object.")
    _validate_tensor_table(header, file_len - 8 - header_len)
    return header, 8 + header_len


def _hash_span(path: Path, offset: int, length: int) -> str:
    h = hashlib.sha256()
    remaining = length
    with open(path, "rb") as f:
        f.seek(offset)
        while remaining > 0:
            chunk = f.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError(f"Unexpected EOF hashing tensor span in {path.name}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def identify_checkpoint(checkpoint_dir: Path | str, *, full: bool = True) -> CheckpointIdentity:
    """Compute layout_sha256, full tensor fingerprint, and config/tokenizer digests."""
    ck = Path(checkpoint_dir)
    if not ck.is_dir():
        raise ValueError("checkpoint_dir must be an existing directory.")

    shards = sorted(ck.glob("*.safetensors"))
    if not shards:
        raise ValueError("No *.safetensors files found in checkpoint_dir.")

    tensor_entries: list[tuple[str, str, list[int], Path, int, int]] = []
    seen_names: set[str] = set()
    for shard in shards:
        header, data_offset = read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in seen_names:
                raise ValueError(f"Duplicate tensor across shards: {name}")
            seen_names.add(name)
            dtype = str(meta["dtype"])
            shape = [int(d) for d in meta["shape"]]
            start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            if end < start:
                raise ValueError(f"Invalid data_offsets for tensor {name}")
            tensor_entries.append((name, dtype, shape, shard, data_offset + start, end - start))

    tensor_entries.sort(key=lambda item: item[0])

    layout_h = hashlib.sha256()
    fp_h = hashlib.sha256()
    for name, dtype, shape, shard, abs_off, nbytes in tensor_entries:
        shape_str = ",".join(str(d) for d in shape)
        layout_h.update(f"{name}\t{dtype}\t{shape_str}\n".encode("utf-8"))
        if full:
            t_digest = _hash_span(shard, abs_off, nbytes)
            fp_h.update(f"{name}\t{dtype}\t{shape_str}\t{t_digest}\n".encode("utf-8"))

    config_hashes: dict[str, str] = {}
    for fname in CONFIG_FILES:
        fpath = ck / fname
        if fpath.is_file():
            config_hashes[fname] = sha256_file(fpath)

    tok_hashes: dict[str, str] = {}
    for fname in TOKENIZER_FILES:
        fpath = ck / fname
        if fpath.is_file():
            tok_hashes[fname] = sha256_file(fpath)

    return CheckpointIdentity(
        layout_sha256=layout_h.hexdigest(),
        fingerprint=fp_h.hexdigest() if full else "",
        n_tensors=len(tensor_entries),
        config=config_hashes,
        tokenizer=tok_hashes,
    )


def hash_plane1_sources(root: Path | str, relative_paths: Sequence[str]) -> dict[str, str]:
    """Hash pinned Plane 1 source documents strictly confined under root."""
    base = Path(root).resolve()
    out: dict[str, str] = {}
    for rel in sorted(set(relative_paths)):
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(f"Invalid non-canonical Plane 1 path: {rel!r}")
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Plane 1 path escapes root: {rel!r}") from None
        if not target.is_file():
            raise ValueError(f"Plane 1 source file not found: {rel!r}")
        out[rel] = sha256_file(target)
    return out


def build_binding_manifest(
    *,
    adapter_name: str,
    adapter_class: str,
    checkpoint_dir: Path | str,
    plane1_root: Optional[Path | str] = None,
    plane1_files: Sequence[str] = (),
    plane1_hashes: Optional[Mapping[str, str]] = None,
    derivation_recipe: Optional[dict[str, Any]] = None,
    runtime_flags: Optional[RuntimeFlags] = None,
    factor_digest: str,
    certificate_sha256: str,
) -> BindingManifest:
    """Create a 5-layer binding manifest pinning checkpoint, Plane 1, and runtime state."""
    if adapter_class not in (ADAPTER_CLASS_1, ADAPTER_CLASS_2):
        raise ValueError(f"Unknown adapter_class: {adapter_class!r}")
    if adapter_class == ADAPTER_CLASS_1 and not derivation_recipe:
        raise ValueError("Hard Veto 1: Class 1 adapters require a non-empty derivation_recipe.")

    ident = identify_checkpoint(checkpoint_dir, full=True)
    if plane1_hashes is not None:
        p1 = {str(k): str(v) for k, v in sorted(plane1_hashes.items())}
    elif plane1_root is not None and plane1_files:
        p1 = hash_plane1_sources(plane1_root, plane1_files)
    else:
        p1 = {}

    if adapter_class == ADAPTER_CLASS_1 and not p1:
        raise ValueError("Hard Veto 1: Class 1 adapters must pin at least one Plane 1 source hash.")

    flags = runtime_flags if runtime_flags is not None else RuntimeFlags()
    return BindingManifest(
        schema=BINDING_SCHEMA,
        adapter_name=adapter_name,
        adapter_class=adapter_class,
        layout_sha256=ident.layout_sha256,
        checkpoint_fingerprint=ident.fingerprint,
        n_tensors=ident.n_tensors,
        config_hashes=ident.config,
        tokenizer_hashes=ident.tokenizer,
        plane1_hashes=p1,
        derivation_recipe=derivation_recipe,
        runtime_flags=flags.canonical_dict(),
        runtime_flags_sha256=flags.sha256(),
        factor_digest=factor_digest,
        certificate_sha256=certificate_sha256,
    )


def verify_binding(
    manifest: BindingManifest,
    *,
    checkpoint_dir: Path | str,
    runtime_flags: RuntimeFlags,
    plane1_root: Optional[Path | str] = None,
    expected_factor_digest: Optional[str] = None,
    expected_certificate_sha256: Optional[str] = None,
    require_class1_derivation: bool = True,
) -> BindingVerification:
    """Fail-closed verifier checking all 5 layers before GPU mount."""
    problems: list[str] = []

    if manifest.schema != BINDING_SCHEMA:
        problems.append(f"unsupported_schema:{manifest.schema}")

    if manifest.adapter_class not in (ADAPTER_CLASS_1, ADAPTER_CLASS_2):
        problems.append(f"unknown_adapter_class:{manifest.adapter_class}")

    if require_class1_derivation and manifest.adapter_class == ADAPTER_CLASS_1:
        if not isinstance(manifest.derivation_recipe, dict) or not manifest.derivation_recipe:
            problems.append("missing_class1_derivation")
        if not manifest.plane1_hashes:
            problems.append("missing_plane1_lineage")

    # Layer 1-3: checkpoint layout, full tensor bytes, config, tokenizer/template
    try:
        live = identify_checkpoint(checkpoint_dir, full=True)
    except Exception as exc:
        problems.append(f"checkpoint_read_error:{exc}")
        return BindingVerification(ok=False, problems=problems)

    if live.layout_sha256 != manifest.layout_sha256 or live.n_tensors != manifest.n_tensors:
        problems.append("incompatible_layout")
    if live.fingerprint != manifest.checkpoint_fingerprint:
        problems.append("checkpoint_fingerprint_mismatch")
    if live.config != manifest.config_hashes:
        problems.append("config_drift")
    if live.tokenizer != manifest.tokenizer_hashes:
        problems.append("tokenizer_or_chat_template_drift")

    # Layer 4: live Plane 1 source freshness check (stale_lineage detection)
    if plane1_root is not None and manifest.plane1_hashes:
        try:
            live_p1 = hash_plane1_sources(plane1_root, list(manifest.plane1_hashes.keys()))
            if live_p1 != manifest.plane1_hashes:
                problems.append("stale_lineage")
        except Exception:
            problems.append("stale_lineage")

    # Layer 5: serving runtime flags (verify internal manifest consistency AND live flags)
    manifest_flags_raw = json.dumps(
        manifest.runtime_flags, sort_keys=True, separators=(",", ":")
    )
    manifest_flags_sha = hashlib.sha256(manifest_flags_raw.encode("utf-8")).hexdigest()
    if manifest_flags_sha != manifest.runtime_flags_sha256:
        problems.append("manifest_runtime_flags_digest_mismatch")
    if runtime_flags.sha256() != manifest.runtime_flags_sha256:
        problems.append("runtime_flag_drift")

    # Payload binding checks
    if expected_factor_digest is not None and manifest.factor_digest != expected_factor_digest:
        problems.append("factor_digest_mismatch")
    if (
        expected_certificate_sha256 is not None
        and manifest.certificate_sha256 != expected_certificate_sha256
    ):
        problems.append("certificate_digest_mismatch")

    return BindingVerification(ok=not problems, problems=problems)
