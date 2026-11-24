"""Fail-closed real-weight provenance and synthetic-fallback guards for Experiments E3-E5.

Closes siliconcordat/weft#6:
- Prevents synthetic fallback weights from silently masquerading as real pretrained
  ``roneneldan/TinyStories-1M`` weights.
- Enforces fail-closed rejection (``RealWeightsRequiredError``) whenever
  ``require_real_weights=True``, CLI ``--require-real-weights``, ``--expected-sha256``,
  or ``WEFT_REQUIRE_REAL_WEIGHTS=1`` is active and verified real weights are absent
  or a synthetic fixture is supplied.
- Detects stock ``roneneldan/TinyStories-1M`` ``main`` caches (``pytorch_model.bin``
  without ``model.safetensors``) and reports the required ``refs/pr/12`` conversion
  and pinned SHA-256 explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Any

TINYSTORIES_1M_REPO = "roneneldan/TinyStories-1M"
TINYSTORIES_1M_REVISION = "refs/pr/12"
TINYSTORIES_1M_SAFETENSORS_SHA256 = (
    "eb9c5893cb5dc6f78b9b5703ab77b9e94f0b056b7bd920c8fa0146c78af5b55b"
)
SYNTHETIC_FALLBACK_LABEL = "synthetic_gptneo_8layer_fallback"
SYNTHETIC_SAFETENSORS_SHA256 = (
    "cd62ff6e11c703e627bfd8a362aba9fe3a027f1a8dfb08f4ceb976ca8c46fe92"
)
ENV_REQUIRE_REAL_WEIGHTS = "WEFT_REQUIRE_REAL_WEIGHTS"
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "config.json", "vocab.json")
_OPTIONAL_CACHE_FILES = ("config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja")
_HEX_LOWER = frozenset("0123456789abcdef")
_EXACT_PATH_TYPES = (Path, type(Path(".")))
_MAX_SYNTHETIC_TOKENIZER_BYTES = 262144


class RealWeightsRequiredError(RuntimeError):
    """Fail-closed provenance error raised when real pretrained weights cannot be verified."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CheckpointProvenance:
    checkpoint_source: str
    provenance_tier: str
    pin_source: str
    is_synthetic_fallback: bool
    real_weights_verified: bool
    weights_evidence_claim: str
    safetensors_sha256: str
    expected_known_sha256: str
    matches_known_tinystories_1m: bool
    require_real_weights_enforced: bool
    cache_miss_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_source": self.checkpoint_source,
            "provenance_tier": self.provenance_tier,
            "pin_source": self.pin_source,
            "is_synthetic_fallback": self.is_synthetic_fallback,
            "real_weights_verified": self.real_weights_verified,
            "weights_evidence_claim": self.weights_evidence_claim,
            "safetensors_sha256": self.safetensors_sha256,
            "expected_known_sha256": self.expected_known_sha256,
            "matches_known_tinystories_1m": self.matches_known_tinystories_1m,
            "require_real_weights_enforced": self.require_real_weights_enforced,
            "cache_miss_reason": self.cache_miss_reason,
        }


def validate_sha256_hex(digest: Any, label: str = "expected_safetensors_sha256") -> str:
    """Require an exact 64-character lowercase hexadecimal SHA-256 string."""
    if type(digest) is not str or len(digest) != 64:
        raise RealWeightsRequiredError("INVALID_SHA256", f"{label} must be 64-char lowercase hex str")
    for ch in digest:
        if ch not in _HEX_LOWER:
            raise RealWeightsRequiredError("INVALID_SHA256", f"{label} contains non-hex character")
    return digest


def env_requires_real_weights(env: Mapping[str, str] | None = None) -> bool:
    """Parse WEFT_REQUIRE_REAL_WEIGHTS strictly (rejecting unrecognized non-empty strings)."""
    src = os.environ if env is None else env
    raw = src.get(ENV_REQUIRE_REAL_WEIGHTS, "")
    if type(raw) is not str:
        raise RealWeightsRequiredError("INVALID_ENV_FLAG", f"{ENV_REQUIRE_REAL_WEIGHTS} must be str")
    norm = raw.strip().lower()
    if norm in ("", "0", "false", "no"):
        return False
    if norm in ("1", "true", "yes"):
        return True
    raise RealWeightsRequiredError(
        "INVALID_ENV_FLAG",
        f"unrecognized {ENV_REQUIRE_REAL_WEIGHTS}={raw!r}; expected one of 0/1/true/false/yes/no",
    )


def coerce_exact_path(path: Any, label: str = "checkpoint_dir") -> Path:
    """Accept exact str or exact concrete Path (rejecting subclasses, empty strings, NUL bytes)."""
    if type(path) is str:
        if not path or "\x00" in path:
            raise RealWeightsRequiredError("INVALID_PATH", f"{label} must be non-empty str without NUL")
        return Path(path)
    if type(path) in _EXACT_PATH_TYPES:
        if "\x00" in str(path):
            raise RealWeightsRequiredError("INVALID_PATH", f"{label} must not contain NUL")
        return path
    raise RealWeightsRequiredError("INVALID_PATH", f"{label} must be exact str or Path")


def _sha256_regular_file(path: Path, label: str) -> str:
    """Hash a regular file (following valid HF blob symlinks) while rejecting dirs/FIFOs/missing files."""
    try:
        st = os.stat(path)
    except OSError as exc:
        raise RealWeightsRequiredError("MISSING_CHECKPOINT_FILE", f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise RealWeightsRequiredError("NOT_A_REGULAR_FILE", f"{label} is not a regular file")
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as exc:
        raise RealWeightsRequiredError("CHECKPOINT_READ_ERROR", f"failed to read {label}: {exc}") from exc
    return hasher.hexdigest()


def _tokenizer_declares_synthetic(tok_path: Path) -> bool:
    """Detect synthetic fallback markers (case-insensitive ``version == 'synthetic'``) in tokenizer.json."""
    try:
        st = os.stat(tok_path)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    try:
        with open(tok_path, "rb") as fh:
            raw = fh.read(_MAX_SYNTHETIC_TOKENIZER_BYTES + 1)
    except OSError:
        return False
    if len(raw) > _MAX_SYNTHETIC_TOKENIZER_BYTES:
        return False
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    version = parsed.get("version")
    return isinstance(version, str) and version.strip().lower() == "synthetic"


def _stage_symlink(src: Path, dst: Path) -> None:
    """Create a staging symlink and wrap pre-existing destination / OS failures fail-closed."""
    try:
        os.symlink(src, dst)
    except OSError as exc:
        raise RealWeightsRequiredError(
            "STAGING_DIR_ERROR",
            f"cannot stage checkpoint file {dst.name} into {dst.parent}: {exc}",
        ) from exc


def discover_tinystories_cache_detailed(
    env: Mapping[str, str] | None = None,
) -> tuple[Path | None, Path | None, str | None]:
    """Locate cached TinyStories-1M safetensors + config snapshots and diagnose cache misses."""
    src = os.environ if env is None else env
    hf_home_val = src.get("HF_HOME")
    hf_home = Path(hf_home_val) if hf_home_val else (Path.home() / ".cache" / "huggingface")
    repo_dir = hf_home / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots"
    if not repo_dir.is_dir():
        return None, None, "no_tinystories_cache"
    st_file: Path | None = None
    cfg_dir: Path | None = None
    saw_pytorch_bin = False
    for snap in sorted(repo_dir.iterdir()):
        if not snap.is_dir():
            continue
        if (snap / "model.safetensors").is_file():
            st_file = (snap / "model.safetensors").resolve()
        elif (snap / "pytorch_model.bin").is_file():
            saw_pytorch_bin = True
        if (snap / "config.json").is_file() and (snap / "vocab.json").is_file():
            cfg_dir = snap.resolve()
    if st_file is not None and cfg_dir is not None:
        return st_file, cfg_dir, None
    if saw_pytorch_bin:
        return None, None, "tinystories_cache_missing_safetensors_use_refs_pr_12"
    return None, None, "incomplete_tinystories_cache"


def _stage_user_checkpoint_dir(src: Path, dest_dir: Path) -> None:
    if not src.is_dir():
        raise RealWeightsRequiredError("CHECKPOINT_DIR_NOT_FOUND", f"checkpoint_dir does not exist: {src}")
    for req in REQUIRED_CHECKPOINT_FILES:
        req_path = src / req
        try:
            st = os.stat(req_path)
        except OSError as exc:
            raise RealWeightsRequiredError(
                "MISSING_CHECKPOINT_FILE", f"checkpoint_dir missing required file {req}"
            ) from exc
        if not stat.S_ISREG(st.st_mode):
            raise RealWeightsRequiredError("NOT_A_REGULAR_FILE", f"checkpoint_dir {req} is not a regular file")
    for item in sorted(src.iterdir()):
        if item.is_file():
            _stage_symlink(item.resolve(), dest_dir / item.name)


def resolve_experiment_checkpoint(
    dest_dir: Path,
    *,
    build_synthetic_fn: Callable[[Path], None],
    checkpoint_dir: str | Path | None = None,
    require_real_weights: bool | None = None,
    expected_safetensors_sha256: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CheckpointProvenance:
    """Resolve checkpoint files into ``dest_dir`` and return machine-readable weight provenance.

    Fail-closed rules:
    - When ``require_real_weights`` (or ``WEFT_REQUIRE_REAL_WEIGHTS=1``) is True:
      refuses fallback generation, refuses synthetic fixtures (matched by SHA-256 or
      ``tokenizer.json`` marker), and refuses unpinned ``checkpoint_dir`` or HF cache files
      whose SHA-256 does not match ``TINYSTORIES_1M_SAFETENSORS_SHA256`` unless an explicit
      ``expected_safetensors_sha256`` caller pin matches in constant time.
    - ``real_weights_verified=True`` and ``provenance_tier="verified_pretrained_safetensors"``
      are strictly reserved for checkpoints matching ``TINYSTORIES_1M_SAFETENSORS_SHA256``
      (``pin_source="known_pretrained_pin"``); caller-supplied SHA pins report
      ``pin_source="caller_supplied_pin"`` and ``real_weights_verified=False`` so self-pinning
      never conflates with known pretrained weight provenance.
    """
    if require_real_weights is None:
        enforced = env_requires_real_weights(env=env)
    elif type(require_real_weights) is bool:
        enforced = require_real_weights
    else:
        raise RealWeightsRequiredError("INVALID_REQUIRE_FLAG", "require_real_weights must be bool or None")

    expected_sha = (
        None if expected_safetensors_sha256 is None else validate_sha256_hex(expected_safetensors_sha256)
    )

    if checkpoint_dir is not None:
        src = coerce_exact_path(checkpoint_dir, "checkpoint_dir")
        _stage_user_checkpoint_dir(src, dest_dir)
        st_sha = _sha256_regular_file(dest_dir / "model.safetensors", "model.safetensors")
        is_syn = hmac.compare_digest(st_sha, SYNTHETIC_SAFETENSORS_SHA256) or _tokenizer_declares_synthetic(
            dest_dir / "tokenizer.json"
        )
        if is_syn:
            if enforced:
                raise RealWeightsRequiredError(
                    "SYNTHETIC_FIXTURE_REJECTED",
                    "checkpoint_dir contains synthetic fallback weights while real weights are required",
                )
            if expected_sha is not None and not hmac.compare_digest(st_sha, expected_sha):
                raise RealWeightsRequiredError(
                    "SHA256_MISMATCH",
                    f"checkpoint model.safetensors sha256 {st_sha} != expected {expected_sha}",
                )
            return CheckpointProvenance(
                checkpoint_source=SYNTHETIC_FALLBACK_LABEL,
                provenance_tier="synthetic_fallback",
                pin_source="none",
                is_synthetic_fallback=True,
                real_weights_verified=False,
                weights_evidence_claim="synthetic_fallback_only_not_real_weights",
                safetensors_sha256=st_sha,
                expected_known_sha256=TINYSTORIES_1M_SAFETENSORS_SHA256,
                matches_known_tinystories_1m=False,
                require_real_weights_enforced=enforced,
                cache_miss_reason=None,
            )
        if expected_sha is not None and not hmac.compare_digest(st_sha, expected_sha):
            raise RealWeightsRequiredError(
                "SHA256_MISMATCH",
                f"checkpoint model.safetensors sha256 {st_sha} != expected {expected_sha}",
            )
        matches_ts = hmac.compare_digest(st_sha, TINYSTORIES_1M_SAFETENSORS_SHA256)
        if enforced and not matches_ts and expected_sha is None:
            raise RealWeightsRequiredError(
                "UNVERIFIED_CHECKPOINT_SHA256",
                f"checkpoint_dir model.safetensors sha256 {st_sha} != pinned {TINYSTORIES_1M_REPO} "
                f"{TINYSTORIES_1M_REVISION} ({TINYSTORIES_1M_SAFETENSORS_SHA256}) and no --expected-sha256 supplied",
            )
        pin_source = (
            "known_pretrained_pin"
            if matches_ts
            else ("caller_supplied_pin" if expected_sha is not None else "none")
        )
        tier = (
            "verified_pretrained_safetensors"
            if matches_ts
            else ("caller_pinned_checkpoint" if expected_sha is not None else "unpinned_user_checkpoint")
        )
        claim = (
            "verified_tinystories_1m_refs_pr_12"
            if matches_ts
            else (
                "caller_supplied_sha256_pin_not_known_pretrained"
                if expected_sha is not None
                else "unpinned_user_checkpoint"
            )
        )
        return CheckpointProvenance(
            checkpoint_source="user_provided_checkpoint",
            provenance_tier=tier,
            pin_source=pin_source,
            is_synthetic_fallback=False,
            real_weights_verified=matches_ts,
            weights_evidence_claim=claim,
            safetensors_sha256=st_sha,
            expected_known_sha256=TINYSTORIES_1M_SAFETENSORS_SHA256,
            matches_known_tinystories_1m=matches_ts,
            require_real_weights_enforced=enforced,
            cache_miss_reason=None,
        )

    st_file, cfg_dir, miss_reason = discover_tinystories_cache_detailed(env=env)
    if st_file is not None:
        cfg_root = cfg_dir if cfg_dir is not None else st_file.parent
        _stage_symlink(st_file, dest_dir / "model.safetensors")
        for fname in _OPTIONAL_CACHE_FILES:
            if (cfg_root / fname).is_file():
                _stage_symlink((cfg_root / fname).resolve(), dest_dir / fname)
        st_sha = _sha256_regular_file(dest_dir / "model.safetensors", "model.safetensors")
        is_syn = hmac.compare_digest(st_sha, SYNTHETIC_SAFETENSORS_SHA256) or _tokenizer_declares_synthetic(
            dest_dir / "tokenizer.json"
        )
        if is_syn:
            if enforced:
                raise RealWeightsRequiredError(
                    "SYNTHETIC_FIXTURE_REJECTED",
                    "HF cache contains synthetic fallback weights while real weights are required",
                )
            if expected_sha is not None and not hmac.compare_digest(st_sha, expected_sha):
                raise RealWeightsRequiredError(
                    "SHA256_MISMATCH",
                    f"cached model.safetensors sha256 {st_sha} != expected {expected_sha}",
                )
            return CheckpointProvenance(
                checkpoint_source=SYNTHETIC_FALLBACK_LABEL,
                provenance_tier="synthetic_fallback",
                pin_source="none",
                is_synthetic_fallback=True,
                real_weights_verified=False,
                weights_evidence_claim="synthetic_fallback_only_not_real_weights",
                safetensors_sha256=st_sha,
                expected_known_sha256=TINYSTORIES_1M_SAFETENSORS_SHA256,
                matches_known_tinystories_1m=False,
                require_real_weights_enforced=enforced,
                cache_miss_reason=None,
            )
        matches_ts = hmac.compare_digest(st_sha, TINYSTORIES_1M_SAFETENSORS_SHA256)
        if expected_sha is not None and not hmac.compare_digest(st_sha, expected_sha):
            raise RealWeightsRequiredError(
                "SHA256_MISMATCH",
                f"cached model.safetensors sha256 {st_sha} != expected {expected_sha}",
            )
        if enforced and not matches_ts and expected_sha is None:
            raise RealWeightsRequiredError(
                "UNVERIFIED_CACHE_SHA256",
                f"cached model.safetensors sha256 {st_sha} != pinned {TINYSTORIES_1M_REPO} "
                f"{TINYSTORIES_1M_REVISION} ({TINYSTORIES_1M_SAFETENSORS_SHA256})",
            )
        pin_source = (
            "known_pretrained_pin"
            if matches_ts
            else ("caller_supplied_pin" if expected_sha is not None else "none")
        )
        ckpt_source = (
            f"pretrained_safetensors:{TINYSTORIES_1M_REPO}"
            if matches_ts
            else (
                f"caller_pinned_cache_safetensors:{TINYSTORIES_1M_REPO}"
                if expected_sha is not None
                else f"unverified_cache_safetensors:{TINYSTORIES_1M_REPO}"
            )
        )
        tier = (
            "verified_pretrained_safetensors"
            if matches_ts
            else (
                "caller_pinned_cache_safetensors"
                if expected_sha is not None
                else "unverified_cache_safetensors"
            )
        )
        claim = (
            "verified_tinystories_1m_refs_pr_12"
            if matches_ts
            else (
                "caller_supplied_sha256_pin_not_known_pretrained"
                if expected_sha is not None
                else "unverified_cache_sha256_mismatch"
            )
        )
        return CheckpointProvenance(
            checkpoint_source=ckpt_source,
            provenance_tier=tier,
            pin_source=pin_source,
            is_synthetic_fallback=False,
            real_weights_verified=matches_ts,
            weights_evidence_claim=claim,
            safetensors_sha256=st_sha,
            expected_known_sha256=TINYSTORIES_1M_SAFETENSORS_SHA256,
            matches_known_tinystories_1m=matches_ts,
            require_real_weights_enforced=enforced,
            cache_miss_reason=None,
        )

    if enforced or expected_sha is not None:
        raise RealWeightsRequiredError(
            "REAL_WEIGHTS_REQUIRED",
            f"real {TINYSTORIES_1M_REPO} safetensors weights not found ({miss_reason}); "
            f"download revision {TINYSTORIES_1M_REVISION} (model.safetensors sha256 "
            f"{TINYSTORIES_1M_SAFETENSORS_SHA256}) or pass --checkpoint-dir",
        )

    build_synthetic_fn(dest_dir)
    st_sha = _sha256_regular_file(dest_dir / "model.safetensors", "model.safetensors")
    return CheckpointProvenance(
        checkpoint_source=SYNTHETIC_FALLBACK_LABEL,
        provenance_tier="synthetic_fallback",
        pin_source="none",
        is_synthetic_fallback=True,
        real_weights_verified=False,
        weights_evidence_claim="synthetic_fallback_only_not_real_weights",
        safetensors_sha256=st_sha,
        expected_known_sha256=TINYSTORIES_1M_SAFETENSORS_SHA256,
        matches_known_tinystories_1m=False,
        require_real_weights_enforced=False,
        cache_miss_reason=miss_reason,
    )
