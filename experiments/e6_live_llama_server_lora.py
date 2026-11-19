"""Experiment E6: Live llama-server C++ Multi-LoRA Verification on Real 1.5B GGUF ($0 cloud).

Strictly env-gated (`WEFT_LIVE_BASE_GGUF` and optional `LLAMA_SERVER_BIN`) so
default offline `pytest` stays 100 percent deterministic with zero external
binary or cache dependency and zero hardcoded host paths.

When `WEFT_LIVE_BASE_GGUF` is supplied (for example, a locally cached
`Qwen2.5-1.5B-Instruct` GGUF), E6 proves end-to-end inside the real compiled
`llama-server` C++ binary:
  1. Read-only GGUF v3 geometry inspection (`read_gguf_model_geometry`) to read
     exact `general.architecture`, `block_count`, and final-layer `ffn_up.weight`
     `(d_out, d_in)` dimensions.
  2. Real CPU float64 `BetaEditNullspaceProjector` compilation of two orthogonal
     rank-4 F16 adapters (`adapter_0`, `adapter_1`) plus an exact cancellation
     adapter (`adapter_neg_0 = (-B0, A0)`), with division-free
     `FixedRadixCertificate` checks and 0-byte GGUF v3 re-derivation.
  3. Simultaneous mounting via fail-closed `build_llama_server_lora_flags`
     (`--lora-scaled ...:0.0 --lora-init-without-apply --slot-save-path ...`)
     on an ephemeral localhost port (always terminated cleanly in `finally`).
  4. Deterministic (`temperature = 0`, `seed = 42`, `cache_prompt = False`)
     generation across six live states:
     (a) Pure Base `W0` (separate server with zero `--lora` flags)
     (b) Zero-initialized multi-LoRA `[0.0, 0.0, 0.0]` (must match Pure Base bit-for-bit)
     (c) Solo Adapter 0 `[1.0, 0.0, 0.0]`
     (d) Solo Adapter 1 `[0.0, 1.0, 0.0]`
     (e) Simultaneous 2-adapter forward pass `[1.0, 1.0, 0.0]`
     (f) Exact linear cancellation `[1.0, 0.0, 1.0]` (`+B0 A0 + (-B0) A0 == 0`)
     (g) O(1) kill switch `[0.0, 0.0, 0.0]` + `POST /slots/0?action=erase`
         (must restore Pure Base `W0` tokens and logprobs bit-for-bit).

Run:
  WEFT_LIVE_BASE_GGUF=/path/to/qwen2.5-1.5b-instruct.gguf \
    .venv/bin/python experiments/e6_live_llama_server_lora.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.gguf_lora import (
    build_llama_server_lora_flags,
    read_gguf_lora,
    read_gguf_model_geometry,
    write_gguf_lora,
)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: Any, timeout: float = 30.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_llama_server_version(llama_bin: str) -> str:
    proc = subprocess.run(
        [llama_bin, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    combined = (proc.stdout + "\n" + proc.stderr).splitlines()
    for line in combined:
        if "version:" in line.lower():
            return line.strip()
    return combined[0].strip() if combined else "unknown"


def _start_llama_server(
    llama_bin: str,
    base_gguf: Path,
    port: int,
    extra_args: list[str],
    *,
    timeout_s: float = 30.0,
) -> tuple[subprocess.Popen[str], float]:
    cmd = [
        llama_bin,
        "--model",
        str(base_gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "512",
        "--n-gpu-layers",
        "0",
        "--parallel",
        "1",
        *extra_args,
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            out, _ = proc.communicate(timeout=5)
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}):\n{out[-1200:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                if resp.status == 200:
                    load_ms = (time.perf_counter() - t0) * 1000.0
                    return proc, load_ms
        except OSError:
            time.sleep(0.15)
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    raise RuntimeError(f"Timed out waiting for llama-server /health:\n{out[-1200:]}")


def _stop_llama_server(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


def _max_logprob_diff(probs_a: list[dict[str, Any]], probs_b: list[dict[str, Any]]) -> float:
    if len(probs_a) != len(probs_b):
        return float("inf")
    max_d = 0.0
    for pa, pb in zip(probs_a, probs_b):
        if int(pa["id"]) != int(pb["id"]):
            return float("inf")
        max_d = max(max_d, abs(float(pa["logprob"]) - float(pb["logprob"])))
    return max_d


def run_experiment(
    *,
    base_gguf_path: Optional[str] = None,
    llama_server_bin: Optional[str] = None,
) -> dict[str, Any]:
    gguf_env = base_gguf_path or os.environ.get("WEFT_LIVE_BASE_GGUF", "").strip()
    if not gguf_env:
        return {
            "experiment": "E6_live_llama_server_multi_lora",
            "skipped": True,
            "reason": "WEFT_LIVE_BASE_GGUF not set",
        }

    base_gguf = Path(gguf_env).expanduser().resolve()
    if not base_gguf.is_file():
        raise FileNotFoundError(f"Base GGUF file not found: {base_gguf.name}")

    llama_bin = llama_server_bin or os.environ.get("LLAMA_SERVER_BIN") or shutil.which("llama-server")
    if not llama_bin:
        return {
            "experiment": "E6_live_llama_server_multi_lora",
            "skipped": True,
            "reason": "llama-server binary not found on PATH or LLAMA_SERVER_BIN",
        }

    version_str = _get_llama_server_version(llama_bin)

    # 1. Read-only GGUF model geometry inspection
    geom_all = read_gguf_model_geometry(base_gguf)
    arch = geom_all["architecture"]
    last_blk = max(0, int(geom_all["block_count"]) - 1)
    target_tensor = f"blk.{last_blk}.ffn_up.weight"
    if target_tensor not in geom_all["tensors"]:
        target_tensor = f"blk.{last_blk}.ffn_down.weight"
    if target_tensor not in geom_all["tensors"]:
        raise ValueError(f"Could not find final-layer FFN weight tensor in {base_gguf.name}.")
    d_out, d_in = geom_all["tensors"][target_tensor]["numpy_shape"]

    # 2. Deterministic CPU float64 BetaEdit compilation of 2 orthogonal adapters + cancellation
    rank = 4
    rng = np.random.default_rng(20261119)
    w0_zero = np.zeros((d_out, d_in), dtype=np.float64)
    k0 = rng.standard_normal((d_in, 16))
    keys_0 = rng.standard_normal((d_in, rank))
    res_0 = rng.standard_normal((d_out, rank)) * 0.45
    keys_1 = rng.standard_normal((d_in, rank))
    res_1 = rng.standard_normal((d_out, rank)) * 0.45

    cfg = BetaEditConfig(
        d_in=d_in,
        d_out=d_out,
        max_rank=rank,
        max_slots=8,
        lambda_leakage=500.0,
        lambda_ridge=1.0,
        nullspace_threshold=1e-6,
        threshold_mode="absolute",
        quant_dtype="f16",
    )
    proj = BetaEditNullspaceProjector(w0_zero, cfg)
    proj.initialize_anchor_covariance(k0)

    t_compile = time.perf_counter()
    patch_0 = proj.stage_edit(keys_0, res_0, is_residual=True, rank=rank, slot_name="adapter_0")
    rederive_0_ok, _ = verify_rederivation(
        patch_0,
        base_weight=w0_zero,
        config=cfg,
        k0=k0,
        keys=keys_0,
        target_values_or_residuals=res_0,
        is_residual=True,
        rank=rank,
    )
    policy = GatePolicy(max_frob_sq_num=50, max_frob_sq_den=1, max_leakage_num=1, max_leakage_den=1)
    cert_0_ok, cert_0_probs = patch_0.certificate.verify_division_free(policy)
    proj.commit_patch(patch_0, keys_0, gate_policy=policy)

    patch_1 = proj.stage_edit(keys_1, res_1, is_residual=True, rank=rank, slot_name="adapter_1")
    cert_1_ok, cert_1_probs = patch_1.certificate.verify_division_free(policy)
    proj.commit_patch(patch_1, keys_1, gate_policy=policy)
    compile_ms = (time.perf_counter() - t_compile) * 1000.0

    prompt = "Answer in one short sentence. What is 2 + 2?"
    comp_req = {
        "prompt": prompt,
        "n_predict": 16,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "n_probs": 3,
    }

    with tempfile.TemporaryDirectory(prefix="weft_e6_") as tmp:
        tdp = Path(tmp)
        slot_dir = tdp / "slots"
        slot_dir.mkdir()

        a0_path = tdp / "adapter_0.gguf"
        a0_rebuild = tdp / "adapter_0_rebuild.gguf"
        a1_path = tdp / "adapter_1.gguf"
        a1_rebuild = tdp / "adapter_1_rebuild.gguf"
        a0_neg_path = tdp / "adapter_neg_0.gguf"

        _, sha_0 = write_gguf_lora(
            a0_path,
            architecture=arch,
            lora_alpha=float(rank),
            layer_factors={target_tensor: (patch_0.factor_b, patch_0.factor_a)},
            extra_metadata={
                "weft.slot": "adapter_0",
                "weft.certificate_sha256": patch_0.certificate.sha256(),
            },
        )
        _, sha_0_rb = write_gguf_lora(
            a0_rebuild,
            architecture=arch,
            lora_alpha=float(rank),
            layer_factors={target_tensor: (patch_0.factor_b, patch_0.factor_a)},
            extra_metadata={
                "weft.slot": "adapter_0",
                "weft.certificate_sha256": patch_0.certificate.sha256(),
            },
        )
        _, sha_1 = write_gguf_lora(
            a1_path,
            architecture=arch,
            lora_alpha=float(rank),
            layer_factors={target_tensor: (patch_1.factor_b, patch_1.factor_a)},
            extra_metadata={
                "weft.slot": "adapter_1",
                "weft.certificate_sha256": patch_1.certificate.sha256(),
            },
        )
        _, sha_1_rb = write_gguf_lora(
            a1_rebuild,
            architecture=arch,
            lora_alpha=float(rank),
            layer_factors={target_tensor: (patch_1.factor_b, patch_1.factor_a)},
            extra_metadata={
                "weft.slot": "adapter_1",
                "weft.certificate_sha256": patch_1.certificate.sha256(),
            },
        )
        _, sha_neg = write_gguf_lora(
            a0_neg_path,
            architecture=arch,
            lora_alpha=float(rank),
            layer_factors={target_tensor: (-patch_0.factor_b, patch_0.factor_a)},
            extra_metadata={"weft.slot": "adapter_neg_0"},
        )

        # Verify GGUF round-trip and exact 0-byte re-derivation
        gguf_0_rt = read_gguf_lora(a0_path)
        gguf_1_rt = read_gguf_lora(a1_path)
        gguf_rederive_ok = bool(
            sha_0 == sha_0_rb
            and sha_1 == sha_1_rb
            and hashlib.sha256(a0_path.read_bytes()).hexdigest() == sha_0
            and np.array_equal(gguf_0_rt.tensors[f"{target_tensor}.lora_b"].astype(np.float64), patch_0.factor_b)
            and np.array_equal(gguf_1_rt.tensors[f"{target_tensor}.lora_a"].astype(np.float64), patch_1.factor_a)
        )

        # 3a. Pure Base W0 server (zero --lora flags)
        port_base = _pick_free_port()
        proc_base, base_load_ms = _start_llama_server(
            llama_bin,
            base_gguf,
            port_base,
            ["--slots", "--no-cache-idle-slots", "--slot-prompt-similarity", "0.0", "--ctx-checkpoints", "0"],
        )
        try:
            res_pure = _http_post_json(f"http://127.0.0.1:{port_base}/completion", comp_req)
        finally:
            _stop_llama_server(proc_base)

        # 3b. Simultaneous multi-LoRA server with fail-closed zero-init + slot erase
        port_lora = _pick_free_port()
        lora_flags = build_llama_server_lora_flags(
            [a0_path, a1_path, a0_neg_path],
            initial_scale=0.0,
            slot_save_path=slot_dir,
        )
        proc_lora, lora_load_ms = _start_llama_server(
            llama_bin,
            base_gguf,
            port_lora,
            lora_flags,
        )
        try:
            url_root = f"http://127.0.0.1:{port_lora}"
            init_adapters = _http_get_json(f"{url_root}/lora-adapters")
            init_scales = [float(item["scale"]) for item in init_adapters]
            res_init = _http_post_json(f"{url_root}/completion", comp_req)

            # Solo Adapter 0: [1.0, 0.0, 0.0]
            _http_post_json(
                f"{url_root}/lora-adapters",
                [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}],
            )
            res_solo_0 = _http_post_json(f"{url_root}/completion", comp_req)

            # Solo Adapter 1: [0.0, 1.0, 0.0]
            _http_post_json(
                f"{url_root}/lora-adapters",
                [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}],
            )
            res_solo_1 = _http_post_json(f"{url_root}/completion", comp_req)

            # Simultaneous 2-adapter forward pass: [1.0, 1.0, 0.0]
            _http_post_json(
                f"{url_root}/lora-adapters",
                [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}],
            )
            res_both = _http_post_json(f"{url_root}/completion", comp_req)

            # Exact algebraic cancellation: [1.0, 0.0, 1.0] -> (+B0 A0) + (-B0 A0) == 0
            _http_post_json(
                f"{url_root}/lora-adapters",
                [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 1.0}],
            )
            res_cancel = _http_post_json(f"{url_root}/completion", comp_req)

            # O(1) kill switch + slot erase: [0.0, 0.0, 0.0]
            t_rb = time.perf_counter()
            _http_post_json(
                f"{url_root}/lora-adapters",
                [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}],
            )
            erase_resp = _http_post_json(f"{url_root}/slots/0?action=erase", {})
            rollback_ms = (time.perf_counter() - t_rb) * 1000.0
            res_rollback = _http_post_json(f"{url_root}/completion", comp_req)
        finally:
            _stop_llama_server(proc_lora)

        def _summary(resp: dict[str, Any]) -> dict[str, Any]:
            p0 = resp["completion_probabilities"][0]
            return {
                "content": resp["content"],
                "tok0_id": int(p0["id"]),
                "tok0_text": p0["token"],
                "tok0_logprob": round(float(p0["logprob"]), 6),
            }

        pure_probs = res_pure["completion_probabilities"]
        init_max_diff = _max_logprob_diff(res_init["completion_probabilities"], pure_probs)
        cancel_max_diff = _max_logprob_diff(res_cancel["completion_probabilities"], pure_probs)
        rollback_max_diff = _max_logprob_diff(res_rollback["completion_probabilities"], pure_probs)

        return {
            "experiment": "E6_live_llama_server_multi_lora",
            "skipped": False,
            "llama_server_version": version_str,
            "model_geometry": {
                "architecture": arch,
                "model_name": geom_all["model_name"],
                "block_count": geom_all["block_count"],
                "embedding_length": geom_all["embedding_length"],
                "feed_forward_length": geom_all["feed_forward_length"],
                "target_tensor": target_tensor,
                "target_shape_numpy": [d_out, d_in],
                "rank": rank,
            },
            "compilation_and_gguf": {
                "compile_two_adapters_ms": round(compile_ms, 2),
                "adapter_0_gguf_bytes": a0_path.stat().st_size,
                "adapter_1_gguf_bytes": a1_path.stat().st_size,
                "adapter_0_sha256_prefix": sha_0[:16],
                "adapter_1_sha256_prefix": sha_1[:16],
                "adapter_neg_0_sha256_prefix": sha_neg[:16],
                "compiler_zero_byte_rederivation": bool(rederive_0_ok),
                "gguf_zero_byte_rederivation_and_roundtrip": gguf_rederive_ok,
                "cert_0_ok": bool(cert_0_ok and not cert_0_probs),
                "cert_1_ok": bool(cert_1_ok and not cert_1_probs),
            },
            "server_timings": {
                "pure_base_load_ms": round(base_load_ms, 2),
                "multi_lora_load_ms": round(lora_load_ms, 2),
                "rollback_plus_slot_erase_ms": round(rollback_ms, 2),
                "slot_erase_response": erase_resp,
                "initial_scales": init_scales,
            },
            "states": {
                "pure_base_w0": _summary(res_pure),
                "init_zero_scales": _summary(res_init),
                "solo_adapter_0": _summary(res_solo_0),
                "solo_adapter_1": _summary(res_solo_1),
                "simultaneous_both_0_1": _summary(res_both),
                "cancellation_0_plus_neg0": _summary(res_cancel),
                "rollback_zero_plus_erase": _summary(res_rollback),
            },
            "verifications": {
                "init_zero_scales_all_zero": all(s == 0.0 for s in init_scales),
                "init_matches_pure_base_bit_exact": bool(
                    res_init["content"] == res_pure["content"] and init_max_diff == 0.0
                ),
                "solo_0_shifts_from_base": bool(
                    res_solo_0["completion_probabilities"][0]["logprob"]
                    != pure_probs[0]["logprob"]
                ),
                "solo_1_shifts_from_base": bool(
                    res_solo_1["completion_probabilities"][0]["logprob"]
                    != pure_probs[0]["logprob"]
                ),
                "simultaneous_differs_from_both_solos": bool(
                    res_both["completion_probabilities"][0]["logprob"]
                    != res_solo_0["completion_probabilities"][0]["logprob"]
                    and res_both["completion_probabilities"][0]["logprob"]
                    != res_solo_1["completion_probabilities"][0]["logprob"]
                ),
                "cancellation_matches_pure_base_content": bool(
                    res_cancel["content"] == res_pure["content"]
                ),
                "cancellation_max_logprob_diff": round(cancel_max_diff, 8),
                "rollback_matches_pure_base_bit_exact": bool(
                    res_rollback["content"] == res_pure["content"] and rollback_max_diff == 0.0
                ),
                "rollback_max_logprob_diff": round(rollback_max_diff, 8),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Weft E6 live llama-server multi-LoRA verification")
    parser.add_argument("--base-gguf", default=None, help="Path to base GGUF model (or WEFT_LIVE_BASE_GGUF)")
    parser.add_argument("--llama-server", default=None, help="Path to llama-server binary")
    args = parser.parse_args()
    res = run_experiment(base_gguf_path=args.base_gguf, llama_server_bin=args.llama_server)
    print(json.dumps(res, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
