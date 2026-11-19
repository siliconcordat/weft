"""Experiment E3: Real pretrained checkpoint end-to-end smoke test ($0 cloud).

Loads an actual pretrained 8-layer GPT-Neo Safetensors checkpoint
(`roneneldan/TinyStories-1M`, 124 tensors, hidden_size=64, 8 layers,
vocab_size=50257) from the local Hugging Face cache if present (or --checkpoint-dir),
otherwise falls back transparently to a real 8-layer GPT-Neo architecture with
synthetic weights and labels the boundary explicitly.

Runs the full Weft pipeline end-to-end:
  1. Five-layer checkpoint + Plane 1 lineage + runtime-flag binding
  2. Real layer-7 MLP activation harvesting & K0 singular-value spectrum check
  3. Deterministic CPU float64 BetaEdit compile -> BF16 & F16 quantization
  4. Post-quantization division-free FixedRadixCertificate verification
  5. Deterministic GGUF v3 F16 LoRA binary export & round-trip readback
  6. Exact 0-byte deterministic re-derivation (factors + GGUF binary)
  7. Live forward-pass measurement: target logit shift, K0 retention, O(1) rollback
  8. Tamper rejection battery: 1-byte template drift, 1-byte Plane 1 stale_lineage,
     and runtime-flag drift.

Run:
  .venv/bin/python experiments/e3_real_checkpoint_smoke.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from weft.binding import (
    ADAPTER_CLASS_1,
    RuntimeFlags,
    build_binding_manifest,
    read_safetensors_header,
    verify_binding,
)
from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.gguf_lora import read_gguf_lora, write_gguf_lora


def _gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _softmax(x: np.ndarray) -> np.ndarray:
    xm = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(xm)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def load_safetensors_dict(path: Path) -> dict[str, np.ndarray]:
    header, data_offset = read_safetensors_header(path)
    raw = path.read_bytes()
    dtype_map = {"F32": np.float32, "F16": np.float16, "BOOL": np.bool_, "I64": np.int64}
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = dtype_map[str(meta["dtype"])]
        shape = tuple(int(d) for d in meta["shape"])
        start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        arr = np.frombuffer(raw[data_offset + start : data_offset + end], dtype=dt).reshape(shape)
        out[name] = arr.copy()
    return out


class PureNumpyGPTNeo:
    """Exact pure-NumPy forward pass for GPTNeoForCausalLM (e.g. TinyStories-1M)."""

    def __init__(self, weights: dict[str, np.ndarray], cfg: dict[str, Any]) -> None:
        self.w = weights
        self.n_layers = int(cfg["num_layers"])
        self.d_model = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_heads"])
        self.head_dim = self.d_model // self.n_heads
        self.eps = float(cfg.get("layer_norm_epsilon", 1e-5))
        self.layer_types = list(cfg.get("attention_layers", ["global"] * self.n_layers))
        self.window = int(cfg.get("window_size", 256))
        self.wte = self.w["transformer.wte.weight"].astype(np.float64)
        self.wpe = self.w["transformer.wpe.weight"].astype(np.float64)

    def _attn(self, h: np.ndarray, layer_idx: int) -> np.ndarray:
        seq_len, _ = h.shape
        pfx = f"transformer.h.{layer_idx}.attn.attention"
        q = (h @ self.w[f"{pfx}.q_proj.weight"].astype(np.float64).T).reshape(
            seq_len, self.n_heads, self.head_dim
        )
        k = (h @ self.w[f"{pfx}.k_proj.weight"].astype(np.float64).T).reshape(
            seq_len, self.n_heads, self.head_dim
        )
        v = (h @ self.w[f"{pfx}.v_proj.weight"].astype(np.float64).T).reshape(
            seq_len, self.n_heads, self.head_dim
        )
        # GPT-Neo does NOT divide QK^T by sqrt(head_dim) in HF GPTNeoSelfAttention
        scores = np.einsum("shd,thd->hst", q, k)
        causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        if self.layer_types[layer_idx] == "local":
            idx = np.arange(seq_len)
            local = (idx[:, None] - idx[None, :]) < self.window
            mask = causal & local
        else:
            mask = causal
        scores = np.where(mask[None, :, :], scores, -1e9)
        probs = _softmax(scores)
        ctx = np.einsum("hst,thd->shd", probs, v).reshape(seq_len, self.d_model)
        return (
            ctx @ self.w[f"{pfx}.out_proj.weight"].astype(np.float64).T
            + self.w[f"{pfx}.out_proj.bias"].astype(np.float64)
        )

    def forward_last_token(
        self,
        token_ids: list[int],
        *,
        edit_layer: int = 7,
        projector: BetaEditNullspaceProjector | None = None,
        slot_scales: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full 8-layer forward pass; return (last_token_logits, edit_layer_mlp_in_last)."""
        ids = np.array(token_ids, dtype=np.int64)
        pos = np.arange(len(token_ids), dtype=np.int64)
        h = self.wte[ids] + self.wpe[pos]
        captured_mlp_in = np.zeros(1, dtype=np.float64)

        for i in range(self.n_layers):
            pfx = f"transformer.h.{i}"
            h_ln1 = _layer_norm(
                h,
                self.w[f"{pfx}.ln_1.weight"].astype(np.float64),
                self.w[f"{pfx}.ln_1.bias"].astype(np.float64),
                self.eps,
            )
            h = h + self._attn(h_ln1, i)
            h_ln2 = _layer_norm(
                h,
                self.w[f"{pfx}.ln_2.weight"].astype(np.float64),
                self.w[f"{pfx}.ln_2.bias"].astype(np.float64),
                self.eps,
            )
            mlp_mid = _gelu_new(
                h_ln2 @ self.w[f"{pfx}.mlp.c_fc.weight"].astype(np.float64).T
                + self.w[f"{pfx}.mlp.c_fc.bias"].astype(np.float64)
            )
            if i == edit_layer:
                captured_mlp_in = mlp_mid[-1].copy()
                if projector is not None:
                    mlp_out = projector.forward(mlp_mid, slot_scales=slot_scales) + self.w[
                        f"{pfx}.mlp.c_proj.bias"
                    ].astype(np.float64)
                else:
                    mlp_out = (
                        mlp_mid @ self.w[f"{pfx}.mlp.c_proj.weight"].astype(np.float64).T
                        + self.w[f"{pfx}.mlp.c_proj.bias"].astype(np.float64)
                    )
            else:
                mlp_out = (
                    mlp_mid @ self.w[f"{pfx}.mlp.c_proj.weight"].astype(np.float64).T
                    + self.w[f"{pfx}.mlp.c_proj.bias"].astype(np.float64)
                )
            h = h + mlp_out

        h_f = _layer_norm(
            h[-1:],
            self.w["transformer.ln_f.weight"].astype(np.float64),
            self.w["transformer.ln_f.bias"].astype(np.float64),
            self.eps,
        )
        logits = (h_f @ self.wte.T)[0]
        return logits, captured_mlp_in


class SimpleWordTokenizer:
    """Minimal deterministic tokenizer using GPT-2 vocab.json word/prefix lookup."""

    def __init__(self, vocab_path: Path) -> None:
        self.vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.id_to_tok = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        vocab_size = max(1, len(self.vocab))
        for idx, word in enumerate(text.split(" ")):
            candidate = word if idx == 0 else f"\u0120{word}"
            if candidate in self.vocab:
                ids.append(self.vocab[candidate])
            elif word in self.vocab:
                ids.append(self.vocab[word])
            else:
                # Greedy character/byte fallback, or deterministic hash for synthetic vocabs
                matched_any = False
                for ch in candidate:
                    if ch in self.vocab:
                        ids.append(self.vocab[ch])
                        matched_any = True
                if not matched_any:
                    digest = hashlib.sha256(candidate.encode("utf-8")).digest()
                    ids.append(int.from_bytes(digest[:4], "little") % vocab_size)
        return ids or [0]

    def decode_token(self, tok_id: int) -> str:
        return self.id_to_tok.get(int(tok_id), f"<id:{tok_id}>").replace("\u0120", " ")


def discover_tinystories_cache() -> tuple[Path | None, Path | None]:
    """Locate cached roneneldan/TinyStories-1M safetensors + config snapshots without hardcoding paths."""
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    repo_dir = hf_home / "hub" / "models--roneneldan--TinyStories-1M" / "snapshots"
    if not repo_dir.is_dir():
        return None, None
    st_file: Path | None = None
    cfg_dir: Path | None = None
    for snap in sorted(repo_dir.iterdir()):
        if not snap.is_dir():
            continue
        if (snap / "model.safetensors").is_file():
            st_file = (snap / "model.safetensors").resolve()
        if (snap / "config.json").is_file() and (snap / "vocab.json").is_file():
            cfg_dir = snap.resolve()
    return st_file, cfg_dir


def _build_synthetic_gptneo_dir(dest: Path) -> None:
    """Fallback real 8-layer GPT-Neo safetensors fixture when no local cache exists."""
    rng = np.random.default_rng(20261119)
    cfg = {
        "architectures": ["GPTNeoForCausalLM"],
        "attention_layers": ["global", "local"] * 4,
        "hidden_size": 64,
        "layer_norm_epsilon": 1e-5,
        "model_type": "gpt_neo",
        "num_heads": 8,
        "num_layers": 8,
        "vocab_size": 512,
        "window_size": 256,
    }
    (dest / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    vocab = {f"tok{i}": i for i in range(512)}
    (dest / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (dest / "tokenizer.json").write_text('{"version":"synthetic"}\n', encoding="utf-8")
    (dest / "chat_template.jinja").write_text("{{ prompt }}", encoding="utf-8")

    tensors: dict[str, np.ndarray] = {
        "transformer.wte.weight": (rng.standard_normal((512, 64)) * 0.05).astype(np.float32),
        "transformer.wpe.weight": (rng.standard_normal((256, 64)) * 0.05).astype(np.float32),
        "transformer.ln_f.weight": np.ones(64, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(64, dtype=np.float32),
    }
    for i in range(8):
        p = f"transformer.h.{i}"
        tensors[f"{p}.ln_1.weight"] = np.ones(64, dtype=np.float32)
        tensors[f"{p}.ln_1.bias"] = np.zeros(64, dtype=np.float32)
        tensors[f"{p}.ln_2.weight"] = np.ones(64, dtype=np.float32)
        tensors[f"{p}.ln_2.bias"] = np.zeros(64, dtype=np.float32)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{p}.attn.attention.{proj}.weight"] = (
                rng.standard_normal((64, 64)) * 0.05
            ).astype(np.float32)
        tensors[f"{p}.attn.attention.out_proj.bias"] = np.zeros(64, dtype=np.float32)
        tensors[f"{p}.mlp.c_fc.weight"] = (rng.standard_normal((256, 64)) * 0.05).astype(np.float32)
        tensors[f"{p}.mlp.c_fc.bias"] = np.zeros(256, dtype=np.float32)
        tensors[f"{p}.mlp.c_proj.weight"] = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)
        tensors[f"{p}.mlp.c_proj.bias"] = np.zeros(64, dtype=np.float32)

    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    off = 0
    for k, arr in sorted(tensors.items()):
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[k] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(dest / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def run_experiment(checkpoint_dir: str | None = None) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="weft_e3_") as tmp:
        tmp_path = Path(tmp)
        ck_dir = tmp_path / "checkpoint"
        ck_dir.mkdir()
        p1_dir = tmp_path / "plane1"
        (p1_dir / "principles").mkdir(parents=True)
        (p1_dir / "principles" / "writing.md").write_text(
            "# Writing\nUse plain language. No em dashes.\n", encoding="utf-8"
        )

        # Resolve checkpoint source cleanly without exposing host user paths
        if checkpoint_dir is not None:
            src = Path(checkpoint_dir)
            for item in src.iterdir():
                if item.is_file():
                    os.symlink(item.resolve(), ck_dir / item.name)
            checkpoint_label = "user_provided_checkpoint"
        else:
            st_file, cfg_dir = discover_tinystories_cache()
            if st_file is not None and cfg_dir is not None:
                os.symlink(st_file, ck_dir / "model.safetensors")
                for fname in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
                    if (cfg_dir / fname).is_file():
                        os.symlink((cfg_dir / fname).resolve(), ck_dir / fname)
                checkpoint_label = "pretrained_safetensors:roneneldan/TinyStories-1M"
            else:
                _build_synthetic_gptneo_dir(ck_dir)
                checkpoint_label = "synthetic_gptneo_8layer_fallback"

        if not (ck_dir / "chat_template.jinja").exists():
            (ck_dir / "chat_template.jinja").write_text("{{ prompt }}\n", encoding="utf-8")

        cfg_json = json.loads((ck_dir / "config.json").read_text(encoding="utf-8"))
        weights = load_safetensors_dict(ck_dir / "model.safetensors")
        model = PureNumpyGPTNeo(weights, cfg_json)
        tok = SimpleWordTokenizer(ck_dir / "vocab.json")

        # Real narrative preservation prompts (K0) vs style-edit prompts
        k0_prompts = [
            "Once upon a time there was a little girl named",
            "The big brown dog ran across the green",
            "Mom put the warm cake on the kitchen",
            "Tom found a shiny red ball under the",
            "The sun was shining bright in the blue",
            "A tiny bird sat on the tall tree",
            "Lily wanted to play with her best",
            "The cat jumped onto the soft bed and",
            "Dad read a funny book before they went to",
            "Ben built a tall tower out of wooden",
            "The fish swam fast in the clear cold",
            "She opened the door and saw a cute",
            "Sam smiled when he heard the happy",
            "They walked down the quiet road to the",
            "The rain fell softly on the window",
            "Max shared his sweet apple with his",
        ]
        edit_train_prompts = [
            "Draft status note: tests pass on the branch",
            "Summary report: the build failed with an error",
            "Release note: the new parser checks every file",
            "Audit record: zero secrets found in the diff",
            "Technical note: memory stayed flat during the run",
            "Review note: the patch keeps old behavior intact",
        ]
        edit_val_prompts = [
            "Status note: all checks passed on main",
            "Change note: the server verifies every hash",
            "Audit note: no tests failed in the suite",
            "Release record: rollback restores exact bytes",
        ]

        edit_layer = model.n_layers - 1
        k0_acts = []
        k0_base_logits = []
        sample_next_token = ""
        for idx, pr in enumerate(k0_prompts):
            lg, act = model.forward_last_token(tok.encode(pr), edit_layer=edit_layer)
            k0_acts.append(act)
            k0_base_logits.append(lg)
            if idx == 0:
                sample_next_token = tok.decode_token(int(np.argmax(lg)))

        k0_mat = np.stack(k0_acts, axis=1)  # (256, 16)
        k0_eigvals = np.linalg.eigvalsh(k0_mat @ k0_mat.T)
        k0_eigvals_pos = np.maximum(k0_eigvals, 0.0)

        train_acts = np.stack(
            [model.forward_last_token(tok.encode(pr), edit_layer=edit_layer)[1] for pr in edit_train_prompts],
            axis=1,
        )
        val_base_logits = [
            model.forward_last_token(tok.encode(pr), edit_layer=edit_layer)[0] for pr in edit_val_prompts
        ]

        # Choose a clean period token "." (or token 13) to boost and an em-dash/semicolon
        # token to suppress in readout direction (demonstrating deterministic readout steering
        # compiled into layer-7 MLP down-projection inside Null(K0))
        period_id = tok.vocab.get(".", 13)
        semi_id = tok.vocab.get(";", 26)
        wte = model.wte
        target_dir = wte[period_id] - wte[semi_id]
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12) * 3.0
        target_residuals = np.tile(target_dir[:, np.newaxis], (1, train_acts.shape[1]))

        w0_down = weights[f"transformer.h.{edit_layer}.mlp.c_proj.weight"].astype(np.float64)
        beta_cfg = BetaEditConfig(
            d_in=w0_down.shape[1],
            d_out=w0_down.shape[0],
            max_rank=4,
            lambda_leakage=500.0,
            lambda_ridge=1.0,
            nullspace_threshold=1e-4,
            threshold_mode="relative_max",
            quant_dtype="f16",
        )
        proj = BetaEditNullspaceProjector(w0_down, beta_cfg)
        null_dim = proj.initialize_anchor_covariance(k0_mat)
        patch = proj.stage_edit(
            train_acts, target_residuals, is_residual=True, rank=4, slot_name="corpus_writing"
        )

        # 0-byte re-derivation check (Hard Veto 1)
        rederive_ok, rederive_info = verify_rederivation(
            patch,
            base_weight=w0_down,
            config=beta_cfg,
            k0=k0_mat,
            keys=train_acts,
            target_values_or_residuals=target_residuals,
            is_residual=True,
            rank=4,
        )

        # Division-free integer certificate check
        policy = GatePolicy(
            max_frob_sq_num=50,
            max_frob_sq_den=1,
            max_leakage_num=1,
            max_leakage_den=10,
        )
        cert_ok, cert_probs = patch.certificate.verify_division_free(policy)

        # 5-layer binding manifest
        flags = RuntimeFlags(
            quantization_mode="f32_base_f16_lora",
            mamba_cache_mode="none",
            enable_thinking=False,
            tool_call_parser="none",
            ctx_size=2048,
        )
        recipe = {
            "compiler": "weft/0.1.0",
            "target_tensor": f"transformer.h.{edit_layer}.mlp.c_proj.weight",
            "rank": 4,
            "lambda_leakage": beta_cfg.lambda_leakage,
            "lambda_ridge": beta_cfg.lambda_ridge,
            "nullspace_threshold": beta_cfg.nullspace_threshold,
            "quant_dtype": beta_cfg.quant_dtype,
        }
        manifest = build_binding_manifest(
            adapter_name="corpus_writing",
            adapter_class=ADAPTER_CLASS_1,
            checkpoint_dir=ck_dir,
            plane1_root=p1_dir,
            plane1_files=["principles/writing.md"],
            derivation_recipe=recipe,
            runtime_flags=flags,
            factor_digest=patch.factor_digest,
            certificate_sha256=patch.certificate.sha256(),
        )
        bind_clean = verify_binding(
            manifest,
            checkpoint_dir=ck_dir,
            runtime_flags=flags,
            plane1_root=p1_dir,
            expected_factor_digest=patch.factor_digest,
            expected_certificate_sha256=patch.certificate.sha256(),
        )

        # Export GGUF v3 F16 LoRA binary twice and check 0-byte determinism + readback
        gguf_path1 = tmp_path / "tinystories_corpus_1.gguf"
        gguf_path2 = tmp_path / "tinystories_corpus_2.gguf"
        tname = f"blk.{edit_layer}.ffn_down.weight"
        extra_meta = {
            "weft.adapter_class": ADAPTER_CLASS_1,
            "weft.binding_sha256": manifest.sha256(),
            "weft.certificate_sha256": patch.certificate.sha256(),
            "weft.factor_digest": patch.factor_digest,
        }
        _, g_sha1 = write_gguf_lora(
            gguf_path1,
            architecture="gptneo",
            lora_alpha=4.0,
            layer_factors={tname: (patch.factor_b, patch.factor_a)},
            extra_metadata=extra_meta,
        )
        _, g_sha2 = write_gguf_lora(
            gguf_path2,
            architecture="gptneo",
            lora_alpha=4.0,
            layer_factors={tname: (patch.factor_b, patch.factor_a)},
            extra_metadata=extra_meta,
        )
        gguf_loaded = read_gguf_lora(gguf_path1)
        gguf_roundtrip_ok = (
            g_sha1 == g_sha2
            and np.array_equal(gguf_loaded.tensors[f"{tname}.lora_a"], patch.factor_a.astype(np.float16))
            and np.array_equal(gguf_loaded.tensors[f"{tname}.lora_b"], patch.factor_b.astype(np.float16))
        )

        # Commit patch, evaluate target shift & K0 preservation, then test O(1) rollback
        slot = proj.commit_patch(patch, train_acts, gate_policy=policy)

        val_margin_gains = []
        for idx, pr in enumerate(edit_val_prompts):
            lg_new, _ = model.forward_last_token(
                tok.encode(pr), edit_layer=edit_layer, projector=proj
            )
            base_m = float(val_base_logits[idx][period_id] - val_base_logits[idx][semi_id])
            new_m = float(lg_new[period_id] - lg_new[semi_id])
            val_margin_gains.append(new_m - base_m)

        k0_top1_matches = 0
        k0_max_logit_drift = 0.0
        for idx, pr in enumerate(k0_prompts):
            lg_new, _ = model.forward_last_token(
                tok.encode(pr), edit_layer=edit_layer, projector=proj
            )
            if int(np.argmax(lg_new)) == int(np.argmax(k0_base_logits[idx])):
                k0_top1_matches += 1
            k0_max_logit_drift = max(
                k0_max_logit_drift, float(np.max(np.abs(lg_new - k0_base_logits[idx])))
            )

        k0_linear_leak = proj.knowledge_leakage_norm(k0_mat)

        # O(1) rollback verification: bit-exact restoration of W0 outputs
        proj.rollback_slot(slot, revert_covariance=True)
        rollback_max_diff = 0.0
        for idx, pr in enumerate(k0_prompts):
            lg_rb, _ = model.forward_last_token(
                tok.encode(pr), edit_layer=edit_layer, projector=proj
            )
            rollback_max_diff = max(
                rollback_max_diff, float(np.max(np.abs(lg_rb - k0_base_logits[idx])))
            )

        # Tamper rejection battery (3/3 must reject)
        (ck_dir / "chat_template.jinja").write_text("{{ prompt }} ", encoding="utf-8")
        res_tpl = verify_binding(manifest, checkpoint_dir=ck_dir, runtime_flags=flags, plane1_root=p1_dir)
        (ck_dir / "chat_template.jinja").write_text("{{ prompt }}\n", encoding="utf-8")

        (p1_dir / "principles" / "writing.md").write_text(
            "# Writing\nUse plain language. Allow em dashes.\n", encoding="utf-8"
        )
        res_p1 = verify_binding(manifest, checkpoint_dir=ck_dir, runtime_flags=flags, plane1_root=p1_dir)
        (p1_dir / "principles" / "writing.md").write_text(
            "# Writing\nUse plain language. No em dashes.\n", encoding="utf-8"
        )

        bad_flags = RuntimeFlags(
            quantization_mode="f32_base_f16_lora",
            mamba_cache_mode="none",
            enable_thinking=True,
            tool_call_parser="none",
            ctx_size=2048,
        )
        res_flags = verify_binding(manifest, checkpoint_dir=ck_dir, runtime_flags=bad_flags, plane1_root=p1_dir)

        return {
            "experiment": "E3_real_checkpoint_smoke",
            "checkpoint_source": checkpoint_label,
            "n_tensors_fingerprinted": manifest.n_tensors,
            "checkpoint_fingerprint_prefix": manifest.checkpoint_fingerprint[:16],
            "sanity_next_token_prompt0": sample_next_token,
            "k0_spectrum": {
                "d_in": int(w0_down.shape[1]),
                "n_k0_prompts": len(k0_prompts),
                "max_eigval": round(float(k0_eigvals_pos[-1]), 4),
                "rank_gt_1e-4_rel": int(np.sum(k0_eigvals_pos >= 1e-4 * k0_eigvals_pos[-1])),
                "usable_nullspace_dim_m": null_dim,
            },
            "certificate_ok": cert_ok,
            "certificate_problems": cert_probs,
            "binding_clean_ok": bind_clean.ok,
            "gguf_lora_bytes": gguf_path1.stat().st_size,
            "gguf_sha256_prefix": g_sha1[:16],
            "gguf_roundtrip_and_zero_byte_match": gguf_roundtrip_ok,
            "compiler_zero_byte_rederivation": rederive_ok,
            "rederivation_digests_match": rederive_info["expected_factor_digest"]
            == rederive_info["rebuilt_factor_digest"],
            "val_target_margin_gain_mean": round(float(np.mean(val_margin_gains)), 4),
            "val_target_margin_gain_min": round(float(np.min(val_margin_gains)), 4),
            "k0_layer7_linear_leakage_fro": round(k0_linear_leak, 6),
            "k0_top1_token_retention": f"{k0_top1_matches}/{len(k0_prompts)}",
            "k0_max_abs_logit_drift": round(k0_max_logit_drift, 6),
            "rollback_max_abs_logit_diff": rollback_max_diff,
            "tamper_rejections": {
                "chat_template_1byte_rejected": (not res_tpl.ok)
                and ("tokenizer_or_chat_template_drift" in res_tpl.problems),
                "plane1_1byte_stale_lineage_rejected": (not res_p1.ok)
                and ("stale_lineage" in res_p1.problems),
                "runtime_flag_drift_rejected": (not res_flags.ok)
                and ("runtime_flag_drift" in res_flags.problems),
            },
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()
    print(json.dumps(run_experiment(args.checkpoint_dir), indent=2))
