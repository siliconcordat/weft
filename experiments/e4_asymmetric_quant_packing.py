"""Experiment E4: Asymmetric (B, A) Factor Quantization and siliquant#12 Probes on Pretrained Weights.

Validates the Asymmetric Factor Leakage Theorem (siliconcordat/loom docs/05) live
on real pretrained 8-layer GPT-Neo weights (`roneneldan/TinyStories-1M`, falling
back to a synthetic 8-layer GPT-Neo fixture when no local cache is present):

  For Delta_j = B_j A_j with A_j K0 = 0 in float64:
    Q(B_j) Q(A_j) K0 = Q(B_j) (A_j + E_A) K0 = Q(B_j) (E_A K0)

  Quantization error E_B on the output factor B drops out to first order while
  quantization error E_A on the null-space factor A drives all K0 leakage.

Runs three live measurements:
  1. Sequential N=1..3 BetaEdit composition on real Layer-7 mlp.c_proj (64 x 256)
     across 8 precision pairs: f32/f32, bf16/bf16, f16/f16, int8/int8, int4/int4,
     int4/int8 (asymmetric), int4/f16 (asymmetric), and int8/int4 (reversed control).
  2. Wide-output companion check on real Layer-7 mlp.c_fc (256 x 64, d_out = 4*d_in)
     showing both the leakage separation and the ~38% byte savings vs int8/int8
     (~69% vs f16/f16) when B is the wide factor (as in W_up / W_gate).
  3. Two siliquant#12 probes on the same real checkpoint:
     (a) Weight-identifiable loud columns in mlp.c_proj (top-10 columns kept whole
         under INT4 base quantization vs uniform INT4).
     (b) Noise-only K0 covariance harvesting from random vocabulary tokens (zero
         real text read) evaluated on held-out real story prompts.

Run:
  .venv/bin/python experiments/e4_asymmetric_quant_packing.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e3_real_checkpoint_smoke import (
    PureNumpyGPTNeo,
    SimpleWordTokenizer,
    _build_synthetic_gptneo_dir,
    _gelu_new,
    _layer_norm,
    discover_tinystories_cache,
    load_safetensors_dict,
)
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.quant import dequantize_int4, dequantize_int8, quantize_int4

REGIMES: list[tuple[str, str, str]] = [
    ("f32/f32", "f32", "f32"),
    ("bf16/bf16", "bf16", "bf16"),
    ("f16/f16", "f16", "f16"),
    ("int8/int8", "int8", "int8"),
    ("int4/int4", "int4", "int4"),
    ("int4/int8", "int4", "int8"),
    ("int4/f16", "int4", "f16"),
    ("int8/int4", "int8", "int4"),
]


def _harvest_layer_ln2_last(
    model: PureNumpyGPTNeo, token_ids: list[int], target_layer: int
) -> np.ndarray:
    """Capture the exact pre-MLP LayerNorm output h_ln2[-1] (input to mlp.c_fc)."""
    ids = np.array(token_ids, dtype=np.int64)
    pos = np.arange(len(token_ids), dtype=np.int64)
    h = model.wte[ids] + model.wpe[pos]
    for i in range(target_layer + 1):
        pfx = f"transformer.h.{i}"
        h_ln1 = _layer_norm(
            h,
            model.w[f"{pfx}.ln_1.weight"].astype(np.float64),
            model.w[f"{pfx}.ln_1.bias"].astype(np.float64),
            model.eps,
        )
        h = h + model._attn(h_ln1, i)
        h_ln2 = _layer_norm(
            h,
            model.w[f"{pfx}.ln_2.weight"].astype(np.float64),
            model.w[f"{pfx}.ln_2.bias"].astype(np.float64),
            model.eps,
        )
        if i == target_layer:
            return h_ln2[-1].copy()
        mlp_mid = _gelu_new(
            h_ln2 @ model.w[f"{pfx}.mlp.c_fc.weight"].astype(np.float64).T
            + model.w[f"{pfx}.mlp.c_fc.bias"].astype(np.float64)
        )
        mlp_out = (
            mlp_mid @ model.w[f"{pfx}.mlp.c_proj.weight"].astype(np.float64).T
            + model.w[f"{pfx}.mlp.c_proj.bias"].astype(np.float64)
        )
        h = h + mlp_out
    raise RuntimeError("target_layer out of range")


def _verify_wire_decode(patch: Any, b_dtype: str, a_dtype: str) -> bool:
    """Verify packed INT4/INT8 factor wire payloads decode bit-for-bit to patch factors."""
    ok = True
    if b_dtype == "int4":
        ok = ok and np.array_equal(
            dequantize_int4(patch.b_bytes, patch.factor_b.shape, axis=0), patch.factor_b
        )
    elif b_dtype == "int8":
        ok = ok and np.array_equal(
            dequantize_int8(patch.b_bytes, patch.factor_b.shape, axis=0), patch.factor_b
        )
    if a_dtype == "int4":
        ok = ok and np.array_equal(
            dequantize_int4(patch.a_bytes, patch.factor_a.shape, axis=1), patch.factor_a
        )
    elif a_dtype == "int8":
        ok = ok and np.array_equal(
            dequantize_int8(patch.a_bytes, patch.factor_a.shape, axis=1), patch.factor_a
        )
    return bool(ok)


def _token_id(tok: SimpleWordTokenizer, word: str, fallback: int) -> int:
    for cand in (f"\u0120{word}", word):
        if cand in tok.vocab:
            return int(tok.vocab[cand])
    return fallback % len(tok.vocab)


def run_experiment(checkpoint_dir: str | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="weft_e4_") as tmp:
        ck_dir = Path(tmp) / "checkpoint"
        ck_dir.mkdir()
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

        cfg_json = json.loads((ck_dir / "config.json").read_text(encoding="utf-8"))
        weights = load_safetensors_dict(ck_dir / "model.safetensors")
        model = PureNumpyGPTNeo(weights, cfg_json)
        tok = SimpleWordTokenizer(ck_dir / "vocab.json")
        edit_layer = model.n_layers - 1

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
        domains = [
            {
                "name": "corpus",
                "train": [
                    "Draft status note: tests pass on the branch",
                    "Summary report: the build failed with an error",
                    "Release note: the new parser checks every file",
                    "Audit record: zero secrets found in the diff",
                ],
                "val": [
                    "Status note: all checks passed on main",
                    "Change note: the server verifies every hash",
                    "Audit note: no tests failed in the suite",
                ],
                "pos_tok": _token_id(tok, ".", 13),
                "neg_tok": _token_id(tok, ";", 26),
            },
            {
                "name": "eido",
                "train": [
                    "Theme audit: check foreground contrast on bright",
                    "Color check: text contrast must stay clean and",
                    "Layout note: keep card spacing flat and",
                    "Surface rule: avoid low contrast gray on",
                ],
                "val": [
                    "Contrast audit: verify button label stays",
                    "Palette check: dark text reads cleanly on",
                    "Visual note: keep every border sharp and",
                ],
                "pos_tok": _token_id(tok, "clean", 40),
                "neg_tok": _token_id(tok, "dark", 41),
            },
            {
                "name": "formal",
                "train": [
                    "Proof step: rewrite the lemma using exact",
                    "Theorem check: prove the bound without any",
                    "Goal state: close the subgoal with pure",
                    "Kernel check: verify every axiom stays",
                ],
                "val": [
                    "Lemma check: finish the proof by exact",
                    "Bound proof: show the norm stays small",
                    "Axiom audit: keep every theorem sound",
                ],
                "pos_tok": _token_id(tok, "true", 50),
                "neg_tok": _token_id(tok, "false", 51),
            },
        ]

        # Harvest real Layer-7 mlp.c_proj inputs (dim 256) & baseline logits
        k0_ids = [tok.encode(p) for p in k0_prompts]
        k0_acts_list = []
        k0_base_logits = []
        for ids in k0_ids:
            lg, act = model.forward_last_token(ids, edit_layer=edit_layer)
            k0_base_logits.append(lg)
            k0_acts_list.append(act)
        k0_mat = np.stack(k0_acts_list, axis=1)  # (256, 16)

        domain_data = []
        for dom in domains:
            tr_acts = np.stack(
                [model.forward_last_token(tok.encode(p), edit_layer=edit_layer)[1] for p in dom["train"]],
                axis=1,
            )
            val_ids = [tok.encode(p) for p in dom["val"]]
            val_base = [model.forward_last_token(ids, edit_layer=edit_layer)[0] for ids in val_ids]
            direction = model.wte[dom["pos_tok"]] - model.wte[dom["neg_tok"]]
            direction = direction / (np.linalg.norm(direction) + 1e-12) * 3.0
            residuals = np.tile(direction[:, np.newaxis], (1, tr_acts.shape[1]))
            domain_data.append(
                {
                    "name": dom["name"],
                    "train_acts": tr_acts,
                    "residuals": residuals,
                    "val_ids": val_ids,
                    "val_base": val_base,
                    "pos_tok": dom["pos_tok"],
                    "neg_tok": dom["neg_tok"],
                }
            )

        # ------------------------------------------------------------------
        # Part 1: Real Layer-7 mlp.c_proj (d_out=64, d_in=256), N=1..3
        # ------------------------------------------------------------------
        w0_down = weights[f"transformer.h.{edit_layer}.mlp.c_proj.weight"].astype(np.float64)
        c_proj_results: dict[str, Any] = {}
        for label, b_dt, a_dt in REGIMES:
            cfg = BetaEditConfig(
                d_in=w0_down.shape[1],
                d_out=w0_down.shape[0],
                max_rank=4,
                max_slots=8,
                lambda_leakage=1000.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e-6,
                threshold_mode="absolute",
                refresh_period=1,
                quant_dtype="f16",
                b_quant_dtype=b_dt,  # type: ignore[arg-type]
                a_quant_dtype=a_dt,  # type: ignore[arg-type]
            )
            proj = BetaEditNullspaceProjector(w0_down, cfg)
            proj.initialize_anchor_covariance(k0_mat)

            by_n: dict[str, Any] = {}
            total_bytes = 0
            total_cert_leak = 0
            all_wire_ok = True
            rederive_n1_ok = False

            for step_idx, ddata in enumerate(domain_data, start=1):
                patch = proj.stage_edit(
                    ddata["train_acts"],
                    ddata["residuals"],
                    is_residual=True,
                    rank=4,
                    slot_name=ddata["name"],
                )
                all_wire_ok = all_wire_ok and _verify_wire_decode(patch, b_dt, a_dt)
                if step_idx == 1:
                    rederive_n1_ok, _ = verify_rederivation(
                        patch,
                        base_weight=w0_down,
                        config=cfg,
                        k0=k0_mat,
                        keys=ddata["train_acts"],
                        target_values_or_residuals=ddata["residuals"],
                        is_residual=True,
                        rank=4,
                    )
                proj.commit_patch(patch, ddata["train_acts"])
                total_bytes += len(patch.b_bytes) + len(patch.a_bytes)
                total_cert_leak += patch.certificate.leakage_energy_scaled

                k0_lin_leak = proj.knowledge_leakage_norm(k0_mat)

                # Live 8-layer forward pass on all 16 K0 prompts
                retained = 0
                max_logit_drift = 0.0
                mean_logit_drifts = []
                for ids, base_lg in zip(k0_ids, k0_base_logits):
                    new_lg, _ = model.forward_last_token(
                        ids, edit_layer=edit_layer, projector=proj
                    )
                    if int(np.argmax(new_lg)) == int(np.argmax(base_lg)):
                        retained += 1
                    diff = np.abs(new_lg - base_lg)
                    max_logit_drift = max(max_logit_drift, float(np.max(diff)))
                    mean_logit_drifts.append(float(np.mean(diff)))

                # Live unseen validation margin gain across compiled domains 1..step_idx
                val_gains = []
                for prev in domain_data[:step_idx]:
                    for ids, base_lg in zip(prev["val_ids"], prev["val_base"]):
                        new_lg, _ = model.forward_last_token(
                            ids, edit_layer=edit_layer, projector=proj
                        )
                        base_m = float(base_lg[prev["pos_tok"]] - base_lg[prev["neg_tok"]])
                        new_m = float(new_lg[prev["pos_tok"]] - new_lg[prev["neg_tok"]])
                        val_gains.append(new_m - base_m)

                by_n[f"N{step_idx}"] = {
                    "payload_bytes_total": total_bytes,
                    "k0_linear_leakage_fro": round(k0_lin_leak, 6),
                    "k0_top1_retained": retained,
                    "k0_total": len(k0_ids),
                    "k0_max_abs_logit_drift": round(max_logit_drift, 6),
                    "k0_mean_abs_logit_drift": round(float(np.mean(mean_logit_drifts)), 6),
                    "val_mean_margin_gain": round(float(np.mean(val_gains)), 4),
                    "cert_leakage_energy_scaled_sum": int(total_cert_leak),
                }

            c_proj_results[label] = {
                "b_dtype": b_dt,
                "a_dtype": a_dt,
                "wire_roundtrip_exact": all_wire_ok,
                "n1_zero_byte_rederivation": bool(rederive_n1_ok),
                "steps": by_n,
            }

        # ------------------------------------------------------------------
        # Part 2: Wide-output companion check on real Layer-7 mlp.c_fc (256 x 64)
        # ------------------------------------------------------------------
        w0_up = weights[f"transformer.h.{edit_layer}.mlp.c_fc.weight"].astype(np.float64)
        k0_up = np.stack(
            [_harvest_layer_ln2_last(model, ids, edit_layer) for ids in k0_ids], axis=1
        )  # (64, 16)
        rng_up = np.random.default_rng(20261119)
        up_domains = []
        for dom in domains:
            tr_u = np.stack(
                [_harvest_layer_ln2_last(model, tok.encode(p), edit_layer) for p in dom["train"]],
                axis=1,
            )
            res_u = rng_up.standard_normal((w0_up.shape[0], tr_u.shape[1])) * 0.4
            up_domains.append((dom["name"], tr_u, res_u))

        c_fc_results: dict[str, Any] = {}
        for label, b_dt, a_dt in REGIMES:
            cfg_u = BetaEditConfig(
                d_in=w0_up.shape[1],
                d_out=w0_up.shape[0],
                max_rank=4,
                max_slots=8,
                lambda_leakage=1000.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e-6,
                threshold_mode="absolute",
                refresh_period=1,
                quant_dtype="f16",
                b_quant_dtype=b_dt,  # type: ignore[arg-type]
                a_quant_dtype=a_dt,  # type: ignore[arg-type]
            )
            proj_u = BetaEditNullspaceProjector(w0_up, cfg_u)
            proj_u.initialize_anchor_covariance(k0_up)
            tot_b = 0
            for dname, tr_u, res_u in up_domains:
                p_u = proj_u.stage_edit(tr_u, res_u, is_residual=True, rank=4, slot_name=dname)
                proj_u.commit_patch(p_u, tr_u)
                tot_b += len(p_u.b_bytes) + len(p_u.a_bytes)
            c_fc_results[label] = {
                "n3_payload_bytes_total": tot_b,
                "n3_k0_linear_leakage_fro": round(proj_u.knowledge_leakage_norm(k0_up), 6),
            }

        # ------------------------------------------------------------------
        # Part 3a: siliquant#12 "loud columns" check on real Layer-7 mlp.c_proj
        # ------------------------------------------------------------------
        layer_col_ratios = []
        for l_idx in range(model.n_layers):
            w_l = weights[f"transformer.h.{l_idx}.mlp.c_proj.weight"].astype(np.float64)
            col_norms = np.linalg.norm(w_l, axis=0)
            layer_col_ratios.append(round(float(np.max(col_norms) / np.mean(col_norms)), 3))

        col_norms_l7 = np.linalg.norm(w0_down, axis=0)
        top10_idx = np.argsort(col_norms_l7)[-10:]
        _, w_down_int4 = quantize_int4(w0_down, axis=1)
        w_down_kept10 = w_down_int4.copy()
        w_down_kept10[:, top10_idx] = w0_down[:, top10_idx]

        def _eval_base_c_proj(w_override: np.ndarray) -> dict[str, Any]:
            saved = model.w[f"transformer.h.{edit_layer}.mlp.c_proj.weight"]
            model.w[f"transformer.h.{edit_layer}.mlp.c_proj.weight"] = w_override.astype(np.float32)
            try:
                ret = 0
                max_d = 0.0
                mean_ds = []
                for ids, base_lg in zip(k0_ids, k0_base_logits):
                    lg, _ = model.forward_last_token(ids, edit_layer=edit_layer)
                    if int(np.argmax(lg)) == int(np.argmax(base_lg)):
                        ret += 1
                    diff = np.abs(lg - base_lg)
                    max_d = max(max_d, float(np.max(diff)))
                    mean_ds.append(float(np.mean(diff)))
                return {
                    "k0_top1_retained": ret,
                    "k0_max_abs_logit_drift": round(max_d, 6),
                    "k0_mean_abs_logit_drift": round(float(np.mean(mean_ds)), 6),
                    "weight_fro_error": round(float(np.linalg.norm(w_override - w0_down, "fro")), 6),
                }
            finally:
                model.w[f"transformer.h.{edit_layer}.mlp.c_proj.weight"] = saved

        loud_col_probe = {
            "all_layers_c_proj_max_to_mean_col_norm": layer_col_ratios,
            "layer7_uniform_int4_base": _eval_base_c_proj(w_down_int4),
            "layer7_int4_plus_top10_kept_cols": _eval_base_c_proj(w_down_kept10),
        }

        # ------------------------------------------------------------------
        # Part 3b: siliquant#12 noise-only K0 covariance harvesting probe
        # ------------------------------------------------------------------
        rng_noise = np.random.default_rng(777)
        vocab_size = int(model.wte.shape[0])
        noise_acts = []
        for _ in range(64):
            rand_ids = rng_noise.integers(0, vocab_size, size=8).tolist()
            _, act_n = model.forward_last_token(rand_ids, edit_layer=edit_layer)
            noise_acts.append(act_n)
        k0_noise = np.stack(noise_acts, axis=1)  # (256, 64), ZERO real text read
        k0_real_train = k0_mat[:, :12]
        k0_real_holdout = k0_mat[:, 12:]
        k0_holdout_ids = k0_ids[12:]
        k0_holdout_base = k0_base_logits[12:]

        def _eval_k0_source(source_k0: np.ndarray | None, lam: float) -> dict[str, Any]:
            cfg_s = BetaEditConfig(
                d_in=w0_down.shape[1],
                d_out=w0_down.shape[0],
                max_rank=4,
                lambda_leakage=lam,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,  # pure soft covariance penalty
                threshold_mode="absolute",
                quant_dtype="f16",
            )
            p_s = BetaEditNullspaceProjector(w0_down, cfg_s)
            if source_k0 is not None:
                # Normalize by 1/sqrt(n_samples) so Gram = mean outer product (fair across sample counts)
                norm_k0 = source_k0 / np.sqrt(float(source_k0.shape[1]))
                p_s.initialize_anchor_covariance(norm_k0)
            else:
                p_s.initialize_anchor_covariance(np.zeros((w0_down.shape[1], 1)))
            d0 = domain_data[0]
            patch_s = p_s.stage_edit(
                d0["train_acts"], d0["residuals"], is_residual=True, rank=4, slot_name="corpus"
            )
            p_s.commit_patch(patch_s, d0["train_acts"])
            holdout_leak = p_s.knowledge_leakage_norm(k0_real_holdout)
            max_hd = 0.0
            for ids, blg in zip(k0_holdout_ids, k0_holdout_base):
                nlg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=p_s)
                max_hd = max(max_hd, float(np.max(np.abs(nlg - blg))))
            val_gains = []
            for ids, blg in zip(d0["val_ids"], d0["val_base"]):
                nlg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=p_s)
                bm = float(blg[d0["pos_tok"]] - blg[d0["neg_tok"]])
                nm = float(nlg[d0["pos_tok"]] - nlg[d0["neg_tok"]])
                val_gains.append(nm - bm)
            return {
                "holdout_real_k0_leakage_fro": round(holdout_leak, 6),
                "holdout_real_k0_max_logit_drift": round(max_hd, 6),
                "val_target_margin_gain": round(float(np.mean(val_gains)), 4),
            }

        naive_res = _eval_k0_source(None, 0.0)
        noise_res = _eval_k0_source(k0_noise, 60.0)
        real_res = _eval_k0_source(k0_real_train, 60.0)
        naive_l = naive_res["holdout_real_k0_leakage_fro"]
        noise_l = noise_res["holdout_real_k0_leakage_fro"]
        real_l = real_res["holdout_real_k0_leakage_fro"]
        recovery_pct = (
            round(100.0 * (naive_l - noise_l) / max(naive_l - real_l, 1e-12), 1)
            if naive_l > real_l
            else 0.0
        )

        # Headline ratios for N=3 on c_proj and c_fc
        c_proj_n3_44 = c_proj_results["int4/int4"]["steps"]["N3"]["k0_linear_leakage_fro"]
        c_proj_n3_48 = c_proj_results["int4/int8"]["steps"]["N3"]["k0_linear_leakage_fro"]
        c_proj_n3_84 = c_proj_results["int8/int4"]["steps"]["N3"]["k0_linear_leakage_fro"]
        c_proj_n3_88 = c_proj_results["int8/int8"]["steps"]["N3"]["k0_linear_leakage_fro"]

        c_fc_n3_44 = c_fc_results["int4/int4"]["n3_k0_linear_leakage_fro"]
        c_fc_n3_48 = c_fc_results["int4/int8"]["n3_k0_linear_leakage_fro"]
        c_fc_n3_84 = c_fc_results["int8/int4"]["n3_k0_linear_leakage_fro"]
        c_fc_n3_88 = c_fc_results["int8/int8"]["n3_k0_linear_leakage_fro"]

        return {
            "experiment": "E4_asymmetric_quant_packing_and_siliquant_probes",
            "checkpoint_source": checkpoint_label,
            "edit_layer": edit_layer,
            "c_proj_shape": list(w0_down.shape),
            "c_fc_shape": list(w0_up.shape),
            "c_proj_headline_n3": {
                "int4_int4_vs_int4_int8_leakage_ratio": round(
                    c_proj_n3_44 / max(c_proj_n3_48, 1e-12), 2
                ),
                "int8_int4_vs_int4_int8_leakage_ratio": round(
                    c_proj_n3_84 / max(c_proj_n3_48, 1e-12), 2
                ),
                "int4_int8_vs_int8_int8_leakage_ratio": round(
                    c_proj_n3_48 / max(c_proj_n3_88, 1e-12), 3
                ),
            },
            "c_fc_headline_n3": {
                "int4_int4_vs_int4_int8_leakage_ratio": round(
                    c_fc_n3_44 / max(c_fc_n3_48, 1e-12), 2
                ),
                "int8_int4_vs_int4_int8_leakage_ratio": round(
                    c_fc_n3_84 / max(c_fc_n3_48, 1e-12), 2
                ),
                "int4_int8_vs_int8_int8_leakage_ratio": round(
                    c_fc_n3_48 / max(c_fc_n3_88, 1e-12), 3
                ),
                "int4_int8_byte_savings_vs_int8_int8_pct": round(
                    100.0
                    * (
                        1.0
                        - c_fc_results["int4/int8"]["n3_payload_bytes_total"]
                        / c_fc_results["int8/int8"]["n3_payload_bytes_total"]
                    ),
                    1,
                ),
                "int4_int8_byte_savings_vs_f16_f16_pct": round(
                    100.0
                    * (
                        1.0
                        - c_fc_results["int4/int8"]["n3_payload_bytes_total"]
                        / c_fc_results["f16/f16"]["n3_payload_bytes_total"]
                    ),
                    1,
                ),
            },
            "c_proj_regimes": c_proj_results,
            "c_fc_regimes": c_fc_results,
            "siliquant12_loud_columns_probe": loud_col_probe,
            "siliquant12_noise_k0_probe": {
                "naive_unregularized": naive_res,
                "noise_only_k0_soft": noise_res,
                "real_text_k0_soft": real_res,
                "noise_k0_holdout_leakage_recovery_pct": recovery_pct,
            },
        }


def main() -> None:
    res = run_experiment()
    print(json.dumps(res, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
