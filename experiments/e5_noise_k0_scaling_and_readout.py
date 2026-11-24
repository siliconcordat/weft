"""Experiment E5: Seeded-Noise K0 Marchenko-Pastur Scaling and Wire-Dialect Readout Bridge.

Runs live on pretrained `roneneldan/TinyStories-1M` (with transparent fallback to
an 8-layer synthetic GPT-Neo fixture when no local HF cache is present):

  Part A (Seeded-Noise K0 Marchenko-Pastur Curve):
    Sweeps zero-data seeded random-vocabulary noise K0 across
    n_noise in {32, 64, 256, 512, 1024, 2048} (zero real calibration text read),
    comparing unseen real-story K0 holdout leakage, top-1 retention, and target
    margin against naive unregularized editing, 16-prompt real-text K0, and
    Hybrid (seeded-noise n=2048 + 16 register-scoped y=0 anchors). Also verifies
    0-byte deterministic re-derivation from seed alone.

  Part B (tokenpage#5 x Weft Parametric Wire-Dialect Readout Adapter):
    Reproduces the linear-probe vs unadapted-readout gap on a 4-class compact
    symbolic dialect, then repairs readout accuracy using a rank-4 asymmetric
    `int4 / int8` adapter compiled with seeded-noise K0 only (zero real text)
    and packaged in a verified `weft-slab/v0.1` binary container with exact
    size comparison against F16 GGUF LoRA.

Run:
  .venv/bin/python experiments/e5_noise_k0_scaling_and_readout.py
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments.checkpoint_provenance import (
    RealWeightsRequiredError,
    resolve_experiment_checkpoint,
)
from experiments.e3_real_checkpoint_smoke import (
    PureNumpyGPTNeo,
    SimpleWordTokenizer,
    _build_synthetic_gptneo_dir,
    _gelu_new,
    _layer_norm,
    _softmax,
    load_safetensors_dict,
)
from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation
from weft.gguf_lora import write_gguf_lora
from weft.slab_wire import read_slab_wire, verify_slab_wire, write_slab_wire

NOISE_COUNTS = (32, 64, 256, 512, 1024, 2048)
SAMPLE_ZONE_MANIFEST_ID = "sha256:4e00656cb47fce97f687ae638afba22c569b9680b3780eeb2e5ea71a62c55eca"


def _harvest_batch_mlp_in_last(
    model: PureNumpyGPTNeo, token_matrix: np.ndarray, edit_layer: int, chunk_size: int = 256
) -> np.ndarray:
    """Vectorized exact batch forward pass harvesting edit_layer MLP inputs (d_ff, N)."""
    n_seq, seq_len = token_matrix.shape
    pos = np.arange(seq_len, dtype=np.int64)
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    idx = np.arange(seq_len)
    local_mask = causal & ((idx[:, None] - idx[None, :]) < model.window)

    layer_weights = []
    for i in range(edit_layer + 1):
        pfx = f"transformer.h.{i}"
        apfx = f"{pfx}.attn.attention"
        layer_weights.append(
            {
                "ln1_w": model.w[f"{pfx}.ln_1.weight"].astype(np.float64),
                "ln1_b": model.w[f"{pfx}.ln_1.bias"].astype(np.float64),
                "ln2_w": model.w[f"{pfx}.ln_2.weight"].astype(np.float64),
                "ln2_b": model.w[f"{pfx}.ln_2.bias"].astype(np.float64),
                "q_wt": model.w[f"{apfx}.q_proj.weight"].astype(np.float64).T,
                "k_wt": model.w[f"{apfx}.k_proj.weight"].astype(np.float64).T,
                "v_wt": model.w[f"{apfx}.v_proj.weight"].astype(np.float64).T,
                "o_wt": model.w[f"{apfx}.out_proj.weight"].astype(np.float64).T,
                "o_b": model.w[f"{apfx}.out_proj.bias"].astype(np.float64),
                "fc_wt": model.w[f"{pfx}.mlp.c_fc.weight"].astype(np.float64).T,
                "fc_b": model.w[f"{pfx}.mlp.c_fc.bias"].astype(np.float64),
                "proj_wt": model.w[f"{pfx}.mlp.c_proj.weight"].astype(np.float64).T,
                "proj_b": model.w[f"{pfx}.mlp.c_proj.bias"].astype(np.float64),
                "mask": local_mask if model.layer_types[i] == "local" else causal,
            }
        )

    chunks = []
    for start in range(0, n_seq, chunk_size):
        ids = token_matrix[start : start + chunk_size]
        bsz = ids.shape[0]
        h = model.wte[ids] + model.wpe[pos][None, :, :]
        for i, lw in enumerate(layer_weights):
            h_ln1 = _layer_norm(h, lw["ln1_w"], lw["ln1_b"], model.eps)
            q = (h_ln1 @ lw["q_wt"]).reshape(bsz, seq_len, model.n_heads, model.head_dim)
            k = (h_ln1 @ lw["k_wt"]).reshape(bsz, seq_len, model.n_heads, model.head_dim)
            v = (h_ln1 @ lw["v_wt"]).reshape(bsz, seq_len, model.n_heads, model.head_dim)
            scores = np.einsum("bshd,bthd->bhst", q, k)
            scores = np.where(lw["mask"][None, None, :, :], scores, -1e9)
            probs = _softmax(scores)
            ctx = np.einsum("bhst,bthd->bshd", probs, v).reshape(bsz, seq_len, model.d_model)
            h = h + (ctx @ lw["o_wt"] + lw["o_b"])

            h_ln2 = _layer_norm(h, lw["ln2_w"], lw["ln2_b"], model.eps)
            mlp_mid = _gelu_new(h_ln2 @ lw["fc_wt"] + lw["fc_b"])
            if i == edit_layer:
                chunks.append(mlp_mid[:, -1, :].copy())
                break
            h = h + (mlp_mid @ lw["proj_wt"] + lw["proj_b"])

    return np.concatenate(chunks, axis=0).T  # (d_ff, n_seq)


def _effective_rank(mat: np.ndarray) -> float:
    """Participation ratio (Tr G)^2 / Tr(G^2) of sample Gram matrix G = mat @ mat.T / n."""
    gram = (mat @ mat.T) / max(1, mat.shape[1])
    tr = float(np.trace(gram))
    tr2 = float(np.sum(gram * gram))
    return round((tr * tr) / max(tr2, 1e-18), 2)


def _token_id(tok: SimpleWordTokenizer, word: str, fallback: int) -> int:
    for cand in (f"\u0120{word}", word):
        if cand in tok.vocab:
            return int(tok.vocab[cand])
    return fallback % len(tok.vocab)


def run_experiment(
    checkpoint_dir: str | None = None,
    *,
    require_real_weights: bool | None = None,
    expected_safetensors_sha256: str | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="weft_e5_") as tmp:
        tmp_path = Path(tmp)
        ck_dir = tmp_path / "checkpoint"
        ck_dir.mkdir()
        provenance = resolve_experiment_checkpoint(
            ck_dir,
            build_synthetic_fn=_build_synthetic_gptneo_dir,
            checkpoint_dir=checkpoint_dir,
            require_real_weights=require_real_weights,
            expected_safetensors_sha256=expected_safetensors_sha256,
        )
        checkpoint_label = provenance.checkpoint_source

        cfg_json = json.loads((ck_dir / "config.json").read_text(encoding="utf-8"))
        weights = load_safetensors_dict(ck_dir / "model.safetensors")
        model = PureNumpyGPTNeo(weights, cfg_json)
        tok = SimpleWordTokenizer(ck_dir / "vocab.json")
        edit_layer = model.n_layers - 1
        w0_down = weights[f"transformer.h.{edit_layer}.mlp.c_proj.weight"].astype(np.float64)
        base_weights_sha256 = hashlib.sha256(
            (ck_dir / "model.safetensors").read_bytes()
        ).hexdigest()

        # 32 disjoint real narrative prompts: 16 calibration train + 16 strictly unseen holdout
        k0_real_train_prompts = [
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
        k0_real_holdout_prompts = [
            "Anna saw a bright yellow flower in the",
            "The small white rabbit hopped into the",
            "Grandma baked ten fresh cookies for the",
            "Jack dropped his blue hat near the old",
            "A friendly frog sat beside the deep",
            "The wind blew leaves across the quiet",
            "Mia painted a big house with a red",
            "The puppy barked loudly when it saw the",
            "Leo climbed up the hill to watch the",
            "Two ducks swam slowly across the warm",
            "She picked up the silver spoon from the",
            "The baby laughed when the balloon flew",
            "Owen drew a fast train on the white",
            "A gentle deer stood behind the tall",
            "The moon glowed softly over the dark",
            "Nora carried her heavy basket to the",
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

        k0_train_mat = np.stack(
            [model.forward_last_token(tok.encode(p), edit_layer=edit_layer)[1] for p in k0_real_train_prompts],
            axis=1,
        )
        k0_holdout_ids = [tok.encode(p) for p in k0_real_holdout_prompts]
        k0_holdout_acts = []
        k0_holdout_base_logits = []
        for ids in k0_holdout_ids:
            lg, act = model.forward_last_token(ids, edit_layer=edit_layer)
            k0_holdout_base_logits.append(lg)
            k0_holdout_acts.append(act)
        k0_holdout_mat = np.stack(k0_holdout_acts, axis=1)  # (256, 16)

        edit_train_acts = np.stack(
            [model.forward_last_token(tok.encode(p), edit_layer=edit_layer)[1] for p in edit_train_prompts],
            axis=1,
        )
        edit_val_ids = [tok.encode(p) for p in edit_val_prompts]
        edit_val_base = [model.forward_last_token(ids, edit_layer=edit_layer)[0] for ids in edit_val_ids]

        period_id = _token_id(tok, ".", 13)
        semi_id = _token_id(tok, ";", 26)
        target_dir = model.wte[period_id] - model.wte[semi_id]
        # Physically matched residual scale (~2x natural Layer-7 ||mlp_out|| ~ 0.17)
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12) * 0.40
        target_residuals = np.tile(target_dir[:, np.newaxis], (1, edit_train_acts.shape[1]))

        # Harvest 2048 deterministic random-vocab noise sequences in one fast vectorized pass
        noise_seed = 20261119
        rng_noise = np.random.default_rng(noise_seed)
        vocab_size = int(model.wte.shape[0])
        noise_tokens = rng_noise.integers(0, vocab_size, size=(max(NOISE_COUNTS), 8), dtype=np.int64)
        k0_noise_all = _harvest_batch_mlp_in_last(model, noise_tokens, edit_layer=edit_layer)

        # Verify vectorized batch forward matches scalar forward_last_token bit-near-identically
        _, scalar_check = model.forward_last_token(noise_tokens[0].tolist(), edit_layer=edit_layer)
        batch_forward_max_diff = float(np.max(np.abs(k0_noise_all[:, 0] - scalar_check)))

        def _eval_part_a(
            source_k0: np.ndarray | None,
            lam: float,
            *,
            pres_keys: np.ndarray | None = None,
            pres_weight: float = 0.0,
        ) -> tuple[dict[str, Any], BetaEditConfig, Any]:
            cfg = BetaEditConfig(
                d_in=w0_down.shape[1],
                d_out=w0_down.shape[0],
                max_rank=4,
                lambda_leakage=lam,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,  # pure soft covariance penalty
                threshold_mode="absolute",
                quant_dtype="f16",
                b_quant_dtype="int4",
                a_quant_dtype="f16",
            )
            proj = BetaEditNullspaceProjector(w0_down, cfg)
            if source_k0 is not None:
                norm_k0 = source_k0 / np.sqrt(float(source_k0.shape[1]))
                proj.initialize_anchor_covariance(norm_k0)
                eff_rank = _effective_rank(source_k0)
            else:
                proj.initialize_anchor_covariance(np.zeros((w0_down.shape[1], 1)))
                eff_rank = 0.0

            patch = proj.stage_edit(
                edit_train_acts,
                target_residuals,
                is_residual=True,
                rank=4,
                preservation_keys=pres_keys,
                preservation_weight=pres_weight,
                slot_name="corpus",
            )
            proj.commit_patch(patch, edit_train_acts)

            holdout_leak = proj.knowledge_leakage_norm(k0_holdout_mat)
            retained = 0
            max_drift = 0.0
            mean_drifts = []
            for ids, blg in zip(k0_holdout_ids, k0_holdout_base_logits):
                nlg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=proj)
                if int(np.argmax(nlg)) == int(np.argmax(blg)):
                    retained += 1
                diff = np.abs(nlg - blg)
                max_drift = max(max_drift, float(np.max(diff)))
                mean_drifts.append(float(np.mean(diff)))

            val_gains = []
            for ids, blg in zip(edit_val_ids, edit_val_base):
                nlg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=proj)
                bm = float(blg[period_id] - blg[semi_id])
                nm = float(nlg[period_id] - nlg[semi_id])
                val_gains.append(nm - bm)

            metrics = {
                "k0_effective_rank": eff_rank,
                "holdout_k0_linear_leakage_fro": round(holdout_leak, 6),
                "holdout_k0_top1_retained": retained,
                "holdout_k0_total": len(k0_holdout_ids),
                "holdout_k0_max_abs_logit_drift": round(max_drift, 6),
                "holdout_k0_mean_abs_logit_drift": round(float(np.mean(mean_drifts)), 6),
                "val_target_margin_gain": round(float(np.mean(val_gains)), 4),
            }
            return metrics, cfg, patch

        lam_soft = 60.0
        naive_m, _, _ = _eval_part_a(None, 0.0)
        real_m, _, _ = _eval_part_a(k0_train_mat, lam_soft)

        part_a_arms: dict[str, Any] = {"naive_unregularized": naive_m}
        noise_2048_cfg = None
        noise_2048_patch = None
        for n_cnt in NOISE_COUNTS:
            sub_k0 = k0_noise_all[:, :n_cnt]
            m, c, p = _eval_part_a(sub_k0, lam_soft)
            m["n_over_d_in"] = round(n_cnt / float(w0_down.shape[1]), 3)
            m["leakage_recovery_vs_real_pct"] = round(
                100.0
                * (naive_m["holdout_k0_linear_leakage_fro"] - m["holdout_k0_linear_leakage_fro"])
                / max(
                    naive_m["holdout_k0_linear_leakage_fro"]
                    - real_m["holdout_k0_linear_leakage_fro"],
                    1e-12,
                ),
                1,
            )
            part_a_arms[f"noise_n{n_cnt}"] = m
            if n_cnt == 2048:
                noise_2048_cfg = c
                noise_2048_patch = p

        part_a_arms["real_text_n16"] = real_m
        hybrid_m, _, _ = _eval_part_a(
            k0_noise_all,
            lam_soft,
            pres_keys=k0_train_mat,
            pres_weight=2.0,
        )
        hybrid_m["leakage_recovery_vs_real_pct"] = round(
            100.0
            * (naive_m["holdout_k0_linear_leakage_fro"] - hybrid_m["holdout_k0_linear_leakage_fro"])
            / max(
                naive_m["holdout_k0_linear_leakage_fro"]
                - real_m["holdout_k0_linear_leakage_fro"],
                1e-12,
            ),
            1,
        )
        part_a_arms["hybrid_noise2048_plus_16_y0"] = hybrid_m

        # Verify 0-byte re-derivation from seed alone (zero external calibration files!)
        rng_check = np.random.default_rng(noise_seed)
        noise_tokens_rebuild = rng_check.integers(
            0, vocab_size, size=(max(NOISE_COUNTS), 8), dtype=np.int64
        )
        k0_noise_rebuild = _harvest_batch_mlp_in_last(
            model, noise_tokens_rebuild, edit_layer=edit_layer
        )
        seed_rederive_ok, _ = verify_rederivation(
            noise_2048_patch,
            base_weight=w0_down,
            config=noise_2048_cfg,
            k0=k0_noise_rebuild / np.sqrt(float(k0_noise_rebuild.shape[1])),
            keys=edit_train_acts,
            target_values_or_residuals=target_residuals,
            is_residual=True,
            rank=4,
        )

        # ------------------------------------------------------------------
        # Part B: tokenpage#5 x Weft Parametric Wire-Dialect Readout Adapter
        # ------------------------------------------------------------------
        # Varying unseen prefixes ending in compact symbolic dialect tags (@A..@D):
        # hidden states linearly separate the 4 tags, yet unadapted TinyStories-1M
        # has no mapping from (@A, @B, @C, @D) -> (yes, no, stop, run).
        class_specs = [
            {
                "label": "class_A_yes",
                "tok_id": _token_id(tok, "yes", 60),
                "train": ["msg 10 : @A", "job 14 : @A", "ref 21 : @A", "cmd 33 : @A"],
                "val": ["log 45 : @A", "pkt 52 : @A", "req 68 : @A", "sig 79 : @A"],
            },
            {
                "label": "class_B_no",
                "tok_id": _token_id(tok, "no", 61),
                "train": ["msg 11 : @B", "job 15 : @B", "ref 24 : @B", "cmd 36 : @B"],
                "val": ["log 46 : @B", "pkt 55 : @B", "req 69 : @B", "sig 82 : @B"],
            },
            {
                "label": "class_C_stop",
                "tok_id": _token_id(tok, "stop", 62),
                "train": ["msg 12 : @C", "job 18 : @C", "ref 27 : @C", "cmd 39 : @C"],
                "val": ["log 48 : @C", "pkt 57 : @C", "req 71 : @C", "sig 85 : @C"],
            },
            {
                "label": "class_D_run",
                "tok_id": _token_id(tok, "run", 63),
                "train": ["msg 13 : @D", "job 19 : @D", "ref 29 : @D", "cmd 41 : @D"],
                "val": ["log 49 : @D", "pkt 58 : @D", "req 73 : @D", "sig 88 : @D"],
            },
        ]
        # Ensure 4 distinct target token IDs even on tiny synthetic fallback vocabs
        class_tok_ids = [c["tok_id"] for c in class_specs]
        if len(set(class_tok_ids)) != 4:
            class_tok_ids = [60 % vocab_size, 61 % vocab_size, 62 % vocab_size, 63 % vocab_size]

        train_x_list, train_y_list, train_res_list = [], [], []
        val_x_list, val_y_list, val_ids_list, val_base_lg_list = [], [], [], []
        class_dirs = []
        for c_idx in range(4):
            other_mean = np.mean([model.wte[class_tok_ids[j]] for j in range(4) if j != c_idx], axis=0)
            d_vec = model.wte[class_tok_ids[c_idx]] - other_mean
            # Physically matched residual scale (~2x natural Layer-7 ||mlp_out||)
            d_vec = d_vec / (np.linalg.norm(d_vec) + 1e-12) * 0.35
            class_dirs.append(d_vec)

        for c_idx, spec in enumerate(class_specs):
            for pr in spec["train"]:
                _, act = model.forward_last_token(tok.encode(pr), edit_layer=edit_layer)
                train_x_list.append(act)
                train_y_list.append(c_idx)
                train_res_list.append(class_dirs[c_idx])
            for pr in spec["val"]:
                ids = tok.encode(pr)
                lg, act = model.forward_last_token(ids, edit_layer=edit_layer)
                val_x_list.append(act)
                val_y_list.append(c_idx)
                val_ids_list.append(ids)
                val_base_lg_list.append(lg)

        x_tr = np.stack(train_x_list, axis=1)  # (256, 16)
        r_tr = np.stack(train_res_list, axis=1)  # (64, 16)
        x_va = np.stack(val_x_list, axis=1)  # (256, 16)

        # Linear hidden-state probe accuracy on unseen validation messages (tokenpage#5 check)
        y_onehot = np.zeros((4, x_tr.shape[1]), dtype=np.float64)
        for col_i, cls_i in enumerate(train_y_list):
            y_onehot[cls_i, col_i] = 1.0
        probe_w = y_onehot @ x_tr.T @ np.linalg.inv(x_tr @ x_tr.T + 0.1 * np.eye(x_tr.shape[0]))
        probe_preds = np.argmax(probe_w @ x_va, axis=0)
        linear_probe_val_acc = float(np.mean(probe_preds == np.array(val_y_list)))

        def _score_readout(proj: BetaEditNullspaceProjector | None) -> dict[str, Any]:
            correct = 0
            margins = []
            for ids, true_c in zip(val_ids_list, val_y_list):
                lg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=proj)
                c_logits = np.array([lg[tid] for tid in class_tok_ids], dtype=np.float64)
                if int(np.argmax(c_logits)) == true_c:
                    correct += 1
                other_max = float(np.max([c_logits[j] for j in range(4) if j != true_c]))
                margins.append(float(c_logits[true_c] - other_max))
            return {
                "val_readout_acc": round(correct / len(val_y_list), 4),
                "val_correct": correct,
                "val_total": len(val_y_list),
                "val_mean_class_margin": round(float(np.mean(margins)), 4),
            }

        base_readout = _score_readout(None)

        def _compile_readout_adapter(
            *,
            use_noise_k0: bool,
            use_hybrid_y0: bool = False,
        ) -> tuple[BetaEditNullspaceProjector, Any, dict[str, Any]]:
            cfg_r = BetaEditConfig(
                d_in=w0_down.shape[1],
                d_out=w0_down.shape[0],
                max_rank=4,
                lambda_leakage=1500.0 if use_noise_k0 else 0.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,
                threshold_mode="absolute",
                quant_dtype="f16",
                b_quant_dtype="int4",
                a_quant_dtype="int8",
            )
            proj_r = BetaEditNullspaceProjector(w0_down, cfg_r)
            if use_noise_k0:
                proj_r.initialize_anchor_covariance(k0_noise_all / np.sqrt(float(k0_noise_all.shape[1])))
            else:
                proj_r.initialize_anchor_covariance(np.zeros((w0_down.shape[1], 1)))
            patch_r = proj_r.stage_edit(
                x_tr,
                r_tr,
                is_residual=True,
                rank=4,
                preservation_keys=k0_train_mat if use_hybrid_y0 else None,
                preservation_weight=4.0 if use_hybrid_y0 else 0.0,
                slot_name="wire_dialect",
            )
            slot_id = proj_r.commit_patch(patch_r, x_tr)

            ro_metrics = _score_readout(proj_r)
            holdout_leak = proj_r.knowledge_leakage_norm(k0_holdout_mat)
            ret = 0
            max_d = 0.0
            for ids, blg in zip(k0_holdout_ids, k0_holdout_base_logits):
                nlg, _ = model.forward_last_token(ids, edit_layer=edit_layer, projector=proj_r)
                if int(np.argmax(nlg)) == int(np.argmax(blg)):
                    ret += 1
                max_d = max(max_d, float(np.max(np.abs(nlg - blg))))
            ro_metrics["holdout_k0_linear_leakage_fro"] = round(holdout_leak, 6)
            ro_metrics["holdout_k0_top1_retained"] = ret
            ro_metrics["holdout_k0_total"] = len(k0_holdout_ids)
            ro_metrics["holdout_k0_max_abs_logit_drift"] = round(max_d, 6)
            ro_metrics["slot_id"] = slot_id
            return proj_r, patch_r, ro_metrics

        _, _, readout_naive_m = _compile_readout_adapter(use_noise_k0=False)
        proj_ro, patch_ro, readout_noise_m = _compile_readout_adapter(use_noise_k0=True)
        _, _, readout_hybrid_m = _compile_readout_adapter(use_noise_k0=True, use_hybrid_y0=True)

        # Rollback check on readout adapter
        proj_ro.rollback_slot(readout_noise_m["slot_id"], revert_covariance=True)
        rollback_lg, _ = model.forward_last_token(
            k0_holdout_ids[0], edit_layer=edit_layer, projector=proj_ro
        )
        readout_rollback_diff = float(np.max(np.abs(rollback_lg - k0_holdout_base_logits[0])))

        # Package readout adapter into canonical weft-slab/v0.1 & compare vs F16 GGUF LoRA
        slab_path = tmp_path / "wire_dialect.wfsl"
        slab_blob = write_slab_wire(
            patch_ro,
            target_tensor=f"transformer.h.{edit_layer}.mlp.c_proj.weight",
            base_weights_sha256=base_weights_sha256,
            zone_manifest_id=SAMPLE_ZONE_MANIFEST_ID,
            output_path=slab_path,
        )
        slab_blob_2 = write_slab_wire(
            patch_ro,
            target_tensor=f"transformer.h.{edit_layer}.mlp.c_proj.weight",
            base_weights_sha256=base_weights_sha256,
            zone_manifest_id=SAMPLE_ZONE_MANIFEST_ID,
        )
        slab_ok, slab_probs, decoded_slab = verify_slab_wire(
            slab_path,
            gate_policy=GatePolicy(max_frob_sq_num=500, max_frob_sq_den=1),
            expected_zone_manifest_id=SAMPLE_ZONE_MANIFEST_ID,
            expected_base_weights_sha256=base_weights_sha256,
        )
        slab_roundtrip_exact = bool(
            decoded_slab is not None
            and slab_blob == slab_blob_2
            and np.array_equal(decoded_slab.factor_b, patch_ro.factor_b)
            and np.array_equal(decoded_slab.factor_a, patch_ro.factor_a)
        )
        # Verify 1-hex zone manifest rotation rejects container
        zone_drift_ok, zone_drift_probs, _ = verify_slab_wire(
            slab_path,
            expected_zone_manifest_id="sha256:" + ("0" * 64),
        )

        gguf_c_proj = tmp_path / "wire_dialect_c_proj.gguf"
        write_gguf_lora(
            gguf_c_proj,
            layer_factors={f"blk.{edit_layer}.ffn_down.weight": (patch_ro.factor_b, patch_ro.factor_a)},
            architecture="gpt_neo",
            lora_alpha=4.0,
        )

        # Wide-output 4B-scale container comparison (W_up: 10240 x 2560, rank 16)
        rng_4b = np.random.default_rng(42)
        b_4b = rng_4b.standard_normal((10240, 16)) * 0.02
        a_4b = rng_4b.standard_normal((16, 2560)) * 0.02
        from weft.certificate import build_certificate
        from weft.compiler import StagedPatch
        from weft.quant import quantize_array, sha256_factor_pair

        b4_bytes, b4_q = quantize_array(b_4b, "int4", axis=0)
        a4_bytes, a4_q = quantize_array(a_4b, "int8", axis=1)
        d4 = sha256_factor_pair(b4_bytes, a4_bytes, b4_q.shape, a4_q.shape, "f16", b_dtype="int4", a_dtype="int8")
        c4 = build_certificate(
            b4_q,
            a4_q,
            np.zeros((2560, 2560), dtype=np.float64),
            factor_digest=d4,
            quant_dtype="int4/int8",
            projector_epoch=1,
        )
        patch_4b = StagedPatch(
            factor_b=b4_q,
            factor_a=a4_q,
            b_bytes=b4_bytes,
            a_bytes=a4_bytes,
            singular_values=np.ones(16),
            effective_rank=16,
            truncation_rel_error=0.0,
            factor_digest=d4,
            certificate=c4,
            slot_name="corpus",
        )
        slab_4b_bytes = write_slab_wire(
            patch_4b,
            target_tensor="blk.16.ffn_up.weight",
            base_weights_sha256=base_weights_sha256,
            zone_manifest_id=SAMPLE_ZONE_MANIFEST_ID,
        )
        gguf_4b_path = tmp_path / "w_up_4b.gguf"
        write_gguf_lora(
            gguf_4b_path,
            layer_factors={"blk.16.ffn_up.weight": (b4_q, a4_q)},
            architecture="qwen2",
            lora_alpha=16.0,
        )
        gguf_4b_size = gguf_4b_path.stat().st_size
        decoded_4b = read_slab_wire(slab_4b_bytes)
        slab_4b_exact = bool(
            np.array_equal(decoded_4b.factor_b, b4_q) and np.array_equal(decoded_4b.factor_a, a4_q)
        )

        return {
            "experiment": "E5_noise_k0_scaling_and_wire_dialect_readout",
            "checkpoint_source": checkpoint_label,
            "is_synthetic_fallback": provenance.is_synthetic_fallback,
            "real_weights_verified": provenance.real_weights_verified,
            "weights_provenance": provenance.to_dict(),
            "edit_layer": edit_layer,
            "batch_forward_parity_max_diff": batch_forward_max_diff,
            "part_a_seeded_noise_k0_curve": {
                "noise_seed": noise_seed,
                "seed_only_zero_byte_rederivation_n2048": bool(seed_rederive_ok),
                "arms": part_a_arms,
            },
            "part_b_wire_dialect_readout_bridge": {
                "linear_probe_val_acc": round(linear_probe_val_acc, 4),
                "base_unadapted_readout": base_readout,
                "naive_int4_int8_readout": readout_naive_m,
                "seeded_noise_int4_int8_readout": readout_noise_m,
                "hybrid_noise_plus_16_y0_int4_int8_readout": readout_hybrid_m,
                "rollback_max_abs_logit_diff": readout_rollback_diff,
            },
            "part_c_slab_wire_container": {
                "c_proj_slab_bytes": len(slab_blob),
                "c_proj_gguf_f16_bytes": gguf_c_proj.stat().st_size,
                "c_proj_savings_vs_gguf_pct": round(
                    100.0 * (1.0 - len(slab_blob) / gguf_c_proj.stat().st_size), 1
                ),
                "c_proj_verify_ok": bool(slab_ok and slab_probs == []),
                "c_proj_roundtrip_and_zero_byte_exact": slab_roundtrip_exact,
                "zone_manifest_drift_rejected": bool(
                    (not zone_drift_ok) and ("zone_manifest_drift" in zone_drift_probs)
                ),
                "scale_4b_w_up_shape": [10240, 2560, 16],
                "scale_4b_slab_bytes": len(slab_4b_bytes),
                "scale_4b_gguf_f16_bytes": gguf_4b_size,
                "scale_4b_savings_vs_gguf_pct": round(
                    100.0 * (1.0 - len(slab_4b_bytes) / gguf_4b_size), 1
                ),
                "scale_4b_roundtrip_exact": slab_4b_exact,
            },
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument(
        "--require-real-weights",
        action="store_true",
        default=None,
        help="Fail closed instead of falling back to synthetic weights (or set WEFT_REQUIRE_REAL_WEIGHTS=1).",
    )
    ap.add_argument(
        "--expected-sha256",
        default=None,
        help="Pin the exact expected 64-hex SHA-256 digest of model.safetensors.",
    )
    args = ap.parse_args(argv)
    try:
        res = run_experiment(
            args.checkpoint_dir,
            require_real_weights=True if args.require_real_weights else None,
            expected_safetensors_sha256=args.expected_sha256,
        )
    except RealWeightsRequiredError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
