"""Experiment E8: SwiGLU Gate Metric, RMSNorm Radial Shrinkage, and Multi-Layer TangentBetaEdit.

Reproduces and resolves from first principles two coupled multi-LoRA failure modes
observed on 4B-scale pre-norm SwiGLU Transformers:

1. The 99.85% Linear Subspace vs 0.0% Live Behavioral Retention Paradox:
   Right-singular scalar projection `<Delta_tot V_solo, Delta_solo V_solo>_F / ||Delta_solo V_solo||_F^2`
   stays near 1.0 whenever sibling adapters write into orthogonal `B_other` columns,
   even when `alpha_other * W_down(silu(W_gate x) * (Delta_other x))` dominates the
   target domain's post-RMSNorm logit shift by >10x.

2. Single-Layer RMSNorm Radial Base-Logit Shrinkage (Effective Temperature Inflation):
   Even for a strictly tangent perturbation `<delta_perp, h0> = 0` with normalized
   energy `eta = ||delta_perp||_2^2 / R0^2`, `RMSNorm(h0 + delta_perp)` attenuates the
   entire base logit vector `W_embed RMSNorm(h0)` by `s(eta) = 1 - 1 / sqrt(1 + eta)`
   (`eta / (2*(1 + eta)) <= s(eta) <= eta / 2`). Concentrating full edit impulse into
   a single upper FFN layer inflates `s(eta)` quadratically; splitting across `L`
   upper layers (`1/L` per layer) cuts cumulative per-layer quadratic radial
   distortion by an exact factor of `1/L` while multiplying effective rank by `L`.

Run:
  .venv/bin/python experiments/e8_swiglu_rmsnorm_tangent_coupling.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector
from weft.gguf_lora import read_gguf_lora
from weft.tangent import (
    LayerPackSpec,
    TangentBetaEditConfig,
    compare_subspace_vs_functional_retention_f64,
    compile_multilayer_tangent_suite,
    decompose_rmsnorm_perturbation_f64,
    project_up_residual_rmsnorm_tangent_f64,
    rational_shrinkage_bounds,
    rmsnorm_f64,
    verify_multilayer_suite_rederivation,
)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _log_softmax_cols(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=0, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=0, keepdims=True))


def _step0_logprob_shift_mean(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    lp_a = _log_softmax_cols(logits_a)
    lp_b = _log_softmax_cols(logits_b)
    top_b = np.argmax(lp_b, axis=0)
    cols = np.arange(logits_b.shape[1])
    return float(np.mean(np.abs(lp_a[top_b, cols] - lp_b[top_b, cols])))


def _build_synthetic_4layer_swiglu_stack(seed: int = 20261119) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_model = 128
    d_ff = 384
    n_vocab = 64
    layers = (0, 1, 2, 3)

    w_norm_layers = {l: 1.0 + 0.1 * rng.standard_normal(d_model) for l in layers}
    w_gate = {l: rng.standard_normal((d_ff, d_model)) / math.sqrt(d_model) for l in layers}
    w_up = {l: rng.standard_normal((d_ff, d_model)) / math.sqrt(d_model) for l in layers}
    w_down = {l: rng.standard_normal((d_model, d_ff)) / math.sqrt(d_ff) for l in layers}
    gamma_final = 1.0 + 0.15 * rng.standard_normal(d_model)
    w_embed = rng.standard_normal((n_vocab, d_model)) / math.sqrt(d_model)
    return {
        "d_model": d_model,
        "d_ff": d_ff,
        "n_vocab": n_vocab,
        "layers": layers,
        "w_norm_layers": w_norm_layers,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "gamma_final": gamma_final,
        "w_embed": w_embed,
    }


def _forward_stack(
    stack: Mapping[str, Any],
    h_in: np.ndarray,
    *,
    deltas_by_pack: Mapping[str, Mapping[int, np.ndarray]] | None = None,
    scales: Mapping[str, float] | None = None,
    pack_order: tuple[str, ...] = ("corpus", "weft-eido", "formal-verification"),
    eps: float = 1e-6,
) -> dict[str, Any]:
    h = np.ascontiguousarray(h_in, dtype=np.float64).copy()
    up_in_by_layer: dict[int, np.ndarray] = {}
    gate_by_layer: dict[int, np.ndarray] = {}
    per_layer_shrinkage_sum = 0.0
    active_deltas = deltas_by_pack or {}
    active_scales = scales or {}

    for lid in stack["layers"]:
        x_l = rmsnorm_f64(h, stack["w_norm_layers"][lid], eps=eps)
        g_l = _silu(stack["w_gate"][lid] @ x_l)
        u_base = stack["w_up"][lid] @ x_l
        u_delta = np.zeros_like(u_base)
        for p_name in pack_order:
            sc = float(active_scales.get(p_name, 0.0))
            if sc != 0.0 and p_name in active_deltas and lid in active_deltas[p_name]:
                u_delta = u_delta + sc * (active_deltas[p_name][lid] @ x_l)

        ffn_base = stack["w_down"][lid] @ (g_l * u_base)
        ffn_delta = stack["w_down"][lid] @ (g_l * u_delta)
        h_base_next = h + ffn_base
        if float(np.linalg.norm(ffn_delta)) > 0.0:
            sh_list = [
                decompose_rmsnorm_perturbation_f64(
                    h_base_next[:, i],
                    ffn_delta[:, i],
                    stack["gamma_final"],
                    eps=eps,
                ).pure_tangent_shrinkage_s
                for i in range(h.shape[1])
            ]
            per_layer_shrinkage_sum += float(np.mean(sh_list))
        h = h_base_next + ffn_delta
        up_in_by_layer[lid] = x_l
        gate_by_layer[lid] = g_l

    h_norm = rmsnorm_f64(h, stack["gamma_final"], eps=eps)
    logits = stack["w_embed"] @ h_norm
    return {
        "h_pre": h,
        "logits": logits,
        "up_in": up_in_by_layer,
        "gate": gate_by_layer,
        "cumulative_layer_shrinkage": per_layer_shrinkage_sum,
    }


def run_experiment() -> dict[str, Any]:
    rng = np.random.default_rng(20261119)
    stack = _build_synthetic_4layer_swiglu_stack(seed=20261119)
    d_model = int(stack["d_model"])
    d_ff = int(stack["d_ff"])
    layers = tuple(stack["layers"])
    eps = 1e-6

    # Part A: Exact RMSNorm finite-perturbation identity, rational bounds, and 1/L law
    h0_probe = rng.standard_normal(d_model) * 3.0
    raw_delta = rng.standard_normal(d_model) * 1.1
    dec_gen = decompose_rmsnorm_perturbation_f64(
        h0_probe, raw_delta, stack["gamma_final"], eps=eps
    )
    delta_pure_tan = raw_delta - (float(np.dot(raw_delta, h0_probe)) / float(np.dot(h0_probe, h0_probe))) * h0_probe
    dec_tan = decompose_rmsnorm_perturbation_f64(
        h0_probe, delta_pure_tan, stack["gamma_final"], eps=eps
    )

    multilayer_shrinkage_curve: dict[str, dict[str, float]] = {}
    r0_sq = float(np.dot(h0_probe, h0_probe)) + float(d_model) * eps
    d_sq = float(np.dot(delta_pure_tan, delta_pure_tan))
    for l_cnt in (1, 2, 4, 8):
        eta_l = (d_sq / float(l_cnt * l_cnt)) / r0_sq
        low_l, s_l, up_l = rational_shrinkage_bounds(eta_l)
        multilayer_shrinkage_curve[f"L{l_cnt}"] = {
            "per_layer_eta": eta_l,
            "cum_exact_shrinkage": float(l_cnt) * s_l,
            "cum_quadratic_upper_bound": float(l_cnt) * up_l,
            "cum_rational_lower_bound": float(l_cnt) * low_l,
        }

    # Part B: 3-pack composition on upper 4 FFN blocks with shared carrier prototype + register prototypes
    # Mirrors E1's 75% shared-carrier geometry across all 4 SwiGLU + W_down + RMSNorm blocks
    q_orth, _ = np.linalg.qr(rng.standard_normal((d_model, 64)))
    b_shared = q_orth[:, 0:12]
    b_corpus = q_orth[:, 12:24]
    b_eido = q_orth[:, 24:36]
    b_fv = q_orth[:, 36:48]
    b_anchor = q_orth[:, 48:64]

    carrier_proto = np.mean(b_shared[:, :4], axis=1)
    carrier_proto = carrier_proto / float(np.linalg.norm(carrier_proto))

    def _sample_domain_inputs(b_dom: np.ndarray, n_cols: int, shared_frac: float = 0.75) -> np.ndarray:
        reg_proto = b_dom[:, 0]
        c_sh = rng.standard_normal((b_shared.shape[1], n_cols)) * (0.18 / math.sqrt(b_shared.shape[1]))
        c_dm = rng.standard_normal((b_dom.shape[1], n_cols)) * (0.18 / math.sqrt(b_dom.shape[1]))
        return (
            math.sqrt(shared_frac) * (carrier_proto[:, np.newaxis] + b_shared @ c_sh)
            + math.sqrt(1.0 - shared_frac) * (reg_proto[:, np.newaxis] + b_dom @ c_dm)
        ) * 3.2

    def _sample_anchors(n_cols: int) -> np.ndarray:
        anc_proto = b_anchor[:, 0]
        c_an = rng.standard_normal((b_anchor.shape[1], n_cols)) * (0.18 / math.sqrt(b_anchor.shape[1]))
        return (anc_proto[:, np.newaxis] + b_anchor @ c_an) * 3.2

    h_in_noise = rng.standard_normal((d_model, 512)) * 2.5
    h_in_anchors = _sample_anchors(24)
    # Intentionally unbalanced sample counts (48 formal states vs 24 corpus/eido states)
    h_in_corpus = _sample_domain_inputs(b_corpus, 24, shared_frac=0.75)
    h_in_eido = _sample_domain_inputs(b_eido, 24, shared_frac=0.75)
    h_in_fv = _sample_domain_inputs(b_fv, 48, shared_frac=0.75)

    # Disjoint held-out probes per domain and held-out anchors
    h_val_anchors = _sample_anchors(24)
    h_val = {
        "corpus": _sample_domain_inputs(b_corpus, 12, shared_frac=0.75),
        "weft-eido": _sample_domain_inputs(b_eido, 12, shared_frac=0.75),
        "formal-verification": _sample_domain_inputs(b_fv, 12, shared_frac=0.75),
    }

    base_noise = _forward_stack(stack, h_in_noise)
    base_anchors = _forward_stack(stack, h_in_anchors)
    base_train = {
        "corpus": _forward_stack(stack, h_in_corpus),
        "weft-eido": _forward_stack(stack, h_in_eido),
        "formal-verification": _forward_stack(stack, h_in_fv),
    }
    base_val_anchors = _forward_stack(stack, h_val_anchors)
    base_val = {dom: _forward_stack(stack, h_val[dom]) for dom in h_val}

    # Target token pairs: corpus (0 vs 1), weft-eido (10 vs 11, subtle weak residual in raw single-layer),
    # formal-verification (20 vs 21)
    domain_tok_pairs = {
        "corpus": (0, 1, 3.2),
        "weft-eido": (10, 11, 0.25),
        "formal-verification": (20, 21, 3.4),
    }

    raw_up_res: dict[str, dict[int, np.ndarray]] = {d: {} for d in domain_tok_pairs}
    for dom, (pos_t, neg_t, strength) in domain_tok_pairs.items():
        v_dir = stack["w_embed"][pos_t] - stack["w_embed"][neg_t]
        v_dir = v_dir / max(float(np.linalg.norm(v_dir)), 1e-12)
        hp = base_train[dom]["h_pre"]
        # Intentionally include a radial component on raw residuals to mirror naive logit pullback
        h_unit = hp / np.maximum(np.linalg.norm(hp, axis=0, keepdims=True), 1e-12)
        d_hidden = (v_dir[:, np.newaxis] + 0.65 * h_unit) * strength
        for lid in layers:
            g_l = base_train[dom]["gate"][lid]
            u_l = g_l * (stack["w_down"][lid].T @ d_hidden)
            u_norm = np.maximum(np.linalg.norm(u_l, axis=0, keepdims=True), 1e-12)
            raw_up_res[dom][lid] = (u_l / u_norm) * (strength * 2.2)

    rank = 4
    last_l = 3
    k0_noise_last = _forward_stack(stack, h_in_noise)["up_in"][last_l]
    tr_last = float(np.sum(k0_noise_last * k0_noise_last))
    k0_eff_last = k0_noise_last * math.sqrt(float(d_model) / tr_last)

    be_single_cfg = BetaEditConfig(
        d_in=d_model,
        d_out=d_ff,
        max_rank=rank,
        max_slots=8,
        lambda_leakage=20.0,
        lambda_ridge=0.5,
        nullspace_threshold=1e6,
        min_nullspace_dim=d_model,
        quant_dtype="f16",
    )
    w0_zero = np.zeros((d_ff, d_model), dtype=np.float64)

    # Arm 1: Single-layer independent BetaEdit (omitting sibling keys & radial-null projection)
    arm1_deltas: dict[str, dict[int, np.ndarray]] = {}
    for dom in ("corpus", "weft-eido", "formal-verification"):
        proj1 = BetaEditNullspaceProjector(w0_zero, be_single_cfg)
        proj1.initialize_anchor_covariance(k0_eff_last)
        p1 = proj1.stage_edit(
            base_train[dom]["up_in"][last_l],
            raw_up_res[dom][last_l],
            is_residual=True,
            rank=rank,
            preservation_keys=base_anchors["up_in"][last_l],
            preservation_weight=20.0,
            slot_name=dom,
        )
        arm1_deltas[dom] = {last_l: p1.factor_b @ p1.factor_a}

    # Arm 2: Single-layer sequential commit_patch (order-dependent!)
    def _compile_sequential(order: tuple[str, ...]) -> dict[str, dict[int, np.ndarray]]:
        proj_seq = BetaEditNullspaceProjector(w0_zero, be_single_cfg)
        proj_seq.initialize_anchor_covariance(k0_eff_last)
        out_d: dict[str, dict[int, np.ndarray]] = {}
        for dom in order:
            ps = proj_seq.stage_edit(
                base_train[dom]["up_in"][last_l],
                raw_up_res[dom][last_l],
                is_residual=True,
                rank=rank,
                preservation_keys=base_anchors["up_in"][last_l],
                preservation_weight=20.0,
                slot_name=dom,
            )
            proj_seq.commit_patch(ps, base_train[dom]["up_in"][last_l] * 5.0)
            out_d[dom] = {last_l: ps.factor_b @ ps.factor_a}
        return out_d

    arm2_fwd = _compile_sequential(("corpus", "weft-eido", "formal-verification"))
    arm2_rev = _compile_sequential(("formal-verification", "weft-eido", "corpus"))
    seq_order_diff_fro = max(
        float(np.linalg.norm(arm2_fwd[d][last_l] - arm2_rev[d][last_l]))
        for d in ("corpus", "weft-eido", "formal-verification")
    )

    # Arm 3: Single-layer symmetric TangentBetaEdit (L = 1 on layer 3)
    packs_1l = {
        dom: LayerPackSpec(
            keys_by_layer={last_l: base_train[dom]["up_in"][last_l]},
            up_residuals_by_layer={last_l: raw_up_res[dom][last_l]},
            gate_by_layer={last_l: base_train[dom]["gate"][last_l]},
            h_pre=base_train[dom]["h_pre"],
            target_total_post_wdown_rms=2.6,
        )
        for dom in ("corpus", "weft-eido", "formal-verification")
    }
    tcfg_1l = TangentBetaEditConfig(
        d_in=d_model,
        d_out=d_ff,
        rank=rank,
        lambda_leakage=20.0,
        lambda_ridge=0.25,
        anchor_preservation_weight=4000.0,
        sibling_preservation_weight=4000.0,
        project_radial_null=True,
        split_by_num_layers=True,
        max_per_layer_radial_eta=0.20,
        max_gain_multiplier=64.0,
        quant_dtype="f16",
    )
    suite_1l = compile_multilayer_tangent_suite(
        layer_ids=(last_l,),
        w_down_by_layer={last_l: stack["w_down"][last_l]},
        k0_noise_by_layer={last_l: base_noise["up_in"][last_l]},
        k0_anchors_by_layer={last_l: base_anchors["up_in"][last_l]},
        packs=packs_1l,
        config=tcfg_1l,
        rms_eps=eps,
    )
    arm3_deltas = {dom: suite_1l[dom].delta_by_layer() for dom in suite_1l}

    # Arm 4: 4-Layer Symmetric TangentBetaEdit across layers (0, 1, 2, 3)
    packs_4l = {
        dom: LayerPackSpec(
            keys_by_layer={lid: base_train[dom]["up_in"][lid] for lid in layers},
            up_residuals_by_layer={lid: raw_up_res[dom][lid] for lid in layers},
            gate_by_layer={lid: base_train[dom]["gate"][lid] for lid in layers},
            h_pre=base_train[dom]["h_pre"],
            target_total_post_wdown_rms=2.6,
        )
        for dom in ("corpus", "weft-eido", "formal-verification")
    }
    tcfg_4l = TangentBetaEditConfig(
        d_in=d_model,
        d_out=d_ff,
        rank=rank,
        lambda_leakage=20.0,
        lambda_ridge=0.25,
        anchor_preservation_weight=4000.0,
        sibling_preservation_weight=4000.0,
        project_radial_null=True,
        split_by_num_layers=True,
        max_per_layer_radial_eta=0.05,
        max_gain_multiplier=64.0,
        quant_dtype="f16",
    )
    suite_4l = compile_multilayer_tangent_suite(
        layer_ids=layers,
        w_down_by_layer=stack["w_down"],
        k0_noise_by_layer=base_noise["up_in"],
        k0_anchors_by_layer=base_anchors["up_in"],
        packs=packs_4l,
        config=tcfg_4l,
        rms_eps=eps,
    )
    arm4_deltas = {dom: suite_4l[dom].delta_by_layer() for dom in suite_4l}

    rederive_ok, rederive_report = verify_multilayer_suite_rederivation(
        suite_4l,
        layer_ids=layers,
        w_down_by_layer=stack["w_down"],
        k0_noise_by_layer=base_noise["up_in"],
        k0_anchors_by_layer=base_anchors["up_in"],
        packs=packs_4l,
        config=tcfg_4l,
        rms_eps=eps,
    )

    # Verify deterministic multi-layer GGUF v3 F16 emission and roundtrip
    with tempfile.TemporaryDirectory(prefix="weft_e8_") as tmp:
        tdp = Path(tmp)
        gguf_roundtrip_ok = True
        for dom, cpack in suite_4l.items():
            p_a, sha_a = cpack.write_gguf(tdp / f"{dom}.a.gguf", architecture="qwen35", rank=rank)
            p_b, sha_b = cpack.write_gguf(tdp / f"{dom}.b.gguf", architecture="qwen35", rank=rank)
            parsed = read_gguf_lora(p_a)
            if sha_a != sha_b or p_a.read_bytes() != p_b.read_bytes() or len(parsed.tensors) != 2 * len(layers):
                gguf_roundtrip_ok = False

    def _evaluate_arm(deltas: Mapping[str, Mapping[int, np.ndarray]], arm_layers: tuple[int, ...]) -> dict[str, Any]:
        eido_scales = {"corpus": 0.2, "weft-eido": 1.0, "formal-verification": 0.0}
        rpage_scales = {"corpus": 0.0, "weft-eido": 1.0, "formal-verification": 1.0}

        # Subspace vs functional retention on weft-eido validation states
        ret_cmp = compare_subspace_vs_functional_retention_f64(
            solo_name="weft-eido",
            scales=eido_scales,
            layer_deltas_by_pack=deltas,
            probe_keys_by_layer={l: base_val["weft-eido"]["up_in"][l] for l in arm_layers},
            probe_gates_by_layer={l: base_val["weft-eido"]["gate"][l] for l in arm_layers},
            w_down_by_layer={l: stack["w_down"][l] for l in arm_layers},
            h_pre=base_val["weft-eido"]["h_pre"],
            gamma=stack["gamma_final"],
            eps=eps,
        )

        # Full 4-layer nonlinear stack behavioral evaluation on weft-eido under [0.2, 1.0, 0.0]
        solo_eido = _forward_stack(
            stack, h_val["weft-eido"], deltas_by_pack=deltas, scales={"weft-eido": 1.0}
        )
        ui_routed = _forward_stack(
            stack, h_val["weft-eido"], deltas_by_pack=deltas, scales=eido_scales
        )
        rpage_eido = _forward_stack(
            stack, h_val["weft-eido"], deltas_by_pack=deltas, scales=rpage_scales
        )
        solo_shift = _step0_logprob_shift_mean(solo_eido["logits"], base_val["weft-eido"]["logits"])
        ui_drift = _step0_logprob_shift_mean(ui_routed["logits"], solo_eido["logits"])
        rpage_drift = _step0_logprob_shift_mean(rpage_eido["logits"], solo_eido["logits"])
        ui_behav_ret = max(0.0, 1.0 - (ui_drift / max(solo_shift, 1e-12)))
        rpage_behav_ret = max(0.0, 1.0 - (rpage_drift / max(solo_shift, 1e-12)))

        # Anchor top-1 retention under simultaneous [1.0, 1.0, 1.0]
        all_ones = {"corpus": 1.0, "weft-eido": 1.0, "formal-verification": 1.0}
        anc_out = _forward_stack(stack, h_val_anchors, deltas_by_pack=deltas, scales=all_ones)
        anc_top1 = int(
            np.sum(
                np.argmax(anc_out["logits"], axis=0)
                == np.argmax(base_val_anchors["logits"], axis=0)
            )
        )

        # Solo formal-verification cumulative per-layer radial shrinkage
        fv_solo = _forward_stack(
            stack,
            h_val["formal-verification"],
            deltas_by_pack=deltas,
            scales={"formal-verification": 1.0},
        )
        return {
            "linear_scalar_projection_retention": ret_cmp["linear_scalar_projection_retention"],
            "post_rmsnorm_functional_retention": ret_cmp["post_rmsnorm_functional_retention"],
            "ui_module_02_10_00_step0_retention": ui_behav_ret,
            "research_page_00_10_10_step0_retention": rpage_behav_ret,
            "crosstalk_to_solo_post_wdown_ratio": ret_cmp["crosstalk_to_solo_post_wdown_ratio"],
            "anchor_top1_retained_of_24": anc_top1,
            "fv_cumulative_layer_shrinkage": fv_solo["cumulative_layer_shrinkage"],
        }

    arms = {
        "arm1_single_layer_unsym_betaedit": _evaluate_arm(arm1_deltas, (last_l,)),
        "arm2_single_layer_seq_commit_fwd": _evaluate_arm(arm2_fwd, (last_l,)),
        "arm3_single_layer_sym_tangent_betaedit": _evaluate_arm(arm3_deltas, (last_l,)),
        "arm4_multilayer_4l_sym_tangent_betaedit": _evaluate_arm(arm4_deltas, layers),
    }

    # Verify load-order permutation invariance (all 6 permutations) & O(1) rollback on Arm 4
    ref_logits = _forward_stack(
        stack,
        h_val["weft-eido"],
        deltas_by_pack=arm4_deltas,
        scales={"corpus": 0.2, "weft-eido": 1.0, "formal-verification": 0.5},
        pack_order=("corpus", "weft-eido", "formal-verification"),
    )["logits"]
    max_perm_diff = 0.0
    for perm in itertools.permutations(("corpus", "weft-eido", "formal-verification")):
        p_logits = _forward_stack(
            stack,
            h_val["weft-eido"],
            deltas_by_pack=arm4_deltas,
            scales={"corpus": 0.2, "weft-eido": 1.0, "formal-verification": 0.5},
            pack_order=perm,
        )["logits"]
        max_perm_diff = max(max_perm_diff, float(np.max(np.abs(p_logits - ref_logits))))

    rb_logits = _forward_stack(
        stack,
        h_val["weft-eido"],
        deltas_by_pack=arm4_deltas,
        scales={"corpus": 0.0, "weft-eido": 0.0, "formal-verification": 0.0},
    )["logits"]
    rollback_max_diff = float(np.max(np.abs(rb_logits - base_val["weft-eido"]["logits"])))

    # Verify exact radial-null projection inner product == 0
    u_proj_check = project_up_residual_rmsnorm_tangent_f64(
        raw_up_res["corpus"][last_l],
        base_train["corpus"]["gate"][last_l],
        stack["w_down"][last_l],
        base_train["corpus"]["h_pre"],
    )
    dh_proj_check = stack["w_down"][last_l] @ (base_train["corpus"]["gate"][last_l] * u_proj_check)
    max_radial_inner_prod = float(
        np.max(np.abs(np.sum(dh_proj_check * base_train["corpus"]["h_pre"], axis=0)))
    )

    return {
        "experiment": "E8_swiglu_rmsnorm_tangent_coupling",
        "part_a_rmsnorm_identity_and_multilayer_bounds": {
            "general_reconstruction_l2_error": dec_gen.reconstruction_l2_error,
            "pure_tangent_reconstruction_l2_error": dec_tan.reconstruction_l2_error,
            "pure_tangent_radial_coord_c": dec_tan.radial_coord_c,
            "pure_tangent_rational_lower": dec_tan.rational_lower_bound,
            "pure_tangent_exact_s": dec_tan.pure_tangent_shrinkage_s,
            "pure_tangent_rational_upper": dec_tan.rational_upper_bound,
            "radial_null_projection_max_abs_dot": max_radial_inner_prod,
            "multilayer_shrinkage_curve": multilayer_shrinkage_curve,
        },
        "part_b_three_pack_composition": {
            "sequential_commit_compile_order_fro_diff": seq_order_diff_fro,
            "arms": arms,
            "arm4_zero_byte_rederivation": rederive_ok,
            "arm4_reversed_pack_order_zero_byte_match": rederive_report[
                "reversed_pack_order_zero_byte_match"
            ],
            "arm4_gguf_roundtrip_and_zero_byte_match": gguf_roundtrip_ok,
            "arm4_six_permutation_max_logit_diff": max_perm_diff,
            "arm4_rollback_max_logit_diff": rollback_max_diff,
        },
    }


def main() -> None:
    res = run_experiment()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
