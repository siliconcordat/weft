"""Experiment E1: Shared-token null-space collapse vs register-scoped composition.

Tests simultaneous 3-pack composition (corpus prose, weft-eido UI modules,
formal-verification proof blocks) on a deterministic 4-layer SwiGLU transformer
stack where all three packs share high-frequency carrier tokens.

Measures four concrete phenomena:
  1. Marchenko-Pastur K0 calibration boundary (n_k0 = 256 vs n_k0 = 2048 for d_ff = 256):
     undersampled K0 cuts train leakage while worsening unseen holdout leakage.
  2. Null-space dimension collapse m(t) across absolute vs relative thresholds.
  3. Five composition arms on well-conditioned K0 (n_k0 = 2048):
     A. Naive unconstrained linear sum
     B1. Hard-threshold BetaEdit
     B2. Soft-covariance BetaEdit (smooth C^(t) ridge, no hard clipping)
     C. Soft-covariance BetaEdit + register-scoped y=0 preservation (simultaneous sum)
     D. Soft-covariance BetaEdit + register-scoped y=0 + L5 zone-routed scale gating
  4. Cross-layer nonlinear drift (Layer-2 linear leakage vs Layer-3 RMSNorm/SwiGLU drift).

Run:
  .venv/bin/python experiments/e1_shared_token_composition.py
"""

from __future__ import annotations

import json
import numpy as np

from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class SwiGLUStack:
    """4-layer deterministic SwiGLU transformer MLP stack in CPU float64."""

    def __init__(self, d_model: int = 128, d_ff: int = 256, n_vocab: int = 48, seed: int = 20261119):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_vocab = n_vocab
        self.n_layers = 4
        scale = 1.0 / np.sqrt(d_model)
        self.w_gate = [rng.standard_normal((d_ff, d_model)) * scale for _ in range(self.n_layers)]
        self.w_up = [rng.standard_normal((d_ff, d_model)) * scale for _ in range(self.n_layers)]
        self.w_down = [
            rng.standard_normal((d_model, d_ff)) * (0.6 / np.sqrt(d_ff))
            for _ in range(self.n_layers)
        ]
        self.w_out = rng.standard_normal((n_vocab, d_model)) * (1.5 / np.sqrt(d_model))

    def forward_until_layer_mlp_in(self, x: np.ndarray, target_layer: int) -> np.ndarray:
        h = x.astype(np.float64, copy=True)
        for idx in range(target_layer):
            hn = rms_norm(h)
            mid = silu(hn @ self.w_gate[idx].T) * (hn @ self.w_up[idx].T)
            h = h + mid @ self.w_down[idx].T
        hn = rms_norm(h)
        return silu(hn @ self.w_gate[target_layer].T) * (hn @ self.w_up[target_layer].T)

    def forward_logits(
        self,
        x: np.ndarray,
        *,
        target_layer: int = 2,
        projector: BetaEditNullspaceProjector | None = None,
        slot_scales: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = x.astype(np.float64, copy=True)
        for idx in range(self.n_layers):
            hn = rms_norm(h)
            mid = silu(hn @ self.w_gate[idx].T) * (hn @ self.w_up[idx].T)
            if idx == target_layer and projector is not None:
                delta_h = projector.forward(mid, slot_scales=slot_scales)
            else:
                delta_h = mid @ self.w_down[idx].T
            h = h + delta_h
        h_final = rms_norm(h)
        return h_final @ self.w_out.T, h_final


def build_shared_carrier_dataset(d_model: int = 128, seed: int = 20261119) -> dict[str, np.ndarray]:
    """Generate structured inputs with a heavy shared carrier subspace + disjoint registers."""
    rng = np.random.default_rng(seed)
    q_full, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))

    b_shared = q_full[:, 0:16]
    b_k0 = q_full[:, 16:40]
    b_corpus = q_full[:, 40:52]
    b_eido = q_full[:, 52:64]
    b_formal = q_full[:, 64:76]

    carrier_proto = np.mean(b_shared[:, :4], axis=1)
    carrier_proto = carrier_proto / np.linalg.norm(carrier_proto)

    def sample_k0(n: int) -> np.ndarray:
        c = rng.standard_normal((n, b_k0.shape[1])) / np.sqrt(b_k0.shape[1])
        return c @ b_k0.T

    def sample_pack(basis: np.ndarray, n: int, shared_weight: float = 0.75) -> np.ndarray:
        reg_proto = basis[:, 0]
        c_sh = rng.standard_normal((n, b_shared.shape[1])) * (0.20 / np.sqrt(b_shared.shape[1]))
        c_reg = rng.standard_normal((n, basis.shape[1])) * (0.20 / np.sqrt(basis.shape[1]))
        return (
            np.sqrt(shared_weight) * (carrier_proto[np.newaxis, :] + c_sh @ b_shared.T)
            + np.sqrt(1.0 - shared_weight) * (reg_proto[np.newaxis, :] + c_reg @ basis.T)
        )

    mixed_c_sh = rng.standard_normal((30, b_shared.shape[1])) * (0.20 / np.sqrt(b_shared.shape[1]))
    mixed_formal_in_prose = (
        np.sqrt(0.60) * (carrier_proto[np.newaxis, :] + mixed_c_sh @ b_shared.T)
        + np.sqrt(0.20) * b_corpus[:, 0][np.newaxis, :]
        + np.sqrt(0.20) * b_formal[:, 0][np.newaxis, :]
    )

    return {
        "k0_train_256": sample_k0(256),
        "k0_train_2048": sample_k0(2048),
        "k0_holdout": sample_k0(64),
        "corpus_train": sample_pack(b_corpus, 48),
        "corpus_holdout": sample_pack(b_corpus, 48),
        "eido_train": sample_pack(b_eido, 48),
        "eido_holdout": sample_pack(b_eido, 48),
        "formal_train": sample_pack(b_formal, 48),
        "formal_holdout": sample_pack(b_formal, 48),
        "mixed_formal_zone": mixed_formal_in_prose,
    }


def make_pack_targets(
    stack: SwiGLUStack,
    n_samples: int,
    boost_tokens: list[int],
    suppress_tokens: list[int],
    strength: float = 2.2,
) -> np.ndarray:
    direction = np.zeros(stack.d_model, dtype=np.float64)
    for t in boost_tokens:
        direction += stack.w_out[t]
    for t in suppress_tokens:
        direction -= stack.w_out[t]
    direction = direction / (np.linalg.norm(direction) + 1e-12) * strength
    return np.tile(direction[:, np.newaxis], (1, n_samples))


def score_pack_compliance(
    logits: np.ndarray,
    base_logits: np.ndarray,
    boost_tokens: list[int],
    suppress_tokens: list[int],
    threshold: float = 0.75,
) -> tuple[float, float]:
    base_margin = np.mean(base_logits[:, boost_tokens], axis=1) - np.mean(
        base_logits[:, suppress_tokens], axis=1
    )
    new_margin = np.mean(logits[:, boost_tokens], axis=1) - np.mean(
        logits[:, suppress_tokens], axis=1
    )
    gain = new_margin - base_margin
    return float(np.mean(gain >= threshold)), float(np.mean(gain))


def run_experiment() -> dict[str, object]:
    stack = SwiGLUStack(d_model=128, d_ff=256, n_vocab=48, seed=20261119)
    data = build_shared_carrier_dataset(d_model=128, seed=20261119)
    target_layer = 2

    acts = {k: stack.forward_until_layer_mlp_in(v, target_layer).T for k, v in data.items()}
    # Scale K0 Gram matrices by 1/sqrt(n_k0/256) so total energy is matched across sample sizes
    k0_256 = acts["k0_train_256"]
    k0_2048 = acts["k0_train_2048"] / np.sqrt(2048.0 / 256.0)

    base_logits = {k: stack.forward_logits(v, target_layer=target_layer)[0] for k, v in data.items()}
    base_h_final = {k: stack.forward_logits(v, target_layer=target_layer)[1] for k, v in data.items()}
    k0_base_argmax = np.argmax(base_logits["k0_holdout"], axis=1)

    pack_specs = {
        "corpus": {"boost": [0, 1], "suppress": [2, 3]},
        "eido": {"boost": [4, 5], "suppress": [6, 7]},
        "formal": {"boost": [2, 3], "suppress": [0, 1]},
    }
    pack_residuals = {
        name: make_pack_targets(
            stack, data[f"{name}_train"].shape[0], spec["boost"], spec["suppress"], strength=2.2
        )
        for name, spec in pack_specs.items()
    }
    w0 = stack.w_down[target_layer]

    # 1. Marchenko-Pastur K0 calibration boundary (n_k0=256 vs n_k0=2048 for d_ff=256)
    mp_check: dict[str, dict[str, float]] = {}
    for label, k0_tr, lam_l in [
        ("naive_n256", k0_256, 0.0),
        ("beta_undersampled_n256", k0_256, 200.0),
        ("beta_oversampled_n2048", k0_2048, 200.0),
    ]:
        p_mp = BetaEditNullspaceProjector(
            w0,
            BetaEditConfig(
                d_in=256,
                d_out=128,
                max_rank=8,
                lambda_leakage=lam_l,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,
                quant_dtype="bf16",
            ),
        )
        p_mp.initialize_anchor_covariance(k0_tr)
        pt = p_mp.stage_edit(acts["corpus_train"], pack_residuals["corpus"], is_residual=True, rank=6)
        p_mp.commit_patch(pt, acts["corpus_train"])
        dw = pt.factor_b @ pt.factor_a
        lg_k0, _ = stack.forward_logits(data["k0_holdout"], target_layer=target_layer, projector=p_mp)
        mp_check[label] = {
            "delta_frob_norm": round(float(np.linalg.norm(dw, "fro")), 4),
            "k0_train_leakage_fro": round(p_mp.knowledge_leakage_norm(k0_tr), 4),
            "k0_holdout_leakage_fro": round(p_mp.knowledge_leakage_norm(acts["k0_holdout"]), 4),
            "k0_holdout_top1_retention": round(
                float(np.mean(np.argmax(lg_k0, axis=1) == k0_base_argmax)), 4
            ),
        }

    # 2. Null-space dimension collapse m(t) across thresholds on oversampled K0
    collapse_trace: dict[str, list[int]] = {}
    for label, mode, thresh in [
        ("abs_2e-2", "absolute", 2e-2),
        ("relmax_1e-2", "relative_max", 1e-2),
        ("relmax_1e-3", "relative_max", 1e-3),
    ]:
        p_probe = BetaEditNullspaceProjector(
            w0,
            BetaEditConfig(
                d_in=256,
                d_out=128,
                max_rank=8,
                lambda_leakage=200.0,
                lambda_ridge=1.0,
                nullspace_threshold=thresh,
                threshold_mode=mode,  # type: ignore[arg-type]
                refresh_period=1,
                quant_dtype="bf16",
            ),
        )
        dims = [p_probe.initialize_anchor_covariance(k0_2048)]
        for name in ("corpus", "eido", "formal"):
            patch = p_probe.stage_edit(
                acts[f"{name}_train"], pack_residuals[name], is_residual=True, rank=6, slot_name=name
            )
            p_probe.commit_patch(patch, acts[f"{name}_train"])
            dims.append(p_probe.nullspace_dim)
        collapse_trace[label] = dims

    # 3. Five composition arms using well-conditioned K0 (n_k0 = 2048)
    def build_arm(arm: str) -> tuple[BetaEditNullspaceProjector, dict[str, int], bool]:
        if arm == "A_naive_sum":
            cfg = BetaEditConfig(
                d_in=256,
                d_out=128,
                max_rank=8,
                lambda_leakage=0.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,
                threshold_mode="absolute",
                refresh_period=1000,
                quant_dtype="bf16",
            )
        elif arm == "B1_hard_thresh_betaedit":
            cfg = BetaEditConfig(
                d_in=256,
                d_out=128,
                max_rank=8,
                lambda_leakage=200.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e-2,
                threshold_mode="relative_max",
                refresh_period=1,
                quant_dtype="bf16",
            )
        else:
            # Soft-covariance BetaEdit with balanced lambda_leakage so K0 stays protected
            # while packs retain enough signal to pass compliance thresholds.
            cfg = BetaEditConfig(
                d_in=256,
                d_out=128,
                max_rank=8,
                lambda_leakage=40.0,
                lambda_ridge=1.0,
                nullspace_threshold=1e9,
                threshold_mode="absolute",
                refresh_period=1,
                quant_dtype="bf16",
            )

        proj = BetaEditNullspaceProjector(w0, cfg)
        proj.initialize_anchor_covariance(k0_2048)
        slots: dict[str, int] = {}
        rederive_all = True

        for name in ("corpus", "eido", "formal"):
            if arm in ("C_soft_reg_scoped", "D_soft_reg_plus_zone"):
                others = [acts[f"{o}_train"] for o in ("corpus", "eido", "formal") if o != name]
                pres_keys = np.concatenate(others, axis=1)
                pres_w = 10.0
            else:
                pres_keys = None
                pres_w = 1.0

            patch = proj.stage_edit(
                acts[f"{name}_train"],
                pack_residuals[name],
                is_residual=True,
                rank=6,
                preservation_keys=pres_keys,
                preservation_weight=pres_w,
                slot_name=name,
            )
            if name == "corpus":
                ok_rd, _ = verify_rederivation(
                    patch,
                    base_weight=w0,
                    config=cfg,
                    k0=k0_2048,
                    keys=acts["corpus_train"],
                    target_values_or_residuals=pack_residuals["corpus"],
                    is_residual=True,
                    rank=6,
                    preservation_keys=pres_keys,
                    preservation_weight=pres_w,
                )
                rederive_all = rederive_all and ok_rd
            slots[name] = proj.commit_patch(patch, acts[f"{name}_train"])
        return proj, slots, rederive_all

    arm_names = (
        "A_naive_sum",
        "B1_hard_thresh_betaedit",
        "B2_soft_cov_betaedit",
        "C_soft_reg_scoped",
        "D_soft_reg_plus_zone",
    )
    arm_results: dict[str, dict[str, object]] = {}
    for arm in arm_names:
        proj, slots, ok_rd = build_arm(arm)

        solo_rates: dict[str, float] = {}
        for name, spec in pack_specs.items():
            sc = {slots[name]: 1.0}
            lg, _ = stack.forward_logits(
                data[f"{name}_holdout"], target_layer=target_layer, projector=proj, slot_scales=sc
            )
            rate, _ = score_pack_compliance(
                lg, base_logits[f"{name}_holdout"], spec["boost"], spec["suppress"]
            )
            solo_rates[name] = round(rate, 4)

        comp_rates: dict[str, float] = {}
        comp_margins: dict[str, float] = {}
        for name, spec in pack_specs.items():
            sc = {slots[name]: 1.0} if arm == "D_soft_reg_plus_zone" else {s: 1.0 for s in slots.values()}
            lg, _ = stack.forward_logits(
                data[f"{name}_holdout"], target_layer=target_layer, projector=proj, slot_scales=sc
            )
            rate, mg = score_pack_compliance(
                lg, base_logits[f"{name}_holdout"], spec["boost"], spec["suppress"]
            )
            comp_rates[name] = round(rate, 4)
            comp_margins[name] = round(mg, 4)

        mixed_sc = {slots["formal"]: 1.0} if arm == "D_soft_reg_plus_zone" else {s: 1.0 for s in slots.values()}
        lg_mixed, _ = stack.forward_logits(
            data["mixed_formal_zone"], target_layer=target_layer, projector=proj, slot_scales=mixed_sc
        )
        mixed_formal_rate, mixed_formal_margin = score_pack_compliance(
            lg_mixed,
            base_logits["mixed_formal_zone"],
            pack_specs["formal"]["boost"],
            pack_specs["formal"]["suppress"],
        )

        lg_corpus_on_formal, _ = stack.forward_logits(
            data["formal_holdout"],
            target_layer=target_layer,
            projector=proj,
            slot_scales={slots["corpus"]: 1.0},
        )
        _, corpus_harm_on_formal = score_pack_compliance(
            lg_corpus_on_formal,
            base_logits["formal_holdout"],
            pack_specs["formal"]["boost"],
            pack_specs["formal"]["suppress"],
        )

        k0_scales = {s: 1.0 for s in slots.values()}
        lg_k0, h_k0 = stack.forward_logits(
            data["k0_holdout"], target_layer=target_layer, projector=proj, slot_scales=k0_scales
        )
        k0_ret = float(np.mean(np.argmax(lg_k0, axis=1) == k0_base_argmax))
        linear_leak = proj.knowledge_leakage_norm(acts["k0_holdout"], slot_scales=k0_scales)
        nonlinear_drift = float(np.linalg.norm(h_k0 - base_h_final["k0_holdout"], "fro"))

        arm_results[arm] = {
            "solo_pass_rates": solo_rates,
            "composed_pass_rates": comp_rates,
            "composed_margin_gain": comp_margins,
            "mixed_formal_zone_pass_rate": round(mixed_formal_rate, 4),
            "mixed_formal_zone_margin_gain": round(mixed_formal_margin, 4),
            "corpus_margin_shift_on_formal": round(corpus_harm_on_formal, 4),
            "k0_top1_retention": round(k0_ret, 4),
            "k0_layer2_linear_leakage_fro": round(linear_leak, 4),
            "k0_layer3_nonlinear_drift_fro": round(nonlinear_drift, 4),
            "zero_byte_rederivation": ok_rd,
        }

    return {
        "experiment": "E1_shared_token_composition",
        "seed": 20261119,
        "stack": {"layers": 4, "d_model": 128, "d_ff": 256, "n_vocab": 48, "quant": "bf16"},
        "marchenko_pastur_k0_boundary": mp_check,
        "nullspace_dim_collapse_m_of_t": collapse_trace,
        "arms": arm_results,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
