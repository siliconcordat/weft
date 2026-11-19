"""Experiment E7: Latent Lattice Language Models (arXiv:2611.21608) x Weft MDE.

Implements a faithful 4-layer Latent Lattice Transformer in deterministic CPU
float64 NumPy following Duggal et al. (arXiv:2611.21608v1, Sec. 3.2.1-3.2.2 and
App. A.2.1):
  - Layer blocks: early E = {0}, intermediate Lattice L = {1, 2}, late D = {3}.
  - Horizontal affine gated recurrence (parallel-scan form):
      Delta_t^l = softplus(W_Delta^l * RMSNorm(h_t^{l-1}) + b_Delta^l)
      g_t^l     = exp(-exp(a^l) * Delta_t^l)
      c_t^l     = SiLU(W_r^l * RMSNorm(h_t^{l-1}) + b_r^l)
      r_t^l     = g_t^l * r_{t-1}^l + (1 - g_t^l) * c_t^l
      r_hat_t^l = RMSNorm(r_t^l)
  - Vertical Attention + KV-cache write FiLM conditioning:
      q_t^l = (W_q^l h_hat) * (1 + tanh(W_gq^l r_hat)) + W_bq^l r_hat
      k_t^l = (W_k^l h_hat) * (1 + tanh(W_gk^l r_hat)) + W_bk^l r_hat  -> KV cache M_t^l
      v_t^l = (W_v^l h_hat) * (1 + tanh(W_gv^l r_hat)) + W_bv^l r_hat  -> KV cache M_t^l
  - SwiGLU MLP modulation:
      gate = W_gate^l m + W_bgate^l r_hat,  up = W_up^l m + W_bup^l r_hat
      h_t^l = h_bar_t^l + W_down^l (SiLU(gate) * up)
  - Optional top-Lattice decoding modulation (App. A.2.1):
      logits = W_U (RMSNorm(h_t^N) + P_r RMSNorm(r_t^{top}))

Tests three falsifiable hypotheses with zero external downloads:
  Part A: Horizontal Lattice cartridges (capability-partitioned banks in r_t^l)
          vs vertical BetaEdit slabs vs coupled vertical + horizontal MDE across
          3 shared-carrier domains (prose, ui, formal) + unseen K0 preservation.
  Part B: KV-write contamination & atomic rollback proof comparing
          (1) adapter-scale-zero only, (2) KV-cache flush only (proving surviving
          poisoned horizontal latents re-contaminate future KV writes!), vs
          (3) atomic KV + horizontal-lattice bank restore (0.0 exact diff).
  Part C: Hard Veto 1 0-byte re-derivation of compiled L_0 cartridges and
          horizontal-to-vertical conditioning adapters + Asymmetric Factor Leakage
          Theorem (int4/int8 & int4/f16 vs int4/int4 & reversed int8/int4 control)
          on wide-output Lattice MLP/KV conditioning projections.

Run:
  .venv/bin/python experiments/e7_latent_lattice_mde.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weft.certificate import GatePolicy
from weft.compiler import BetaEditConfig, BetaEditNullspaceProjector, verify_rederivation


def _rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def _silu(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -40.0, 40.0)
    return xc / (1.0 + np.exp(-xc))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -40.0, 30.0))))


def _softmax(x: np.ndarray) -> np.ndarray:
    xm = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(xm)
    return ex / np.sum(ex, axis=-1, keepdims=True)


@dataclass
class LatticeLayerParams:
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray
    is_lattice: bool
    # Horizontal recurrence & FiLM/MLP conditioning (present when is_lattice=True)
    w_delta: Optional[np.ndarray] = None
    b_delta: Optional[np.ndarray] = None
    a_log: Optional[np.ndarray] = None
    w_r: Optional[np.ndarray] = None
    b_r: Optional[np.ndarray] = None
    r0: Optional[np.ndarray] = None
    w_gq: Optional[np.ndarray] = None
    w_bq: Optional[np.ndarray] = None
    w_gk: Optional[np.ndarray] = None
    w_bk: Optional[np.ndarray] = None
    w_gv: Optional[np.ndarray] = None
    w_bv: Optional[np.ndarray] = None
    w_bgate: Optional[np.ndarray] = None
    w_bup: Optional[np.ndarray] = None


@dataclass
class LatticeRuntimeState:
    """Mutable per-sequence state separated into KV cache and horizontal lattice banks."""

    kv_keys: dict[int, list[np.ndarray]]
    kv_vals: dict[int, list[np.ndarray]]
    r_horiz: dict[int, np.ndarray]

    def clone(self) -> "LatticeRuntimeState":
        return LatticeRuntimeState(
            kv_keys={k: [v.copy() for v in vs] for k, vs in self.kv_keys.items()},
            kv_vals={k: [v.copy() for v in vs] for k, vs in self.kv_vals.items()},
            r_horiz={k: v.copy() for k, v in self.r_horiz.items()},
        )


class LatentLatticeTransformer:
    """Deterministic CPU float64 4-layer Latent Lattice Transformer (arXiv:2611.21608)."""

    def __init__(
        self,
        *,
        n_layers: int = 4,
        lattice_layers: tuple[int, ...] = (1, 2),
        d_model: int = 64,
        d_ff: int = 128,
        d_r: int = 48,
        n_heads: int = 4,
        n_vocab: int = 48,
        seed: int = 20261119,
    ) -> None:
        self.n_layers = n_layers
        self.lattice_layers = lattice_layers
        self.top_lattice = max(lattice_layers)
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_r = d_r
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_vocab = n_vocab

        rng = np.random.default_rng(seed)
        self.w_emb = rng.standard_normal((n_vocab, d_model)) * 0.25
        self.w_lm = rng.standard_normal((n_vocab, d_model)) * 0.25
        # Decoding modulation head P_r (App. A.2.1): maps top horizontal latent to readout space
        self.p_r = rng.standard_normal((d_model, d_r)) * 0.08
        self.layers: list[LatticeLayerParams] = []

        for l_idx in range(n_layers):
            is_lat = l_idx in lattice_layers
            lp = LatticeLayerParams(
                wq=rng.standard_normal((d_model, d_model)) * (1.0 / np.sqrt(d_model)),
                wk=rng.standard_normal((d_model, d_model)) * (1.0 / np.sqrt(d_model)),
                wv=rng.standard_normal((d_model, d_model)) * (1.0 / np.sqrt(d_model)),
                wo=rng.standard_normal((d_model, d_model)) * (0.5 / np.sqrt(d_model)),
                w_gate=rng.standard_normal((d_ff, d_model)) * (1.0 / np.sqrt(d_model)),
                w_up=rng.standard_normal((d_ff, d_model)) * (1.0 / np.sqrt(d_model)),
                w_down=rng.standard_normal((d_model, d_ff)) * (0.5 / np.sqrt(d_ff)),
                is_lattice=is_lat,
            )
            if is_lat:
                # Bank 0 [0:12] is active base horizontal stream; banks 1..3 [12:48] are
                # isolated MDE cartridge slots driven by L_0 and zone-routed channel masks.
                lp.w_delta = rng.standard_normal((d_r, d_model)) * 0.15
                lp.b_delta = np.full(d_r, -1.2, dtype=np.float64)
                lp.a_log = np.full(d_r, -1.0, dtype=np.float64)
                lp.w_r = rng.standard_normal((d_r, d_model)) * 0.20
                lp.b_r = np.zeros(d_r, dtype=np.float64)
                # Keep capability banks 1..3 slow-decaying and isolated from unmasked base drift
                lp.w_r[12:, :] = 0.0
                lp.w_delta[12:, :] = 0.0
                lp.b_delta[12:] = -4.0
                lp.a_log[12:] = -3.0
                r0 = np.zeros(d_r, dtype=np.float64)
                r0[:12] = rng.standard_normal(12) * 0.25
                lp.r0 = r0

                # Small nonzero trained-like FiLM & MLP projections on base bank 0;
                # cartridge banks 1..3 start zero-initialized (Sec. 3.2.1) and are compiled cleanly.
                scale_f = 0.12
                lp.w_gq = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_bq = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_gk = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_bk = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_gv = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_bv = np.zeros((d_model, d_r), dtype=np.float64)
                lp.w_bgate = np.zeros((d_ff, d_r), dtype=np.float64)
                lp.w_bup = np.zeros((d_ff, d_r), dtype=np.float64)

                lp.w_gq[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_bq[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_gk[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_bk[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_gv[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_bv[:, :12] = rng.standard_normal((d_model, 12)) * scale_f
                lp.w_bgate[:, :12] = rng.standard_normal((d_ff, 12)) * scale_f
                lp.w_bup[:, :12] = rng.standard_normal((d_ff, 12)) * scale_f
                # Small capability-bank FiLM K/V/MLP hooks (active only when bank mask = 1.0)
                lp.w_bk[:, 12:] = rng.standard_normal((d_model, d_r - 12)) * 0.04
                lp.w_bv[:, 12:] = rng.standard_normal((d_model, d_r - 12)) * 0.04
                lp.w_bgate[:, 12:] = rng.standard_normal((d_ff, d_r - 12)) * 0.04
            self.layers.append(lp)

    def init_state(
        self,
        *,
        l0_overrides: Optional[dict[int, np.ndarray]] = None,
    ) -> LatticeRuntimeState:
        r_h: dict[int, np.ndarray] = {}
        for l_idx in self.lattice_layers:
            base_r0 = self.layers[l_idx].r0
            assert_free = np.zeros(self.d_r, dtype=np.float64) if base_r0 is None else base_r0.copy()
            if l0_overrides is not None and l_idx in l0_overrides:
                assert_free = assert_free + l0_overrides[l_idx]
            r_h[l_idx] = assert_free
        return LatticeRuntimeState(
            kv_keys={l_idx: [] for l_idx in range(self.n_layers)},
            kv_vals={l_idx: [] for l_idx in range(self.n_layers)},
            r_horiz=r_h,
        )

    def step_token(
        self,
        x_embed: np.ndarray,
        state: LatticeRuntimeState,
        *,
        channel_mask: Optional[np.ndarray] = None,
        vertical_projector: Optional[BetaEditNullspaceProjector] = None,
        vertical_layer: int = 2,
        vertical_scales: Optional[dict[int, float]] = None,
        horiz_poison_delta: Optional[dict[int, np.ndarray]] = None,
        decode_p_r_override: Optional[np.ndarray] = None,
        use_decode_mod: bool = False,
    ) -> dict[str, Any]:
        """Advance one token through all 4 layers, mutating KV cache and horizontal latents."""
        h = x_embed.astype(np.float64).copy()
        captured_mlp_act = np.zeros(self.d_ff, dtype=np.float64)
        captured_r_hat: dict[int, np.ndarray] = {}
        written_kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for l_idx, lp in enumerate(self.layers):
            h_hat = _rmsnorm(h)
            q = lp.wq @ h_hat
            k = lp.wk @ h_hat
            v = lp.wv @ h_hat

            r_hat_masked: Optional[np.ndarray] = None
            if lp.is_lattice:
                # Exact arXiv:2611.21608 Sec. 3.2.2 affine horizontal update
                n_in = _rmsnorm(h)
                delta = _softplus(lp.w_delta @ n_in + lp.b_delta)  # type: ignore[operator]
                gate_g = np.exp(-np.exp(lp.a_log) * delta)  # type: ignore[arg-type]
                cand_c = _silu(lp.w_r @ n_in + lp.b_r)  # type: ignore[operator]
                r_next = gate_g * state.r_horiz[l_idx] + (1.0 - gate_g) * cand_c
                if horiz_poison_delta is not None and l_idx in horiz_poison_delta:
                    r_next = r_next + horiz_poison_delta[l_idx]
                state.r_horiz[l_idx] = r_next
                r_eff = r_next if channel_mask is None else (r_next * channel_mask)
                # Normalize active subspace without inflating zero-masked banks
                r_hat_masked = _rmsnorm(r_eff) if float(np.dot(r_eff, r_eff)) > 0.0 else np.zeros_like(r_eff)
                if channel_mask is not None:
                    r_hat_masked = r_hat_masked * channel_mask
                captured_r_hat[l_idx] = r_hat_masked.copy()

                # Exact Sec. 3.2.1 FiLM conditioning on Q, K, V (directly conditioning KV writes!)
                gq = np.tanh(lp.w_gq @ r_hat_masked)  # type: ignore[operator]
                bq = lp.w_bq @ r_hat_masked  # type: ignore[operator]
                gk = np.tanh(lp.w_gk @ r_hat_masked)  # type: ignore[operator]
                bk = lp.w_bk @ r_hat_masked  # type: ignore[operator]
                gv = np.tanh(lp.w_gv @ r_hat_masked)  # type: ignore[operator]
                bv = lp.w_bv @ r_hat_masked  # type: ignore[operator]

                q = q * (1.0 + gq) + bq
                k = k * (1.0 + gk) + bk
                v = v * (1.0 + gv) + bv

            # Write FiLM-conditioned (k_t^l, v_t^l) to layer KV cache M_t^l
            state.kv_keys[l_idx].append(k.copy())
            state.kv_vals[l_idx].append(v.copy())
            written_kv[l_idx] = (k.copy(), v.copy())

            # Multi-head causal attention over M_t^l
            k_hist = np.stack(state.kv_keys[l_idx], axis=0).reshape(-1, self.n_heads, self.head_dim)
            v_hist = np.stack(state.kv_vals[l_idx], axis=0).reshape(-1, self.n_heads, self.head_dim)
            q_h = q.reshape(self.n_heads, self.head_dim)
            scores = np.einsum("hd,shd->hs", q_h, k_hist) / np.sqrt(float(self.head_dim))
            probs = _softmax(scores)
            ctx = np.einsum("hs,shd->hd", probs, v_hist).reshape(self.d_model)
            h_bar = h + (lp.wo @ ctx)

            # SwiGLU MLP with horizontal modulation (Sec. 3.2.1)
            m_in = _rmsnorm(h_bar)
            gate_pre = lp.w_gate @ m_in
            up_pre = lp.w_up @ m_in
            if lp.is_lattice and r_hat_masked is not None:
                gate_pre = gate_pre + (lp.w_bgate @ r_hat_masked)  # type: ignore[operator]
                up_pre = up_pre + (lp.w_bup @ r_hat_masked)  # type: ignore[operator]
            mlp_mid = _silu(gate_pre) * up_pre

            if l_idx == vertical_layer:
                captured_mlp_act = mlp_mid.copy()
                if vertical_projector is not None:
                    mlp_out = vertical_projector.forward(
                        mlp_mid[np.newaxis, :], slot_scales=vertical_scales
                    )[0]
                else:
                    mlp_out = lp.w_down @ mlp_mid
            else:
                mlp_out = lp.w_down @ mlp_mid
            h = h_bar + mlp_out

        # Decoding (App. A.2.1 optional horizontal modulation)
        readout = _rmsnorm(h)
        if use_decode_mod and self.top_lattice in captured_r_hat:
            pr_mat = self.p_r if decode_p_r_override is None else decode_p_r_override
            readout = readout + (pr_mat @ captured_r_hat[self.top_lattice])
        logits = self.w_lm @ readout
        return {
            "logits": logits,
            "mlp_act": captured_mlp_act,
            "r_hat": captured_r_hat,
            "written_kv": written_kv,
        }


def _bank_mask(active_banks: tuple[int, ...], d_r: int = 48, width: int = 12) -> np.ndarray:
    mask = np.zeros(d_r, dtype=np.float64)
    for b in active_banks:
        mask[b * width : (b + 1) * width] = 1.0
    return mask


def _compile_horizontal_cartridge_l0(
    target_dir: np.ndarray,
    w_lm: np.ndarray,
    p_r_bank: np.ndarray,
    *,
    ridge: float = 0.05,
) -> tuple[np.ndarray, str]:
    """Deterministically compile a Plane-1 horizontal L_0 cartridge in CPU float64.

    Solves min_u ||P_r_bank u - target_dir||_2^2 + ridge * ||u||_2^2 in exact CPU float64
    so identical Plane-1 targets yield a 0-byte SHA-256 identical cartridge vector.
    """
    dim = p_r_bank.shape[1]
    lhs = (p_r_bank.T @ p_r_bank) + ridge * np.eye(dim, dtype=np.float64)
    rhs = p_r_bank.T @ target_dir
    u = np.linalg.solve(lhs, rhs)
    # Normalize and scale to controlled cartridge norm
    norm_u = float(np.linalg.norm(u))
    if norm_u > 0.0:
        u = (u / norm_u) * 1.0
    digest = hashlib.sha256(np.ascontiguousarray(u, dtype="<f8").tobytes()).hexdigest()
    return u, digest


def run_experiment() -> dict[str, Any]:
    model = LatentLatticeTransformer(seed=20261119)
    rng = np.random.default_rng(20261120)
    d_model = model.d_model
    d_ff = model.d_ff

    # Shared carrier + domain-specific inputs (75% shared carrier energy, matching E1 stress)
    shared_basis, _ = np.linalg.qr(rng.standard_normal((d_model, 16)))
    domain_bases = {}
    for dname in ("k0", "prose", "ui", "formal"):
        q_mat, _ = np.linalg.qr(rng.standard_normal((d_model, 12)))
        domain_bases[dname] = q_mat

    def _make_sequences(dname: str, n_seq: int, seq_len: int = 3) -> np.ndarray:
        out = []
        for _ in range(n_seq):
            seq = []
            for _t in range(seq_len):
                c = shared_basis @ (rng.standard_normal(16) * 0.75)
                s = domain_bases[dname] @ (rng.standard_normal(12) * 0.45)
                seq.append(c + s)
            out.append(np.stack(seq, axis=0))
        return np.stack(out, axis=0)  # (n_seq, seq_len, d_model)

    def _make_mixed_sequences(n_seq: int, seq_len: int = 3) -> np.ndarray:
        """Mixed prompts: prose carrier + formal final token (zone switch at last step)."""
        out = []
        for _ in range(n_seq):
            t0 = shared_basis @ (rng.standard_normal(16) * 0.75) + domain_bases["prose"] @ (rng.standard_normal(12) * 0.45)
            t1 = shared_basis @ (rng.standard_normal(16) * 0.75) + domain_bases["prose"] @ (rng.standard_normal(12) * 0.45)
            t2 = shared_basis @ (rng.standard_normal(16) * 0.75) + domain_bases["formal"] @ (rng.standard_normal(12) * 0.45)
            out.append(np.stack([t0, t1, t2], axis=0))
        return np.stack(out, axis=0)

    # Disjoint train and unseen test splits
    k0_train = _make_sequences("k0", 64)
    k0_test = _make_sequences("k0", 24)
    dom_train = {d: _make_sequences(d, 20) for d in ("prose", "ui", "formal")}
    dom_test = {d: _make_sequences(d, 16) for d in ("prose", "ui", "formal")}
    mixed_test = _make_mixed_sequences(16)

    # Opposing prose vs formal target token directions (shared token interference!)
    targets = {
        "prose": (0, 1),   # boost tok 0, suppress tok 1
        "ui": (2, 3),      # boost tok 2, suppress tok 3
        "formal": (1, 0),  # boost tok 1, suppress tok 0 (direct conflict with prose!)
    }
    target_dirs = {}
    for dname, (pos_t, neg_t) in targets.items():
        v = model.w_lm[pos_t] - model.w_lm[neg_t]
        target_dirs[dname] = (v / (np.linalg.norm(v) + 1e-12)) * 1.4

    # Harvest base Layer-2 MLP inputs and base logits
    def _run_seq_last(
        seq: np.ndarray,
        *,
        l0_overrides: Optional[dict[int, np.ndarray]] = None,
        per_step_masks: Optional[list[np.ndarray]] = None,
        vertical_projector: Optional[BetaEditNullspaceProjector] = None,
        vertical_scales: Optional[dict[int, float]] = None,
        decode_p_r_override: Optional[np.ndarray] = None,
        use_decode_mod: bool = False,
    ) -> dict[str, Any]:
        st = model.init_state(l0_overrides=l0_overrides)
        out: dict[str, Any] = {}
        for t_i in range(seq.shape[0]):
            cm = None if per_step_masks is None else per_step_masks[t_i]
            out = model.step_token(
                seq[t_i],
                st,
                channel_mask=cm,
                vertical_projector=vertical_projector,
                vertical_layer=2,
                vertical_scales=vertical_scales,
                decode_p_r_override=decode_p_r_override,
                use_decode_mod=use_decode_mod,
            )
        return out

    base_mask = _bank_mask((0,))
    k0_train_acts = np.stack(
        [_run_seq_last(s, per_step_masks=[base_mask] * 3)["mlp_act"] for s in k0_train], axis=1
    )  # (128, 64)
    k0_test_base = [
        _run_seq_last(s, per_step_masks=[base_mask] * 3)["logits"] for s in k0_test
    ]
    dom_train_acts = {
        d: np.stack([_run_seq_last(s, per_step_masks=[base_mask] * 3)["mlp_act"] for s in dom_train[d]], axis=1)
        for d in ("prose", "ui", "formal")
    }
    dom_test_base = {
        d: [_run_seq_last(s, per_step_masks=[base_mask] * 3)["logits"] for s in dom_test[d]]
        for d in ("prose", "ui", "formal")
    }
    mixed_test_base = [
        _run_seq_last(s, per_step_masks=[base_mask] * 3)["logits"] for s in mixed_test
    ]

    # ------------------------------------------------------------------
    # Part A: Vertical BetaEdit vs Horizontal Lattice Cartridges vs Coupled
    # ------------------------------------------------------------------
    # 1. Vertical BetaEdit slabs compiled sequentially on Layer-2 W_down
    v_cfg = BetaEditConfig(
        d_in=d_ff,
        d_out=d_model,
        max_rank=8,
        max_slots=8,
        lambda_leakage=25.0,
        lambda_ridge=1.0,
        nullspace_threshold=1e9,  # soft covariance (E1 finding on full-rank SwiGLU)
        threshold_mode="absolute",
        quant_dtype="f16",
    )
    v_proj = BetaEditNullspaceProjector(model.layers[2].w_down, v_cfg)
    v_proj.initialize_anchor_covariance(k0_train_acts / np.sqrt(float(k0_train_acts.shape[1])))
    v_slots: dict[str, int] = {}
    for dname in ("prose", "ui", "formal"):
        other_keys = np.concatenate(
            [dom_train_acts[o] for o in ("prose", "ui", "formal") if o != dname], axis=1
        )
        res_mat = np.tile(target_dirs[dname][:, np.newaxis], (1, dom_train_acts[dname].shape[1]))
        patch_v = v_proj.stage_edit(
            dom_train_acts[dname],
            res_mat,
            is_residual=True,
            rank=4,
            preservation_keys=other_keys,
            preservation_weight=1.0,
            slot_name=dname,
        )
        v_slots[dname] = v_proj.commit_patch(patch_v, dom_train_acts[dname])

    # 2. Horizontal Lattice L_0 cartridges compiled deterministically in banks 1, 2, 3
    # Keep Bank 0 decoding head zero-initialized (App. A.2.1) so K0 Bank 0 equals Base W0 exactly.
    p_r_compiled = np.zeros_like(model.p_r)
    l0_top_vec = np.zeros(model.d_r, dtype=np.float64)
    cartridge_digests: dict[str, str] = {}
    bank_map = {"prose": 1, "ui": 2, "formal": 3}

    for dname, b_idx in bank_map.items():
        sl = slice(b_idx * 12, (b_idx + 1) * 12)
        # Orthonormal bank readout frame derived deterministically from seed
        bank_rng = np.random.default_rng(1000 + b_idx)
        q_bank, _ = np.linalg.qr(bank_rng.standard_normal((d_model, 12)))
        # Ensure bank span contains target direction cleanly while remaining bounded
        q_bank[:, 0] = target_dirs[dname] / (np.linalg.norm(target_dirs[dname]) + 1e-12)
        q_bank, _ = np.linalg.qr(q_bank)
        p_r_compiled[:, sl] = q_bank * 0.65

        u_l0, dig_l0 = _compile_horizontal_cartridge_l0(target_dirs[dname], model.w_lm, p_r_compiled[:, sl])
        l0_top_vec[sl] = u_l0
        cartridge_digests[dname] = dig_l0

    # Verify 0-byte re-derivation of compiled L_0 cartridge from Plane 1 target alone
    _, dig_prose_rb = _compile_horizontal_cartridge_l0(
        target_dirs["prose"], model.w_lm, p_r_compiled[:, 12:24]
    )
    l0_zero_byte_rederive = bool(dig_prose_rb == cartridge_digests["prose"])

    l0_overrides = {model.top_lattice: l0_top_vec}

    def _eval_arm(
        *,
        use_vertical: bool,
        use_horizontal: bool,
        zone_routed_horizontal: bool,
        zone_routed_vertical: bool = False,
    ) -> dict[str, Any]:
        all_banks = _bank_mask((0, 1, 2, 3))
        v_all = {0: 1.0, 1: 1.0, 2: 1.0}

        # K0 evaluation
        k0_mask = base_mask if zone_routed_horizontal else (all_banks if use_horizontal else base_mask)
        k0_v_scales = {0: 0.0, 1: 0.0, 2: 0.0} if zone_routed_vertical else v_all
        k0_ret = 0
        k0_drifts = []
        for s, blg in zip(k0_test, k0_test_base):
            out = _run_seq_last(
                s,
                l0_overrides=l0_overrides if use_horizontal else None,
                per_step_masks=[k0_mask] * 3,
                vertical_projector=v_proj if use_vertical else None,
                vertical_scales=k0_v_scales if use_vertical else None,
                decode_p_r_override=p_r_compiled,
                use_decode_mod=use_horizontal,
            )
            if int(np.argmax(out["logits"])) == int(np.argmax(blg)):
                k0_ret += 1
            k0_drifts.append(float(np.max(np.abs(out["logits"] - blg))))

        # Pure domain margins & pass rates (margin gain > 0.10)
        dom_gains: dict[str, float] = {}
        dom_pass: dict[str, float] = {}
        for dname in ("prose", "ui", "formal"):
            pos_t, neg_t = targets[dname]
            d_mask = (
                _bank_mask((0, bank_map[dname]))
                if zone_routed_horizontal
                else (all_banks if use_horizontal else base_mask)
            )
            d_v_scales = (
                {v_slots[o]: (1.0 if o == dname else 0.0) for o in ("prose", "ui", "formal")}
                if zone_routed_vertical
                else v_all
            )
            gains = []
            for s, blg in zip(dom_test[dname], dom_test_base[dname]):
                out = _run_seq_last(
                    s,
                    l0_overrides=l0_overrides if use_horizontal else None,
                    per_step_masks=[d_mask] * 3,
                    vertical_projector=v_proj if use_vertical else None,
                    vertical_scales=d_v_scales if use_vertical else None,
                    decode_p_r_override=p_r_compiled,
                    use_decode_mod=use_horizontal,
                )
                bm = float(blg[pos_t] - blg[neg_t])
                nm = float(out["logits"][pos_t] - out["logits"][neg_t])
                gains.append(nm - bm)
            dom_gains[dname] = round(float(np.mean(gains)), 4)
            dom_pass[dname] = round(float(np.mean([g > 0.10 for g in gains])), 4)

        # Mixed prompt evaluation: steps 0,1 = prose zone, step 2 = formal zone
        pos_f, neg_f = targets["formal"]
        mix_masks = (
            [_bank_mask((0, 1)), _bank_mask((0, 1)), _bank_mask((0, 3))]
            if zone_routed_horizontal
            else ([all_banks] * 3 if use_horizontal else [base_mask] * 3)
        )
        mix_v_scales = (
            {v_slots["prose"]: 0.0, v_slots["ui"]: 0.0, v_slots["formal"]: 1.0}
            if zone_routed_vertical
            else v_all
        )
        mix_gains = []
        for s, blg in zip(mixed_test, mixed_test_base):
            out = _run_seq_last(
                s,
                l0_overrides=l0_overrides if use_horizontal else None,
                per_step_masks=mix_masks,
                vertical_projector=v_proj if use_vertical else None,
                vertical_scales=mix_v_scales if use_vertical else None,
                decode_p_r_override=p_r_compiled,
                use_decode_mod=use_horizontal,
            )
            bm = float(blg[pos_f] - blg[neg_f])
            nm = float(out["logits"][pos_f] - out["logits"][neg_f])
            mix_gains.append(nm - bm)

        return {
            "k0_top1_retained": k0_ret,
            "k0_total": len(k0_test),
            "k0_mean_max_logit_drift": round(float(np.mean(k0_drifts)), 6),
            "domain_margin_gains": dom_gains,
            "domain_pass_rates": dom_pass,
            "mixed_formal_margin_gain": round(float(np.mean(mix_gains)), 4),
            "mixed_formal_pass_rate": round(float(np.mean([g > 0.10 for g in mix_gains])), 4),
        }

    part_a_arms = {
        "arm1_vertical_betaedit_simultaneous": _eval_arm(
            use_vertical=True, use_horizontal=False, zone_routed_horizontal=False
        ),
        "arm1b_vertical_betaedit_zone_routed": _eval_arm(
            use_vertical=True,
            use_horizontal=False,
            zone_routed_horizontal=False,
            zone_routed_vertical=True,
        ),
        "arm2_horizontal_cartridges_unrouted": _eval_arm(
            use_vertical=False, use_horizontal=True, zone_routed_horizontal=False
        ),
        "arm3_horizontal_cartridges_zone_routed": _eval_arm(
            use_vertical=False, use_horizontal=True, zone_routed_horizontal=True
        ),
        "arm4_coupled_vertical_plus_horizontal_routed": _eval_arm(
            use_vertical=True,
            use_horizontal=True,
            zone_routed_horizontal=True,
            zone_routed_vertical=True,
        ),
    }

    # ------------------------------------------------------------------
    # Part B: KV-Write Contamination & Atomic Multi-Bank Rollback Proof
    # ------------------------------------------------------------------
    # 6-token rollout: attack injects poisoned vertical adapter + poisoned horizontal update
    # during steps t=0,1,2; revoked before t=3; evaluate contamination at t=3,4,5.
    rollout_seq = _make_sequences("k0", 1, seq_len=6)[0]  # (6, d_model)
    poison_delta = {
        1: np.concatenate([rng.standard_normal(12) * 1.2, np.zeros(36)]),
        2: np.concatenate([rng.standard_normal(12) * 1.2, np.zeros(36)]),
    }

    # Clean reference rollout (t=0..5)
    st_clean = model.init_state()
    clean_steps = []
    pre_attack_snapshot = st_clean.clone()
    clean_t2_snapshot = None
    for t_i in range(6):
        out_c = model.step_token(rollout_seq[t_i], st_clean, channel_mask=base_mask)
        clean_steps.append(out_c)
        if t_i == 2:
            clean_t2_snapshot = st_clean.clone()

    # Poisoned prefix (t=0,1,2)
    st_poison = model.init_state()
    for t_i in range(3):
        model.step_token(
            rollout_seq[t_i],
            st_poison,
            channel_mask=base_mask,
            vertical_projector=v_proj,
            vertical_scales={0: 1.0, 1: 0.0, 2: 0.0},
            horiz_poison_delta=poison_delta,
        )

    def _finish_rollout_and_diff(st_after_rb: LatticeRuntimeState) -> dict[str, Any]:
        logit_drifts = []
        kv_write_drifts = []
        horiz_drifts = []
        for t_i in range(3, 6):
            out_r = model.step_token(
                rollout_seq[t_i],
                st_after_rb,
                channel_mask=base_mask,
                vertical_projector=v_proj,
                vertical_scales={0: 0.0, 1: 0.0, 2: 0.0},  # adapter revoked!
            )
            logit_drifts.append(float(np.max(np.abs(out_r["logits"] - clean_steps[t_i]["logits"]))))
            # Measure KV-write drift on NEW keys & values written at step t_i AFTER revocation!
            k_new, v_new = out_r["written_kv"][2]
            k_ref, v_ref = clean_steps[t_i]["written_kv"][2]
            kv_write_drifts.append(
                float(np.linalg.norm(k_new - k_ref) + np.linalg.norm(v_new - v_ref))
            )
            horiz_drifts.append(
                float(np.linalg.norm(out_r["r_hat"][2] - clean_steps[t_i]["r_hat"][2]))
            )
        return {
            "max_abs_logit_drift_t3_t5": round(max(logit_drifts), 6),
            "max_new_kv_write_drift_t3_t5": round(max(kv_write_drifts), 6),
            "max_horiz_latent_drift_t3_t5": round(max(horiz_drifts), 6),
        }

    # Strategy 1: Adapter scale zeroed ONLY (KV cache + horizontal state untouched)
    rb_scale_only = _finish_rollout_and_diff(st_poison.clone())

    # Strategy 2: KV cache flushed/restored to clean prefix ONLY, horizontal lattice left poisoned!
    st_kv_only = st_poison.clone()
    st_kv_only.kv_keys = {k: [v.copy() for v in vs] for k, vs in clean_t2_snapshot.kv_keys.items()}  # type: ignore[union-attr]
    st_kv_only.kv_vals = {k: [v.copy() for v in vs] for k, vs in clean_t2_snapshot.kv_vals.items()}  # type: ignore[union-attr]
    rb_kv_flush_only = _finish_rollout_and_diff(st_kv_only)

    # Strategy 3: Atomic KV + horizontal-lattice bank restore (from clean pre-attack snapshot + clean prefix)
    st_atomic = pre_attack_snapshot.clone()
    for t_i in range(3):
        model.step_token(rollout_seq[t_i], st_atomic, channel_mask=base_mask)
    rb_atomic = _finish_rollout_and_diff(st_atomic)

    # ------------------------------------------------------------------
    # Part C: Hard Veto 1 & Asymmetric Quantization on Lattice Conditioning
    # ------------------------------------------------------------------
    # Compile a wide-output horizontal-to-MLP conditioning map Delta_cond = B_c A_c
    # (shape d_ff=128 x d_r=48) orthogonal to the base K0 horizontal subspace R_0_base!
    r0_base_samples = np.stack(
        [_run_seq_last(s, per_step_masks=[base_mask] * 3)["r_hat"][2] for s in k0_train[:16]], axis=1
    )  # (48, 16) — rank 12 inside Bank 0 [0:12]
    r_dom_keys = rng.standard_normal((model.d_r, 4))
    # Remove any accidental bank-0 component from domain triggers to isolate quantization leakage
    r_dom_keys[:12, :] = 0.0
    mlp_cond_targets = rng.standard_normal((d_ff, 4)) * 0.5
    w_cond_base = model.layers[2].w_bgate.copy()  # type: ignore[union-attr]

    asym_regimes = [
        ("f32/f32", "f32", "f32"),
        ("f16/f16", "f16", "f16"),
        ("bf16/bf16", "bf16", "bf16"),
        ("int8/int8", "int8", "int8"),
        ("int4/f16", "int4", "f16"),
        ("int4/int8", "int4", "int8"),
        ("int8/int4", "int8", "int4"),
        ("int4/int4", "int4", "int4"),
    ]
    part_c_regimes: dict[str, Any] = {}
    for lbl, b_dt, a_dt in asym_regimes:
        cfg_c = BetaEditConfig(
            d_in=model.d_r,
            d_out=d_ff,
            max_rank=4,
            lambda_leakage=1000.0,
            lambda_ridge=1.0,
            nullspace_threshold=1e-6,
            threshold_mode="absolute",
            quant_dtype="f16",
            b_quant_dtype=b_dt,  # type: ignore[arg-type]
            a_quant_dtype=a_dt,  # type: ignore[arg-type]
        )
        proj_c = BetaEditNullspaceProjector(w_cond_base, cfg_c)
        proj_c.initialize_anchor_covariance(r0_base_samples)
        patch_c = proj_c.stage_edit(
            r_dom_keys, mlp_cond_targets, is_residual=True, rank=4, slot_name="lattice_cond"
        )
        rederive_c_ok, _ = verify_rederivation(
            patch_c,
            base_weight=w_cond_base,
            config=cfg_c,
            k0=r0_base_samples,
            keys=r_dom_keys,
            target_values_or_residuals=mlp_cond_targets,
            is_residual=True,
            rank=4,
        )
        cert_ok, cert_probs = patch_c.certificate.verify_division_free(
            GatePolicy(max_frob_sq_num=50, max_frob_sq_den=1)
        )
        proj_c.commit_patch(patch_c, r_dom_keys)
        leak_r0 = proj_c.knowledge_leakage_norm(r0_base_samples)
        part_c_regimes[lbl] = {
            "payload_bytes": len(patch_c.b_bytes) + len(patch_c.a_bytes),
            "r0_base_leakage_fro": round(leak_r0, 6),
            "cert_leakage_energy_scaled": int(patch_c.certificate.leakage_energy_scaled),
            "cert_ok": bool(cert_ok and not cert_probs),
            "zero_byte_rederivation": bool(rederive_c_ok),
        }

    # Note: because r_dom_keys[:12] == 0, A_c has exact zeros in bank 0! So ALSO test the realistic
    # NON-disjoint case where domain horizontal activations overlap all 48 dims (shared carrier in r_t^l)
    # so A_c K0 = 0 is enforced purely by BetaEdit null-space projection (not trivial block zeros)!
    r0_full = rng.standard_normal((model.d_r, 12))
    r_dom_full = rng.standard_normal((model.d_r, 4))
    part_c_non_disjoint: dict[str, Any] = {}
    for lbl, b_dt, a_dt in asym_regimes:
        cfg_nd = BetaEditConfig(
            d_in=model.d_r,
            d_out=d_ff,
            max_rank=4,
            lambda_leakage=1000.0,
            lambda_ridge=1.0,
            nullspace_threshold=1e-6,
            threshold_mode="absolute",
            quant_dtype="f16",
            b_quant_dtype=b_dt,  # type: ignore[arg-type]
            a_quant_dtype=a_dt,  # type: ignore[arg-type]
        )
        proj_nd = BetaEditNullspaceProjector(w_cond_base, cfg_nd)
        proj_nd.initialize_anchor_covariance(r0_full)
        patch_nd = proj_nd.stage_edit(
            r_dom_full, mlp_cond_targets, is_residual=True, rank=4, slot_name="lattice_cond_nd"
        )
        proj_nd.commit_patch(patch_nd, r_dom_full)
        part_c_non_disjoint[lbl] = {
            "payload_bytes": len(patch_nd.b_bytes) + len(patch_nd.a_bytes),
            "r0_leakage_fro": round(proj_nd.knowledge_leakage_norm(r0_full), 6),
            "cert_leakage_energy_scaled": int(patch_nd.certificate.leakage_energy_scaled),
        }

    nd_44 = part_c_non_disjoint["int4/int4"]["r0_leakage_fro"]
    nd_48 = part_c_non_disjoint["int4/int8"]["r0_leakage_fro"]
    nd_84 = part_c_non_disjoint["int8/int4"]["r0_leakage_fro"]
    nd_88 = part_c_non_disjoint["int8/int8"]["r0_leakage_fro"]

    return {
        "experiment": "E7_latent_lattice_mde",
        "paper_ref": "arXiv:2611.21608v1",
        "architecture": {
            "n_layers": model.n_layers,
            "lattice_layers": list(model.lattice_layers),
            "d_model": d_model,
            "d_ff": d_ff,
            "d_r": model.d_r,
            "capability_banks": 4,
            "bank_width": 12,
        },
        "part_a_composition": {
            "l0_cartridge_zero_byte_rederivation": l0_zero_byte_rederive,
            "arms": part_a_arms,
        },
        "part_b_kv_write_contamination_and_rollback": {
            "strategy1_scale_zero_only": rb_scale_only,
            "strategy2_kv_flush_only_lattice_poisoned": rb_kv_flush_only,
            "strategy3_atomic_kv_plus_lattice_restore": rb_atomic,
        },
        "part_c_lattice_conditioning_asymmetric_quant": {
            "disjoint_bank_regimes": part_c_regimes,
            "shared_subspace_regimes": part_c_non_disjoint,
            "shared_subspace_headline": {
                "int4_int4_vs_int4_int8_ratio": round(nd_44 / max(nd_48, 1e-12), 2),
                "int8_int4_vs_int4_int8_ratio": round(nd_84 / max(nd_48, 1e-12), 2),
                "int4_int8_vs_int8_int8_ratio": round(nd_48 / max(nd_88, 1e-12), 3),
                "int4_int8_byte_savings_vs_int8_int8_pct": round(
                    100.0
                    * (
                        1.0
                        - part_c_non_disjoint["int4/int8"]["payload_bytes"]
                        / part_c_non_disjoint["int8/int8"]["payload_bytes"]
                    ),
                    1,
                ),
                "int4_int8_byte_savings_vs_f16_f16_pct": round(
                    100.0
                    * (
                        1.0
                        - part_c_non_disjoint["int4/int8"]["payload_bytes"]
                        / part_c_non_disjoint["f16/f16"]["payload_bytes"]
                    ),
                    1,
                ),
            },
        },
    }


def main() -> None:
    res = run_experiment()
    print(json.dumps(res, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
