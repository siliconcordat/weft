# Weft v0.1 Experiment Results

All runs below are labeled `verified` within their stated scope and executed locally on CPU (`Python 3.12.15`, `NumPy 2.5.5`, `pytest 9.2.0`, seed `20261119`, zero paid cloud compute). Negative findings and failure boundaries are recorded verbatim next to passing checks.

## Rerun commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/python experiments/e1_shared_token_composition.py
.venv/bin/python experiments/e2_anti_goodhart_gate.py
.venv/bin/python experiments/e3_real_checkpoint_smoke.py
.venv/bin/python experiments/e4_asymmetric_quant_packing.py
```

## E1: Shared-token null-space collapse vs register-scoped composition

Scope (`verified` on a deterministic 4-layer SwiGLU MLP stack, `d_model = 128`, `d_ff = 256`, `n_vocab = 48`, BF16 slabs, editing Layer 2 `W_down`): three packs (`corpus`, `weft-eido`, `formal`) share 75 percent carrier energy while `corpus` and `formal` push opposing token sets.

### Finding 1: Marchenko-Pastur `K0` calibration boundary (`d_ff = 256`)

| Calibration Setup | `||dW||_F` | `K0` Train Leakage | `K0` Unseen Holdout Leakage | `K0` Holdout Top-1 Retention |
| --- | --- | --- | --- | --- |
| `naive_n256` (`lambda_leakage = 0`) | 0.4380 | 4.2510 | 2.0231 | 0.9531 |
| `beta_undersampled_n256` (`n / d = 1.0`) | 1.7841 | 0.2598 | 8.6295 | 0.7656 |
| `beta_oversampled_n2048` (`n / d = 8.0`) | 0.0504 | 0.3731 | 0.1586 | 1.0000 |

Honest failure mode: when `n_k0` equals `d_ff` (`n / d = 1`), empirical `K0` covariance creates spurious tail null directions. BetaEdit cuts calibration leakage 16x (`4.251 -> 0.260`) while inflating `||dW||_F` 4x and making unseen `K0` holdout leakage 4.3x worse (`2.023 -> 8.630`). Oversampling `K0` (`n / d = 8`) flips the outcome, cutting unseen leakage 12.8x (`2.023 -> 0.159`) with 100 percent top-1 retention.

### Finding 2: Hard-threshold collapse vs soft covariance, register scoping, and L5 zone gating

On full-rank SwiGLU `K0` (`n_k0 = 2048`), hard eigenspace thresholds (`abs_2e-2`, `relmax_1e-2`, `relmax_1e-3`) all collapse usable null-space dimension `m(t)` to `[1, 1, 1, 1]`.

| Arm (on `n_k0 = 2048`) | Solo Pass (`C/E/F`) | Composed Pass (`C/E/F`) | Mixed Formal Margin | `corpus` Shift on `formal` | `K0` Top-1 Retention | Layer-2 Linear Leak | Layer-3 Nonlinear Drift | 0-Byte Re-derive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_naive_sum` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `+0.0329` (6.7%) | `-0.0918` | 0.8906 | 3.7962 | 6.5580 | `true` |
| `B1_hard_thresh_betaedit` | `0.0 / 0.0 / 0.0` | `0.0 / 0.0 / 0.0` | `-0.0000` (0.0%) | `+0.0019` | 1.0000 | 0.0429 | 0.0750 | `true` |
| `B2_soft_cov_betaedit` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `-0.3984` (0.0%) | `-0.1797` | 0.9688 | 1.0311 | 1.8675 | `true` |
| `C_soft_reg_scoped` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `-0.2835` (0.0%) | `+0.0438` | 0.9531 | 1.0085 | 1.7915 | `true` |
| `D_soft_reg_plus_zone` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `+0.5627` (10.0%) | `+0.0438` | 0.9531 | 1.0085 | 1.7915 | `true` |

Takeaways and limits:

- Hard eigenspace clipping (`B1`) destroys all target editing (`0.0%` pass) when SwiGLU lifts `K0` to full rank. Soft covariance (`B2`) recovers 100 percent solo and pure-domain compliance while cutting `K0` leakage 3.7x vs naive (`3.796 -> 1.031`).
- Register-scoped `y = 0` preservation (`C`) eliminates harmful prose bleed onto pure formal prompts (`-0.1797 -> +0.0438`), yet simultaneous summation still goes negative (`-0.2835`) on mixed prompts containing both prose and formal cues.
- Adding L5 zone-routed scale gating (`D`) flips mixed-zone formal margin positive (`-0.2835 -> +0.5627`). Even then, only 10 percent of mixed prompts clear the strict `+0.75` margin threshold because mixed carrier energy dilutes signal, proving dynamic routing helps without removing the underlying capacity bound.
- Downstream Layer-3 RMSNorm and SwiGLU amplify Layer-2 linear leakage by 1.73x to 1.81x across all arms (`1.0085 -> 1.7915`). Linear single-layer bounds never replace end-to-end behavioral verification.

## E2: Anti-Goodhart verification matrix

Scope (`verified` on `d_in = 32`, `d_out = 8`, rank-12 `K0`, 4 safety canaries, 4 visible target canaries, 40 unseen private holdouts, BF16 quantization, 10x flake stability verified with 0 flaky IDs):

| Candidate | `||dW||_F` (cap 10.61) | Cert Leakage Energy (cap 5.0) | Safety | Visible Target | Private Holdouts | Gate 1 Norm Only | Gate 2 Visible Only | Gate 3 Canary + Cert | Gate 4 Unified Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `honest_large` | 7.8318 | 0.4641 | 4/4 | 4/4 | 40/40 | PASS | PASS | PASS | `tier1_promote` |
| `honest_small` | 5.4098 | 0.1673 | 4/4 | 3/4 | 40/40 | PASS | FAIL | FAIL | `tier2_stepping_stone` |
| `stepping_stone_3of4` | 3.6676 | 0.0434 | 4/4 | 3/4 | 40/40 | PASS | FAIL | FAIL | `tier2_stepping_stone` |
| `rank4_sabotage` | 2.5465 | 26505.6422 | 0/4 | 0/4 | 0/40 | **PASS** | FAIL | FAIL | `reject` |
| `goodhart_overfit` | 6.3939 | 0.1451 | 4/4 | 4/4 | 0/40 | **PASS** | **PASS** | **PASS** | `reject` (`goodhart_holdout_failure`) |
| `forged_self_anchor` | 2.5465 | 26505.6422 | 0/4 | 0/4 | 0/40 | **PASS** | FAIL | FAIL | `reject` (`forbidden_candidate_supplied_anchors`) |

Takeaway: a low-norm sabotage patch (`2.55` vs cap `10.61`) fools norm-only gating. An in-nullspace Goodhart patch (`cert_leakage_energy = 0.145`, 4/4 safety, 4/4 visible target) fools Gates 1, 2, and 3 simultaneously. Only Gate 4 (adding private holdouts and banning candidate-supplied anchors) rejects every attack while separating Tier 1 promotion from Tier 2 stepping stones.

## E3: Real pretrained checkpoint end-to-end smoke test

Scope (`verified` on real pretrained `roneneldan/TinyStories-1M` Safetensors checkpoint loaded offline from local cache: 8-layer `GPTNeoForCausalLM`, 124 `F32` tensors, `hidden_size = 64`, `d_ff = 256`, `vocab_size = 50257`, editing Layer-7 `mlp.c_proj.weight` at rank 4 in F16):

| Check | Measured Output |
| --- | --- |
| Checkpoint source | `pretrained_safetensors:roneneldan/TinyStories-1M` |
| Base tensors fingerprinted | 124 (`fingerprint` prefix `2dfb3aba3be97ab7`) |
| Sanity forward pass (prompt 0) | `"Once upon a time there was a little girl named"` -> `" Lily"` |
| Real Layer-7 `K0` spectrum (`n = 16`, `d_in = 256`) | `max_eigval = 41.1757`, rank `>= 1e-4` rel = `16`, usable `m = 240` |
| `FixedRadixCertificate` & 5-layer binding | `certificate_ok = true`, `binding_clean_ok = true` |
| GGUF v3 F16 LoRA binary | 3,328 bytes (`sha256` prefix `a75cbea13463d40a`), 0-byte match on rebuild and round-trip |
| Compiler 0-byte re-derivation (Veto 1) | `true` (exact factor and certificate SHA-256 match) |
| Unseen validation target logit margin gain | Mean `+25.3306`, min `+10.2644` (period `.` vs semicolon `;`) |
| `K0` preservation (16 narrative prompts) | Linear leakage `0.000408`, top-1 token retention `16/16`, max abs logit drift `0.020682` |
| `O(1)` rollback verification (Veto 3) | Max abs logit diff vs base = `0.0` |
| Tamper rejection battery | `chat_template.jinja` 1-byte = rejected, Plane 1 `writing.md` 1-byte = rejected (`stale_lineage`), `enable_thinking` flag = rejected (`runtime_flag_drift`) |

Honest boundary on E3: `TinyStories-1M` is a 1M-parameter narrative model, not an instruction-following coding model. E3 verifies exact plumbing, real Safetensors activation harvesting, GGUF binary emission, 0-byte re-derivation, rollback, and tamper rejection on real pretrained weights. It does not prove complex SOP generalization on a 4B to 27B instruct checkpoint, which remains Milestone 3.

## E4: Asymmetric factor quantization and zero-data calibration probes

Scope (`verified` on real pretrained `roneneldan/TinyStories-1M`, editing Layer-7 `mlp.c_proj` of shape `64 x 256` and Layer-7 `mlp.c_fc` of shape `256 x 64`, `N = 3` sequential rank-4 adapters for `corpus`, `eido`, and `formal`, 16 real `K0` narrative prompts, 9 unseen validation prompts).

### Finding 1: Live validation of the Asymmetric Factor Leakage Theorem

For a null-space adapter `Delta = B A` with `A K0 = 0`, separate factor quantization yields `Q(B) Q(A) K0 = Q(B) (E_A K0)`. Rounding error `E_B` drops out to first order, while rounding error `E_A` drives all `K0` leakage. Per-rank-vector scales (`axis = 0` on `B`, `axis = 1` on `A`) keep integer scale overhead to `4 * r` bytes per factor.

Measured at `N = 3` sequential adapters on real Layer-7 `mlp.c_proj` (`64 x 256`, live 8-layer forward pass across 50,257 vocab tokens, 100 percent exact wire readback and 0-byte re-derivation across all 8 regimes):

| Regime (`B / A`) | `N = 3` Bytes | `K0` Linear Leak | `K0` Top-1 Retained | `K0` Max Logit Drift | Val Margin Gain | Cert Leakage Sum |
| --- | --- | --- | --- | --- | --- | --- |
| `f32 / f32` | 15,360 | 0.000000 | 16/16 | 0.000002 | +18.5257 | 3 |
| `f16 / f16` | 7,680 | 0.000597 | 16/16 | 0.022598 | +18.5256 | 5,448 |
| `int4 / f16` | 6,576 | 0.000605 | 16/16 | 0.023368 | +18.3238 | 5,606 |
| `bf16 / bf16` | 7,680 | 0.004471 | 15/16 | 0.178002 | +18.5234 | 318,769 |
| `int8 / int8` | 3,936 | 0.028787 | 15/16 | 1.328532 | +18.5267 | 13,596,890 |
| `int4 / int8` | 3,552 | 0.029345 | 15/16 | 1.353259 | +18.3219 | 14,115,373 |
| `int8 / int4` (control) | 2,400 | 0.533395 | 6/16 | 11.938732 | +18.6557 | 4,479,267,316 |
| `int4 / int4` | 2,016 | 0.539110 | 6/16 | 11.811741 | +18.4366 | 4,584,581,757 |

Key observations on `mlp.c_proj` (`N = 3`):

- `int4 / int8` cuts `K0` leakage by 18.37x vs uniform `int4 / int4` (`0.0293` vs `0.5391`) and by 18.18x vs the reversed control `int8 / int4` (`0.5334`), staying within 1.9 percent of `int8 / int8` (`0.0288`) and preserving 15/16 top-1 predictions vs 6/16.
- `int4 / f16` beats uniform `bf16 / bf16` on every metric: smaller payload (`6,576` vs `7,680` bytes), 7.4x lower linear leakage (`0.000605` vs `0.004471`), 7.6x lower max logit drift (`0.0234` vs `0.1780`), and 16/16 vs 15/16 top-1 retention.
- The post-quantization integer certificate detects the `int4 / int4` and `int8 / int4` failure before GPU load (`4.58e9` vs `1.41e7` scaled leakage energy, a 325x jump).

### Finding 2: Shape asymmetry on wide-output projections (`mlp.c_fc`, `256 x 64`)

On `mlp.c_proj`, `A` (`4 x 256`) is 4x larger than `B` (`64 x 4`), so shrinking `B` saves only 9.8 percent vs `int8 / int8`. On wide-output layers like `mlp.c_fc` (`256 x 64`, matching `W_up` and `W_gate` in larger transformers), `B` is 4x larger than `A`:

| Regime (`B / A`) on `mlp.c_fc` | `N = 3` Bytes | Savings vs `int8/int8` | Savings vs `f16/f16` | `K0` Linear Leak (`N = 3`) |
| --- | --- | --- | --- | --- |
| `f16 / f16` | 7,680 | -95.1% | 0.0% | 0.003876 |
| `bf16 / bf16` | 7,680 | -95.1% | 0.0% | 0.048433 |
| `int8 / int8` | 3,936 | 0.0% | 48.8% | 0.147815 |
| `int4 / f16` | 3,120 | 20.7% | 59.4% | 0.003916 |
| `int4 / int8` | 2,400 | 39.0% | 68.8% | 0.148809 |
| `int8 / int4` (control) | 3,552 | 9.8% | 53.8% | 2.946486 |
| `int4 / int4` | 2,016 | 48.8% | 73.8% | 2.981155 |

On `mlp.c_fc`, `int4 / int8` cuts payload bytes 39.0 percent vs `int8 / int8` and 68.8 percent vs `f16 / f16` while staying within 0.7 percent of `int8 / int8` leakage and beating both `int4 / int4` (20.03x) and `int8 / int4` (19.80x). `int4 / f16` cuts bytes 59.4 percent vs `bf16 / bf16` while cutting leakage 12.4x.

### Finding 3: Zero-data `K0` covariance and `ffn_down` loud-column probes

Two probes tested whether quantization findings from 1B-scale models transfer to `TinyStories-1M`:

| Probe Arm (Layer-7 `mlp.c_proj`) | Held-out Real `K0` Leak | Held-out Max Logit Drift | Val Target Margin Gain | Notes |
| --- | --- | --- | --- | --- |
| `naive_unregularized` | 2.382173 | 20.890732 | +21.8208 | No `K0` covariance |
| `noise_only_k0_soft` (64 random seqs) | 1.121438 | 16.956105 | +18.9183 | Zero real text read, 83.9% leakage recovery |
| `real_text_k0_soft` (12 real stories) | 0.878790 | 17.122625 | +20.7740 | Trace-normalized (`K K^T / n`, `lambda = 60`) |

- **Noise-only `K0` covariance works on `TinyStories-1M`:** Running 64 random vocabulary token sequences (zero real text) recovers 83.9 percent of real-text holdout `K0` leakage protection at matched trace, with a 9 percent target-margin tax (`+18.92` vs `+20.77`).
- **Honest scale boundary on `ffn_down` loud columns:** On `TinyStories-1M`, `mlp.c_proj` column L2 norms are nearly flat (max-to-mean ratio `1.21` to `1.41` across all 8 layers), unlike 1B-scale Llama models with massive outlier units. Leaving the top-10 weight-norm columns whole under INT4 base quantization lowers Layer-7 `K0` max logit drift only modestly (`4.1657` to `4.0039`, top-1 retention unchanged at `12/16`). Loud-column gains must be tested on 1.5B to 8B checkpoints in Milestone 3.
