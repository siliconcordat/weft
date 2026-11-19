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
.venv/bin/python experiments/e5_noise_k0_scaling_and_readout.py
# Opt-in live 1.5B llama-server test (supply any local Qwen2.5-1.5B GGUF path):
WEFT_LIVE_BASE_GGUF=/path/to/qwen2.5-1.5b-instruct.gguf .venv/bin/python experiments/e6_live_llama_server_lora.py
.venv/bin/python experiments/e7_latent_lattice_mde.py
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

## E5: Seeded-noise `K0` scaling, wire-dialect readout repair, and `weft-slab/v0.1`

Scope (`verified` on real pretrained `roneneldan/TinyStories-1M`, Layer-7 `mlp.c_proj` of shape `64 x 256`, 32 disjoint real narrative prompts split into 16 train and 16 strictly unseen holdout prompts, physically matched residual scale `0.40` / `0.35`, vectorized batch parity max diff `0.0`).

### Part A: Seeded-noise `K0` Marchenko-Pastur scaling curve (zero real calibration text)

Random vocabulary token sequences (`seed = 20261119`, `n_noise = 32` to `2048`, trace-normalized `K K^T / n`, `lambda_leakage = 60.0`, `int4 / f16` factors) vs 16 real story prompts and a hybrid arm adding 16 register-scoped `y = 0` anchors:

| Arm (evaluated on 16 unseen stories) | `n / d_in` | `K0` Eff. Rank | Holdout `K0` Leak | Top-1 Retained | Max Logit Drift | Val Target Margin | Recovery vs Real |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `naive_unregularized` | 0.0 | 0.00 | 0.879786 | 4/16 | 14.7901 | +15.2067 | 0.0% |
| `noise_n32` (0 real text) | 0.125 | 2.28 | 0.667700 | 3/16 | 14.0971 | +7.1733 | 35.8% |
| `noise_n64` (0 real text) | 0.250 | 2.30 | 0.571813 | 6/16 | 13.3046 | +6.0371 | 52.0% |
| `noise_n256` (0 real text) | 1.000 | 2.59 | 0.583776 | 5/16 | 12.6347 | +7.0158 | 50.0% |
| `noise_n512` (0 real text) | 2.000 | 2.51 | 0.556300 | 5/16 | 12.6633 | +6.9700 | 54.6% |
| `noise_n1024` (0 real text) | 4.000 | 2.44 | 0.603129 | 4/16 | 13.3051 | +7.0662 | 46.7% |
| `noise_n2048` (0 real text) | 8.000 | 2.42 | 0.564394 | 5/16 | 13.0303 | +6.9445 | 53.2% |
| `real_text_n16` | 0.063 | 4.04 | 0.287497 | 13/16 | 10.9856 | +9.8143 | 100.0% |
| `hybrid_noise2048_plus_16_y0` | 8.000 | 2.42 | 0.217666 | 10/16 | 7.7799 | +4.4852 | 111.8% |

Honest finding: increasing seeded random-token noise from `n = 64` to `n = 2048` does **not** close the remaining gap to real-text calibration. Uniform random vocab tokens saturate at participation-ratio effective rank `2.42` vs `4.04` for English stories, plateauing at 52 to 55 percent holdout leakage recovery (`0.564` vs `0.287`).

Seed-only 0-byte re-derivation passes (`true`), making seeded noise a reliable zero-corpus floor. Pairing it with 16 public `y = 0` anchors beats `real_text_n16` on unseen leakage (`0.218`) and max logit drift (`7.78` vs `10.99`) at a lower target margin (`+4.49` vs `+9.81`).

### Part B: Parametric wire-dialect readout adapter (`int4 / int8`)

Sixteen train and 16 unseen validation messages pair varying prefixes with 4 compact symbolic dialect tags (`@A` to `@D`) mapping to `[yes, no, stop, run]`. On the base model, a linear probe on Layer-7 hidden states scores `1.000` (`16/16`) while the unadapted readout scores `0.125` (`2/16`, mean margin `-2.3031`), reproducing the hidden-state vs readout gap on real pretrained weights:

| Readout Arm (`int4 / int8`, rank 4) | Unseen Dialect Acc | Mean Class Margin | Unseen `K0` Leak | Unseen `K0` Top-1 | Unseen `K0` Max Drift |
| --- | --- | --- | --- | --- | --- |
| `base_unadapted` (linear probe `1.000`) | 0.125 (2/16) | -2.3031 | 0.000000 | 16/16 | 0.0000 |
| `naive_int4_int8` (`lambda = 0`) | 1.000 (16/16) | +16.8420 | 0.838508 | 1/16 | 18.0652 |
| `seeded_noise_int4_int8` (0 real text) | 1.000 (16/16) | +9.9781 | 0.360385 | 8/16 | 13.2972 |
| `hybrid_noise_plus_16_y0_int4_int8` | 1.000 (16/16) | +9.4899 | 0.141029 | 13/16 | 5.1677 |

All adapted arms recover `16/16` unseen dialect accuracy with `0.0` rollback logit diff. Seeded-noise `K0` alone cuts unseen narrative leakage 2.33x and raises unseen story top-1 retention from `1/16` to `8/16`. Adding 16 public `y = 0` anchors cuts leakage 5.95x (`0.839` to `0.141`), raises retention to `13/16`, and cuts max drift 3.50x while keeping class margin at `+9.49`.

### Part C: `weft-slab/v0.1` binary container vs F16 GGUF LoRA

| Target Tensor (Asymmetric `int4 / int8`) | `weft-slab/v0.1` Bytes | F16 GGUF LoRA Bytes | Size Reduction | Verify / Drift Reject / Round-trip |
| --- | --- | --- | --- | --- |
| `TinyStories-1M` `mlp.c_proj` (`64 x 256`, `r = 4`) | 2,121 | 2,976 | 28.7% | `true` / `true` / `true` |
| 4B-scale `W_up` (`10240 x 2560`, `r = 16`) | 123,933 | 410,016 | 69.8% | `true` / `true` / `true` |

`weft-slab/v0.1` embeds the full `FixedRadixCertificate`, `base_weights_sha256`, L5 `zone_manifest_id` pin, duplicate-key-rejected canonical JSON header, and 32-byte SHA-256 trailer while still cutting 4B-scale wide-output adapter size 69.8 percent below standard F16 GGUF LoRA.

## E6: Live `llama-server` C++ multi-LoRA verification on `Qwen2.5-1.5B-Instruct`

Scope (`verified` on real locally cached `Qwen2.5-1.5B-Instruct` GGUF inside compiled `llama-server` `0.5.0` build `11146`, commit `7fe450e19`, Darwin arm64 CPU, zero paid cloud compute, env-gated via `WEFT_LIVE_BASE_GGUF`). Two orthogonal rank-4 F16 adapters plus an algebraic cancellation adapter `(-B0, A0)` compile onto final-layer non-square `blk.27.ffn_up.weight` (`8960 x 1536`) and run on one forward pass (`temperature = 0`, `seed = 42`, `cache_prompt = False`, 16 tokens).

| Live `llama-server` Check / State | Measured Output |
| --- | --- |
| Model geometry (`read_gguf_model_geometry`) | `qwen2`, `Qwen2.5 1.5B Instruct`, 28 blocks, `d_model = 1536`, `d_ff = 8960` |
| BetaEdit compile (2 rank-4 F16 adapters on `8960 x 1536`) | `559.42 ms`, `cert_0_ok = true`, `cert_1_ok = true`, compiler 0-byte re-derive = `true` |
| GGUF v3 F16 LoRA binaries | `84,512` bytes each (`3cd412e2cab91634` and `648e4b58c8f27b02`), 0-byte rebuild & round-trip = `true` |
| Server load timings | Pure Base `W0`: `1,730.73 ms`, 3-LoRA mounted: `1,699.78 ms`, initial scales `[0.0, 0.0, 0.0]` |
| Pure Base `W0` (separate server, zero `--lora` flags) | Token 0 `' '` (`id = 220`), `logprob = -0.304419` |
| Zero-init `[0.0, 0.0, 0.0]` | Token 0 `' '` (`logprob = -0.304419`), bit-exact content and `0.0` logprob diff vs Pure Base |
| Solo Adapter 0 `[1.0, 0.0, 0.0]` | Token 0 `' '` (`logprob = -0.310336`, shift `-0.005917`) |
| Solo Adapter 1 `[0.0, 1.0, 0.0]` | Token 0 `' '` (`logprob = -0.324380`, shift `-0.019961`) |
| Simultaneous 2-adapter `[1.0, 1.0, 0.0]` | Token 0 `' '` (`logprob = -0.341411`, shift `-0.036992`, distinct from both solos) |
| Linear cancellation `[1.0, 0.0, 1.0]` (`+B0 A0 + (-B0) A0`) | Token 0 `' '` (`logprob = -0.304419`), exact content match, max 16-token logprob diff `1.2e-7` |
| `O(1)` kill switch `[0.0, 0.0, 0.0]` + `POST /slots/0?action=erase` | **`0.99 ms`** (`n_erased = 29`), bit-exact content match, **`0.0`** max logprob diff vs Pure Base |

Two real `llama-server` `0.5.0` runtime findings (captured in `build_llama_server_lora_flags()`): passing repeated `--lora` flags with `--lora-init-without-apply` still initializes adapters at `scale = 1.0`, so fail-closed zero-init requires comma-separated `--lora-scaled <path>:0.0,...`. In addition, `POST /slots/0?action=erase` returns HTTP 501 unless `--slot-save-path` is configured.

Honest boundary on E6: E6 proves real `llama.cpp` C++ GGUF v3 multi-LoRA loading, non-square `ggml` factor layout, simultaneous additive composition, algebraic cancellation, and sub-millisecond bit-exact rollback on `Qwen2.5-1.5B-Instruct`. The two adapters in E6 use seeded synthetic residuals to isolate linear-algebra and runtime mechanics rather than measuring SOP writing compliance, which remains Milestone 3.

## E7: Latent Lattice Language Models (`arXiv:2611.21608`) x Weft MDE

Scope (`verified` on a deterministic CPU `float64` 4-layer Latent Lattice Transformer implementing Duggal et al., `arXiv:2611.21608v1` Sec. 3.2.1 to 3.2.2 and App. A.2.1: early layer `0`, Lattice layers `1` and `2`, late layer `3`, `d_model = 64`, `d_ff = 128`, `d_r = 48` split into 4 width-12 capability banks `[base_k0, prose, ui, formal]`, exact Mamba-style affine horizontal recurrence, FiLM Q/K/V modulation writing into KV cache, SwiGLU MLP modulation, and top-Lattice decoding modulation).

### Part A: Horizontal Lattice cartridges vs vertical `BetaEdit` vs coupled composition

Three shared-carrier domains (75 percent shared carrier energy, opposing `prose` vs `formal` target tokens, plus 24 unseen `K0` sequences and 16 mixed `prose -> formal` switch prompts). `L_0` cartridge 0-byte re-derivation passed (`true`):

| Composition Arm | `K0` Top-1 Retained | `K0` Mean Max Drift | Pure Pass (`P / U / F`) | Pure Margins (`P / U / F`) | Mixed Formal Switch |
| --- | --- | --- | --- | --- | --- |
| `arm1_vertical_betaedit_simultaneous` | 19/24 | 1.388620 | `56% / 50% / 56%` | `+0.35 / +0.01 / +1.12` | `+0.0619` (50.0%) |
| `arm1b_vertical_betaedit_zone_routed` | 24/24 | 0.000000 | `56% / 50% / 63%` | `+0.66 / -0.02 / +0.80` | `+0.6330` (62.5%) |
| `arm2_horizontal_cartridges_unrouted` | 5/24 | 4.525512 | `50% / 100% / 50%` | `+0.22 / +7.94 / -0.17` | `-0.5272` (18.8%) |
| `arm3_horizontal_cartridges_zone_routed` | 24/24 | 0.000000 | `100% / 100% / 100%` | `+9.90 / +10.67 / +8.62` | `+8.4463` (100.0%) |
| `arm4_coupled_vertical_plus_horizontal_routed` | 24/24 | 0.000000 | `100% / 100% / 100%` | `+10.33 / +10.32 / +9.36` | `+9.0875` (100.0%) |

Honest failure mode: unrouted persistent horizontal banks (`arm2`) collapse `K0` (`5/24`, drift `4.53`) and mixed-zone formal output (`-0.527`) because opposing horizontal states collide across every token. Adding L5 zone-routed channel masking (`arm3`) restores exact `0.0` `K0` drift (`24/24`) and 100 percent pass across all domains, while coupling vertical `BetaEdit` + routed horizontal cartridges (`arm4`) lifts mixed-zone formal margin further (`+8.45` to `+9.09`) with zero `K0` drift.

### Part B: KV-write contamination and atomic multi-bank rollback (`t = 3..5` after revocation)

An adversarial adapter plus horizontal latent injection runs at steps `t = 0..2`, then revokes at `t = 3`. All metrics measure steps `t = 3..5` against the clean Base `W0` trajectory:

| Rollback Strategy after Revocation at `t = 3` | Max Abs Logit Drift (`t = 3..5`) | Max NEW KV-Write Drift (`t = 3..5`) | Max Horizontal Latent Drift (`t = 3..5`) |
| --- | --- | --- | --- |
| `strategy1_scale_zero_only` (adapter scale = 0 only) | 6.165156 | 31.812266 | 11.149887 |
| `strategy2_kv_flush_only_lattice_poisoned` (KV flushed, lattice kept) | 8.129843 | 30.260386 | 11.106646 |
| `strategy3_atomic_kv_plus_lattice_restore` (KV + lattice atomic) | 0.000000 | 0.000000 | 0.000000 |

Takeaway: flushing only the KV cache (`strategy2`) fails completely on Latent Lattice architectures. The surviving poisoned horizontal latent `r_2^l` immediately re-poisons newly written `(k_t^l, v_t^l)` pairs (`30.26` norm drift) and downstream logits (`8.13` max drift). Exact `0.0` restoration requires atomic multi-bank restore of both `kv_cache` and `lattice_horizontal_streams`.

### Part C: Hard Veto 1 and asymmetric `(int4 / int8)` quantization on Lattice conditioning (`128 x 48`)

Compiling the wide-output horizontal-to-MLP conditioning projection `W_bgate` (`d_ff = 128`, `d_r = 48`, rank 4) passes `cert_ok = true` and 0-byte re-derivation across all 8 precision regimes. Under disjoint capability banks, symmetric zero-centered quantization keeps `0.0` leakage across every precision. Under shared 48-dim horizontal carriers (`A_c R_0 = 0` via BetaEdit):

| Shared-Subspace Conditioning Regime (`B / A`) | Bytes | `R_0` Base Leakage | Cert Leakage Scaled | Savings vs `f16/f16` |
| --- | --- | --- | --- | --- |
| `f16 / f16` | 1,408 | 0.001334 | 29,850 | 0.0% |
| `bf16 / bf16` | 1,408 | 0.011524 | 2,228,082 | 0.0% |
| `int8 / int8` | 736 | 0.040783 | 27,904,899 | 47.7% |
| `int4 / f16` | 656 | 0.001338 | 30,041 | 53.4% |
| `int4 / int8` | 480 | 0.040734 | 27,837,481 | 65.9% |
| `int8 / int4` (control) | 640 | 0.584591 | 5,733,549,567 | 54.5% |
| `int4 / int4` | 384 | 0.587362 | 5,788,034,681 | 72.7% |

`int4 / int8` cuts shared-subspace horizontal-to-vertical leakage 14.42x vs `int4 / int4` and 14.35x vs reversed control `int8 / int4` (matching `int8 / int8` at ratio `0.999`) while cutting bytes 34.8 percent vs `int8 / int8` and 65.9 percent vs `f16 / f16`. `int4 / f16` cuts bytes 53.4 percent vs `bf16 / bf16` with 8.6x lower leakage.

Honest boundary on E7: E7 uses a synthetic 4-layer CPU `float64` Latent Lattice Transformer to isolate exact recurrence, KV-write coupling, rollback, and quantization physics. Pretrained 406M+ LLLM checkpoints are not yet publicly released, so language-modeling perplexity evaluation on trained Lattice weights awaits open checkpoints.
