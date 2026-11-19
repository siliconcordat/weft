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
