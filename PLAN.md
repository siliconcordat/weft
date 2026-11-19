# Weft Implementation Plan

This plan defines the build milestones, verification tests, and falsification experiments for `siliconcordat/weft` v0.1. Every step runs locally with zero paid cloud compute and records exact commands, seeds, and negative results alongside passing ones.

## Milestone 1: Core deterministic compiler and binary packaging

Build the `weft` Python package (`pyproject.toml`, `src/weft/`, `tests/`) inside an isolated local `.venv` with zero global environment pollution.

1. **`src/weft/certificate.py` (`FixedRadixCertificate`):** Compute squared Frobenius norm, squared spectral norm, `K0` leakage energy, and squared safety-direction dot products on post-quantized BF16/F16 factors. Verify all policy thresholds using exact integer cross-multiplication with no floating-point division.
2. **`src/weft/compiler.py` (`BetaEditNullspaceProjector`):** Run history-aware covariance tracking `C^(t) = lambda_1 * K0 @ K0.T + sum_{i<t} K_i @ K_i.T` in deterministic CPU `float64` with exact `m x m` SPD Cholesky solves, skinny-QR rank-`r` Eckart-Young factorization, canonical IEEE-754 round-to-nearest-even BF16/F16 quantization, frozen-base SGMV additive multi-slot inference, `O(1)` rollback, and 0-byte re-derivation checks.
3. **`src/weft/binding.py` (Five-layer binding):** Hash tensor layout, full checkpoint tensor bytes, config and tokenizer files (including `chat_template.jinja`), Plane 1 rule digests with adapter class tag, and runtime serving flags. Fail closed on any one-byte mismatch, stale Plane 1 lineage, or missing Class 1 recipe.
4. **`src/weft/gguf_lora.py` (GGUF v3 F16 LoRA codec):** Pure-stdlib binary writer and reader emitting deterministic 32-byte aligned GGUF v3 LoRA adapter files (`adapter.type = "lora"`, `general.type = "adapter"`, `weft.binding_sha256`, `weft.certificate_sha256`, and F16 `<tensor>.lora_a` / `<tensor>.lora_b` tensors) ready for `llama.cpp`.
5. **`src/weft/anchors.py` (Anti-Goodhart verifier harness):** Enforce commit-before-probes, reject candidate-supplied expected answers on sight, require 10-run zero-variance flake checks, separate Tier 1 canonical promotion from Tier 2 stepping-stone retention (never relaxing safety canaries), and implement deterministic shuffled-order lexicase selection.

## Milestone 2: Runnable mathematical and real-checkpoint experiments

Every experiment lives under `experiments/`, uses fixed seeds, and writes verbatim outputs to `experiments/RESULTS.md` with clear claim labels (`verified` vs `proposed`).

### Experiment E1: Shared-token null-space collapse vs register-scoped composition

- **Goal:** Test simultaneous 3-pack composition (`corpus` prose, `weft-eido` UI modules, `formal-verification` proof tokens) on a deterministic 4-layer SwiGLU transformer block stack where all three packs share high-frequency carrier tokens.
- **Measurements:** Track effective null-space dimension `m(t)` as adapters stack, cross-layer nonlinear logit drift through RMSNorm and SwiGLU, and `T-C1` through `T-C3` across four arms: (a) naive linear sum, (b) sequential `BetaEdit` without register scoping, (c) register-scoped `y = 0` preservation keys, and (d) register-scoped preservation plus L5 zone-routed scale gating.
- **Kill condition:** Record exact points where shared-token rank collapse breaks pure linear projection so L4/L5 zone routing is proven necessary rather than assumed.

### Experiment E2: Anti-Goodhart verification matrix

- **Goal:** Red-team load-time verification using five candidate adapters: honest small null-space delta, honest large null-space delta, low-norm rank-4 targeted sabotage, a visible-canary-overfitted Goodhart attack (passes 100 percent of visible anchors while sabotaging holdouts), and a 3/4 stepping-stone candidate.
- **Measurements:** Compare four gates: norm-only, visible-canary-only, verifier-owned lexicase canaries plus `FixedRadixCertificate`, and the full unified gate (commit-before-probes, integer certificate, verifier-owned canaries, and private holdout). Record exact false positives and false negatives.

### Experiment E3: Real pretrained checkpoint end-to-end smoke test

- **Goal:** Run the complete pipeline (5-layer binding, activation harvesting, deterministic CPU `float64` BetaEdit compilation, `FixedRadixCertificate`, GGUF v3 LoRA binary export, 0-byte re-derivation, apply, rollback, and tamper rejection).
- **Target checkpoint:** Run on an actual pretrained open-weights transformer (`roneneldan/TinyStories-1M`, 8-layer `GPTNeoForCausalLM`, 46 MB Safetensors) when cached locally, or fall back to a real multi-layer GPT architecture with the boundary labeled.
- **Pass criteria:** Exact 0-byte re-derivation on both quantized factors and GGUF binary, 100 percent tamper rejection on template, Plane 1 lineage, and runtime-flag edits, verified `K0` preservation, and bit-exact `W0` restoration on rollback.

## Milestone 3: Next-stage hardware benchmarks (post-v0.1)

After Milestone 2 clears independent verification:

1. Run the 160-prompt `clean-writing` solo proof experiment (64 compile, 32 validation, 64 private holdout) on a 1.7B to 4B open checkpoint (`LFM3-1.7B` or `Qwen3.5-4B`) on the 8 GB RTX 3080 lab machine, including a constant logit-bias control and vendored `check-prose.py --dupes` oracle.
2. Only if solo compliance cuts violations by at least 50 percent with at most 1.0 percentage point `K0` regression, stack a second pack (`soko` or `weft-eido`) and run `T-C1` through `T-C5`.
3. Evaluate `Qwen3.8-27B` `Q8_0` with F16 GGUF LoRAs on the 64 GB Apple M1 Max hub via read-only `llama-server` to measure quantization retention and multi-LoRA latency.
