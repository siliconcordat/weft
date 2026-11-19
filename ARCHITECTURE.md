# Weft Architecture

Weft implements Layer 2 (model and parametric software) of the Siliconcordat stack. Every design choice keeps human authority in readable text (Plane 1) while compiling repeatable behavioral defaults into verifiable, revocable weight adapters (Plane 2) on self-hosted open-weights hardware.

## Two planes and five hard vetoes

Plane 1 holds committed plain-text rules, revision history, and human approvals. Plane 2 holds compiled weight adapters derived from Plane 1. A Plane 2 adapter is always a disposable cache bound to exact Plane 1 and base-checkpoint digests.

Five hard vetoes bound every component in Weft:

1. **No opaque deltas.** Every Plane 2 adapter must rebuild byte-for-byte from published Plane 1 inputs using a public, deterministic compiler.
2. **No silent conflict resolution.** Contested human preferences or overlapping style rules halt compilation and escalate to the owner.
3. **No closed-weight injection.** Plane 2 runs only where the host runtime enforces load checks, logs active digests, and supports instant rollback to frozen base weights.
4. **No premature protocol standard.** Weight-slot metadata stays inside vendor-namespaced `_meta` fields until external users test it.
5. **Plane 1 holds all human authority.** Spend limits, credentials, external posting approvals, and merge rules never move into weights.

## Two adapter classes (Veto 1 reconciliation)

Deterministic closed-form compilation and stochastic reinforcement learning solve different problems. Mixing them hides provenance failure. Weft enforces a hard boundary between two classes:

| Property | Class 1: Compiled Plane 2 Adapter | Class 2: Open-Recipe Base Checkpoint (`W_base'`) |
| --- | --- | --- |
| Build method | Deterministic CPU `float64` closed-form solve (BetaEdit null-space projection, ridge readout, or steering) | Stochastic open-recipe training (for example, SFT or online RL LoRA such as AsyncGRPO harness adapters) |
| Re-derivability | Strict 0-byte match on re-run from pinned public inputs | Reproducible recipe and open datasets, not byte-exact weights |
| Allowed scope | High-frequency style, format, and domain priors (`corpus`, design, formal syntax habits) | General agentic tool-loop fluency and token compression |
| Human authority | Zero. Must pass all five hard vetoes | Zero. Owner sign-off required to pin as a new base checkpoint |
| Composition role | Stacked simultaneously in orthogonal slots on top of `W0` or `W_base'` | Frozen base substrate underneath Class 1 adapters |

Class 2 adapters are never labeled Plane 2 ethos compilations. When an owner approves a Class 2 open-recipe adapter, it is pinned by exact hash as a frozen base `W_base'`. Class 1 adapters then compile deterministically against `W_base'`.

## Three composable domain packs

Weft targets simultaneous composition of an omnipresent base pack alongside task-specific packs on the same forward pass:

1. **Base operating pack (`corpus`):** Always active for natural prose. Targets high-token-tax writing and evidence habits such as plain language, zero em dashes, zero banned filler phrases, and explicit claim labels. Checked by deterministic prose and duplicate-reference scripts.
2. **Design pack (`weft-eido`):** Activated on interface and visual tasks. Targets clean module structure, accessible color contrast pairing, and adaptive layout conventions. Checked by deterministic contrast, static DOM safety, and module contract tests.
3. **Formal verification pack (`formal-verification`):** Activated on proof and high-assurance code tasks. Targets proof structure, explicit obligations, and strict tool-check discipline grounded in zero-dependency verifier engines and formal checkers.

## Solving multi-adapter interference

Null-space projection (AlphaEdit and BetaEdit) protects sparse declarative facts because distinct entity keys span separate subspaces. Behavioral packs break that assumption because prose, UI modules, and formal proofs share common punctuation, connectors, and markdown tokens.

Weft combines three defenses to stop shared-token collapse and cross-layer drift:

- **History-aware covariance accumulation (BetaEdit):** Each sequential pack `t` compiles inside the low-eigenvalue subspace of `C^(t) = lambda_1 * K0 @ K0.T + sum_{i<t} K_i @ K_i.T`. An exact `m x m` symmetric positive-definite Cholesky reduction avoids non-symmetric full-width inversions.
- **Register-scoped preservation keys (`y = 0`):** Compiling the prose pack places code-block, formal-proof, and UI-module activations into the preservation set with zero target shift. The prose adapter learns near-zero response on non-prose hidden states.
- **Dynamic zone-routed scale gating:** Layer 5 syntax zones (`prose`, `code`, `formal_proof`, `ui_module`) let the Layer 4 harness toggle per-adapter scale vectors (`scale_i in {0.0, 1.0}`) at `O(1)` cost without reloading weights. Explicit rule conflicts (such as static print HTML versus reactive live UI) require an explicit Plane 1 precedence flag rather than silent blending.

## Five-layer binding and integer certificates

Every Weft adapter carries a five-layer binding manifest checked by Layer 3 (`siliconcordat/warp`) before any weights reach a GPU:

1. **Tensor layout hash:** SHA-256 over sorted `name`, `dtype`, and `shape` across all base shards.
2. **Full checkpoint fingerprint:** SHA-256 over exact raw tensor bytes of every base weight tensor.
3. **Config and tokenizer hashes:** SHA-256 over `config.json`, `generation_config.json`, `tokenizer.json`, vocabulary files, and `chat_template.jinja`.
4. **Plane 1 lineage and derivation:** SHA-256 map of every source rule file, adapter class tag, and deterministic compiler recipe digest. Any one-byte edit to a Plane 1 source marks the cache stale.
5. **Runtime serving flags:** Hash over quantization mode, hybrid cache alignment (`mamba_cache_mode`), thinking toggle, tool parser ID, RoPE scaling, and context limit.

After canonical round-to-nearest-even quantization to BF16 or F16, the compiler emits a `FixedRadixCertificate`. This record stores scaled integers for squared Frobenius norm, squared spectral norm, preserved-key leakage energy, and squared safety-direction dot products. The load verifier checks every policy limit using exact integer cross-multiplication with zero floating-point division.

## Unified anti-Goodhart verification primitive

Adaptive optimizers and malicious patches defeat naive norm caps and visible test suites. Previous research measured visible-probe gaming up to 23x on MLPs and 459x on transformers, while self-carried candidate anchors allow trivial forgery.

Weft unifies four defenses into one load and promotion gate:

- **Commit-before-probes:** The adapter digest and integer certificate commit before any behavioral probe runs.
- **Verifier-owned canaries:** Expected anchor inputs and answers live strictly with the verifier, never inside the candidate packet. Any candidate trying to supply its own expected anchors is rejected on sight.
- **Two-tier lexicase gating:** Canaries span a difficulty ladder from easy to tight margins and must pass 10 repeated deterministic runs with zero variance before admission.
- **Strict safety floor:** Tier 1 (active promotion) requires 100 percent safety and capability retention, 100 percent target canaries, and passing scores on unseen private holdouts. Tier 2 (shadow stepping-stone pool) relaxes only target canaries to `(K - 1)/K` while keeping safety retention fixed at 100 percent.
- **Dual-oracle and external checks:** Objective external scripts and proof checkers score outputs directly, with zero model-as-judge shortcuts.

## Deterministic Zone A and Zone B boundary

| Zone | Operations | Replay Guarantee |
| --- | --- | --- |
| **Zone A (Deterministic Core)** | CPU `float64` BetaEdit solve, canonical BF16/F16 rounding, `FixedRadixCertificate` math, GGUF LoRA binary serialization, 5-layer binding verification, seeded lexicase selection | Exact 0-byte match across machines and runs |
| **Zone B (Hardware Runtime)** | GPU or Metal activation harvesting and greedy (`temperature = 0`) generation on real checkpoints | Recorded content-addressed transcripts tagged with hardware, driver, and kernel digests plus explicit tolerance bounds |

## Falsifiable composition matrix (`T-C1` to `T-C5`)

No multi-adapter combination is declared working until it passes five checks on greedy decoding across bare baseline `B0`, solo adapters (`C`, `E`, `F`), pairs (`C+E`, `C+F`), full stack (`C+E+F`), and Plane 1 prompt ceiling `B1`:

- **T-C1 Holdout retention:** Composed pairs keep at least 90 percent of each solo adapter's gain on separate private holdouts.
- **T-C2 Mode switching without bleed:** Mixed prompts combining prose, formal blocks, and UI modules show zero corrupted formal tokens and zero prose-rule leaks.
- **T-C3 Base capability retention (`K0`):** Preserved coding, math, and refusal holdouts stay within 1.5 percentage points of `B0` with 100 percent safety canaries intact, beating naive linear averaging.
- **T-C4 Replay and tamper proof:** Re-running the compiler yields a 0-byte diff. Flipping one byte in `chat_template.jinja`, a Plane 1 rule file, or a runtime flag triggers immediate rejection.
- **T-C5 Instant rollback:** Toggling adapter scales to zero restores bit-exact `W0` outputs and clears hybrid recurrent and KV caches in `O(1)` time.

## Hardware-matched serving topology

Stock `vLLM` and `SGLang` multiplex one LoRA per request rather than summing three adapters on the same forward pass. Weft targets `llama.cpp` (`llama-server`) for cross-platform simultaneous multi-LoRA serving (`--lora`, `--lora-init-without-apply`, per-request scale vectors) alongside a PyTorch/NumPy SGMV arena for exact null-space research:

- **Dedicated Linux compute lab (8 GiB VRAM RTX 3080 Laptop, 32 GiB RAM):** Runs full BF16/Q4 experiments on 1.7B to 4B checkpoints (`LFM3-1.7B`, `Qwen3.5-4B`) and Q4 8B checkpoints (`TwIL-8B`), plus CPU `float64` compilation and automated tamper verification.
- **Stable macOS hub (Apple M1 Max, 64 GiB unified memory):** Stays untouched as a clean daily workstation while serving larger 24B to 27B quantized checkpoints (`Qwen3.8-27B` `Q8_0`/`Q4_K_XL` or `LFM2.5-24B-A2B`) with simultaneous F16 GGUF LoRAs via read-only `llama-server`.
- **Lightweight Linux client (CPU or integrated GPU):** Runs Layer 4 harness and Layer 5 protocol tooling locally and routes larger multi-adapter inference over an authenticated local link.
