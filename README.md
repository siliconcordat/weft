# weft

Weft is the Layer 2 model and weight-space layer of the Siliconcordat stack. It compiles public written rules and domain conventions into small, verifiable weight adapters for local open-weights models. It is for teams and engineers who want repeatable, inspectable model adaptation instead of hidden fine-tunes.

## What Weft builds

Weft sits between written human rules (Plane 1) and local model inference runtimes. Instead of dumping tens of kilobytes of repeated instructions into every prompt, a public compiler turns approved text rules into bounded low-rank adapters that load behind strict verification gates.

- **Deterministic compilation:** A public CPU `float64` compiler builds null-space projected adapters using history-aware covariance tracking (BetaEdit) and emits exact quantized factors. Running the compiler twice on the same inputs gives identical bytes.
- **Exact checkpoint and lineage binding:** Every adapter binds to five checks: tensor layout, full checkpoint weights, config and tokenizer files, source rule digests, and runtime serving flags. Any one-byte drift causes rejection before GPU load.
- **Division-free integer certificates:** Post-quantization norm, spectral, leakage, and safety-direction bounds are stored as scaled integers checked without floating-point division.
- **Binary GGUF LoRA and `weft-slab/v0.1` wire containers:** Adapters export directly to F16 GGUF LoRA files for simultaneous multi-adapter serving in `llama.cpp`, or to compact `weft-slab/v0.1` asymmetric `INT4`/`INT8`/`F16` containers with embedded certificates, L5 `zone_manifest_id` pins, and 32-byte SHA-256 trailers.
- **Verifier-owned behavioral gates:** Expected canary outputs live with the verifier, never inside untrusted adapter headers. Promotion requires both visible canaries and private holdouts.

## Non-negotiable rules

Weft enforces five hard vetoes inherited from Model Domain Expansion research:

1. No hidden weight deltas. Every Plane 2 adapter must be re-derivable from public Plane 1 inputs with a public, simple compiler.
2. No automatic resolution of conflicting human rules. Contested rules stop compilation and return to the owner.
3. No closed-weight injection. Adapters run only on self-hosted open-weights runtimes with load-time verification, audit logs, and an instant kill switch.
4. No premature protocol standard. Wire extensions stay vendor-namespaced until outside users exercise them.
5. Human authority stays in plain text (Plane 1). Every weight adapter is a disposable cache.

## Stack position

| Layer | Repository | Role |
| --- | --- | --- |
| L5 | `siliconcordat/mde` | Universal structured agent protocol and read-before-write locks |
| L4 | `siliconcordat/exo` | Continuous harness, human approval gates, causal memory, zone routing |
| L3 | `siliconcordat/warp` | Operating system boundary, sandboxes, and fail-closed load verifier |
| L2 | `siliconcordat/weft` | Deterministic weight compiler, integer certificates, multi-adapter composition |
| L1 | `siliconcordat/loom` | Hardware theory and verification physics research |

## Documentation and status

- [ARCHITECTURE.md](ARCHITECTURE.md): Two-class adapter policy, three-pack composition, verification primitive, and hardware layout.
- [PLAN.md](PLAN.md): Phased build milestones, test suite, runnable experiments, and kill criteria.
- [experiments/RESULTS.md](experiments/RESULTS.md): Verbatim measured outputs from Experiments E1 to E8, including failure boundaries, live `llama-server` results, Latent Lattice MDE probes, and multi-layer SwiGLU/RMSNorm tangent projection.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/python experiments/e1_shared_token_composition.py
.venv/bin/python experiments/e2_anti_goodhart_gate.py
.venv/bin/python experiments/e3_real_checkpoint_smoke.py
.venv/bin/python experiments/e4_asymmetric_quant_packing.py
.venv/bin/python experiments/e5_noise_k0_scaling_and_readout.py
# Opt-in live 1.5B llama-server multi-LoRA check:
WEFT_LIVE_BASE_GGUF=/path/to/qwen2.5-1.5b-instruct.gguf .venv/bin/python experiments/e6_live_llama_server_lora.py
.venv/bin/python experiments/e7_latent_lattice_mde.py
.venv/bin/python experiments/e8_swiglu_rmsnorm_tangent_coupling.py
```

Status: v0.1 core compiler (with asymmetric `INT4`/`INT8`/`F16` packing, `weft-slab/v0.1` containers, and multi-layer symmetric `TangentBetaEdit`), `llama.cpp`-verified GGUF v3 F16 LoRA codec, five-layer binding, anti-Goodhart anchor harness, 104 tests, and Experiments E1 to E8 are verified locally. For E8 that covers Parts A and B only, and Part C is an unverified `source claim`.

Experiment E6 verifies simultaneous multi-LoRA serving and `0.99 ms` bit-exact rollback inside live `llama-server` on `Qwen2.5-1.5B-Instruct`, Experiment E7 verifies Latent Lattice capability banks and atomic multi-bank rollback, and Experiment E8 verifies exact SwiGLU/RMSNorm finite-perturbation coupling and multi-layer symmetric `TangentBetaEdit` on a synthetic CPU `float64` stack.

The live `Qwen3.5-4B-Q8_0` E8 Part C numbers are an unverified `source claim`, because no re-runnable script or base-model hash is committed. Claims carry explicit labels (`verified`, `source claim`, or `proposed`).

## License

Code is licensed under Apache-2.0 (see [LICENSE](LICENSE)). Documentation is licensed under CC BY 4.0.
