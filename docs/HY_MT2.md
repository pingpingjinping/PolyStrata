# Experimental Hy-MT2-30B-A3B (HY-V3) support

This branch contains an **experimental native HyV3 engine implementation** that has NOT been built with CUDA or checked against the actual GGUF. It is not ready for production translation.
The `hy_v3` architecture now has an experimental registration in `Engine::open()` solely for development testing. **A successful load does not prove correct inference**; do not merge until both CUDA compilation and llama.cpp parity tests pass.

## Verified upstream references

- Official weights: https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF
- llama.cpp reference: https://github.com/ggml-org/llama.cpp/blob/master/src/models/hy-v3.cpp
- GGUF architecture: `hy_v3`; official Q4_K_M checkpoint: approximately 18.2 GB.
- PolyStrata: `strata-poly` handles some architectures using the shared `Engine`
  class, but `hy_v3` is not currently registered.

## Current status

- [x] Fork-specific feature branch, leaving `main` untouched
- [x] Independent CPU reference for HY-V3 sigmoid/selection-bias/top-k routing
- [x] CPU-only deterministic routing smoke test
- [x] Header-only Python GGUF inspector and synthetic-file tests (both expert-bias spellings)
- [x] Standalone GGUF metadata and tensor inventory preflight (without loading weights)
- [x] Initial model-specific metadata checks and GGUF tensor-loading code in a new `HyV3 : Engine` subclass (NOT RUN on the real GGUF)
- [x] Implemented full-attention/Q-K RMSNorm/RoPE/KV code by adapting Flint's existing inference paths (NOT parity-tested)
- [x] Implemented dense lead block and SwiGLU (NOT parity-tested)
- [x] Connected routed experts to the shared PolyStrata RAM/VRAM cache and inherited shared-expert execution (NOT parity-tested)
- [x] Connected existing GPU sigmoid/biased-top-k router path with HY-V3 selection bias and scaling (NOT parity-tested)
- [ ] Model tokenizer/chat template and installer integration
- [ ] Build actual Windows/Linux CUDA binary and verify output tokens/logits against llama.cpp
- [ ] Windows CUDA build + RTX 5060 8 GB / RAM 32 GB smoke test

The first tests isolate correctness-sensitive parts of the model. The standalone routing reference in `src/poly/hy_v3_router.hpp` is not called by the inference engine; the engine uses the shared GPU `route_rows()` path, which must still be checked numerically against the reference.

## Run CPU-only test (no model download)

```sh
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 tests/hy_v3_router_test.cpp -o hy_v3_router_test
./hy_v3_router_test
```

## Compare translation quality with llama.cpp (temporary baseline)

On Windows, use a recent llama.cpp release with HY-V3 support:

```powershell
llama-server -hf tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M -c 8192 -ngl 20 --port 18080
```

Adjust `-ngl` for available VRAM. The baseline API is `http://127.0.0.1:18080/v1`.
Do **not** point it to PolyStrata's usual port 8080. Measure the same Japanese subtitle
lines and compare meaning, formatting, prompt processing and generation speed against Gemma 4.
This fallback is not native PolyStrata acceleration.

## Notes for native implementation

HY-V3 from llama.cpp has 48 trunk layers in the 30B-A3B variant.
Its full-attention blocks have per-head Q/K norms and separate RoPE; the routed MoE path
uses sigmoid gating with a *selection-only bias*, and the always-active shared expert
is summed with the routed ones. Some layers may have an ordinary dense FFN.
Optional MTP/NextN weights must not be executed as ordinary trunk layers.

Port the full tensor shapes and routing flags from the upstream reference.
Validate prefill and decode **token-by-token** against llama.cpp before benchmarking.

## Validate an existing downloaded GGUF (no GPU needed)

```powershell
python tools/inspect_hy_v3_gguf.py "C:\\models\\Hy-MT2-30B-A3B-Q4_K_M.gguf" --json
```

The inspector reads **only the GGUF header and tensor inventory** (not 18.2 GB of weights).
It verifies architecture, attention tensors, dense lead block, MoE expert tensor sets and
both `blk.N.exp_probs_b` and `blk.N.exp_probs_b.bias` names. The latter spelling is seen
in older HY-V3 quantizations, per https://github.com/ggml-org/llama.cpp/issues/25657.
A PASS is not evidence that the model runs or produces correct translations.

```sh
python -m unittest discover -s tests -p 'test_hy_v3_gguf.py' -v
```

## Native prototype status (NOT production ready)

`src/poly/hy_v3.cpp` and `hy_v3.hpp` implement a first-pass experimental decoder using
PolyStrata's tested Flint attention and shared-expert MoE code as a base, adapted to
Hy-MT2's **full causal attention**, **RoPE on every layer**, **no sink logits**, Q/K RMSNorm,
one dense leading block and 47 MoE blocks. The decoder is registered in `Engine::open()`
and added to all three CMake targets that link `engine.cpp`.

The native implementation hard-checks the known Hy-MT2-30B-A3B geometry (48 layers,
2048 hidden, 32/4 attention heads, head dim 128, 128 experts, 8 active experts,
6912 dense FFN, 768 expert FFN). It requires selection bias tensors in either of
the two known GGUF spellings. It disables MTP/NextN draft decoding until correctly ported.

**Unverified:** CUDA compilation, real checkpoint tensor shapes, GPU routing parity,
exact output logits, tokenizer compatibility, Windows inference, memory and speed.
Neither the synthetic GGUF CI nor the source-wiring CI checks these. Run the inspector
on the real checkpoint first; then compile and compare greedy output against llama.cpp.
Do not merge this draft or use it for subtitle production until those checks pass.

The upstream official configuration is documented at
https://huggingface.co/tencent/Hy-MT2-30B-A3B/blob/main/config.json .
