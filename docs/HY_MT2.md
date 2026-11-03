# Experimental Hy-MT2-30B-A3B (HY-V3) support

This branch is a **feasibility and routing-test stage**, NOT a working PolyStrata backend.
Do not add `hy_v3` to `Engine::open()` before the implementation is correct:
it would misleadingly claim support and can silently generate incorrect translations.

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
- [ ] Load and validate HY-V3 GGUF metadata and tensors in a new `Engine` subclass
- [ ] Full attention with Q/K RMS norms, RoPE, KV cache, and output projection
- [ ] Dense lead block handling and gated SwiGLU
- [ ] Routed-expert weight caching and always-on shared expert
- [ ] Correct bias selection, unbiased weights, normalization and scaling on CUDA
- [ ] Tokenizer/chat template and PolyStrata setup integration
- [ ] Verify output tokens/logits against llama.cpp on identical prompts
- [ ] Windows CUDA build + RTX 5060 8 GB / RAM 32 GB smoke test

This first test isolates a correctness-sensitive part of the model.
The routing reference in `src/poly/hy_v3_router.hpp` is **not yet used by the inference engine**.

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
