# Experimental Hy-MT2-30B-A3B (HY-V3) support

This branch contains an **experimental native HyV3 engine implementation** whose Windows CUDA 13.2/sm_120 **compile has passed**, but whose native decode has been smoke-tested on the official Q4_K_M GGUF, but whose output correctness has NOT been checked against llama.cpp. It is not ready for production translation.
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
- [x] Compile Windows CUDA 13.2 sm_120 binary in GitHub Actions (no GPU)
- [x] Load actual Q4_K_M model and generate 1 then 16 tokens on RTX 5060 8 GB (load/decode smoke test only)
- [ ] Verify output tokens/logits against llama.cpp on identical input token IDs
- [x] Windows CUDA build + RTX 5060 8 GB / RAM 32 GB first smoke test (numerical correctness NOT yet established)

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

**Unverified:** actual model load and runtime tensor shapes, GPU routing parity,
exact output logits, tokenizer compatibility, Windows inference, memory and speed.
Neither the synthetic GGUF CI, source-wiring CI, nor Windows **compile-only** CI checks these. Run the inspector
on the real checkpoint first; then compile and compare greedy output against llama.cpp.
Do not merge this draft or use it for subtitle production until those checks pass.

The upstream official configuration is documented at
https://huggingface.co/tencent/Hy-MT2-30B-A3B/blob/main/config.json .

## Windows build result and first native model smoke test

The first Windows CI attempt failed at CMake configuration because the CI runner lacked
`cublas_dev`. After adding that package, [Windows CUDA 13.2 / sm_120 build #41648984135](https://github.com/pingpingjinping/PolyStrata/actions/runs/41648984135)
completed successfully, including `strata-poly.exe` verification. The runner has no GPU;
**this is compile-only success, not inference success**.

A follow-up workflow revision uploads the experimental `strata-poly.exe` as an
artifact for manual testing (see the most recent `HY-V3 Windows CUDA compile` run).
Do not replace the normal PolyStrata executable or merge the PR yet.

From a separate PowerShell directory containing the downloaded `strata-poly.exe`:

```powershell
.\\strata-poly.exe --model "C:\\Users\\yjhmo\\Desktop\\PolyStrata\\Hy-MT2-30B-A3B-Q4_K_M.gguf" --tokens 1 --max-new 1 --max-context 512 --prompt-batch 16 --stage-mib 64 --ram-experts-mib 4096 --no-draft --window 1
```

Run after stopping existing GPU inference servers to free memory. On Windows, you may need
CUDA 13 DLL directories on `PATH` (the compiled engine links NVIDIA CUDA libraries).
If DLLs are missing, capture the exact error instead of mixing random versions.
This command tests only a single token as a load and inference smoke test. **One generated
ID does not establish output correctness**; compare full greedy token sequences against
llama.cpp with identical token IDs in a later step.


## First real-device results (reported 2026-11-04 KST)

Windows RTX 5060 8 GB, 32 GB system RAM, official
`Hy-MT2-30B-A3B-Q4_K_M.gguf`, with CUDA 13 DLL path supplied from the
PolyStrata Python venv. The GGUF inspector reported 48 layers, 1 dense,
47 MoE, 766 tensors, 47 selection-bias tensors named `exp_probs_b`,
no warnings/errors.

`strata-poly.exe --tokens 1 --max-new 16 --max-context 512 --prompt-batch 16
--stage-mib 64 --ram-experts-mib 4096 --no-draft --window 1 --temperature 0`

produced:
- Loaded in 2.1 s (warm cache); initial one-token run loaded in 8.0 s.
- 1,991 of 6,016 routed experts resident in 5.3 GiB of VRAM.
- 12 CPU threads; 4,096 MiB explicitly reserved for external experts.
- 16 generated token IDs: `206,341,245,636,13,206,478,628,206,15271,206,279,206,252,206,26887`.
- Measured 7.86 generated tokens/s over a 16-token sample.
- GPU expert hit share 48.5%, 478 experts swapped, 119.5 ms/token CPU-expert time.

This establishes native loading and sustained token generation but **not**
text correctness, translation quality or representative benchmark speed.
The next gate is identical-token greedy parity with llama.cpp. Do not
merge or replace the user's production PolyStrata environment yet.

## One-window Japanese-to-Korean test (no llama.cpp)

A **standalone Tkinter UI** is in `tools/hy_v3_translate_gui.py`. This is an
experimental, direct client for `strata-poly.exe --serve`: it loads the real GGUF
tokenizer, follows Tencent's one-user-message no-thinking template, uses the
model's special EOS tokens, streams decoded Korean output, and keeps the
native engine loaded between translations. It does **not** replace PolyStrata's
production settings or require the regular web server.

Run from the **experimental clone** on Windows:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_translate_gui.py
```

The defaults match `Desktop\PolyStrata\strata-poly.exe` and
`Desktop\PolyStrata\Hy-MT2-30B-A3B-Q4_K_M.gguf`. To override:

```powershell
python .\tools\hy_v3_translate_gui.py --exe "D:\test\strata-poly.exe" --model "D:\models\Hy-MT2-30B-A3B-Q4_K_M.gguf"
```

The UI finds the existing `nvidia\cu13\bin\x86_64` CUDA 13 pip-wheel DLL
folder automatically. The Python interpreter must have the `regex` package,
which the normal PolyStrata .venv already uses. A standard Python for Windows
provides `tkinter`.

The Hunyuan pretokenizer uses the Qwen2 regex in llama.cpp; experimental support
was added in `tools/strata_tokenizer.py`. Unsupported GGUF pretokenizers cause
an explicit error rather than silently corrupting the input. The visible
text being translated can be checked directly in the GUI.

First try a single simple Japanese sentence. Greedy temperature=0 is chosen
for debugging; the 30B model card recommends 0.7 for normal use. **Native
model output correctness is still unverified and the PR stays a draft.**
