# Experimental Hy-MT2-30B-A3B (HY-V3)

Native `hy_v3` support runs through `strata-poly` and the existing PolyStrata
web/API server. The feature stays opt-in in this fork. **Keep PR #1 a draft
until the real-device gates below pass.**

## Verified status, 2026-11-04 KST

| Area | Evidence | Remaining gate |
| --- | --- | --- |
| Windows CUDA build | CUDA 13.2 / sm_120 compile succeeded; the first failure was a missing CI `cublas_dev` package | Latest-head CI must be green |
| Real model load/decode | User loaded official Q4_K_M on RTX 5060 8 GB / Windows / 32 GB RAM; 48 layers, 1 dense + 47 MoE, 766 tensors | Numerical parity with llama.cpp |
| Translation | User's 100-line test returned 100 Korean lines, 1,327 output tokens, reported 68.3 generation tok/s | Full semantic review; controlled throughput comparison |
| Python regression | Existing Gemma/Qwen config, tokenizer, web API, cancellation and recovery suites; no GPU required | Actual Gemma 4 and Qwen3.6 load/translation on the new executable |
| Performance settings | Launcher exposes batch, staging and host-expert RAM controls | Same-input 512/1024/2048 sweep and memory samples |
| Stability | Offline context-overflow, cancellation and engine recovery tests | Long input, repeated requests, restart and RAM/VRAM pressure on RTX 5060 |

The 68.3 tok/s report used the former experimental settings (batch 32,
stage 64 MiB, no prompt checkpoints). The first 16-token ID-only smoke test
reported 7.86 tok/s and 48.5% VRAM expert hits; it was a different workload
and is not a translation benchmark. Neither number proves numerical parity.
Current larger-batch defaults have not yet been measured on that PC.

References:

- [Official model and configuration](https://huggingface.co/tencent/Hy-MT2-30B-A3B)
- [Official GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) (Q4_K_M approximately 18.2 GB)
- [llama.cpp reference graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/hy-v3.cpp)

## Run the isolated web/API launcher

Close other model servers to free VRAM. In PowerShell:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py
```

The default engine/model paths point to `Desktop\PolyStrata`. Override with
`--exe` and `--model`. CUDA 13 DLLs are found in the existing Python venv.
The model's tokenizer and chat template are extracted from its GGUF header.
The generated config, tokenizer and log stay under the ignored `.hy-v3-web/`.

Open http://127.0.0.1:18081/ ; the API is
http://127.0.0.1:18081/v1/chat/completions . Close with Ctrl+C.
The launcher does not overwrite the normal Gemma/Qwen run config or port 8080.
Hy-MT2 has no default system prompt: put the translation instruction in the
user message, followed by the Japanese text. Tools and vision are unverified.

Current experimental settings: 4096 context, prompt batch 1024, stage 128 MiB,
temporary read stage up to 2048 MiB, 6 prompt checkpoints, all off-VRAM experts
in RAM (`--ram-experts-mib -1`), temperature 0.7, no MTP drafts or lookup.
The read stage borrows expert-cache VRAM during prefill; the engine restores
that space afterwards. A bigger batch still needs more permanent scratch.

For allocation failures try `--prompt-batch 512 --stage-mib 64 --stage-read-mib 0`,
then batch 256. Restore the initial mode with
`--prompt-batch 32 --stage-mib 64 --stage-read-mib 0 --prompt-cache 0`.
Use `--ram-experts-mib 4096` to cap explicitly held host experts at 4 GiB.
These are manual fallback settings, not an automatic OOM recovery guarantee.
The -1 setting may reduce disk reads, but cannot remove CPU expert compute.

## What can affect speed

- Q4_K_M and the earlier Qwen UD-Q3_K_M are different quantizations. Compare
  cache occupancy/hit rates instead of treating total parameter count or A3B
  as equal memory traffic. The official Hy checkpoint is about 18.2 GB.
- Hy uses 48 full-attention layers, with 47 top-8 MoE blocks and a shared
  expert. Qwen3.6 uses a different mixture of recurrent and full-attention
  blocks. Equal active parameter labels do not imply equal decoding work.
- Qwen's `fattn`/`fdec` kernel currently requires head dimension 256
  (`src/poly/fattn.cu`); Hy has dimension 128 and uses `attend_batch`/`attend`.
  Setting `STRATA_FATTN=1` alone does not accelerate Hy. Extending those kernels
  is a separate native change requiring CUDA/HIP builds and numerical checks.
- Batch 512/1024/2048 trades prefill work against expert-cache space. Measure
  both prompt and generation speed; do not choose a winner by prompt speed alone.
- Hy's fp16 KV cache uses 96 KiB per context position across all layers:
  about 384 MiB at 4096 populated positions and 768 MiB at 8192, before scratch
  and checkpoints. Longer sequences may displace experts as the cache grows.
  Hy does not yet support the engine's Qwen q8_0 or Gemma int8 KV options.

Two optional launcher controls reuse existing engine features without changing
model math:

```powershell
# Learn on representative translation requests; restart to use the saved cache ranking.
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --learn-experts
# Optional CPU thread comparison: replace N with this PC's physical-core count.
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --threads N
```

`--learn-experts` passes a private `.hy-v3-web/hy-mt2-experts.bin` to the
existing native `--expert-profile` loader/saver. Its counts are saved after
answers (at most once per 10 seconds) and on clean exit. It may improve the
initial resident expert selection for the translation workload; no speedup is
claimed before measurement. Leave it off for the first batch sweep, so a
changing learned profile does not confound the comparison. Do not reuse this
file for other checkpoints. Thread count also remains automatic unless selected.

## Controlled benchmark on the user's PC

Save the same numbered Japanese text as UTF-8 `hy-input.txt`. Start the launcher
with `--prompt-batch 512`; after testing, stop it and repeat at 1024 and 2048.
Keep context, staging, RAM mode, temperature and source fixed. Do not run another
client against the endpoint during measurements. In a second PowerShell window:

```powershell
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_bench.py --input .\hy-input.txt --label hy-q4-b512 --output .\.hy-v3-web\bench\hy-b512.json --cancel --run
```

Change the label and output filename for each batch. Add `--long-input PATH`
for a longer numbered sample; leave room for its output in the context, or
record the context error explicitly. The client defaults to 2048 output tokens;
increase `--max-tokens` and context together when required. Without `--run`, it
prints a plan without connecting or writing results.

The report keeps three fresh-prefix/warm request pairs, native prompt and decode
timings, client TTFT and wall time, the request's expert VRAM hit rate, and sampled
RAM/VRAM readings. Expert hit rate is the server's request-wide statistic, not a
separately instrumented prefill/decode rate. Missing telemetry stays missing.
Warm results are separate; excessive cold-prefix reuse is rejected. A truncated
stream or `length` finish is not a successful complete translation. Numbered-line
coverage checks missing/duplicate/reordered IDs; meaning still needs manual review.
Reports contain translation text and machine details; they are local and not uploaded.

The cancellation probe closes an active stream after 32 characters, checks the
engine observed cancellation/disconnection, waits for idle, then sends a fresh
translation. If the engine finishes before the disconnect, the probe is
inconclusive and does not count as a cancellation pass. Partial results are saved
after each request, including failures. Cancellation tests do not test process restart.

For Gemma 4 and Qwen3.6 regression, start each model separately with its normal
config and run the same client with `--url http://127.0.0.1:8080` and a model-specific
label/output. Compare the old and new executable with the same checkpoint,
config, prompt, and greedy output. Also retain each launcher/config and executable
hash: labels alone do not establish the actual batch/build. Compare time for the
whole correctly translated input as well as tok/s; tokenizers differ across models.

## Release gates and remaining stability checks

- [ ] Same-token greedy Hy output/logit comparison against llama.cpp, same GGUF.
- [ ] Old/new Gemma 4 and Qwen3.6 real loading, complete translation, web UI and
  streaming/non-streaming API, with tokenizer and settings held fixed.
- [ ] 512/1024/2048 measurements with completed translations, cache hits and memory.
- [ ] Long translation and at least ten consecutive varying requests, including A/B/A
  cache restoration; inspect output and memory growth after warm-up.
- [ ] Stop while reading a prompt and while generating; next request succeeds.
- [ ] Ctrl+C, relaunch, and translate again; model process and VRAM are released.
- [ ] Naturally encountered allocation failures or a bounded low-memory test: save
  the error, stop the failed process, restart with the fallback settings and verify
  a subsequent request. Capping expert RAM alone does not simulate system RAM OOM.
  Do not allocate all system RAM/VRAM simply to force an error.
- [ ] Latest Windows CUDA and Python CI pass; attach actual results and clear Draft.

The current Python CI covers Gemma/Qwen settings and mock/fake-engine HTTP paths,
not real GPU loading or translation quality. A container-only telemetry unit-test
failure was reproduced on unchanged main: real cgroup memory replaced its fake
psutil fixture. The fixture now isolates cgroup limits; production telemetry is unchanged.
Local validation ran 480 existing Python tests (477 passed, 3 skipped because
reference vocabularies were unavailable), 24 HY-V3 tests (all passed), and the
standalone C++ router smoke test (passed). These are not real-device release gates.

Local CPU checks:

```sh
python -m unittest discover -s tests -p 'test_hy_v3*.py' -v
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 tests/hy_v3_router_test.cpp -o /tmp/hy_v3_router_test
/tmp/hy_v3_router_test
```

`tools/inspect_hy_v3_gguf.py MODEL.gguf --json` checks headers/tensor inventory,
including both expert-bias spellings, without reading model weights. Synthetic
header, routing and source-wiring tests do not prove native inference correctness.
