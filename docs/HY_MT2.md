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
Later user single-run reports (same 100-line task, web sampling not initially
held fixed): batch 512 at 65.5 tok/s / 78.5% expert hits; 1024 at 66.4 / 76.0%;
2048 at 68.1 / 75.2%. After explicitly setting web Temperature=0, batch 2048
reported 69.4 tok/s. An expert-learning-enabled run later reported 76.0 tok/s
and 75.2% hits, but whether that was after profile reload was not established.
These reports do not yet isolate causality or establish a validated new default.

The same user later compared CPU thread counts at batch 2048, with decode
profiling enabled. Each row is a single run, not a repeated median:

| Threads | Generation tok/s | Expert hits | Attention GPU ms/step | CPU host phase ms/step | Router wait ms/step | Decode wall ms/step |
| --- | --- | --- | --- | --- | --- | --- |
| Automatic | 67.1 | 74.2% | 3.470 | 7.375 | 4.531 | 14.907 |
| 4 | 73.0 | 75.7% | 3.337 | 6.744 | 4.603 | 13.704 |
| 6 | 79.1 | 75.7% | 3.298 | 5.792 | 4.408 | 12.657 |
| 8 | 76.4 | 75.9% | 3.411 | 5.720 | 4.749 | 13.105 |

A subsequent **6-thread run without profiling** returned 1,323 tokens at
**76.7 tok/s**, with 75.8% expert hits. Use 6 threads as this PC's provisional
comparison setting. The difference between that run and 79.1 is not an estimate
of profiling overhead; these are separate requests with run-to-run variation.
Host phase and router wait overlap other work; they are not additive.

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
  The existing `attend()` dispatch already has a fused split/join kernel for
  dimension 128 and eight query heads per KV head. Hy's 32/4 head geometry
  matches it when workspace fits and there is no window/ring/whole-context
  override or quantized KV. This is a source-dispatch finding; individual
  kernel execution has not been traced on the user's GPU. The attention timer
  also includes Q/K/V projections and output projection, not just attention.
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

### Step 1: generation bottleneck diagnostics (opt-in)

`--profile-decode` enables native `--hy-v3-profile` in the isolated launcher.
**Download the new Windows sm_120 `strata-poly.exe` CI artifact first**; the
previous executable does not recognize the new flag. Keep the existing model,
tokenizer and CUDA DLL environment. Close the old server before replacing its EXE.

```powershell
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --temperature 0 --learn-experts --profile-decode
```

Set the web Chat Temperature to **0 and save it**: a browser-stored request
temperature overrides the launcher's default. Use a new conversation, the same
100 Japanese lines and no other API clients. Send the final `[hy-v3 profile]`
line together with the normal `done` line. Remove `--profile-decode` for normal use.

The profile covers **decode eval steps only**: prefill and its final one-token
eval are excluded. The first generated token is picked from prefill logits and
needs no decode eval, so `decode_steps` is normally generated tokens minus one.
An answer with no decode eval emits a zero-step note rather than a fake rate.

| Field | What it measures |
| --- | --- |
| attention GPU | CUDA event interval across each layer's Q/K/V projection, Q/K norm, RoPE, KV store, attention and output projection; summed over 48 layers and averaged over sampled decode steps |
| CPU expert host phase | Existing host elapsed time in `cpu_experts()`, including parallel GPU-launch callbacks, weight-read waits and worker synchronization; not CPU utilization or exclusive CPU compute time |
| router host wait | Existing host time waiting for router-copy readiness; can include preceding queued GPU work, not just the router kernel |
| decode wall | Whole generation-loop elapsed time per model eval step, including sampling, streaming and profiling overhead |
| profile collection | Host time spent reading completed timing events per step; excludes GPU event-node overhead |

**These intervals overlap. Do not add them or subtract their sum from wall time.**
Attention time is a stream interval, not a sum of exclusive kernel durations, and
does not identify which attention kernel is slow. Cached experts, shared experts,
logits, transfers and all other work are not individually timed in this first step.

GPU samples are taken on steps 1, 17, 33, ... . Profile mode uses distinct graph
keys for sampled and ordinary heads, so the intervening steps execute recordings
without timing nodes. `gpu_samples` reports the actual sample count; host phase
averages cover all decode steps. Timing events use external event-record nodes;
graph replay stays enabled. Events are read after the ordinary logits synchronization,
without adding layer-by-layer stream waits. Profiling itself adds event nodes and
host calls, so its tok/s is diagnostic and must not be presented as an optimized
speed measurement. The user has now collected the timing reports above; native
output/logit parity against the reference engine remains unverified.

The first version supports **CUDA, one GPU, one sequence, window 1, lookup 0**.
Other model families and unsupported profile combinations fail explicitly; HIP
profiling is not supported. Without the flag, no timing events are allocated or
recorded and existing model math/graphs/protocol fields are unchanged. Profile
JSON goes to native stderr, while the launcher tails just those new lines into
its console. No new stdout protocol message or common web-server change is needed.

### Step 2: CPU down-projection scheduling experiment (opt-in)

`--cpu-down32` passes native `--hy-v3-cpu-down32`. It changes only the CPU
down-projection work distribution from 16 output rows per task to 32: at Hy's
2048-wide hidden state, 128 down tasks become 64. Fewer tasks may reduce worker
bookkeeping but can also reduce load balancing. The small user-reported
improvement below is preliminary, not a validated default for other hardware.
The stock CPU function is retained unchanged. The experimental copy uses the
same quantizers, dot products, activation math and per-output expert sum order.
The option defaults off and is rejected for other model architectures.

Download the **new** Windows CI EXE for this change. Stop the old server and
replace `Desktop\PolyStrata\strata-poly.exe`; leave the model and venv there.
Run a baseline on that same new executable first:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --threads 6 --temperature 0 --profile-decode
```

Then stop the server and append **`--cpu-down32`** to the same command. Verify
the native log says `HY-V3 CPU down tasks: 32 rows`. Use the same new chat input,
web Temperature 0 and cache warm-up state for both. Leave `--learn-experts`
off for both runs, as in the command above, to exclude learned cache ranking
and profile updates from this comparison; this does not delete its saved file.
Repeat the pair if the change is small. Report each `done` and profile line and
compare output too. Remove `--profile-decode` from both commands for a final
pair without profiling.
Remove `--cpu-down32` to return to the stock scheduling.

User-reported results on the same Windows RTX 5060 8 GB / 32 GB RAM PC, native
revision `8625d11c`, prompt batch 2048, 6 CPU threads, context 4096, requested
temperature 0, expert learning off, and the same 100-line input as instructed:

| Run order | Down rows/task | Profiling | Generation tok/s | CPU host ms/step | Attention GPU ms/step | Router wait ms/step | Decode wall ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A) | 16 | On | 73.8 | 6.338 | 3.418 | 4.685 | 13.568 |
| 2 (B) | 32 | On | 75.5 | 6.048 | 3.398 | 4.700 | 13.259 |
| 3 (A repeat) | 16 | On | 74.6 | 6.276 | 3.355 | 4.661 | 13.417 |
| 4 (A) | 16 | Off | 74.9 | — | — | — | — |
| 5 (B) | 32 | Off | 76.1 | — | — | — | — |

Every run reported 1,322 output tokens, `stop`, `cancel=False`, and 73.6% expert
cache hits. Profiled runs each had 1,321 decode steps and 83 GPU samples. The
unprofiled pair is **+1.6% reported generation throughput** for 32 rows; the
profiled B is +1.2% to +2.3% against the two A runs. CPU host phase is shorter
in that B run, but it includes overlap and is not exclusive CPU compute time.
This is one short unprofiled pair, not repeated medians or a statistical claim.
The console's rounded whole-second duration is 19 seconds for both unprofiled
runs; percentages use the reported tok/s. Equal output counts and hit rates
do not establish identical token IDs, translated text or numerical parity.
Keep the option experimental/default-off; no fresh EXE is needed for these
documentation updates.

The standalone CPU test uses the pinned real ggml backend, synthetic quantized
Q4_K/Q5_K/Q6_K and F32 matrices, Hy-sized 2048/768 dimensions and partial output
tasks. It compares output bits with the stock function, reused experts, two
token rows, SiLU/GELU, launch/wait callbacks and 1/4/6/8 worker threads. This is
CPU scheduling evidence, not whole-model CUDA parity or a speed benchmark.
Local Release CPU build passed **720 checks**, all **31 HY-V3 Python tests**
passed, and the existing regression suite ran **480 tests: 477 passed, 3 skipped**
for absent reference tokenizer data. Router/profile accounting tests and the
host `main.cpp` syntax check passed. CUDA runtime, HIP and SYCL were unavailable
locally. At native revision `8625d11c`, both the Windows sm_120 CUDA compile and
compatibility CI (including the real ggml CPU parity job) passed:
[Windows build](https://github.com/pingpingjinping/PolyStrata/actions/runs/41826631122),
[CPU/Python checks](https://github.com/pingpingjinping/PolyStrata/actions/runs/41826631069).

To reproduce the quantized CPU check without a GPU:

```sh
cmake -S . -B build-hy-cpu -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_HY_CPU_TEST=ON -DSTRATA_PORTABLE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hy-cpu --target poly_hy_cpu_down32_test --parallel 2
./build-hy-cpu/poly_hy_cpu_down32_test
```

### Remaining gates

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
The pre-profiling revision ran 480 existing Python tests (477 passed, 3 skipped
because reference vocabularies were unavailable), 24 HY-V3 tests (all passed),
and the standalone C++ router smoke test (passed). Profiling adds CPU accounting
and log tests; real CUDA event timing/parity remains a device validation gate.

Local CPU checks:

```sh
python -m unittest discover -s tests -p 'test_hy_v3*.py' -v
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 tests/hy_v3_router_test.cpp -o /tmp/hy_v3_router_test
/tmp/hy_v3_router_test
```

`tools/inspect_hy_v3_gguf.py MODEL.gguf --json` checks headers/tensor inventory,
including both expert-bias spellings, without reading model weights. Synthetic
header, routing and source-wiring tests do not prove native inference correctness.
