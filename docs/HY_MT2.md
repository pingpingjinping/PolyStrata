# Experimental Hy-MT2-30B-A3B (HY-V3)

Native `hy_v3` support runs through `strata-poly` and the existing PolyStrata
web/API server. The feature stays opt-in in this fork. **Keep PR #1 a draft
until the real-device gates below pass.**

## Verified status, 2026-11-05 KST

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
current profiling version adds QKV/core/output detail. Older profiling builds
print only the aggregate line; pre-profiling builds reject the flag. Keep the
existing model, tokenizer and CUDA DLL environment. Close the old server before
replacing its EXE.

```powershell
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --threads 6 --temperature 0 --cpu-down32 --profile-decode
```

Set the web Chat Temperature to **0 and save it**: a browser-stored request
temperature overrides the launcher's default. Use a new conversation, the same
100 Japanese lines and no other API clients. Send the final `[hy-v3 profile]`
and **`[hy-v3 attention]`** lines together with the normal `done` line. Keep
expert learning off for this comparison. Remove `--profile-decode` for normal use.

The profile covers **decode eval steps only**: prefill and its final one-token
eval are excluded. The first generated token is picked from prefill logits and
needs no decode eval, so `decode_steps` is normally generated tokens minus one.
An answer with no decode eval emits a zero-step note rather than a fake rate.

| Field | What it measures |
| --- | --- |
| attention GPU | CUDA event interval across each layer's Q/K/V projection, Q/K norm, RoPE, KV store, attention and output projection; summed over 48 layers and averaged over sampled decode steps |
| QKV+norm/RoPE/KV | GPU interval from attention begin through all Q/K/V projections, Q/K head norms, RoPE and fp16 K/V cache stores; excludes the input RMSNorm before attention begin |
| attention core | GPU interval containing `attend()` (decode) or `attend_batch()` (prefill is excluded from this report); excludes Q/K/V and output projections |
| output projection | GPU interval containing the final `wo` projection from attention output into the residual width |
| CPU expert host phase | Existing host elapsed time in `cpu_experts()`, including parallel GPU-launch callbacks, weight-read waits and worker synchronization; not CPU utilization or exclusive CPU compute time |
| router host wait | Existing host time waiting for router-copy readiness; can include preceding queued GPU work, not just the router kernel |
| decode wall | Whole generation-loop elapsed time per model eval step, including sampling, streaming and profiling overhead |
| profile collection | Host time spent reading completed timing events per step; excludes GPU event-node overhead |

**CPU/router host intervals overlap GPU work. Do not add those fields or subtract
their sum from wall time.** The three GPU stages are adjacent intervals within
`attention GPU`, so their sum should approximately match it (subject to event
precision/rounding); adding them to the aggregate would double-count attention.
They are stream intervals including event overhead and any gaps, not exclusive
kernel durations or a kernel trace. Cached experts, shared experts, logits,
transfers and all other work are not individually timed here.

GPU samples are taken on steps 1, 17, 33, ... . Profile mode uses distinct graph
keys for sampled and ordinary heads, so the intervening steps execute recordings
without timing nodes. `gpu_samples` reports the actual sample count; host phase
averages cover all decode steps. Timing events use external event-record nodes;
graph replay stays enabled. Events are read after the ordinary logits synchronization,
without adding layer-by-layer stream waits. Profiling itself adds event nodes and
host calls, so its tok/s is diagnostic and must not be presented as an optimized
speed measurement. The user has now collected the timing reports above; native
output/logit parity against the reference engine remains unverified.

The attention breakdown (native JSON `profile_version: 2`) uses four timing
boundaries per sampled layer rather than two: begin, after QKV/cache, after
attention core, end. It retains the aggregate fields and the same 1/17/33/...
sampling schedule. It adds no event nodes to ordinary decode steps or normal
non-profile mode. Extra sampled event nodes can perturb timings; compare stage
shares on this version, not a few-percent throughput change against the older
profiler. No inference speedup is implemented by this diagnostic change.
The Python log reader accepts older aggregate-only JSON and prints the extra
line only when the complete new stage field group is present.
Local checks for this diagnostic extension passed: all 33 HY-V3 Python tests,
480 existing regression tests (477 passed, 3 skipped for absent reference
tokenizer data), standalone C++ profile/router tests and the host `main.cpp`
syntax check. After removing only the two new timing-marker blocks, the
attention operations/order match the prior source. GPU timestamps and real
output parity still require device validation; no CUDA runtime, HIP or SYCL
build/runtime validation was available locally. Keep the PR a Draft.

The Windows CUDA build and compatibility CI passed at native revision
`f6a3a638` ([Windows EXE](https://github.com/pingpingjinping/PolyStrata/actions/runs/41831932345),
[CPU/Python](https://github.com/pingpingjinping/PolyStrata/actions/runs/41831932330)).
The user then reported these version-2 GPU sample averages on the same PC:

| Attention stage | GPU ms/step | Share of aggregate attention |
| --- | --- | --- |
| QKV + head norm/RoPE/KV store | 1.580 | 40.2% |
| Attention core | 1.373 | 34.9% |
| Output projection | 0.979 | 24.9% |
| Aggregate attention | 3.932 | 100% |

The stages sum to the reported aggregate. The request reported 1,320 output
tokens, 72.8 generation tok/s, `stop`, `cancel=False`, and 75.2% expert hits;
the profile had 1,319 decode steps and 83 GPU samples. CPU expert host phase
was 5.847 ms/step, router host wait 5.216, decode wall 13.748, and collection
0.002. The host fields overlap GPU work and must not be summed with the table.
QKV preparation plus output is about 65% of sampled attention time, but QKV
preparation also includes norms, RoPE, cache stores and event overhead: this
does not attribute all of it to matrix multiplication. The core alone is about
35%, so attention-kernel work is only one optimization candidate.

These are diagnostic timestamps, not an unprofiled throughput comparison.
The preceding 76.1 tok/s run had a different output count (1,322), expert hit
rate (73.6%) and no profiling. The new run cannot establish a regression or
quantify profiler overhead. Output IDs/text and exact request/config hashes
were not captured with these console reports.

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

### Step 3: inspect actual projection tensor formats (no new EXE)

Before selecting a matrix-kernel or launch-fusion experiment, inspect the
actual Q/K/V/output tensor types and dimensions. The whole file being called
Q4_K_M does not imply every projection has the same tensor format. Existing
paired/slots kernels have input, output, type and token-count constraints, so
an environment switch alone does not prove a particular path can run.

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\inspect_hy_v3_gguf.py "$HOME\Desktop\PolyStrata\Hy-MT2-30B-A3B-Q4_K_M.gguf" --projection-types
```

Send the four `attn_q.weight`, `attn_k.weight`, `attn_v.weight` and
`attn_output.weight` lines. The optional report groups tensor type and shape
across trunk layers (excluding MTP), rather than assuming layer zero represents
all layers. It reads only the GGUF header and uses no GPU; model weight bytes
are not loaded. `--json --projection-types` adds the same data under
`attention_projections`; without the new flag, the existing report is unchanged.
This is metadata for choosing the next experiment, not a speed/quality test.

The user's actual Q4_K_M header reported 48 trunk layers, no MTP, and:

| Projection | Stored shape (input, output) | Actual types across trunk layers |
| --- | --- | --- |
| Q | 2048, 4096 | Q4_K on all 48 |
| K | 2048, 512 | Q4_K on all 48 |
| V | 2048, 512 | Q4_K on 24; Q6_K on 24 |
| Output | 4096, 2048 | Q4_K on all 48 |

### Step 4: group single-token QKV launches (experimental, off by default)

`--qkv-slots` passes native `--hy-v3-qkv-slots`. The opt-in path views the Q
matrix as eight consecutive 512-row pieces, plus K and (where Q4_K) V, and
passes their existing device pointers to the existing MMVQ slots kernel. The
shared input remains the original Q8_1 activation. No model weights are copied,
repacked or re-quantized, and no new model-weight VRAM is allocated. Output
projection and attention core are unchanged.

On Q4_K V layers, three projection launches become one with ten slots. On
Q6_K V layers, Q/K use nine Q4 slots and V keeps its original Q6_K launch,
so three become two. Q/K normalization follows the completed projections;
their input values and RoPE positions are unchanged. Only single-token evals
use the grouping (including a final one-token prefill eval); larger prompt
batches keep the original path. Unsupported Q/K/V formats, other model
families, HIP, multiple GPUs/sequences, window other than 1 or lookup other
than 0 are rejected when this option is requested. The default-off attention
operations/order match the prior source after removing only the opt-in dispatch.

**The GPU check must pass before testing this option.** Download the new Windows
artifact and put **both** `strata-poly.exe` and `poly_hy_qkv_slots_test.exe` in
`Desktop\PolyStrata`. Close the model server first. Then:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --check-qkv-slots
```

This runs a small GPU-only projection test with synthetic real ggml Q4_K/Q6_K
weights in the actual 2048/4096/512 geometry. It compares every Q/K/V float bit
with separate native launches, checks output guards, and reuses captured CUDA
graphs with zero, small, normal and larger inputs. It is designed to check
24 cases / 122,880 projection values in eager execution and graph replay.
CI compiles the EXE but has no GPU and does **not** run or certify the test.
No real model/server is loaded; the Python runner supplies private CUDA DLL
paths and returns the native exit code. Missing EXE, missing GPU, CUDA errors
or bit differences fail explicitly. If it fails, leave `--qkv-slots` off and
report the failure.

The web launcher also runs this check before loading a model whenever
`--qkv-slots` is enabled, and stops if it fails. This does not establish whole
model/reference-engine parity; compare translation output as well.

After a PASS, compare this same new executable with the option off and on:

```powershell
# First baseline, with the option absent.
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --threads 6 --temperature 0 --cpu-down32 --profile-decode
# Stop the server; repeat with only --qkv-slots added.
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --threads 6 --temperature 0 --cpu-down32 --profile-decode --qkv-slots
```

Use a new chat, saved web Temperature 0, the same 100 Japanese lines and no
expert learning. Report the `done`, profile and attention-detail lines for
each. If promising, repeat without profiling before claiming a throughput
gain. This targets QKV preparation, not the output projection or all of the
65% preparation/output share. Reduced launch count is not proof of speedup;
different scheduling can also be slower. No GPU result or speedup for this
experiment has been measured in the development environment.
Local validation passed all 40 Hy Python tests and the existing 480-test
regression suite (477 passed, 3 skipped for missing reference tokenizer data).
The grouped helper header and `main.cpp` passed host syntax checks, and a
CPU-only CMake configure succeeded. These are not GPU parity results. CUDA
compilation is delegated to the Windows CI build; HIP/SYCL validation was not
available locally and this PR remains a Draft.

#### User device results (2026-11-05 KST)

At native revision `a633c047`, the Windows build and compatibility CI passed:
[EXEs](https://github.com/pingpingjinping/PolyStrata/actions/runs/41840106604),
[CPU/Python](https://github.com/pingpingjinping/PolyStrata/actions/runs/41840106420).
The user then ran the GPU checker on **NVIDIA GeForce RTX 5060**, slots warp
mode 0, and reported **PASS: 24 cases / 122,880 output values bit-identical,
eager + graph replay**. This is actual device evidence for the synthetic
projection check, not whole-model/reference-engine parity.

Same-new-EXE reports with batch 2048, 6 threads, CPU down32 enabled, context
4096, requested/web Temperature 0, expert learning off and the same 100-line
input as instructed:

| QKV grouping | Profiling | Generation tok/s | CPU host ms/step | Router wait ms/step | Decode wall ms/step |
| --- | --- | --- | --- | --- | --- |
| Off | On | 75.6 | 5.970 | 4.652 | 13.245 |
| On | On | 77.0 | 5.901 | 4.596 | 12.996 |
| Off | Off | 75.1 | — | — | — |
| On | Off | 76.4 | — | — | — |

| QKV grouping | QKV + preparation GPU ms/step | Attention core GPU ms/step | Output GPU ms/step | Aggregate attention GPU ms/step |
| --- | --- | --- | --- | --- |
| Off | 1.555 | 1.328 | 0.934 | 3.818 |
| On | 1.476 | 1.279 | 0.922 | 3.677 |

All four requests reported **1,322 output tokens, 73.6% expert hits, stop and
cancel=False**. The two profiled runs each had 1,321 decode steps, 83 GPU
samples and 0.002 ms/step collection time. The GPU stage sum can differ from
the aggregate by 0.001 ms because the console rounds each field separately.
Host phases overlap GPU work and must not be added to the attention values.

The unprofiled pair reports **+1.7% generation throughput** for QKV grouping;
the profiled pair reports +1.9%, with QKV preparation about 5.1% shorter.
Other GPU intervals also shortened, so run variation/clocks can contribute:
do not attribute every observed difference to the grouped kernel. These are
short single pairs, not repeated medians or statistical evidence. Output
token counts alone do not prove equal generated text/IDs. Keep the feature
opt-in and retain the full-model/reference parity gate. Do not add this
percentage to the earlier CPU experiment from a different native build.

The practical combination re-enables `--learn-experts` while retaining
6 threads, CPU down32 and QKV grouping, with profiling off. A pre-existing
`.hy-v3-web/hy-mt2-experts.bin` is read on restart; if absent, the first run only
starts collecting the profile, and another restart is needed to apply it.
This changes cache ranking and is separate from the QKV-only comparison above.

The user subsequently reported:

| Run with cache learning requested | Output tokens | Generation tok/s | Expert hits | Finish |
| --- | --- | --- | --- | --- |
| First combined run | 1324 | 77.5 | 74.1% | stop, cancel=False |
| Repeat after requested restart | 1325 | 77.3 | 74.1% | stop, cancel=False |

The first run's startup log confirmed GPU QKV parity PASS, CPU down32 and QKV
slots ON, batch 2048, context 4096 and config temperature 0. The initial
profile-file existence result and native profile-read confirmation were not
provided, so its prior state is unknown. Both short runs are around 77 tok/s
on this same 100-line task. Compared with the single learning-off result
(76.4 tok/s, 73.6% hits), this is a small nominal difference, not an isolated
or statistically established cache-learning speedup. Output counts differ,
and translated text/request hashes were not captured. Stop repeating this
same short input here; the next validation should use real subtitle work,
varying consecutive requests, cancellation/restart and memory monitoring.
Neither these two runs nor synthetic projection parity establish those
stability gates or whole-model numerical correctness.

Provisional normal-use command for this user's PC (not a global preset):

```powershell
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py --prompt-batch 2048 --threads 6 --temperature 0 --cpu-down32 --qkv-slots --learn-experts
```

Save web Temperature 0 as well. Leave profiling off for normal use. These
measurement/documentation updates require no new executable; engine/launcher
defaults remain unchanged and experimental controls are still opt-in.

### Step 5: consecutive-request and disconnect/recovery check (Python only)

Keep the current tuned server running, with profiling off. The new opt-in
`--stability` mode in `tools/hy_v3_bench.py` uses its loopback API and requires
no new EXE, configuration changes or model restart. Do not send requests from
another client during this check: request accounting must belong to this run.

Save the 100 numbered Japanese source lines as UTF-8
`.hy-v3-web/stability-input.txt` under the source checkout. The source and
result files stay local/ignored; no text is committed or uploaded by this tool.
In a second PowerShell window:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_bench.py --input .\.hy-v3-web\stability-input.txt --stability --label hy-5060-stability --output ".\.hy-v3-web\stability\hy-$(Get-Date -Format yyyyMMdd-HHmmss).json" --run
```

Without `--run`, this prints a plan with no network traffic or result file.
Normal speed mode and its cold/warm accounting remain available without
`--stability`; `--runs` only controls that speed mode.

For 100 source lines, the ten-request sequence is **A/B/A/C/D/E/A/F/C/A**:

| Label | Numbered source excerpt |
| --- | --- |
| A | All 100 lines |
| B | First half (1–50) |
| C | Second half (51–100) |
| D | Second quarter (26–50) |
| E | Third quarter (51–75) |
| F | Fourth quarter (76–100) |

Each A repeat uses the same complete prompt and the original IDs. Continuation
lines stay with their numbered block; text before the first ID is excluded
from the stability excerpts. Other input lengths are split proportionally;
at least four unique numbered lines are required. This exercises different
prompt lengths and cache switching/restoration within one source. It is not
a complete episode, a multi-episode corpus or a numerical-parity check.

The tool then disconnects a full-input stream after 32 output characters,
waits for idle, and sends a fresh-prefix recovery request: **12 HTTP requests,
11 completed translations plus one cancellation probe**. Cancellation only
counts when the server records `cancel`/`disconnect`; a response that finishes
before cancellation can be observed is not a pass. Prompt-reading cancellation
and the web UI stop button are not separately exercised by this probe.

Every completed translation must finish with `stop`, receive the final SSE
marker, include Hangul, and preserve all expected IDs in order without missing,
duplicate or extra IDs. Meaning still needs manual review. Reports record
input/prompt hashes and full outputs, timings, request metrics, active machine
samples, and before/after-idle machine snapshots. Checkpoints are saved after
every request, so earlier successful records survive a later error. A transport,
idle or accounting error stops the sequence and saves the failed index too.

The schema-2 summary reports planned/recorded counts, completed translations,
failed indices, observed cancellation and recovery, and
`all_request_checks_passed`. A complete successful run has 12 recorded requests,
11 completed translations, no failed indices, and all three request-check
booleans true. Individual request/ID failures do not certify engine failure;
inspect the recorded reason and output.

Memory summaries are **system RAM used (GiB)** and **whole-device GPU memory
used (MiB)** from `/v1/status`, not engine RSS. First/last idle values are taken
after completed requests; sampled peaks and raw snapshots are retained.
Missing readings remain null. Sampling can lag and normal model/graph/prompt
cache warm-up can grow memory. `memory_review_required` therefore stays true:
there is no automatic leak threshold or memory-stability pass. Inspect the
trend after warm-up, alongside other applications' activity.

This mode does not stop/restart the process, force low memory, or change its
context. `--long-input PATH` can append a separate numbered request within
the configured budget; naturally returned context/memory errors are recorded,
not silently treated as successful long translation. Long real subtitle work,
process restart/release and bounded allocation failure/recovery remain separate
device gates. Shortened/reused prompts are not speed-benchmark samples.

Local validation: all 48 Hy tests passed, including real loopback HTTP streaming
with MockEngine, idle telemetry capture, sequence/hash checks, failed-cancel and
incomplete-plan rejection, missing telemetry, partial-report persistence and
no-network dry runs. These are client/offline checks. The real-device results
below show that the complete stability gate has not passed.

### Device results: repeatable missing lines, 2026-11-05 KST

The user supplied two complete 12-request reports on RTX 5060 8 GB / Windows /
32 GB RAM, with the same input and corresponding prompt hashes. Both runs used
batch 2048, 6 threads, context 4096, temperature 0, expert learning enabled and
profiling off. In the first run CPU down32 and QKV slots were enabled (confirmed by the
startup log/config); the second followed the command omitting both controls.
The second launch log/config was not attached, so its flags are user-run context,
not verified by the report label or engine version.

| Request | Source IDs | Tuned output lines | Stock-command output lines |
| --- | --- | --- | --- |
| 4 (C) | 051–100 | 35/50 | 36/50 |
| 6 (E) | 051–075 | 10/25 | 11/25 |
| 9 (C repeat) | 051–100 | 35/50 | 39/50 |

Every failed response received final SSE completion and finished with `stop`,
well below the requested 2048 output-token budget. All six failed responses
omitted 053–062 and 065; other missing IDs varied. Each run's four full-input A
requests and the full-input recovery preserved all 100 IDs. Disconnect and the
following recovery passed in both reports. This establishes an output-coverage
failure in the tested combination; it does not identify a model-only cause or
prove whole-model engine correctness. Turning off the two recent controls as
instructed did not remove the reported pattern.

For the four complete A requests only, tuned speeds were 84.9/82.4/85.2/77.7
tok/s (median 83.65); stock-command speeds were 81.8/83.4/83.6/81.8 (median 82.6).
These consecutive runs are not an alternating controlled performance study.
Incomplete C/E outputs are excluded from these medians. System RAM first/last
idle was 24.9/22.8 GiB and 24.3/21.5 GiB; whole-device sampled GPU peaks were
8083 and 7942 MiB. These readings are not a leak-free or OOM-recovery verdict.

### Step 6: isolate numbering and instruction effects (Python only)

`tools/hy_v3_line_probe.py` selects IDs 051–075 from the original input by
default. It sends four variants twice (eight requests), reversing the order in
the second round:

| Variant | IDs | Instruction |
| --- | --- | --- |
| original | 051–075 | Same as stability E |
| renumbered | 001–025 | Same as stability E |
| strict-original | 051–075 | Adds exact count/range and no omission/merging |
| strict-renumbered | 001–025 | Same strengthened rules with matching range |

The baseline prefix is exactly `[stability-E]` in every case; no variant-specific
tag is added. Renumbering replaces only numbered block-start IDs, preserving
dialogue and continuation lines. Reports map missing IDs back to the original
source, retain prompts/full responses/hashes and checkpoint after each request.
The source file, launcher, inference math and ordinary benchmark are unchanged.
No new EXE is needed. Keep one server/settings combination throughout; avoid
other generation requests. In a second PowerShell window:

```powershell
cd "$HOME\PolyStrata-HyV3-Test"
git pull
& "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_line_probe.py --input .\.hy-v3-web\stability-input.txt --output ".\.hy-v3-web\stability\line-probe-$(Get-Date -Format yyyyMMdd-HHmmss).json" --run
```

Without `--run`, the client only validates the excerpt and shows a plan, with
no network traffic or output file. `--start-id`/`--end-id` select another range;
both endpoints must exist and input IDs must be unique/increasing. The same
stream completion and strict ID checks from the benchmark apply. Transport or
accounting errors stop and preserve prior records. This diagnostic does not
cancel/restart or change the server. Exact repeats can reuse prompt state;
cache/order effects and numerical parity need separate investigation. Passing
after a prompt/ID change alone does not establish whether the model, template
or engine caused earlier omissions. Real-device probe results are pending.

Local validation: all 53 Hy Python tests passed, including five new probe tests
for exact excerpt/continuation preservation, independent ID/instruction changes,
repeated prompt identity, missing-ID mapping, no-network dry runs and partial
result persistence. CI now runs the new probe tests explicitly. These are client
checks without actual model inference; the existing benchmark file is unchanged.

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
