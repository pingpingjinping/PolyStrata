"""Display native HY-V3 diagnostic JSON without changing the engine's stdout protocol."""
import json
from pathlib import Path

PREFIX = "strata hy-v3 profile: "


def format_profile(line):
    if not line.startswith(PREFIX):
        return None
    try:
        data = json.loads(line[len(PREFIX):])
        steps = int(data["decode_steps"])
        if steps == 0:
            return "[hy-v3 profile] no decode model step (the first output comes from prefill)"
        fields = [("attention GPU", "attention_gpu_ms_per_step"),
                  ("CPU expert host phase", "cpu_experts_host_ms_per_step"),
                  ("router host wait", "router_wait_host_ms_per_step"),
                  ("decode wall", "decode_wall_ms_per_step"),
                  ("profile collection", "collect_host_ms_per_step")]
        times = " | ".join(f"{label}={float(data[key]):.3f} ms/step" for label, key in fields)
        samples = data.get("gpu_samples", "unknown")
        return f"[hy-v3 profile] {steps} decode steps, {samples} GPU samples | {times} | overlapping timings; diagnostic throughput"
    except (ValueError, TypeError, KeyError):
        return "[hy-v3 profile] invalid diagnostic line; inspect the native log"


def follow_profile(path: Path, stop, offset=0):
    while not stop.is_set():
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                while not stop.is_set():
                    line = stream.readline()
                    if not line:
                        stop.wait(.2)
                        continue
                    shown = format_profile(line)
                    if shown:
                        print(shown, flush=True)
        except OSError:
            stop.wait(.2)
