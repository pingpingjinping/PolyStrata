#!/usr/bin/env python3
"""Measure translation on a running, private PolyStrata endpoint. Standard library only.

Restart hy_v3_web.py with --prompt-batch 512/1024/2048 between runs. This
client never edits a config or changes a running engine's batch size.
Use the same UTF-8 numbered Japanese input for every model and batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.load(response)


def coverage(source, text):
    """Structural check only: never a translation-quality verdict."""
    pattern = r"(?m)^\s*\[(\d+)\]"
    expected = re.findall(pattern, source)
    actual = re.findall(pattern, text)
    wanted, got = set(expected), set(actual)
    return {"expected_lines": len(expected), "output_lines": len(actual),
            "missing_ids": sorted(wanted - got), "extra_ids": sorted(got - wanted),
            "duplicate_ids": sorted(i for i in got if actual.count(i) > 1),
            "numbered_lines_ok": bool(expected) and expected == actual,
            "has_hangul": bool(re.search(r"[가-힣]", text)),
            "semantic_quality": "manual review required"}


def request(base, model, prompt, max_tokens, timeout, cancel_after=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": True, "stream_options": {"include_usage": True},
            "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    result = {"text": "", "timings": None, "usage": None, "finish": None,
              "ttft_s": None, "cancelled_by_client": False, "done_received": False}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            if line[5:].strip() == "[DONE]":
                result["done_received"] = True
                break
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                result["text"] += piece
                if piece and result["ttft_s"] is None:
                    result["ttft_s"] = time.perf_counter() - start
                if choice.get("finish_reason"):
                    result["finish"] = choice["finish_reason"]
            for key in ("timings", "usage"):
                if event.get(key):
                    result[key] = event[key]
            if cancel_after is not None and len(result["text"]) >= cancel_after:
                result["cancelled_by_client"] = True
                break  # Closing the streaming response exercises disconnect cancellation.
    result["wall_s"] = time.perf_counter() - start
    return result


def idle(base, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        metrics = get(base, "/metrics")
        live = metrics.get("live", {})
        if live.get("state") == "idle" and not live.get("queued"):
            return metrics
        time.sleep(.2)
    raise RuntimeError("server did not return to idle after the request")


def measure(base, model, prompt, source, max_tokens, timeout, cancel_after=None):
    before = idle(base)
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            try:
                samples.append(get(base, "/v1/status", timeout=2).get("machine", {}))
            except (OSError, ValueError):
                pass  # Missing telemetry remains missing; it must not become zero usage.
            stop.wait(.5)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        result = request(base, model, prompt, max_tokens, timeout, cancel_after)
        after = idle(base)
    finally:
        stop.set()
        worker.join(3)
    # The endpoint must be dedicated to this benchmark. Reject metrics belonging
    # to another request rather than attaching its cache hit rate to our result.
    count = after.get("totals", {}).get("requests", 0) - before.get("totals", {}).get("requests", 0)
    if count != 1 or not after.get("requests"):
        raise RuntimeError("request accounting mismatch: another client may be using this server")
    result["engine_request"] = after["requests"][0]
    result["coverage"] = coverage(source, result["text"])
    result["machine_samples"] = samples
    result["ok"] = (result["cancelled_by_client"] and result["engine_request"].get("finish") in ("cancel", "disconnect")
                    if cancel_after is not None else
                    bool(result["text"]) and result["done_received"] and result["finish"] == "stop"
                    and result["coverage"]["numbered_lines_ok"] and result["coverage"]["has_hangul"])
    return result


def summary(records):
    """Only completed cold requests count toward the primary throughput median."""
    rows = [r for r in records if r.get("kind") == "cold" and r.get("ok")]

    def median(values):
        values = [v for v in values if isinstance(v, (float, int))]
        return statistics.median(values) if values else None

    return {"completed_cold_runs": len(rows),
            "prompt_tok_s": median((r.get("timings") or {}).get("prompt_per_second") for r in rows),
            "generation_tok_s": median((r.get("timings") or {}).get("predicted_per_second") for r in rows),
            "wall_s": median(r.get("wall_s") for r in rows),
            "expert_vram_hit_rate": median(r.get("engine_request", {}).get("hit_rate") for r in rows),
            "note": "Engine expert hit_rate is request-wide; warm runs and truncated outputs are excluded."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--input", type=Path, required=True, help="UTF-8 text with [001] Japanese dialogue lines")
    ap.add_argument("--long-input", type=Path, help="optional longer numbered text for a separate stability request")
    ap.add_argument("--label", required=True, help="e.g. hy-q4-b512 or gemma4-baseline")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--cancel", action="store_true", help="disconnect after 32 output characters, then probe recovery")
    ap.add_argument("--run", action="store_true", help="send requests; otherwise print a plan without connecting")
    args = ap.parse_args(argv)
    base = args.url.rstrip("/")
    url = urlparse(base)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        ap.error("use a private loopback HTTP endpoint")
    if min(args.runs, args.max_tokens, args.timeout) <= 0:
        ap.error("runs, max-tokens and timeout must be positive")
    source = args.input.read_text(encoding="utf-8-sig")
    if not re.search(r"(?m)^\s*\[\d+\]", source):
        ap.error("input must contain numbered [001] dialogue lines")
    if not args.run:
        print(f"Plan: {args.label}, {args.runs} cold + warm pairs; cancel={args.cancel}; "
              f"long={bool(args.long_input)}. No network traffic or output files. Add --run to measure.")
        return 0
    status = get(base, "/v1/status")
    model = status["model"]
    report = {"schema": 1, "label": args.label, "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "max_tokens": args.max_tokens, "temperature": 0, "status_before": status,
              "records": [], "limitations": ["Translation meaning needs manual review.",
              "Label does not verify batch size: retain the matching launcher log/config.",
              "OOM and process restart need separate real-device tests."]}
    # A fixed tag per run gives the same test corpus across batch sizes. Placing
    # it first defeats long prefix reuse; verify cache_n rather than assuming.
    instruction = "Translate every numbered line into Korean. Preserve all [NNN] IDs in order. Output only the translation."
    def prompt(text, tag):
        return f"[{tag}] {instruction}\n\n{text}"

    def save():
        report["summary"] = summary(report["records"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.output)

    jobs = []
    for run in range(args.runs):
        p = prompt(source, f"bench-{run}")
        jobs.extend([("cold", p, source, None), ("warm", p, source, None)])
    if args.long_input:
        text = args.long_input.read_text(encoding="utf-8-sig")
        jobs.append(("long", prompt(text, "long"), text, None))
    if args.cancel:
        jobs.extend([("cancel", prompt(source, "cancel"), source, 32),
                     ("after-cancel", prompt(source, "recovery"), source, None)])
    failed = False
    for kind, p, text, cancel in jobs:
        try:
            record = measure(base, model, p, text, args.max_tokens, args.timeout, cancel)
            # At most a short common prefix is expected between different runs.
            timing = record.get("timings") or {}
            if kind == "cold" and timing.get("cache_n", 0) > 64:
                record.update(ok=False, error="cold request reused more than 64 tokens; restart before measuring")
        except Exception as exc:
            record = {"ok": False, "error": str(exc)}
        record["kind"] = kind
        report["records"].append(record)
        failed |= not record["ok"]
        save()  # Keep successful results even if the next request fails or OOMs.
        print(f"{args.label} {kind}: {'OK' if record['ok'] else 'FAILED'}", flush=True)
        if record.get("error"):
            break
    print(json.dumps(report["summary"], indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
