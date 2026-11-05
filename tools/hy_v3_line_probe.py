#!/usr/bin/env python3
"""Compare line IDs and instructions on one excerpt using an existing server.

This is an output-coverage diagnostic, not a speed or numerical-parity test.
No engine/config changes; reports and full response text stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import hy_v3_bench as bench


INSTRUCTION = "Translate every numbered line into Korean. Preserve all [NNN] IDs in order. Output only the translation."
TAG = "stability-E"  # Reproduce the earlier 25-line request without adding a variant-specific prefix.
ID = re.compile(r"(?m)^[ \t]*\[(\d+)\]")


def excerpt(source, start_id, end_id):
    matches = list(ID.finditer(source))
    ids = [int(m[1]) for m in matches]
    if ids != sorted(set(ids)):
        raise ValueError("input IDs must be unique and increasing")
    if start_id > end_id or start_id not in ids or end_id not in ids:
        raise ValueError("both requested endpoint IDs must exist, in increasing order")
    first, last = ids.index(start_id), ids.index(end_id)
    end = matches[last + 1].start() if last + 1 < len(matches) else len(source)
    return source[matches[first].start():end]


def probe_jobs(source):
    original_ids = ID.findall(source)
    # Replace only line-start IDs: dialogue numbers and continuation lines stay intact.
    next_id = iter(range(1, len(original_ids) + 1))
    numbered = ID.sub(lambda m: m[0].replace(m[1], f"{next(next_id):03d}", 1), source)
    variants = {}
    for name, text, strict in (("original", source, False), ("renumbered", numbered, False),
                               ("strict-original", source, True), ("strict-renumbered", numbered, True)):
        ids = ID.findall(text)
        instruction = INSTRUCTION
        if strict:
            instruction += (f" There are exactly {len(ids)} input lines, from [{ids[0]}] to [{ids[-1]}]."
                            " Return one translated line for each input line, including short utterances."
                            " Do not skip, merge, summarize, or renumber any line.")
        variants[name] = dict(kind=name, source=text,
                              prompt=f"[{TAG}] {instruction}\n\n{text}",
                              id_map=dict(zip(ids, original_ids)))
    order = list(variants)
    # Repeat every exact prompt twice, reversing order to expose some order effects.
    return [dict(variants[name], round=round_no) for round_no, names in
            ((1, order), (2, order[::-1])) for name in names]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def summarize(records, planned):
    groups = {}
    for r in records:
        group = groups.setdefault(r["kind"], dict(recorded=0, passed=0, missing_original_ids=[]))
        group["recorded"] += 1
        group["passed"] += bool(r.get("ok"))
        group["missing_original_ids"].append(r.get("missing_original_ids"))
    return dict(planned_requests=planned, recorded_requests=len(records), variants=groups,
                all_line_checks_passed=len(records) == planned and all(r.get("ok") for r in records),
                note="Line coverage only; prompt/numbering effects do not establish model or engine correctness.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="original UTF-8 numbered Japanese source")
    ap.add_argument("--start-id", type=int, default=51)
    ap.add_argument("--end-id", type=int, default=75)
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--run", action="store_true", help="send requests; otherwise show a plan only")
    args = ap.parse_args(argv)
    base = args.url.rstrip("/")
    url = urlparse(base)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        ap.error("use a private loopback HTTP endpoint")
    if min(args.start_id, args.end_id, args.max_tokens, args.timeout) <= 0:
        ap.error("IDs, max-tokens and timeout must be positive")
    try:
        full_source = args.input.read_text(encoding="utf-8-sig")
        source = excerpt(full_source, args.start_id, args.end_id)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    jobs = probe_jobs(source)
    if not args.run:
        print(f"Plan: {len(ID.findall(source))} lines, 4 variants x 2 rounds = {len(jobs)} requests."
              " No network traffic or output files. Add --run to test.")
        return 0
    try:
        status = bench.get(base, "/v1/status")
    except (OSError, ValueError) as exc:
        print(f"Cannot reach the running server at {base}: {exc}", file=sys.stderr)
        return 2
    report = dict(schema=1, mode="line-probe", input_sha256=sha(full_source),
                  excerpt_sha256=sha(source), start_id=args.start_id, end_id=args.end_id,
                  temperature=0, max_tokens=args.max_tokens, status_before=status, records=[],
                  limitations=["Not a speed, semantic-quality, or numerical-parity verdict.",
                               "Same server/settings; no automatic engine flag verification.",
                               "Prompt cache and request order may affect behavior; retain launcher config/log."])
    for index, job in enumerate(jobs, 1):
        try:
            record = bench.measure(base, status["model"], job["prompt"], job["source"],
                                   args.max_tokens, args.timeout, capture_idle_memory=True)
        except Exception as exc:
            record = dict(ok=False, error=str(exc))
        coverage = record.get("coverage") or {}
        record.update(request_index=index, kind=job["kind"], round=job["round"],
                      input_sha256=sha(job["source"]), prompt_sha256=sha(job["prompt"]),
                      prompt=job["prompt"], id_map=job["id_map"],
                      missing_original_ids=([job["id_map"][i] for i in coverage["missing_ids"]]
                                            if "missing_ids" in coverage else None))
        report["records"].append(record)
        report["summary"] = summarize(report["records"], len(jobs))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.output)
        print(f"line-probe {index}/{len(jobs)} {job['kind']} round {job['round']}: "
              f"{'OK' if record['ok'] else 'FAILED'}; "
              f"lines={coverage.get('output_lines', '?')}/{coverage.get('expected_lines', '?')}; "
              f"missing original IDs={record['missing_original_ids']}", flush=True)
        if record.get("error"):
            print(record["error"], file=sys.stderr)
            break
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {args.output.resolve()}")
    return int(not report["summary"]["all_line_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
