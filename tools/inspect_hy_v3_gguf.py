#!/usr/bin/env python3
"""Header-only HY-V3 GGUF inspection; never loads model weights into RAM.

Usage: python tools/inspect_hy_v3_gguf.py path/to/model.gguf [--json]
This is a preflight checker, not a PolyStrata backend or a numerical parity test.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

KINDS = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
MAX_TENSORS = 200_000
MAX_META = 2_000_000
MAX_STRING = 32 * 1024 * 1024
INTERESTING_PREFIX = ("hy_v3.", "general.")


class GGUFError(ValueError):
    pass


def read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise GGUFError(f"truncated GGUF header at offset {stream.tell()}")
    return data


def number(stream, fmt: str):
    return struct.unpack("<" + fmt, read_exact(stream, struct.calcsize(fmt)))[0]


def skip(stream, n: int):
    if n < 0:
        raise GGUFError("negative skip")
    stream.seek(n, 1)


def gguf_string(stream, keep: bool = True) -> str | None:
    n = number(stream, "Q")
    if n > MAX_STRING:
        raise GGUFError(f"GGUF string unexpectedly large: {n} bytes")
    if not keep:
        skip(stream, n)
        return None
    return read_exact(stream, n).decode("utf-8", errors="replace")


def typed_value(stream, kind: int, keep: bool = True):
    if kind in KINDS:
        fmt, size = KINDS[kind]
        if keep:
            return number(stream, fmt)
        skip(stream, size)
        return None
    if kind == 8:
        return gguf_string(stream, keep)
    if kind == 9:
        inner = number(stream, "I")
        length = number(stream, "Q")
        if length > 100_000_000:
            raise GGUFError(f"array unexpectedly large: {length}")
        if inner in KINDS:
            fmt, width = KINDS[inner]
            if not keep or length > 64:
                skip(stream, length * width)
                return None
            return [number(stream, fmt) for _ in range(length)]
        if inner in (8, 9):
            if length > 2_000_000:
                raise GGUFError(f"nested/strings array unexpectedly large: {length}")
            # Retain only small arrays; tokenizer arrays should be skipped.
            return [typed_value(stream, inner, keep and length <= 64) for _ in range(length)] if (keep and length <= 64) else _skip_array(stream, inner, length)
        raise GGUFError(f"unknown array element type {inner}")
    raise GGUFError(f"unknown GGUF metadata type {kind}")


def _skip_array(stream, kind: int, n: int):
    for _ in range(n):
        typed_value(stream, kind, False)
    return None


def read_header(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = number(stream, "I")
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} not supported by inspector")
        tensor_count = number(stream, "Q")
        meta_count = number(stream, "Q")
        if tensor_count > MAX_TENSORS or meta_count > MAX_META:
            raise GGUFError("header has unreasonable tensor/metadata count")

        meta = {"_gguf_version": version, "_tensor_count": tensor_count}
        for _ in range(meta_count):
            key = gguf_string(stream)
            kind = number(stream, "I")
            keep = key.startswith(INTERESTING_PREFIX)
            value = typed_value(stream, kind, keep)
            if keep:
                meta[key] = value

        tensors = {}
        for _ in range(tensor_count):
            name = gguf_string(stream)
            dims = number(stream, "I")
            if dims > 8:
                raise GGUFError(f"tensor {name} has {dims} dimensions")
            shape = [number(stream, "Q") for _ in range(dims)]
            ttype = number(stream, "I")
            offset = number(stream, "Q")
            if name in tensors:
                raise GGUFError(f"duplicate tensor: {name}")
            tensors[name] = {"shape": shape, "type": ttype, "offset": offset}
        return meta, tensors


def inspect(meta: dict, tensors: dict) -> dict:
    errors = []
    warnings = []
    arch = meta.get("general.architecture")
    if arch != "hy_v3":
        errors.append(f"expected general.architecture=hy_v3, got {arch!r}")
    prefix = "hy_v3."

    def required_int(key):
        x = meta.get(prefix + key)
        if type(x) is not int or x <= 0:
            errors.append(f"missing/invalid {prefix + key}")
            return None
        return x

    blocks = required_int("block_count")
    n_exp = required_int("expert_count")
    used = required_int("expert_used_count")
    hidden = required_int("embedding_length")
    heads = required_int("attention.head_count")
    kv_heads = required_int("attention.head_count_kv")
    nextn = meta.get(prefix + "nextn_predict_layers", 0)
    if type(nextn) is not int or nextn < 0:
        errors.append("invalid nextn_predict_layers")
        nextn = 0
    trunk = blocks - nextn if blocks is not None else None
    if trunk is not None and trunk <= 0:
        errors.append("nextn_predict_layers must be smaller than block_count")
    if trunk is not None and trunk != 48:
        warnings.append(f"HY-MT2-30B-A3B is expected to have 48 trunk layers; got {trunk}")
    if n_exp and used and used > n_exp:
        errors.append("expert_used_count exceeds expert_count")
    if heads and kv_heads and heads % kv_heads:
        errors.append("attention.head_count not divisible by head_count_kv")

    def need(name):
        if name not in tensors:
            errors.append(f"missing tensor: {name}")

    def need_either(names):
        if not any(n in tensors for n in names):
            errors.append("missing alternative tensor: " + " OR ".join(names))

    need("token_embd.weight")
    need("output_norm.weight")
    if "output.weight" not in tensors:
        warnings.append("output.weight absent: tied embeddings may be used")

    dense = 0
    moe = 0
    bias_aliases = {"bare": 0, "bias": 0, "missing": 0}
    if trunk is not None and 0 < trunk <= 200:
        for il in range(trunk):
            root = f"blk.{il}."
            for suffix in ("attn_norm.weight", "attn_output.weight",
                           "attn_q_norm.weight", "attn_k_norm.weight",
                           "ffn_norm.weight"):
                need(root + suffix)
            need_either([root + "attn_q.weight", root + "attn_qkv.weight"])
            if root + "attn_qkv.weight" not in tensors:
                need(root + "attn_k.weight")
                need(root + "attn_v.weight")
            if root + "ffn_gate_inp.weight" in tensors:
                moe += 1
                need(root + "ffn_down_exps.weight")
                need_either([root + "ffn_gate_up_exps.weight", root + "ffn_gate_exps.weight"])
                if root + "ffn_gate_up_exps.weight" not in tensors:
                    need(root + "ffn_up_exps.weight")
                for s in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
                    need(root + s)
                bare = root + "exp_probs_b"
                suffixed = root + "exp_probs_b.bias"
                if bare in tensors:
                    bias_aliases["bare"] += 1
                elif suffixed in tensors:
                    bias_aliases["bias"] += 1
                else:
                    bias_aliases["missing"] += 1
                    warnings.append(f"{root} expert selection bias absent: verify checkpoint and routing")
            else:
                dense += 1
                for s in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                    need(root + s)
        if dense != 1:
            warnings.append(f"HY-MT2-30B-A3B config expects 1 leading dense block; found {dense}")
        if moe != trunk - dense:
            errors.append("mismatched dense/MoE layer count")

    return {
        "ok": not errors, "architecture": arch, "trunk_layers": trunk,
        "metadata_block_count": blocks, "mtp_layers": nextn,
        "experts": n_exp, "experts_per_token": used, "hidden": hidden,
        "attention_heads": heads, "kv_heads": kv_heads, "dense_layers": dense,
        "moe_layers": moe, "expert_bias_names": bias_aliases,
        "tensors": len(tensors), "errors": errors, "warnings": warnings,
        "note": "Header-only structural inspection; NOT an inference or numerical-correctness test.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        meta, tensors = read_header(args.gguf)
        report = inspect(meta, tensors)
    except (OSError, ValueError, struct.error, OverflowError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("HY-V3 GGUF preflight:", "PASS" if report["ok"] else "FAIL")
        for k in ("architecture", "trunk_layers", "mtp_layers", "experts",
                  "experts_per_token", "dense_layers", "moe_layers", "expert_bias_names"):
            if k in report:
                print(f"  {k}: {report[k]}")
        for w in report.get("warnings", []):
            print("  WARN:", w)
        for e in report.get("errors", []):
            print("  ERROR:", e)
        print(report.get("note", ""))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
