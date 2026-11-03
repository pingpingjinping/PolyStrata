"""Tests for the header-only HY-V3 GGUF preflight (no actual model required)."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "inspect_hy_v3_gguf", Path(__file__).resolve().parents[1] / "tools" / "inspect_hy_v3_gguf.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def fixture(path, alias="bare"):
    meta = {
        "general.architecture": "hy_v3", "hy_v3.block_count": 48,
        "hy_v3.expert_count": 128, "hy_v3.expert_used_count": 8,
        "hy_v3.embedding_length": 2048,
        "hy_v3.attention.head_count": 32,
        "hy_v3.attention.head_count_kv": 4,
        "tokenizer.ggml.tokens": ["a", "b"],
    }
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for layer in range(48):
        root = f"blk.{layer}."
        names += [root + tail for tail in
                  ("attn_norm.weight", "attn_output.weight", "attn_q_norm.weight",
                   "attn_k_norm.weight", "ffn_norm.weight", "attn_q.weight",
                   "attn_k.weight", "attn_v.weight")]
        if layer == 0:
            names += [root + tail for tail in
                      ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")]
        else:
            names += [root + tail for tail in
                      ("ffn_gate_inp.weight", "ffn_down_exps.weight",
                       "ffn_gate_up_exps.weight", "ffn_gate_shexp.weight",
                       "ffn_up_shexp.weight", "ffn_down_shexp.weight")]
            names.append(root + ("exp_probs_b" if alias == "bare" else "exp_probs_b.bias"))

    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(names), len(meta)))
    for key, val in meta.items():
        out += s(key)
        if isinstance(val, str):
            out += struct.pack("<I", 8) + s(val)
        elif isinstance(val, int):
            out += struct.pack("<IQ", 10, val)
        else:
            out += struct.pack("<IIQ", 9, 8, len(val))
            for item in val:
                out += s(item)
    for name in names:
        out += s(name) + struct.pack("<IQQIQ", 2, 2048, 2048, 0, 0)
    path.write_bytes(out)


class PreflightTests(unittest.TestCase):
    def test_hy_v3_with_bare_bias(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.gguf"
            fixture(p, "bare")
            meta, tensors = mod.read_header(p)
            r = mod.inspect(meta, tensors)
            self.assertTrue(r["ok"], r["errors"])
            self.assertEqual(r["expert_bias_names"]["bare"], 47)
            self.assertEqual(r["dense_layers"], 1)
            self.assertEqual(r["moe_layers"], 47)
            self.assertEqual(len(tensors), r["tensors"])

    def test_hy_v3_with_bias_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.gguf"
            fixture(p, "bias")
            meta, tensors = mod.read_header(p)
            r = mod.inspect(meta, tensors)
            self.assertTrue(r["ok"], r["errors"])
            self.assertEqual(r["expert_bias_names"]["bias"], 47)

    def test_wrong_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.gguf"
            fixture(p)
            meta, tensors = mod.read_header(p)
            meta["general.architecture"] = "qwen35moe"
            self.assertFalse(mod.inspect(meta, tensors)["ok"])

    def test_missing_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.gguf"
            fixture(p)
            meta, tensors = mod.read_header(p)
            del tensors["blk.1.attn_q_norm.weight"]
            self.assertFalse(mod.inspect(meta, tensors)["ok"])

    def test_truncated_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.gguf"
            p.write_bytes(b"GGUF")
            with self.assertRaises(mod.GGUFError):
                mod.read_header(p)


if __name__ == "__main__":
    unittest.main()
