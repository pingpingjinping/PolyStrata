"""Tests for the header-only HY-V3 GGUF preflight (no actual model required)."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
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
    def test_projection_formats_count_mixed_trunk_layers_and_exclude_mtp(self):
        tensors = {}
        for il in range(3):
            tensors[f"blk.{il}.attn_q.weight"] = dict(type=12, shape=[2048, 4096])
        tensors["blk.1.attn_q.weight"]["type"] = 14
        tensors["blk.3.attn_q.weight"] = dict(type=8, shape=[2048, 4096])  # MTP
        tensors["blk.0.attn_k.weight"] = dict(type=999, shape=[2048, 512])
        result = mod.projection_types(tensors, 3)
        self.assertEqual(result["attn_q.weight"], [
            dict(type_id=12, type_name="Q4_K", shape=[2048, 4096], layer_count=2),
            dict(type_id=14, type_name="Q6_K", shape=[2048, 4096], layer_count=1)])
        self.assertEqual(result["attn_k.weight"][0]["type_name"], "GGML type 999")
        self.assertEqual(result["attn_v.weight"], [])
        with self.assertRaises(mod.GGUFError):
            mod.projection_types(tensors, 1_000_000)

    def test_projection_report_is_opt_in_and_uses_header_only_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fake.gguf"
            fixture(p)  # no weight data follows the header
            command = [sys.executable, str(Path(mod.__file__)), str(p), "--json"]
            baseline = json.loads(subprocess.check_output(command, text=True))
            self.assertNotIn("attention_projections", baseline)
            enabled = json.loads(subprocess.check_output(command + ["--projection-types"], text=True))
            groups = enabled.pop("attention_projections")
            self.assertEqual(enabled, baseline)
            self.assertEqual(groups["attn_q.weight"], [
                dict(type_id=0, type_name="F32", shape=[2048, 2048], layer_count=48)])

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
