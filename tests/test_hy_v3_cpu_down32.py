"""Check that experimental CPU scheduling is opt-in and scoped to HY-V3."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hy_v3_web import build_config


class CpuDown32Test(unittest.TestCase):
    def test_flag_changes_only_the_requested_native_argument(self):
        p = Path("/example")
        kwargs = dict(exe=p / "engine", model=p / "model.gguf", tokenizer=p / "tok", runtime=p,
                      port=18081, context=4096, lib_dirs=[], temperature=0, threads=6)
        baseline = build_config(**kwargs)
        self.assertNotIn("--hy-v3-cpu-down32", baseline["args"])
        enabled = build_config(**kwargs, cpu_down32=True)
        self.assertEqual(enabled["args"], baseline["args"] + ["--hy-v3-cpu-down32"])
        self.assertEqual(enabled["sampling"], baseline["sampling"])

    def test_native_dispatch_retains_stock_default_and_rejects_other_models(self):
        header = (ROOT / "src/poly/engine.hpp").read_text(encoding="utf-8")
        engine = (ROOT / "src/poly/engine.cpp").read_text(encoding="utf-8")
        experts = (ROOT / "src/poly/cpu_experts.cpp").read_text(encoding="utf-8")
        self.assertIn("bool hy_v3_cpu_down32 = false", header)
        self.assertIn('o.hy_v3_cpu_down32 && name != "hy_v3"', engine)
        self.assertIn("constexpr int kRowsPerDownTask = 16", experts)
        self.assertIn("void cpu_experts_down32(", experts)


if __name__ == "__main__":
    unittest.main()
