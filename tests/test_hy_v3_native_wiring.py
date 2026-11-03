"""Source wiring checks only: NOT a CUDA compile or model parity test."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HyV3NativeWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = (ROOT / "src/poly/engine.cpp").read_text(encoding="utf-8")
        cls.native = (ROOT / "src/poly/hy_v3.cpp").read_text(encoding="utf-8")
        cls.header = (ROOT / "src/poly/hy_v3.hpp").read_text(encoding="utf-8")
        cls.cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    def test_wired_in_model_factory(self):
        self.assertIn('#include "hy_v3.hpp"', self.engine)
        self.assertIn('if (name == "hy_v3") return std::make_unique<HyV3>();', self.engine)
        self.assertIn("class HyV3 final : public Engine", self.header)

    def test_all_engine_targets_include_native_source(self):
        self.assertEqual(self.cmake.count("src/poly/hy_v3.cpp"), 3)

    def test_expected_hy_mt2_geometry(self):
        for fragment in ("n_layer_ == 48", "n_embd_ == 2048", "n_expert_ == 128",
                         "n_used_ == 8", "n_head_ == 32", "n_kv_ == 4",
                         "n_dense_ == 1", "n_ff_exp_ == 768"):
            self.assertIn(fragment, self.native)

    def test_router_and_moe_path(self):
        for fragment in ('load_moe(il, true)', '"exp_probs_b"', '"exp_probs_b.bias"',
                         'route_rows(', 'w_norm_', 'w_scale_', 'moe_router(L, n)',
                         'HyV3::moe_shared('):
            self.assertIn(fragment, self.native)

    def test_attention_and_mtp_guard(self):
        for fragment in ("B.swa = false;", "B.rope = true;", "head_norm_rope(",
                         "attend_batch(", "attend(", "opt_.draft = false;"):
            self.assertIn(fragment, self.native)
        self.assertNotIn('mat("attn_sinks.weight"', self.native)
        self.assertNotIn('fp(B.sinks)', self.native)


if __name__ == "__main__":
    unittest.main()
