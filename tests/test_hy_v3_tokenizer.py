"""Regression for Tencent Hy-MT2 GGUF tokenizer.ggml.pre=hunyuan-dense."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from strata_tokenizer import DEEPSEEK3_SEQUENCE, QWEN2_PATTERN, Tokenizer  # noqa: E402


class HunyuanDenseTest(unittest.TestCase):
    def test_hunyuan_dense_uses_three_stage_llamacpp_regex(self):
        tok = Tokenizer(["x"], [], [1], pre="hunyuan-dense")
        self.assertEqual(tok.pre_sequence, DEEPSEEK3_SEQUENCE)
        self.assertEqual(len(tok._seq), 3)
        self.assertNotEqual(tok.pre_pattern, QWEN2_PATTERN)
        self.assertEqual(tok._seq[0].findall("2026年11月4日"), ["202", "6", "11", "4"])

    def test_gui_accepts_hunyuan_dense(self):
        s = (ROOT / "tools" / "hy_v3_translate_gui.py").read_text(encoding="utf-8")
        self.assertIn('"hunyuan-dense"', s)
        self.assertIn('pre not in ("hunyuan", "hunyuan-dense", "qwen2")', s)


if __name__ == "__main__":
    unittest.main()
