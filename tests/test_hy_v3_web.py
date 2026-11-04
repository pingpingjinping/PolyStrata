"""Offline checks for Hy-MT2's standalone PolyStrata web launcher."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from hy_v3_web import build_config, cuda13_lib_dirs  # noqa: E402


class HyV3WebConfigTest(unittest.TestCase):
    def test_isolated_server_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            cfg = build_config(
                exe=root / "strata-poly.exe",
                model=root / "Hy-MT2-30B-A3B-Q4_K_M.gguf",
                tokenizer=root / "pack" / "tokenizer",
                runtime=root,
                port=18081,
                context=4096,
                lib_dirs=[str(root / "cuda13")],
                temperature=0.7,
            )
            self.assertEqual(cfg["exe"], str(root / "strata-poly.exe"))
            self.assertEqual(cfg["host"], "127.0.0.1")
            self.assertEqual(cfg["port"], 18081)
            self.assertEqual(cfg["arch"], "hy_v3")
            self.assertEqual(cfg["sampling"]["temperature"], 0.7)
            self.assertEqual(cfg["reasoning_effort"], "none")
            self.assertIn("--serve", cfg["args"])
            self.assertEqual(cfg["args"][cfg["args"].index("--model") + 1],
                             str(root / "Hy-MT2-30B-A3B-Q4_K_M.gguf"))
            self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "4096")
            self.assertIn("--no-draft", cfg["args"])
            self.assertTrue(cfg["tokenizer"].endswith("tokenizer"))
            self.assertNotIn("8080", json.dumps(cfg))
            self.assertNotIn("0.0.0.0", cfg["host"])

    def test_cuda13_dll_search(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            venv = root / ".venv"
            exe = root / "strata-poly.exe"
            interpreter = venv / "Scripts" / "python.exe"
            dll = venv / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
            dll.mkdir(parents=True)
            (dll / "cublas64_13.dll").touch()
            self.assertIn(str(dll), cuda13_lib_dirs(exe, interpreter))
            (dll / "cublas64_13.dll").unlink()
            self.assertNotIn(str(dll), cuda13_lib_dirs(exe, interpreter))

    def test_server_native_architecture_recognized(self):
        contents = (ROOT / "serve" / "models.py").read_text(encoding="utf-8")
        self.assertIn('"hy_v3": "hy_v3"', contents)


if __name__ == "__main__":
    unittest.main()
