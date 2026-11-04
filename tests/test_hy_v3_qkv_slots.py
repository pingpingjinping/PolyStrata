"""Opt-in configuration and GPU-check transport; GPU math runs in the bundled EXE."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hy_v3_web import build_config, main, run_qkv_check


class QkvSlotsTest(unittest.TestCase):
    def test_grouped_projections_are_opt_in_without_other_config_changes(self):
        root = Path("/example")
        kwargs = dict(exe=root / "strata-poly.exe", model=root / "model.gguf", tokenizer=root / "tok",
                      runtime=root, port=18081, context=4096, lib_dirs=[], temperature=0,
                      threads=6, cpu_down32=True, profile_decode=True)
        stock = build_config(**kwargs)
        self.assertNotIn("--hy-v3-qkv-slots", stock["args"])
        enabled = build_config(**kwargs, qkv_slots=True)
        self.assertEqual(enabled.pop("args"), stock.pop("args") + ["--hy-v3-qkv-slots"])
        self.assertEqual(enabled, stock)

    def test_checker_uses_private_dll_path_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-poly.exe"
            checker = exe.with_name("poly_hy_qkv_slots_test.exe")
            checker.touch()
            with patch.dict(os.environ, {"PATH": "stock-path"}), patch("hy_v3_web.subprocess.run") as run:
                run.return_value.returncode = 7
                self.assertEqual(run_qkv_check(exe, ["cuda-bin"]), 7)
                args, kwargs = run.call_args
                self.assertEqual(args[0], [str(checker)])
                self.assertEqual(kwargs["env"]["PATH"], "cuda-bin" + os.pathsep + "stock-path")
                self.assertEqual(os.environ["PATH"], "stock-path")
                self.assertFalse(kwargs["check"])

    def test_missing_checker_does_not_start_any_process(self):
        with tempfile.TemporaryDirectory() as td, patch("hy_v3_web.subprocess.run") as run:
            self.assertEqual(run_qkv_check(Path(td) / "strata-poly.exe", []), 1)
            run.assert_not_called()

    def test_check_mode_exits_before_model_or_tokenizer_loading(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-poly.exe"
            exe.touch()
            with patch("hy_v3_web.cuda13_lib_dirs", return_value=["cuda"]), \
                 patch("hy_v3_web.run_qkv_check", return_value=9) as check, \
                 patch("hy_v3_web.prepare_tokenizer") as tokenizer:
                result = main(["--exe", str(exe), "--model", str(Path(td) / "absent.gguf"), "--check-qkv-slots"])
                self.assertEqual(result, 9)
                check.assert_called_once_with(exe.resolve(), ["cuda"])
                tokenizer.assert_not_called()

    def test_enabled_mode_stops_before_loading_if_gpu_check_fails(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-poly.exe"
            model = Path(td) / "model.gguf"
            exe.touch()
            model.touch()
            with patch("hy_v3_web.cuda13_lib_dirs", return_value=[]), \
                 patch("hy_v3_web.run_qkv_check", return_value=1), \
                 patch("hy_v3_web.prepare_tokenizer") as tokenizer:
                self.assertEqual(main(["--exe", str(exe), "--model", str(model), "--qkv-slots"]), 1)
                tokenizer.assert_not_called()
