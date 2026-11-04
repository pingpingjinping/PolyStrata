"""Native diagnostics log integration and disabled-by-default launcher behavior."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hy_v3_profile_log import PREFIX, follow_profile, format_profile
from hy_v3_web import build_config


class ProfileLogTest(unittest.TestCase):
    def test_format_keeps_host_and_device_times_separate(self):
        data = dict(decode_steps=100, attention_gpu_ms_per_step=3.5,
                    cpu_experts_host_ms_per_step=8, router_wait_host_ms_per_step=2,
                    decode_wall_ms_per_step=12, collect_host_ms_per_step=.2)
        line = format_profile(PREFIX + json.dumps(data))
        self.assertIn("attention GPU=3.500 ms/step", line)
        self.assertIn("CPU expert host phase=8.000", line)
        self.assertIn("decode wall=12.000", line)
        self.assertIn("overlapping", line)  # The fields are never added into a bogus total.

    def test_unrelated_malformed_and_zero_step_lines(self):
        self.assertIsNone(format_profile("DONE 100 20 3 5 stop"))
        self.assertIn("invalid", format_profile(PREFIX + "{}"))
        self.assertIn("no decode", format_profile(PREFIX + '{"decode_steps":0}'))

    def test_stage_breakdown_keeps_the_legacy_summary_and_adds_a_gpu_line(self):
        data = dict(decode_steps=33, gpu_samples=3, attention_gpu_ms_per_step=6,
                    cpu_experts_host_ms_per_step=8, router_wait_host_ms_per_step=2,
                    decode_wall_ms_per_step=12, collect_host_ms_per_step=.2)
        legacy = format_profile(PREFIX + json.dumps(data))
        data.update(profile_version=2, attention_qkv_gpu_ms_per_step=2,
                    attention_core_gpu_ms_per_step=3, attention_output_gpu_ms_per_step=1)
        lines = format_profile(PREFIX + json.dumps(data)).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], legacy)
        self.assertIn("QKV+norm/RoPE/KV=2.000 ms/step", lines[1])
        self.assertIn("attention core=3.000 ms/step", lines[1])
        self.assertIn("output projection=1.000 ms/step", lines[1])
        self.assertIn("event overhead included", lines[1])

    def test_partial_stage_breakdown_is_not_shown_as_complete(self):
        data = dict(decode_steps=33, attention_gpu_ms_per_step=6,
                    cpu_experts_host_ms_per_step=8, router_wait_host_ms_per_step=2,
                    decode_wall_ms_per_step=12, collect_host_ms_per_step=.2,
                    attention_qkv_gpu_ms_per_step=2)
        self.assertIn("invalid", format_profile(PREFIX + json.dumps(data)))

    def test_tail_skips_old_requests_and_reads_only_new_profile_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "native.log"
            path.write_text(PREFIX + '{"decode_steps":0}\n', encoding="utf-8")
            offset = path.stat().st_size
            stop = threading.Event()
            seen = threading.Event()
            with patch("builtins.print", side_effect=lambda *a, **kw: seen.set()) as output:
                worker = threading.Thread(target=follow_profile, args=(path, stop, offset))
                worker.start()
                try:
                    self.assertFalse(seen.wait(.05))
                    with path.open("a", encoding="utf-8") as log:
                        log.write("DONE unrelated\n" + PREFIX + '{"decode_steps":0}\n')
                    self.assertTrue(seen.wait(2))
                    self.assertEqual(output.call_count, 1)
                finally:
                    stop.set()
                    worker.join(3)
            self.assertFalse(worker.is_alive())

    def test_new_binary_flag_is_opt_in(self):
        root = Path("/example")
        kwargs = dict(exe=root / "engine", model=root / "model.gguf", tokenizer=root / "tok",
                      runtime=root, port=18081, context=4096, lib_dirs=[], temperature=0)
        baseline = build_config(**kwargs)
        self.assertNotIn("--hy-v3-profile", baseline["args"])
        enabled = build_config(**kwargs, profile_decode=True)
        self.assertEqual(enabled["args"], baseline["args"] + ["--hy-v3-profile"])

    def test_decode_hooks_exclude_prefill_and_follow_completed_eval(self):
        src = (ROOT / "src/poly/main.cpp").read_text()
        begin = src.index("if (profile_decode) engine.decode_profile_begin();")
        self.assertGreater(begin, src.index("r.prompt_ms = ms_since(t0);"))
        self.assertIn("if (profile_decode) engine.decode_profile_prepare();\n                    engine.eval(&t, 1, true);\n                    if (profile_decode) engine.decode_profile_sample();", src)
        self.assertIn("r.decode_ms = ms_since(t0);\n        if (profile_decode) engine.decode_profile_end(r.decode_ms);", src)
        hy = (ROOT / "src/poly/hy_v3.cpp").read_text()
        sample = hy.split("void HyV3::decode_profile_sample()", 1)[1].split("void HyV3::decode_profile_end", 1)[0]
        self.assertNotIn("cudaStreamSynchronize(", sample)
        self.assertIn("cudaEventRecordExternal", hy)
