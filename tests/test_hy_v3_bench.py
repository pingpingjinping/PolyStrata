"""Benchmark accounting and actual streaming transport, without model weights."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hy_v3_bench as B


def stream(finish="stop", done=True):
    events = [{"choices": [{"delta": {"content": "[001] 안녕"}}]},
              {"choices": [{"delta": {}, "finish_reason": finish}],
               "timings": {"prompt_per_second": 100, "predicted_per_second": 20},
               "usage": {"completion_tokens": 5}}]
    text = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
    return io.BytesIO((text + ("data: [DONE]\n\n" if done else "")).encode())


class BenchTest(unittest.TestCase):
    @staticmethod
    def numbered_source():
        return "".join(f"[{i:03d}] こんにちは{i}\n" for i in range(1, 9))

    def test_stability_plan_varies_real_excerpts_and_repeats_exact_a(self):
        source = self.numbered_source().replace("[003]", "続きの行\n[003]")
        jobs = B.stability_jobs(source, lambda text, tag: tag + "\n" + text)
        self.assertEqual([j[0] for j in jobs], ["stability-" + x for x in "ABACDEAFCA"])
        self.assertEqual(len(set(j[2] for j in jobs)), 6)
        for at in (0, 2, 6, 9):
            self.assertEqual(jobs[at][1:3], jobs[0][1:3])
        self.assertEqual(jobs[0][2], source)
        self.assertIn("続きの行", jobs[1][2])
        self.assertEqual(B.coverage(source, jobs[1][2])["output_lines"], 4)
        self.assertEqual(B.coverage(source, jobs[3][2])["output_lines"], 4)
        self.assertEqual(B.coverage(source, jobs[4][2])["output_lines"], 2)

    def test_stability_rejects_too_few_or_duplicate_ids(self):
        for source in ("[001] a", "[001] a\n[002] b\n[003] c\n[001] d"):
            with self.assertRaises(ValueError):
                B.stability_jobs(source, lambda text, tag: text)

    def test_stability_summary_requires_cancel_recovery_and_complete_plan(self):
        rows = [dict(kind="stability-A", ok=True) for _ in range(10)]
        rows += [dict(kind="cancel", ok=True), dict(kind="after-cancel", ok=True)]
        result = B.stability_summary(rows, 12)
        self.assertTrue(result["all_request_checks_passed"])
        self.assertEqual(result["completed_translations"], 11)
        self.assertIsNone(result["gpu_used_mib_device"]["peak_sampled"])
        self.assertTrue(result["memory_review_required"])
        self.assertFalse(B.stability_summary(rows[:10], 12)["all_request_checks_passed"])
        rows[10]["ok"] = False  # A client disconnect without observed server cancellation is not a pass.
        result = B.stability_summary(rows, 12)
        self.assertFalse(result["all_request_checks_passed"])
        self.assertFalse(result["cancellation_observed"])
        self.assertTrue(result["recovery_ok"])
        self.assertEqual(result["failed_request_indices"], [11])

    def test_stability_memory_keeps_units_missing_values_and_idle_scope(self):
        rows = [dict(kind="stability-A", ok=True,
                     machine_before={"ram": {"used_gib": 2}},
                     machine_after={"ram": {"used_gib": 15}, "gpu": None},
                     machine_samples=[{}, {"ram": {"used_gib": 18.5}, "gpu": {"used_mib": 7200}}]),
                dict(kind="after-cancel", ok=True,
                     machine_after={"ram": {"used_gib": 16}, "gpu": {"used_mib": 7100}},
                     machine_samples=[{"ram": {"used_gib": float("nan")}, "gpu": "unavailable"}])]
        result = B.stability_summary(rows, 12)
        self.assertEqual(result["ram_used_gib_system"], dict(first_idle_after_request=15,
                                                          last_idle_after_request=16, peak_sampled=18.5))
        self.assertEqual(result["gpu_used_mib_device"]["peak_sampled"], 7200)
        self.assertEqual(result["gpu_used_mib_device"]["first_idle_after_request"], 7100)
        # An idle snapshot is still an observed sample, even if a short request
        # completed between the active sampler's polls.
        result = B.stability_summary([dict(kind="stability-A", ok=True,
                                           machine_after={"gpu": {"used_mib": 7000}})], 12)
        self.assertEqual(result["gpu_used_mib_device"]["peak_sampled"], 7000)

    def test_stability_dry_run_never_connects_or_writes(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "result.json"
            source.write_text(self.numbered_source(), encoding="utf-8")
            with patch.object(B, "get", side_effect=AssertionError("network traffic")):
                self.assertEqual(B.main(["--input", str(source), "--label", "stability", "--output", str(output),
                                         "--stability", "--long-input", str(Path(td) / "not-yet-present.txt")]), 0)
            self.assertFalse(output.exists())

    def test_stability_cli_records_twelve_jobs_and_exact_repeated_prompt_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "result.json"
            source.write_text(self.numbered_source(), encoding="utf-8")
            def measured(base, model, prompt, text, max_tokens, timeout, cancel_after, **kwargs):
                self.assertTrue(kwargs["capture_idle_memory"])
                return dict(ok=True, text="translation", engine_request={"finish": "cancel" if cancel_after else "stop"})
            with patch.object(B, "get", return_value={"model": "hy"}), \
                 patch.object(B, "measure", side_effect=measured) as measure:
                self.assertEqual(B.main(["--input", str(source), "--label", "stability", "--output", str(output),
                                         "--stability", "--cancel", "--run"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], 2)
            self.assertEqual(measure.call_count, 12)  # --cancel never adds a second cancel/recovery pair.
            self.assertTrue(report["summary"]["all_request_checks_passed"])
            self.assertEqual(len({report["records"][i]["prompt_sha256"] for i in (0, 2, 6, 9)}), 1)
            self.assertEqual([r["kind"] for r in report["records"][-2:]], ["cancel", "after-cancel"])

    def test_stability_error_keeps_previous_success_and_failed_request(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "result.json"
            source.write_text(self.numbered_source(), encoding="utf-8")
            with patch.object(B, "get", return_value={"model": "hy"}), \
                 patch.object(B, "measure", side_effect=[dict(ok=True, text="kept"), RuntimeError("simulated failure")]):
                self.assertEqual(B.main(["--input", str(source), "--label", "stability", "--output", str(output),
                                         "--stability", "--run"]), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["records"]), 2)
            self.assertEqual(report["records"][0]["text"], "kept")
            self.assertIn("simulated failure", report["records"][1]["error"])
            self.assertFalse(report["summary"]["all_request_checks_passed"])

    def test_stream_keeps_engine_timings_and_finish(self):
        with patch.object(B.urllib.request, "urlopen", return_value=stream()):
            result = B.request("http://localhost", "hy", "test", 32, 10)
        self.assertEqual(result["finish"], "stop")
        self.assertTrue(result["done_received"])
        self.assertEqual(result["timings"]["predicted_per_second"], 20)
        self.assertEqual(result["text"], "[001] 안녕")

    def test_disconnect_closes_before_final_event(self):
        response = stream()
        with patch.object(B.urllib.request, "urlopen", return_value=response):
            result = B.request("http://localhost", "hy", "test", 32, 10, cancel_after=1)
        self.assertTrue(response.closed)
        self.assertTrue(result["cancelled_by_client"])
        self.assertFalse(result["done_received"])
        self.assertIsNone(result["finish"])

    def test_premature_eof_is_not_a_completed_stream(self):
        with patch.object(B.urllib.request, "urlopen", return_value=stream(done=False)):
            result = B.request("http://localhost", "hy", "test", 32, 10)
        self.assertFalse(result["done_received"])

    def test_missing_duplicate_and_reordered_lines(self):
        result = B.coverage("[001] a\n[002] b", "[002] 나\n[002] 나\n[003] 다")
        self.assertEqual(result["missing_ids"], ["001"])
        self.assertEqual(result["duplicate_ids"], ["002"])
        self.assertEqual(result["extra_ids"], ["003"])
        self.assertFalse(result["numbered_lines_ok"])
        self.assertFalse(B.coverage("[001] a\n[002] b", "[002] 나\n[001] 가")["numbered_lines_ok"])

    def test_warm_failures_and_missing_timings_do_not_inflate_median(self):
        rows = [{"kind": "cold", "ok": True, "timings": {"predicted_per_second": 60},
                 "engine_request": {"hit_rate": .5}},
                {"kind": "warm", "ok": True, "timings": {"predicted_per_second": 999}},
                {"kind": "cold", "ok": False, "timings": {"predicted_per_second": 999}}]
        result = B.summary(rows)
        self.assertEqual(result["generation_tok_s"], 60)
        self.assertEqual(result["expert_vram_hit_rate"], .5)
        self.assertIsNone(result["prompt_tok_s"])

    def test_concurrent_traffic_rejects_hit_rate_attribution(self):
        with patch.object(B, "idle", side_effect=[{"totals": {"requests": 0}},
                                                   {"totals": {"requests": 2}, "requests": [{}]}]), \
             patch.object(B, "get", return_value={}), \
             patch.object(B, "request", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "accounting mismatch"):
                B.measure("http://localhost", "hy", "p", "s", 32, 10)

    def test_dry_run_never_connects_or_writes(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "result.json"
            source.write_text("[001] こんにちは", encoding="utf-8")
            with patch.object(B, "get", side_effect=AssertionError("network traffic")):
                self.assertEqual(B.main(["--input", str(source), "--label", "hy-b512",
                                         "--output", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_speed_dry_run_does_not_read_optional_long_file(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "result.json"
            source.write_text("[001] こんにちは", encoding="utf-8")
            with patch.object(B, "get", side_effect=AssertionError("network traffic")):
                self.assertEqual(B.main(["--input", str(source), "--label", "speed", "--output", str(output),
                                         "--long-input", str(Path(td) / "not-yet-present.txt")]), 0)
            self.assertFalse(output.exists())

    def test_real_http_stream_and_request_accounting_with_mock_engine(self):
        sys.path.insert(0, str(ROOT))
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "[001] 안녕", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            result = B.measure(base, svc.model, "Translate [001] こんにちは into Korean",
                               "[001] こんにちは", 128, 10, capture_idle_memory=True)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["engine_request"]["finish"], "stop")
            self.assertTrue(result["coverage"]["numbered_lines_ok"])
            self.assertIsNone(result["timings"])  # A mock engine is not measured GPU throughput.
            self.assertIn("ram", result["machine_before"])
            self.assertIn("gpu", result["machine_after"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            svc.telemetry.close()


if __name__ == "__main__":
    unittest.main()
