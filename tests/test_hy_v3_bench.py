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
                               "[001] こんにちは", 128, 10)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["engine_request"]["finish"], "stop")
            self.assertTrue(result["coverage"]["numbered_lines_ok"])
            self.assertIsNone(result["timings"])  # A mock engine is not measured GPU throughput.
        finally:
            httpd.shutdown()
            httpd.server_close()
            svc.telemetry.close()


if __name__ == "__main__":
    unittest.main()
