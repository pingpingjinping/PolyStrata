"""Paired line-ID/prompt diagnostics, without model weights or a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hy_v3_line_probe as P


class LineProbeTest(unittest.TestCase):
    @staticmethod
    def source():
        return "preamble\n" + "".join(f"[{i:03d}] こんにちは {i}\n続き{i}\n" for i in range(1, 101))

    def test_excerpt_keeps_exact_blocks_and_continuations(self):
        text = P.excerpt(self.source(), 51, 75)
        self.assertEqual(P.ID.findall(text), [f"{i:03d}" for i in range(51, 76)])
        self.assertIn("続き75\n", text)
        self.assertNotIn("[076]", text)
        self.assertNotIn("preamble", text)
        for source, start, end in ((self.source(), 51, 101), (self.source(), 75, 51),
                                   ("[051] a\n[051] b", 51, 51), ("[052] a\n[051] b", 51, 52)):
            with self.assertRaises(ValueError):
                P.excerpt(source, start, end)

    def test_pairing_changes_only_ids_or_instruction_and_repeats_exact_prompts(self):
        source = P.excerpt(self.source(), 51, 75)
        jobs = P.probe_jobs(source)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(jobs[0]["prompt"], f"[stability-E] {P.INSTRUCTION}\n\n{source}")
        self.assertEqual(P.ID.findall(jobs[1]["source"]), [f"{i:03d}" for i in range(1, 26)])
        self.assertEqual(P.ID.sub("[ID]", jobs[0]["source"]), P.ID.sub("[ID]", jobs[1]["source"]))
        self.assertEqual(jobs[1]["id_map"]["003"], "053")
        self.assertEqual(jobs[2]["source"], jobs[0]["source"])
        self.assertIn("exactly 25 input lines, from [051] to [075]", jobs[2]["prompt"])
        self.assertIn("from [001] to [025]", jobs[3]["prompt"])
        for first, second in zip(jobs[:4], jobs[4:][::-1]):
            self.assertEqual(first["prompt"], second["prompt"])
            self.assertEqual((first["round"], second["round"]), (1, 2))

    def test_dry_run_has_no_network_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "report.json"
            source.write_text(self.source(), encoding="utf-8")
            with patch.object(P.bench, "get", side_effect=AssertionError("network")):
                self.assertEqual(P.main(["--input", str(source), "--output", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_cli_maps_missing_ids_and_keeps_failures_without_weakening_checks(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "report.json"
            source.write_text(self.source(), encoding="utf-8")
            def measure(base, model, prompt, text, *args, **kwargs):
                ids = P.ID.findall(text)
                self.assertTrue(kwargs["capture_idle_memory"])
                return dict(ok=False, coverage=dict(missing_ids=[ids[2]], expected_lines=25, output_lines=24))
            with patch.object(P.bench, "get", return_value=dict(model="hy")), \
                 patch.object(P.bench, "measure", side_effect=measure) as measured:
                self.assertEqual(P.main(["--input", str(source), "--output", str(output), "--run"]), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(measured.call_count, 8)
            self.assertTrue(all(r["missing_original_ids"] == ["053"] for r in report["records"]))
            self.assertFalse(report["summary"]["all_line_checks_passed"])
            self.assertTrue(all(g["recorded"] == 2 for g in report["summary"]["variants"].values()))

    def test_transport_error_checkpoints_and_stops(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.txt", Path(td) / "report.json"
            source.write_text(self.source(), encoding="utf-8")
            with patch.object(P.bench, "get", return_value=dict(model="hy")), \
                 patch.object(P.bench, "measure", side_effect=[dict(ok=True, text="kept"), RuntimeError("offline")]):
                self.assertEqual(P.main(["--input", str(source), "--output", str(output), "--run"]), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["records"]), 2)
            self.assertEqual(report["records"][0]["text"], "kept")
            self.assertEqual(report["records"][1]["error"], "offline")
            self.assertFalse(report["summary"]["all_line_checks_passed"])


if __name__ == "__main__":
    unittest.main()
