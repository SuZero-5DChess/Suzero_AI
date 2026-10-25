"""Manifest validation and stdin transport for the HC profiling launcher."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "profile_hc.py"
SPEC = importlib.util.spec_from_file_location("profile_hc", SCRIPT)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class ProfileManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.executable = self.root / "tools"
        self.executable.write_bytes(b"not executed: subprocess is mocked")
        self.pgn = self.root / "game with spaces.5dpgn"
        self.pgn.write_text('[Board "Standard"]\n{a multiline\ncomment}\n', encoding="utf-8")
        self.config = self.root / "manifest.json"
        self.output = self.root / "result.json"

    def run_launcher(self, entries, process, extra=()):
        self.config.write_text(json.dumps({
            "executable": str(self.executable), "profile-count": entries,
        }), encoding="utf-8")
        arguments = [str(SCRIPT), "--config", str(self.config), "--output", str(self.output), *extra]
        with patch.object(sys, "argv", arguments), patch.object(PROFILE.subprocess, "Popen", process), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return PROFILE.main()

    def test_comparisons_are_sent_through_stdin_and_preserved(self):
        for expected in ("=20", "<21", ">19"):
            with self.subTest(expected=expected):
                process = MagicMock()
                child = process.return_value.__enter__.return_value
                child.returncode = 0
                child.communicate.return_value = (json.dumps({
                    "cases": [{"workload": "profile-count", "file": "stdin:1",
                               "action_count": 20, "complete": True}], "summary": {},
                }), None)
                self.assertEqual(self.run_launcher([
                    {"file": str(self.pgn), "expected": expected},
                ], process), 0)
                command = process.call_args.args[0]
                self.assertEqual(command[:2], [str(self.executable), "hc-profile"])
                self.assertNotIn("--stdin", command)
                self.assertNotIn(str(self.pgn), command)
                self.assertEqual(process.call_args.kwargs["stdin"], subprocess.PIPE)
                self.assertEqual(child.communicate.call_args.kwargs["input"],
                                 f'EXPECTED{expected} COUNT [Board "Standard"] {{a multiline comment}}\n')
                report = json.loads(self.output.read_text(encoding="utf-8"))
                self.assertEqual(report["cases"][0]["file"], str(self.pgn))
                self.assertEqual(report["run"]["inputs"][0]["expected"], expected)

    def test_missing_malformed_or_overflowing_comparisons_are_rejected(self):
        for entry in (str(self.pgn), {}, {"file": str(self.pgn)},
                      *({"file": str(self.pgn), "expected": value} for value in (
                          None, 20, "20", "=", "=-1", "=+1", "<=20", "<0",
                          "=2147483647", ">2147483647", "<2147483648"))):
            with self.subTest(entry=entry):
                process = MagicMock()
                with self.assertRaises(SystemExit) as error:
                    self.run_launcher([entry], process)
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
