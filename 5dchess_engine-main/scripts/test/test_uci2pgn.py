"""CLI regression coverage for converting 5DUCI position commands."""

from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "uci2pgn.py"
BOARD = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
HISTORY = """(-1T1)c2c4 (0T1)g2g4 submit (0T1)e7e6 (-1T1)g8f6 submit
(-1T2)d1b3 (0T2)g1f3 submit (0T2)h7h5 (-1T2)c7c6 submit
(-1T3)g1h3 (0T3)c2c3 submit (0T3)h5g4 (-1T3)d7d6 submit
(-1T4)h3f4 (0T4)d1a4 submit (0T4)g4f3 (-1T4)d6d5 submit
(-1T5)b1a3 (0T5)h1g1 submit (0T5)f3e2 (-1T5)d5c4 submit
(-1T6)a3c4 (0T6)f1e2 submit (0T6)h8h2 (-1T6)d8d2 submit
(-1T7)c4d2 (0T7)g1g6 submit (0T7)h2f2 (-1T7)f6d5 submit
(-1T8)f4d5 (0T8)g6e6 submit (0T8)f7e6 (-1T8)c6d5 submit
(-1T9)b3b5 (0T9)a4d7 submit"""


class UciToPgnTests(unittest.TestCase):
    def run_converter(self, command):
        return subprocess.run(
            [sys.executable, str(SCRIPT)], input=command,
            capture_output=True, text=True, timeout=10,
        )

    def test_two_timeline_history(self):
        for separator in ("", " ", "\n\t"):
            with self.subTest(separator=separator):
                fen = f"[{BOARD}:-0:1:w]{separator}[{BOARD}:+0:1:w]"
                result = self.run_converter(f"position size 8x8 even fen {fen} moves {HISTORY}")
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = [
                    '[Size "8x8"]', '[Board "Custom - Even"]',
                    f"[{BOARD}:-0:1:w]", f"[{BOARD}:+0:1:w]", "",
                ]
                for index, action in enumerate(HISTORY.split("submit")[:-1]):
                    moves = " ".join(action.split()).replace("(-1T", "(-0T").replace("(0T", "(+0T")
                    expected.append(("1w. " if index == 0 else "/ ") + moves)
                self.assertEqual(result.stdout, "\n".join(expected) + "\n")

    def test_even_jump_timelines_and_promotion(self):
        result = self.run_converter(
            "position even fen [8/8/8/8/8/8/8/8:+0:1:w] moves "
            "(-2T2)a7(-1T1)a8q (0T2)b2(1T1)b3 submit"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1w. (-1T2)a7(-0T1)a8=Q (+0T2)b2(1T1)b3\n", result.stdout)

    def test_odd_timelines(self):
        result = self.run_converter(
            "position odd fen [8/8/8/8/8/8/8/8:-1:1:w] moves (-1T1)a2a3 submit"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1w. (-1T1)a2a3\n", result.stdout)

    def test_startpos(self):
        result = self.run_converter("position startpos moves (0T1)e2e3 submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '[Board "Standard - Turn Zero"]\n\n1w. (0T1)e2e3\n')

    def test_rejects_unparsed_fen_text(self):
        block = "[8/8/8/8/8/8/8/8:0:1:w]"
        for fen in ("junk" + block, block + "junk", block + "junk" + block):
            with self.subTest(fen=fen):
                result = self.run_converter("position fen " + fen)
                self.assertEqual(result.returncode, 2)
                self.assertIn("FEN must be one or more", result.stderr)


if __name__ == "__main__":
    unittest.main()
