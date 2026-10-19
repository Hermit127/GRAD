import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_archived_closed_loop_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "scripts/reproduce.py"),
                            "--task", "t3", "--output", tmp], check=True,
                           capture_output=True, text=True)
            report = json.loads((Path(tmp) / "t3/verification.json").read_text())
            self.assertEqual((report["verified_cases"], report["verified_cells"]), (240, 6))

    def test_reject_output_overwriting_evidence(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce.py"),
                                 "--output", str(ROOT / "evidence")], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_launcher_rejects_external_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "executed"
            script = Path(tmp) / "external.py"
            script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            result = subprocess.run([sys.executable, str(ROOT / "code/run.py"), str(script)],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install requirements-analysis.txt")
    def test_archived_w1_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "scripts/reproduce.py"),
                            "--task", "w1", "--output", tmp], check=True,
                           capture_output=True, text=True)
            report = json.loads((Path(tmp) / "w1/verification.json").read_text())
            self.assertEqual((report["verified_examples"], report["verified_runs"]), (232, 9))


if __name__ == "__main__":
    unittest.main()
