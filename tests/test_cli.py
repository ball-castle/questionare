from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_module_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "questionnaire_analysis", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("demo", result.stdout)
        self.assertIn("pipeline", result.stdout)

    def test_main_wrapper_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("questionnaire-analysis", result.stdout)
        self.assertIn("sem", result.stdout)

    def test_script_wrapper_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_current_pipeline.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--input-xlsx", result.stdout)


if __name__ == "__main__":
    unittest.main()
