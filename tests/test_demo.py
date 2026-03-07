from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from questionnaire_analysis.demo import write_demo_bundle


ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_write_demo_bundle_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "demo-output"
            write_demo_bundle(out_dir)

            self.assertTrue((out_dir / "问卷数据处理与分析_结果摘要.txt").exists())
            self.assertTrue((out_dir / "tables" / "survey_clean.csv").exists())

            manifest = json.loads((out_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["input_format"], "synthetic_demo")

    def test_write_demo_bundle_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "demo-output-cli"
            subprocess.run(
                [sys.executable, "-m", "questionnaire_analysis", "demo", "--output-dir", str(out_dir)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue((out_dir / "run_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
