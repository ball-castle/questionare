#!/usr/bin/env python3
"""Run dual dataset pipeline: analysis -> boosters -> comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(args):
    print("RUN:", " ".join(str(x) for x in args))
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual questionnaire pipeline in isolated outputs.")
    parser.add_argument("--amethyst-xlsx", required=True)
    parser.add_argument("--new-xlsx", required=True)
    parser.add_argument("--out-root", default="output_runs")
    parser.add_argument("--compare-out", default="output_compare")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    compare_out = Path(args.compare_out)

    run_a = out_root / "amethyst"
    run_b = out_root / "new961"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    compare_out.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    run_cmd(
        [
            py,
            str(root / "run_questionnaire_analysis_v2.py"),
            "--input-xlsx",
            args.amethyst_xlsx,
            "--input-format",
            "amethyst_108",
            "--output-dir",
            str(run_a),
        ]
    )
    run_cmd(
        [
            py,
            str(root / "generate_award_boosters_v2.py"),
            "--tables-dir",
            str(run_a / "tables"),
            "--output-dir",
            str(run_a),
        ]
    )

    run_cmd(
        [
            py,
            str(root / "run_questionnaire_analysis_v2.py"),
            "--input-xlsx",
            args.new_xlsx,
            "--input-format",
            "new961_64",
            "--output-dir",
            str(run_b),
        ]
    )
    run_cmd(
        [
            py,
            str(root / "generate_award_boosters_v2.py"),
            "--tables-dir",
            str(run_b / "tables"),
            "--output-dir",
            str(run_b),
        ]
    )

    run_cmd(
        [
            py,
            str(root / "compare_dataset_runs.py"),
            "--run-a",
            str(run_a),
            "--run-b",
            str(run_b),
            "--label-a",
            "amethyst",
            "--label-b",
            "new961",
            "--out-dir",
            str(compare_out),
        ]
    )
    print("dual_pipeline_done")


if __name__ == "__main__":
    main()

