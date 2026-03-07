#!/usr/bin/env python3
"""本脚本用于运行纯分析主流程并串联分析与补强步骤。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qp_io import read_xlsx_first_sheet


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_step(name: str, cmd: list[str], manifest: dict) -> None:
    start = now_iso()
    rec = {"step": name, "command": cmd, "started_at": start, "status": "running"}
    manifest["steps"].append(rec)
    try:
        subprocess.run(cmd, check=True)
        rec["status"] = "ok"
    except subprocess.CalledProcessError as e:
        rec["status"] = "failed"
        rec["return_code"] = e.returncode
        raise
    finally:
        rec["finished_at"] = now_iso()


def detect_input_format(xlsx: Path) -> str:
    headers, _ = read_xlsx_first_sheet(xlsx)
    if len(headers) == 64:
        return "new961_64"
    if len(headers) == 108:
        return "amethyst_108"
    raise ValueError(f"Unsupported xlsx columns: {len(headers)} (expect 64 or 108)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run functional questionnaire pipeline (analysis + boosters).")
    parser.add_argument("--input-xlsx", required=True, help="Path to input questionnaire xlsx.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument(
        "--quality-profile",
        default="balanced_v20260221",
        choices=["balanced_v20260221", "legacy_balanced"],
        help="Quality filtering profile passed to analysis step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    input_xlsx = Path(args.input_xlsx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    input_format = detect_input_format(input_xlsx)

    manifest = {
        "pipeline": "current_single_track",
        "started_at": now_iso(),
        "input_xlsx": str(input_xlsx),
        "input_format": input_format,
        "quality_profile": args.quality_profile,
        "output_dir": str(output_dir),
        "steps": [],
    }

    run_step(
        "analysis",
        [
            py,
            str(root / "run_questionnaire_analysis_v2.py"),
            "--input-xlsx",
            str(input_xlsx),
            "--input-format",
            input_format,
            "--quality-profile",
            args.quality_profile,
            "--output-dir",
            str(output_dir),
        ],
        manifest,
    )
    run_step(
        "boosters",
        [
            py,
            str(root / "generate_award_boosters_v2.py"),
            "--tables-dir",
            str(output_dir / "tables"),
            "--output-dir",
            str(output_dir),
        ],
        manifest,
    )

    manifest["finished_at"] = now_iso()
    manifest["status"] = "ok"
    manifest["key_outputs"] = [
        str(output_dir / "问卷数据处理与分析_结果摘要.txt"),
        str(output_dir / "tables" / "survey_clean.csv"),
        str(output_dir / "tables" / "信度分析表.csv"),
        str(output_dir / "tables" / "效度分析表.csv"),
        str(output_dir / "tables" / "建议落地行动矩阵.csv"),
        str(output_dir / "run_metadata.json"),
        str(output_dir / "pipeline_manifest.json"),
    ]

    (output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"current_pipeline_done: out={output_dir}")


if __name__ == "__main__":
    main()
