#!/usr/bin/env python3
"""Single-track current pipeline for data.xlsx."""

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
    parser = argparse.ArgumentParser(description="Run current single-track questionnaire pipeline.")
    parser.add_argument("--input-xlsx", required=True, help="Path to input questionnaire xlsx.")
    parser.add_argument("--doc-path", required=True, help="Path to pending docx for fill/check.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument(
        "--build-report",
        action="store_true",
        help="Whether to generate chapter 6/7 integrated report (markdown + docx) under <output-dir>/reports.",
    )
    parser.add_argument(
        "--report-outline-md",
        default=None,
        help="Optional override for chapter 6/7 outline markdown path; default <project-root>/六七部分_国奖标准详细大纲_可直接扩写.md.",
    )
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
    doc_path = Path(args.doc_path)
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
        "doc_path": str(doc_path),
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
    run_step(
        "fill_pending_doc",
        [
            py,
            str(root / "fill_pending_doc_v2.py"),
            "--doc-path",
            str(doc_path),
            "--tables-dir",
            str(output_dir / "tables"),
            "--output-dir",
            str(output_dir),
        ],
        manifest,
    )
    run_step(
        "check_pending_doc_consistency",
        [
            py,
            str(root / "check_pending_doc_consistency_v2.py"),
            "--doc-path",
            str(doc_path),
            "--raw-xlsx",
            str(input_xlsx),
            "--tables-dir",
            str(output_dir / "tables"),
            "--output-dir",
            str(output_dir),
        ],
        manifest,
    )

    if args.build_report:
        report_outline = Path(args.report_outline_md) if args.report_outline_md else (root.parent / "六七部分_国奖标准详细大纲_可直接扩写.md")
        run_step(
            "report_generation",
            [
                py,
                str(root / "generate_ch6_ch7_report.py"),
                "--outline-md",
                str(report_outline),
                "--tables-dir",
                str(output_dir / "tables"),
                "--figures-dir",
                str(output_dir / "figures"),
                "--output-dir",
                str(output_dir / "reports"),
                "--base-name",
                "六七部分_完整报告",
                "--missing-policy",
                "keep_placeholder",
            ],
            manifest,
        )

    manifest["finished_at"] = now_iso()
    manifest["status"] = "ok"
    manifest["key_outputs"] = [
        str(output_dir / "问卷数据处理与分析_结果摘要.txt"),
        str(output_dir / "tables" / "待填数据_回填值清单.csv"),
        str(output_dir / "tables" / "待填数据_待补项清单.csv"),
        str(output_dir / "tables" / "待填数据_一致性核查报告.csv"),
        str(output_dir / "pipeline_manifest.json"),
    ]
    if args.build_report:
        manifest["key_outputs"].extend(
            [
                str(output_dir / "reports" / "六七部分_完整报告.md"),
                str(output_dir / "reports" / "六七部分_完整报告.docx"),
                str(output_dir / "reports" / "六七章_证据索引.csv"),
                str(output_dir / "reports" / "六七章_生成日志.json"),
                str(output_dir / "reports" / "六七部分_输入核查报告.json"),
            ]
        )

    (output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"current_pipeline_done: out={output_dir}")


if __name__ == "__main__":
    main()
