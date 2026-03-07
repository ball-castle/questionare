#!/usr/bin/env python3
"""本脚本用于执行961到880样本重筛并固化核心数据产物。"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from qp_io import read_xlsx_first_sheet
from run_questionnaire_analysis_v2 import QUALITY_PROFILE_BALANCED, run_isolated


def detect_input_format(xlsx: Path) -> str:
    headers, _ = read_xlsx_first_sheet(xlsx)
    if len(headers) == 64:
        return "new961_64"
    if len(headers) == 108:
        return "amethyst_108"
    raise ValueError(f"Unsupported xlsx columns: {len(headers)} (expect 64 or 108)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist rescreen artifacts to data/processed_880.")
    parser.add_argument("--input-xlsx", default="data/raw/数据.xlsx", help="Source questionnaire xlsx.")
    parser.add_argument("--quality-profile", default=QUALITY_PROFILE_BALANCED, help="Quality profile for rescreen.")
    parser.add_argument("--out-dir", default="data/processed_880", help="Output folder under data/.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx)
    out_dir = Path(args.out_dir)
    key_files = [
        out_dir / "survey_clean_880.csv",
        out_dir / "样本流转表_重筛.csv",
        out_dir / "run_metadata.json",
    ]
    if not args.force and all(p.exists() for p in key_files):
        print(f"skipped: key outputs already exist at {out_dir}")
        return

    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input xlsx not found: {input_xlsx}")

    input_format = detect_input_format(input_xlsx)
    with tempfile.TemporaryDirectory(prefix="qp_rescreen_") as tmp:
        tmp_out = Path(tmp) / "analysis_output"
        run_isolated(input_xlsx=input_xlsx, input_format=input_format, output_dir=tmp_out, quality_profile=args.quality_profile)

        mapping = {
            tmp_out / "tables" / "survey_raw.csv": out_dir / "survey_raw_961.csv",
            tmp_out / "tables" / "survey_clean.csv": out_dir / "survey_clean_880.csv",
            tmp_out / "tables" / "survey_model.csv": out_dir / "survey_model_880.csv",
            tmp_out / "tables" / "样本流转表_重筛.csv": out_dir / "样本流转表_重筛.csv",
            tmp_out / "tables" / "异常样本清单_重筛.csv": out_dir / "异常样本清单_重筛.csv",
            tmp_out / "tables" / "筛选规则_重筛.txt": out_dir / "筛选规则_重筛.txt",
            tmp_out / "run_metadata.json": out_dir / "run_metadata.json",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in mapping.items():
            if not src.exists():
                raise FileNotFoundError(f"Expected analysis artifact missing: {src}")
            shutil.copy2(src, dst)

        meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
        raw_n = meta.get("n_samples", "")
        remain_n = meta.get("remain_n_revised", "")
        invalid_n = meta.get("invalid_n_revised", "")
        readme = "\n".join(
            [
                "数据目录说明（processed_880）",
                f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
                f"input_xlsx: {input_xlsx}",
                f"input_format: {input_format}",
                f"quality_profile: {meta.get('quality_profile', args.quality_profile)}",
                f"sample_flow: raw={raw_n}, invalid={invalid_n}, remain={remain_n}",
                "files:",
                "- survey_raw_961.csv: 原始样本导出",
                "- survey_clean_880.csv: 重筛后主分析样本",
                "- survey_model_880.csv: 建模宽表",
                "- 样本流转表_重筛.csv: 重筛样本流转",
                "- 异常样本清单_重筛.csv: 异常剔除明细",
                "- 筛选规则_重筛.txt: 当前重筛规则",
                "- run_metadata.json: 核心元数据",
            ]
        )
        (out_dir / "README_数据说明.txt").write_text(readme, encoding="utf-8")

    print(f"done: rescreen artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
