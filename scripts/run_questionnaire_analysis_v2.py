#!/usr/bin/env python3
"""Run questionnaire analysis into an isolated output directory for two input formats."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

from convert_961_to_108 import convert_961_to_108, export_conversion_artifacts
from qp_io import read_xlsx_first_sheet, write_dict_csv


def _load_base_module():
    try:
        import run_questionnaire_analysis as m  # type: ignore

        return m
    except Exception:
        p = Path(__file__).resolve().parent.parent / "archive_legacy" / "scripts" / "run_questionnaire_analysis.py"
        if not p.exists():
            raise
        spec = importlib.util.spec_from_file_location("run_questionnaire_analysis_legacy", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load legacy module spec: {p}")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def _default_audit(n_rows: int) -> dict:
    return {
        "n_rows": int(n_rows),
        "unknown_value_count": 0,
        "unknown_value_rate": 0.0,
        "other_text_count": 0,
        "branch_conflict_count": 0,
        "branch_conflict_rate": 0.0,
        "conversion_integrity": 1.0,
    }


def _run_base_main_with_loader(loader, tmp_root: Path) -> None:
    base = _load_base_module()
    old_loader = base.read_xlsx_first_sheet
    old_cwd = Path.cwd()
    try:
        base.read_xlsx_first_sheet = loader
        os.chdir(tmp_root)
        base.main()
    finally:
        base.read_xlsx_first_sheet = old_loader
        os.chdir(old_cwd)


def _patch_metadata(out_dir: Path, input_xlsx: Path, input_format: str, audit: dict) -> None:
    meta_path = out_dir / "run_metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["input"] = str(input_xlsx)
    meta["input_format"] = input_format
    meta["source_pipeline"] = "run_questionnaire_analysis_v2"
    meta["conversion_unknown_count"] = int(audit.get("unknown_value_count", 0))
    meta["conversion_unknown_rate"] = float(audit.get("unknown_value_rate", 0.0))
    meta["conversion_other_text_count"] = int(audit.get("other_text_count", 0))
    meta["conversion_branch_conflict_count"] = int(audit.get("branch_conflict_count", 0))
    meta["conversion_integrity"] = float(audit.get("conversion_integrity", 1.0))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_input_loader(input_xlsx: Path, input_format: str) -> Tuple[callable, dict, object]:
    if input_format == "amethyst_108":
        headers, rows = read_xlsx_first_sheet(input_xlsx)

        def loader(_):
            return headers, rows

        return loader, _default_audit(len(rows)), None

    if input_format == "new961_64":
        headers, rows = read_xlsx_first_sheet(input_xlsx)
        conv = convert_961_to_108(headers, rows)

        def loader(_):
            return conv.headers_108, conv.rows_108

        return loader, conv.audit, conv

    raise ValueError(f"Unsupported input_format: {input_format}")


def run_isolated(input_xlsx: Path, input_format: str, output_dir: Path) -> None:
    input_xlsx = Path(input_xlsx)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    loader, audit, conv = _prepare_input_loader(input_xlsx, input_format)
    tmp_root = Path(tempfile.mkdtemp(prefix="qp_run_"))
    try:
        _run_base_main_with_loader(loader, tmp_root)
        tmp_out = tmp_root / "output"
        (tmp_out / "tables").mkdir(parents=True, exist_ok=True)

        if conv is not None:
            export_conversion_artifacts(tmp_out, conv)
        else:
            write_dict_csv(
                tmp_out / "tables" / "unknown_value_log.csv",
                ["row_id", "source_col_idx", "source_header", "source_value", "target_col_idx", "reason"],
                [],
            )
            (tmp_out / "conversion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        _patch_metadata(tmp_out, input_xlsx, input_format, audit)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(tmp_out, output_dir)

        tmp_outline = tmp_root / "问卷数据处理与分析_执行大纲.txt"
        if tmp_outline.exists():
            shutil.copy2(tmp_outline, output_dir / "问卷数据处理与分析_执行大纲.txt")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated questionnaire analysis.")
    parser.add_argument("--input-xlsx", required=True, help="Path to source xlsx file.")
    parser.add_argument(
        "--input-format",
        required=True,
        choices=["amethyst_108", "new961_64"],
        help="Input format identifier.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_isolated(Path(args.input_xlsx), args.input_format, Path(args.output_dir))
    print(f"analysis_v2_done: {args.input_format} -> {args.output_dir}")


if __name__ == "__main__":
    main()
