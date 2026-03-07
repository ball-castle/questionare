#!/usr/bin/env python3
"""统一问卷分析入口，兼容64/108列输入并输出标准产物。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Tuple
from .convert_961_to_108 import convert_961_to_108, export_conversion_artifacts
from .qp_io import read_xlsx_first_sheet, write_dict_csv


QUALITY_PROFILE_LEGACY = "legacy_balanced"
QUALITY_PROFILE_BALANCED = "balanced_v20260221"

INPUT_FORMAT_AMETHYST_108 = "amethyst_108"
INPUT_FORMAT_NEW961_64 = "new961_64"


def _core_module():
    from . import questionnaire_analysis_core as core_module

    return core_module


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


def _run_core_main_with_loader(
    loader: Callable[[Path], Tuple[list[str], list[list[str]]]],
    tmp_root: Path,
    quality_profile: str,
    quality_context: dict,
) -> None:
    core = _core_module()
    old_loader = core.read_xlsx_first_sheet
    old_cwd = Path.cwd()
    had_profile = hasattr(core, "FORCE_QUALITY_PROFILE")
    old_profile = getattr(core, "FORCE_QUALITY_PROFILE", None)
    had_context = hasattr(core, "QUALITY_CONTEXT")
    old_context = getattr(core, "QUALITY_CONTEXT", None)
    try:
        core.read_xlsx_first_sheet = loader
        core.FORCE_QUALITY_PROFILE = quality_profile
        core.QUALITY_CONTEXT = quality_context
        os.chdir(tmp_root)
        core.main()
    finally:
        core.read_xlsx_first_sheet = old_loader
        if had_profile:
            core.FORCE_QUALITY_PROFILE = old_profile
        elif hasattr(core, "FORCE_QUALITY_PROFILE"):
            delattr(core, "FORCE_QUALITY_PROFILE")
        if had_context:
            core.QUALITY_CONTEXT = old_context
        elif hasattr(core, "QUALITY_CONTEXT"):
            delattr(core, "QUALITY_CONTEXT")
        os.chdir(old_cwd)


def _patch_metadata(out_dir: Path, input_xlsx: Path, input_format: str, audit: dict, quality_profile: str) -> None:
    meta_path = out_dir / "run_metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["input"] = str(input_xlsx)
    meta["input_format"] = input_format
    meta["quality_profile"] = quality_profile
    meta["source_pipeline"] = "run_questionnaire_analysis"
    meta["conversion_unknown_count"] = int(audit.get("unknown_value_count", 0))
    meta["conversion_unknown_rate"] = float(audit.get("unknown_value_rate", 0.0))
    meta["conversion_other_text_count"] = int(audit.get("other_text_count", 0))
    meta["conversion_branch_conflict_count"] = int(audit.get("branch_conflict_count", 0))
    meta["conversion_integrity"] = float(audit.get("conversion_integrity", 1.0))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_duration_seconds(raw: str):
    txt = str(raw or "").strip()
    if not txt:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", txt)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _build_quality_context_from_64(rows_64) -> dict:
    ctx = {}
    for i, row in enumerate(rows_64, start=1):
        r = list(row) + [""] * max(0, 64 - len(row))
        ctx[i] = {
            "duration_sec": _parse_duration_seconds(r[2] if len(r) > 2 else ""),
            "q8_text": str(r[13] if len(r) > 13 else "").strip(),
            "q14_text": str(r[19] if len(r) > 19 else "").strip(),
            "q15_text": str(r[20] if len(r) > 20 else "").strip(),
            "attention_text": str(r[33] if len(r) > 33 else "").strip(),
            "ip": str(r[5] if len(r) > 5 else "").strip(),
            "open_text": str(r[63] if len(r) > 63 else "").strip(),
        }
    return ctx


def detect_input_format(xlsx: Path) -> str:
    headers, _ = read_xlsx_first_sheet(xlsx)
    if len(headers) == 64:
        return INPUT_FORMAT_NEW961_64
    if len(headers) == 108:
        return INPUT_FORMAT_AMETHYST_108
    raise ValueError(f"Unsupported xlsx columns: {len(headers)} (expect 64 or 108)")


def _prepare_input_loader(input_xlsx: Path, input_format: str):
    if input_format == INPUT_FORMAT_AMETHYST_108:
        headers, rows = read_xlsx_first_sheet(input_xlsx)

        def loader(_: Path):
            return headers, rows

        return loader, _default_audit(len(rows)), None, {}

    if input_format == INPUT_FORMAT_NEW961_64:
        headers, rows = read_xlsx_first_sheet(input_xlsx)
        conv = convert_961_to_108(headers, rows)
        quality_context = _build_quality_context_from_64(rows)

        def loader(_: Path):
            return conv.headers_108, conv.rows_108

        return loader, conv.audit, conv, quality_context

    raise ValueError(f"Unsupported input_format: {input_format}")


def run_analysis(
    input_xlsx: Path,
    input_format: str,
    output_dir: Path,
    quality_profile: str = QUALITY_PROFILE_BALANCED,
) -> None:
    input_xlsx = Path(input_xlsx)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    loader, audit, conv, quality_context = _prepare_input_loader(input_xlsx, input_format)
    tmp_root = Path(tempfile.mkdtemp(prefix="qp_run_"))
    try:
        _run_core_main_with_loader(loader, tmp_root, quality_profile, quality_context)
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

        _patch_metadata(tmp_out, input_xlsx, input_format, audit, quality_profile)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(tmp_out, output_dir)

        tmp_outline = tmp_root / "问卷数据处理与分析_执行大纲.txt"
        if tmp_outline.exists():
            shutil.copy2(tmp_outline, output_dir / "问卷数据处理与分析_执行大纲.txt")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


run_isolated = run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run questionnaire analysis with unified input handling.")
    parser.add_argument("--input-xlsx", default="原始数据_Amethyst.xlsx", help="Path to source xlsx file.")
    parser.add_argument(
        "--input-format",
        choices=[INPUT_FORMAT_AMETHYST_108, INPUT_FORMAT_NEW961_64],
        help="Optional input format override. Default is auto-detect from xlsx columns.",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory for this run.")
    parser.add_argument(
        "--quality-profile",
        default=QUALITY_PROFILE_BALANCED,
        choices=[QUALITY_PROFILE_BALANCED, QUALITY_PROFILE_LEGACY],
        help="Quality filtering profile for main analysis sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx)
    input_format = args.input_format or detect_input_format(input_xlsx)
    run_analysis(input_xlsx, input_format, Path(args.output_dir), quality_profile=args.quality_profile)
    print(f"analysis_done: {input_format} ({args.quality_profile}) -> {args.output_dir}")


if __name__ == "__main__":
    main()

