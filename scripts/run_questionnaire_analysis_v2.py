#!/usr/bin/env python3
"""Run questionnaire analysis into an isolated output directory for two input formats."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

from convert_961_to_108 import convert_961_to_108, export_conversion_artifacts
from qp_io import read_xlsx_first_sheet, write_dict_csv

QUALITY_PROFILE_LEGACY = "legacy_balanced"
QUALITY_PROFILE_BALANCED = "balanced_v20260221"


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


def _run_base_main_with_loader(loader, tmp_root: Path, quality_profile: str, quality_context: dict) -> None:
    base = _load_base_module()
    old_loader = base.read_xlsx_first_sheet
    old_cwd = Path.cwd()
    had_profile = hasattr(base, "FORCE_QUALITY_PROFILE")
    old_profile = getattr(base, "FORCE_QUALITY_PROFILE", None)
    had_context = hasattr(base, "QUALITY_CONTEXT")
    old_context = getattr(base, "QUALITY_CONTEXT", None)
    try:
        base.read_xlsx_first_sheet = loader
        base.FORCE_QUALITY_PROFILE = quality_profile
        base.QUALITY_CONTEXT = quality_context
        os.chdir(tmp_root)
        base.main()
    finally:
        base.read_xlsx_first_sheet = old_loader
        if had_profile:
            base.FORCE_QUALITY_PROFILE = old_profile
        elif hasattr(base, "FORCE_QUALITY_PROFILE"):
            delattr(base, "FORCE_QUALITY_PROFILE")
        if had_context:
            base.QUALITY_CONTEXT = old_context
        elif hasattr(base, "QUALITY_CONTEXT"):
            delattr(base, "QUALITY_CONTEXT")
        os.chdir(old_cwd)


def _patch_metadata(out_dir: Path, input_xlsx: Path, input_format: str, audit: dict, quality_profile: str) -> None:
    meta_path = out_dir / "run_metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["input"] = str(input_xlsx)
    meta["input_format"] = input_format
    meta["quality_profile"] = quality_profile
    meta["source_pipeline"] = "run_questionnaire_analysis_v2"
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
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    if m is None:
        return None
    try:
        return float(m.group(1))
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


def _prepare_input_loader(input_xlsx: Path, input_format: str) -> Tuple[callable, dict, object, dict]:
    if input_format == "amethyst_108":
        headers, rows = read_xlsx_first_sheet(input_xlsx)

        def loader(_):
            return headers, rows

        return loader, _default_audit(len(rows)), None, {}

    if input_format == "new961_64":
        headers, rows = read_xlsx_first_sheet(input_xlsx)
        conv = convert_961_to_108(headers, rows)
        quality_context = _build_quality_context_from_64(rows)

        def loader(_):
            return conv.headers_108, conv.rows_108

        return loader, conv.audit, conv, quality_context

    raise ValueError(f"Unsupported input_format: {input_format}")


def run_isolated(input_xlsx: Path, input_format: str, output_dir: Path, quality_profile: str = QUALITY_PROFILE_BALANCED) -> None:
    input_xlsx = Path(input_xlsx)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    loader, audit, conv, quality_context = _prepare_input_loader(input_xlsx, input_format)
    tmp_root = Path(tempfile.mkdtemp(prefix="qp_run_"))
    try:
        _run_base_main_with_loader(loader, tmp_root, quality_profile, quality_context)
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
    parser.add_argument(
        "--quality-profile",
        default=QUALITY_PROFILE_BALANCED,
        choices=[QUALITY_PROFILE_BALANCED, QUALITY_PROFILE_LEGACY],
        help="Quality filtering profile for main analysis sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_isolated(Path(args.input_xlsx), args.input_format, Path(args.output_dir), quality_profile=args.quality_profile)
    print(f"analysis_v2_done: {args.input_format} ({args.quality_profile}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
