#!/usr/bin/env python3
"""User-specified rescreen pipeline with strict logic checks and Excel delivery."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qp_io import numeric_matrix, read_xlsx_first_sheet, write_dict_csv, write_rows_csv


OPEN_GIBBERISH_TOKENS = {"好", "行", "嗯呢"}
OPEN_GIBBERISH_PUNCT_RE = re.compile(r"[/\\\.\|。\-_=+*~`!@#$%^&()\[\]{}<>?]+")
OPEN_GIBBERISH_SHORT_NUM_RE = re.compile(r"\d{1,3}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run user-defined questionnaire rescreen and deliver a 3-sheet Excel.")
    parser.add_argument("--input-xlsx", default="data/叶开泰问卷数据.xlsx", help="Input questionnaire xlsx path.")
    parser.add_argument("--tables-dir", default="data/_rescreen_tables", help="Directory for intermediate CSV artifacts.")
    parser.add_argument("--output-xlsx", default="data/问卷重筛结果.xlsx", help="Final delivered xlsx path.")
    parser.add_argument(
        "--logic-mode",
        default="strict",
        choices=["strict", "medium", "branch"],
        help="Logic conflict mode: strict=Q8 no but fill stay/spend/visit_exp (and symmetric check), medium=Q8 no but fill visit_exp only, branch=legacy branch marker conflict.",
    )
    return parser.parse_args()


def _safe_text(v) -> str:
    return str(v or "").strip()


def _open_text_is_gibberish(text: str) -> bool:
    t = _safe_text(text)
    if not t:
        return False
    compact = re.sub(r"\s+", "", t)
    if not compact:
        return False
    if compact in OPEN_GIBBERISH_TOKENS:
        return True
    if OPEN_GIBBERISH_PUNCT_RE.fullmatch(compact):
        return True
    if OPEN_GIBBERISH_SHORT_NUM_RE.fullmatch(compact):
        return True
    return False


def _is_filled(arr: np.ndarray) -> np.ndarray:
    return (~np.isnan(arr)) & (arr != 0) & (arr != -3)


def _build_duplicate_exact_flags(rows_dense: list[list[str]]) -> np.ndarray:
    answer_tuples = [tuple(r) for r in rows_dense]
    answer_count = Counter(answer_tuples)
    seen_answer = defaultdict(int)
    duplicate = np.zeros(len(rows_dense), dtype=int)
    for i, row_key in enumerate(answer_tuples):
        seen_answer[row_key] += 1
        duplicate[i] = int(answer_count[row_key] > 1 and seen_answer[row_key] > 1)
    return duplicate


def _build_logic_conflict_flags(
    num_raw: np.ndarray,
    q8: np.ndarray,
    stay: np.ndarray,
    spend: np.ndarray,
    visit_exp: np.ndarray,
    unvisit_reason: np.ndarray,
    logic_mode: str,
) -> np.ndarray:
    if logic_mode == "strict":
        return (
            ((q8 == 2) & (_is_filled(stay) | _is_filled(spend) | np.any(_is_filled(visit_exp), axis=1)))
            | ((q8 == 1) & np.any(_is_filled(unvisit_reason), axis=1))
        ).astype(int)
    if logic_mode == "medium":
        return ((q8 == 2) & np.any(_is_filled(visit_exp), axis=1)).astype(int)
    if logic_mode == "branch":
        visit_block = num_raw[:, 32:42]
        unvisit_block = num_raw[:, 42:51]
        return (
            ((q8 == 1) & np.any(visit_block == -3, axis=1))
            | ((q8 == 1) & np.any((unvisit_block != -3) & (~np.isnan(unvisit_block)), axis=1))
            | ((q8 == 2) & np.any((visit_block != -3) & (~np.isnan(visit_block)), axis=1))
            | ((q8 == 2) & np.any(unvisit_block == -3, axis=1))
        ).astype(int)
    raise ValueError(f"Unsupported logic_mode: {logic_mode}")


def _render_excel_from_tables(tables_dir: Path, output_xlsx: Path) -> None:
    js_renderer = Path(__file__).resolve().parent / "js" / "render_pretty_tables.mjs"
    cmd = [
        "node",
        str(js_renderer),
        "--tables-dir",
        str(tables_dir),
        "--output-xlsx",
        str(output_xlsx),
        "--title",
        "问卷重筛结果（三表合一）",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx)
    tables_dir = Path(args.tables_dir)
    output_xlsx = Path(args.output_xlsx)

    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input xlsx not found: {input_xlsx}")

    headers, rows_dense = read_xlsx_first_sheet(input_xlsx)
    if len(headers) != 108:
        raise ValueError(f"Expected 108 columns, got {len(headers)} for {input_xlsx}")

    num_raw, _ = numeric_matrix(rows_dense)
    n = len(rows_dense)

    q8 = num_raw[:, 7]
    stay = num_raw[:, 23]
    spend = num_raw[:, 24]
    visit_exp = num_raw[:, 25:42]
    unvisit_reason = num_raw[:, 42:51]

    logic_conflict = _build_logic_conflict_flags(
        num_raw=num_raw,
        q8=q8,
        stay=stay,
        spend=spend,
        visit_exp=visit_exp,
        unvisit_reason=unvisit_reason,
        logic_mode=args.logic_mode,
    )

    key_idx = list(range(0, 8)) + list(range(51, 89)) + [89, 90]
    key_missing = np.any(np.isnan(num_raw[:, key_idx]), axis=1).astype(int)

    open_gibberish = np.array(
        [int(_open_text_is_gibberish(rows_dense[i][107] if len(rows_dense[i]) >= 108 else "")) for i in range(n)],
        dtype=int,
    )

    duplicate_exact = _build_duplicate_exact_flags(rows_dense)

    invalid_union = (
        (logic_conflict == 1)
        | (key_missing == 1)
        | (open_gibberish == 1)
        | (duplicate_exact == 1)
    ).astype(int)

    flags = {
        "logic_conflict_flag": logic_conflict,
        "key_missing_flag": key_missing,
        "open_gibberish_flag": open_gibberish,
        "duplicate_exact_flag": duplicate_exact,
        "invalid_union_flag": invalid_union,
    }

    base_header = ["respondent_id"] + [f"C{i:03d}" for i in range(1, 109)]
    flag_header = list(flags.keys())
    keep_header = base_header + flag_header
    remove_header = keep_header + ["remove_reasons"]

    keep_rows = []
    remove_rows = []
    for i in range(n):
        rid = i + 1
        flag_values = [int(flags[name][i]) for name in flag_header]
        base_values = [rid] + rows_dense[i]
        if invalid_union[i] == 0:
            keep_rows.append(base_values + flag_values)
        else:
            reasons = []
            if logic_conflict[i] == 1:
                reasons.append("logic_conflict")
            if key_missing[i] == 1:
                reasons.append("key_missing")
            if open_gibberish[i] == 1:
                reasons.append("open_gibberish")
            if duplicate_exact[i] == 1:
                reasons.append("duplicate_exact")
            remove_rows.append(base_values + flag_values + [",".join(reasons)])

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(tables_dir / "保留样本.csv", keep_header, keep_rows)
    write_rows_csv(tables_dir / "剔除样本.csv", remove_header, remove_rows)

    raw_n = int(n)
    invalid_n = int(np.sum(invalid_union))
    remain_n = int(raw_n - invalid_n)
    summary_rows = [
        {"metric": "raw_n", "value": raw_n},
        {"metric": "invalid_n", "value": invalid_n},
        {"metric": "remain_n", "value": remain_n},
        {"metric": "logic_mode", "value": args.logic_mode},
        {"metric": "logic_conflict_n", "value": int(np.sum(logic_conflict))},
        {"metric": "key_missing_n", "value": int(np.sum(key_missing))},
        {"metric": "open_gibberish_n", "value": int(np.sum(open_gibberish))},
        {"metric": "duplicate_exact_n", "value": int(np.sum(duplicate_exact))},
        {"metric": "duration_rule_skipped", "value": "true"},
        {"metric": "contact_duplicate_rule_skipped", "value": "true"},
        {"metric": "skipped_due_to_missing_fields", "value": "true"},
    ]
    write_dict_csv(tables_dir / "样本流转摘要.csv", ["metric", "value"], summary_rows)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_xlsx": str(input_xlsx),
        "tables_dir": str(tables_dir),
        "output_xlsx": str(output_xlsx),
        "input_columns": len(headers),
        "logic_mode": args.logic_mode,
        "raw_n": raw_n,
        "invalid_n": invalid_n,
        "remain_n": remain_n,
        "flags_n": {
            "logic_conflict_n": int(np.sum(logic_conflict)),
            "key_missing_n": int(np.sum(key_missing)),
            "open_gibberish_n": int(np.sum(open_gibberish)),
            "duplicate_exact_n": int(np.sum(duplicate_exact)),
        },
        "rules": {
            "logic_conflict_mode_details": {
                "strict": "C008=2 with filled C024/C025/C026-C042, or C008=1 with filled C043-C051",
                "medium": "C008=2 with filled C026-C042",
                "branch": "legacy branch marker conflict by -3 in C033-C051",
            },
            "filled_definition": "non-empty numeric and not in {0,-3}",
            "key_missing": "Any missing in C001-C008, C052-C089, C090-C091",
            "open_gibberish": "C108 in {'好','行','嗯呢'} or punct-only or short-number(1-3 digits)",
            "duplicate_exact": "Exact duplicate over full C001-C108 row; keep first, remove later duplicates",
        },
        "skipped_rules": {
            "duration_lt90": True,
            "contact_duplicate": True,
            "reason": "missing required fields in this 108-column file",
        },
    }
    (tables_dir / "重筛元数据.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _render_excel_from_tables(tables_dir=tables_dir, output_xlsx=output_xlsx)
    print(f"rescreen_done: raw={raw_n}, invalid={invalid_n}, remain={remain_n}, xlsx={output_xlsx}")


if __name__ == "__main__":
    main()
