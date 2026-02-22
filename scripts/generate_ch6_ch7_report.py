#!/usr/bin/env python3
"""Generate chapter 6/7 integrated report (Markdown + DOCX)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from report_data_extractors import extract_report_data
from report_docx_exporter import export_docx
from report_input_audit import audit_required_inputs
from report_outline_renderer import build_outline_report
from report_template_renderer import render_markdown, render_report_content

OUTLINE_DEFAULT = "六七大纲.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chapter 6/7 report from outline markdown.")
    parser.add_argument(
        "--outline-md",
        default=OUTLINE_DEFAULT,
        help="Outline markdown path.",
    )
    parser.add_argument(
        "--tables-dir",
        default="output_report/tables",
        help="Tables directory produced by analysis pipeline.",
    )
    parser.add_argument(
        "--figures-dir",
        default="output_report/figures",
        help="Figures directory produced by analysis pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_report",
        help="Report output directory.",
    )
    parser.add_argument(
        "--base-name",
        default="六七部分_完整报告",
        help="Base filename for markdown and docx outputs (without extension).",
    )
    parser.add_argument(
        "--missing-policy",
        default="keep_placeholder",
        choices=["keep_placeholder", "fail"],
        help="How to handle missing inputs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run input audit and write audit/log outputs.",
    )
    parser.add_argument(
        "--auto-migrate-output-current",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-copy output_current to output_report when output_report does not exist yet.",
    )
    parser.add_argument(
        "--render-mode",
        default="data_rich",
        choices=["data_rich", "outline"],
        help="Render mode: data_rich (recommended) or outline (verbatim outline copy).",
    )
    return parser.parse_args()


def resolve_outline_path(path_arg: str, project_root: Path) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd
    return project_root / p


def maybe_migrate_output_current(tables_dir: Path, enabled: bool) -> bool:
    if not enabled:
        return False
    output_root = tables_dir.parent
    if output_root.name not in {"output", "output_report"}:
        return False
    legacy_root = output_root.parent / "output_current"
    if output_root.exists() or not legacy_root.exists():
        return False
    shutil.copytree(legacy_root, output_root)
    return True


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    outline_md = resolve_outline_path(args.outline_md, project_root)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    output_dir = Path(args.output_dir)
    migration_performed = maybe_migrate_output_current(tables_dir, args.auto_migrate_output_current)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "六七部分_输入核查报告.json"
    audit = audit_required_inputs(
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        outline_md=outline_md,
        output_path=audit_path,
        missing_policy=args.missing_policy,
    )
    if not outline_md.exists():
        raise FileNotFoundError(f"missing required outline: {outline_md}")

    log_path = output_dir / "六七章_生成日志.json"
    missing_items = list(audit.get("missing_items", []))

    if args.prepare_only:
        log_data = {
            "generated_at": now_iso(),
            "outline_md": str(outline_md),
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
            "output_dir": str(output_dir),
            "base_name": args.base_name,
            "missing_policy": args.missing_policy,
            "prepare_only": True,
            "render_mode": args.render_mode,
            "auto_migrate_output_current": bool(args.auto_migrate_output_current),
            "migration_performed": migration_performed,
            "input_audit_path": str(audit_path),
            "missing_items": missing_items,
            "missing_count": len(missing_items),
            "outputs": {
                "input_audit": str(audit_path),
                "log": str(log_path),
            },
        }
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_prepare_done: audit={audit_path} log={log_path}")
        return

    if args.render_mode == "outline":
        content, md_text = build_outline_report(
            outline_md=outline_md,
            title=args.base_name,
            missing_items=missing_items,
        )
    else:
        report_data = extract_report_data(
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            outline_md=outline_md,
            missing_policy=args.missing_policy,
        )
        content = render_report_content(
            data=report_data,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )
        md_text = render_markdown(content)

    md_path = output_dir / f"{args.base_name}.md"
    docx_path = output_dir / f"{args.base_name}.docx"
    evidence_path = output_dir / "六七章_证据索引.csv"

    md_path.write_text(md_text, encoding="utf-8")
    export_docx(content, docx_path)

    evidence_rows: list[dict[str, str]] = []
    seen = set()
    for row in content.get("evidence_rows", []):
        rec = {
            "section": str(row.get("section", "")),
            "statement": str(row.get("statement", "")),
            "evidence_path": str(row.get("evidence_path", "")),
        }
        key = (rec["section"], rec["statement"], rec["evidence_path"])
        if key in seen:
            continue
        seen.add(key)
        evidence_rows.append(rec)
    write_csv(evidence_path, ["section", "statement", "evidence_path"], evidence_rows)

    log_data = {
        "generated_at": now_iso(),
        "outline_md": str(outline_md),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "output_dir": str(output_dir),
        "base_name": args.base_name,
        "missing_policy": args.missing_policy,
        "prepare_only": False,
        "render_mode": args.render_mode,
        "auto_migrate_output_current": bool(args.auto_migrate_output_current),
        "migration_performed": migration_performed,
        "input_audit_path": str(audit_path),
        "missing_items": missing_items,
        "missing_count": len(missing_items),
        "outline_exists": bool(content.get("outline_exists", False)),
        "warning_count": len(content.get("warnings", [])),
        "warnings": list(content.get("warnings", [])),
        "outputs": {
            "markdown": str(md_path),
            "docx": str(docx_path),
            "evidence_index": str(evidence_path),
            "input_audit": str(audit_path),
            "log": str(log_path),
        },
        "evidence_rows": len(evidence_rows),
        "block_count": len(content.get("blocks", [])),
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_done: md={md_path} docx={docx_path} evidence={evidence_path} audit={audit_path} log={log_path}")


if __name__ == "__main__":
    main()
