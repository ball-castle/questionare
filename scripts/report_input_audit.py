#!/usr/bin/env python3
"""Input audit helpers for chapter 6/7 report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_TABLE_FILES = [
    "survey_clean.csv",
    "单选题频数百分比表.csv",
    "多选题选择率表.csv",
    "交叉分析卡方汇总.csv",
    "交叉分析列联明细.csv",
    "MCA特征值.csv",
    "MCA类别坐标与贡献.csv",
    "MCA群体解释卡.txt",
    "Logit模型指标.csv",
    "Logit回归结果_主样本.csv",
    "Logit回归结果_敏感性样本.csv",
    "Logit稳健性方向对比.csv",
    "注意力题双口径对比.csv",
    "二阶聚类画像卡.csv",
    "二阶聚类候选K评估.csv",
    "聚类稳定性对比表.csv",
    "IPA结果表.csv",
    "IPA整改优先级表.csv",
    "IPA阈值敏感性表.csv",
    "问题-证据-建议对照表.csv",
    "建议落地行动矩阵.csv",
    "假设变量模型映射表.csv",
]

REQUIRED_FIGURE_FILES = [
    "MCA二维图.png",
    "二阶聚类画像图.png",
    "IPA象限图.png",
    "核心画像图_core_profile.png",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_path(path: Path) -> str:
    return path.as_posix()


def _record(path: Path, category: str, name: str) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "path": _fmt_path(path),
        "exists": path.exists(),
    }


def _collect_records(tables_dir: Path, figures_dir: Path, outline_md: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.append(_record(outline_md, "outline", outline_md.name))
    for filename in REQUIRED_TABLE_FILES:
        records.append(_record(tables_dir / filename, "table", filename))
    for filename in REQUIRED_FIGURE_FILES:
        records.append(_record(figures_dir / filename, "figure", filename))
    return records


def audit_required_inputs(
    tables_dir: Path,
    figures_dir: Path,
    outline_md: Path,
    output_path: Path | None = None,
    missing_policy: str = "keep_placeholder",
) -> dict[str, Any]:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    outline_md = Path(outline_md)

    records = _collect_records(tables_dir=tables_dir, figures_dir=figures_dir, outline_md=outline_md)
    missing_items = [str(rec["path"]) for rec in records if not bool(rec["exists"])]

    result: dict[str, Any] = {
        "audited_at": now_iso(),
        "outline_md": _fmt_path(outline_md),
        "tables_dir": _fmt_path(tables_dir),
        "figures_dir": _fmt_path(figures_dir),
        "required_count": len(records),
        "missing_count": len(missing_items),
        "missing_items": missing_items,
        "items": records,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing_items and missing_policy == "fail":
        missing_text = "\n".join(missing_items)
        raise FileNotFoundError(f"missing required report inputs:\n{missing_text}")

    return result
