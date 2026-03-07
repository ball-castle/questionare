#!/usr/bin/env python3
"""本脚本用于对880样本执行信度与效度分析。"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .qp_io import write_dict_csv
from .qp_stats import cronbach_alpha, kmo_bartlett


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reliability and validity analysis on 880 clean sample.")
    parser.add_argument("--clean-csv", default="data/processed_880/survey_clean_880.csv", help="Path to clean sample CSV.")
    parser.add_argument("--out-dir", default="output_data_analysis", help="Output directory for analysis artifacts.")
    parser.add_argument("--meta-path", default="data/processed_880/run_metadata.json", help="Rescreen metadata path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _safe_float(v):
    txt = str(v or "").strip()
    if not txt:
        return np.nan
    try:
        return float(txt)
    except Exception:
        return np.nan


def _load_clean_matrix(clean_csv: Path) -> np.ndarray:
    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean sample CSV not found: {clean_csv}")
    with clean_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Clean sample CSV has no rows: {clean_csv}")
    mat = np.full((len(rows), 108), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for c in range(1, 109):
            mat[i, c - 1] = _safe_float(row.get(f"C{c:03d}", ""))
    return mat


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    clean_csv = Path(args.clean_csv)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    meta_path = Path(args.meta_path)

    key_files = [
        tables_dir / "信度分析表.csv",
        tables_dir / "效度分析表.csv",
        out_dir / "run_metadata.json",
    ]
    if not args.force and all(p.exists() for p in key_files):
        print(f"skipped: key outputs already exist at {out_dir}")
        return

    mat = _load_clean_matrix(clean_csv)
    blocks = {
        "感知维度(52-63,65)": list(range(52, 64)) + [65],
        "重要度维度(66-75)": list(range(66, 76)),
        "表现维度(76-85)": list(range(76, 86)),
        "认知维度(86-89)": list(range(86, 90)),
        "综合量表(52-63,65,66-85,86-89)": list(range(52, 64)) + [65] + list(range(66, 86)) + list(range(86, 90)),
    }

    reliability_rows = []
    for block, cols in blocks.items():
        alpha, n_complete = cronbach_alpha(mat[:, [c - 1 for c in cols]])
        reliability_rows.append({"block": block, "alpha": alpha, "n_complete": n_complete})
    validity = kmo_bartlett(mat[:, [c - 1 for c in (list(range(52, 64)) + [65] + list(range(66, 86)) + list(range(86, 90)))]])

    write_dict_csv(tables_dir / "信度分析表.csv", ["block", "alpha", "n_complete"], reliability_rows)
    write_dict_csv(tables_dir / "效度分析表.csv", ["n_complete", "kmo", "bartlett_chi2", "bartlett_df", "bartlett_p"], [validity])

    summary_lines = [
        "信效度分析摘要（880主样本）",
        f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"input_clean_csv: {clean_csv}",
        "",
        "信度（Cronbach alpha）:",
    ]
    for row in reliability_rows:
        alpha_val = row["alpha"]
        alpha_text = "nan" if np.isnan(alpha_val) else f"{alpha_val:.4f}"
        summary_lines.append(f"- {row['block']}: alpha={alpha_text}, n_complete={row['n_complete']}")
    summary_lines.extend(
        [
            "",
            "效度（KMO/Bartlett）:",
            f"- n_complete={validity['n_complete']}",
            f"- KMO={validity['kmo']:.4f}" if not np.isnan(validity["kmo"]) else "- KMO=nan",
            f"- Bartlett_chi2={validity['bartlett_chi2']:.3f}" if not np.isnan(validity["bartlett_chi2"]) else "- Bartlett_chi2=nan",
            f"- Bartlett_df={validity['bartlett_df']}",
            f"- Bartlett_p={validity['bartlett_p']:.6g}" if not np.isnan(validity["bartlett_p"]) else "- Bartlett_p=nan",
        ]
    )
    (tables_dir / "信效度摘要.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    src_meta = _load_meta(meta_path)
    out_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": "run_reliability_validity_880.py",
        "input_clean_csv": str(clean_csv),
        "source_meta_path": str(meta_path),
        "quality_profile": src_meta.get("quality_profile", "unknown"),
        "raw_n": src_meta.get("n_samples"),
        "remain_n_revised": src_meta.get("remain_n_revised"),
        "sample_n_from_clean": int(mat.shape[0]),
        "reliability_blocks_n": len(reliability_rows),
        "validity_n_complete": validity["n_complete"],
        "kmo": validity["kmo"],
        "bartlett_chi2": validity["bartlett_chi2"],
        "bartlett_df": validity["bartlett_df"],
        "bartlett_p": validity["bartlett_p"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: reliability/validity -> {out_dir}")


if __name__ == "__main__":
    main()

