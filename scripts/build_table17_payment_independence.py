#!/usr/bin/env python3
"""Build table 17: payment willingness independence tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table 17 independence-test p-values.")
    parser.add_argument(
        "--input-csv",
        default="data/data_analysis/_source_analysis/tables/survey_clean.csv",
        help="Input cleaned survey CSV.",
    )
    parser.add_argument(
        "--out-csv",
        default="new/MCA/表17_知识付费意愿独立性检验_p值.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-json",
        default="new/MCA/表17_知识付费意愿独立性检验_说明.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _p3(p: float) -> str:
    return f"{p:.3f}"


def _table_dict(tab: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in tab.index.tolist():
        rk = str(int(r))
        out[rk] = {}
        for c in tab.columns.tolist():
            ck = str(int(c))
            out[rk][ck] = int(tab.loc[r, c])
    return out


def _chi2_yates(a: pd.Series, b: pd.Series) -> tuple[float, pd.DataFrame]:
    tab = pd.crosstab(a, b)
    p = float(chi2_contingency(tab.values, correction=True)[1])
    return p, tab


def _fisher(a: pd.Series, b: pd.Series) -> tuple[float, pd.DataFrame]:
    tab = pd.crosstab(a, b)
    p = float(fisher_exact(tab.values)[1])
    return p, tab


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    cols = ["C025", "C002", "C003", "C005"]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    sub = sub[
        sub["C025"].between(1, 5)
        & sub["C002"].between(1, 5)
        & sub["C003"].between(1, 5)
        & sub["C005"].between(1, 5)
    ].copy()
    sub = sub.astype(int)

    # Definition used for this table:
    # payment willingness = C025 >= 3 (medium/high spend intent).
    sub["pay_willing_bin"] = (sub["C025"] >= 3).astype(int)
    sub["age_bin"] = (sub["C002"] <= 3).astype(int)
    sub["edu_bin"] = (sub["C003"] <= 3).astype(int)
    sub["income_bin"] = (sub["C005"] <= 3).astype(int)

    p_wa, t_wa = _chi2_yates(sub["pay_willing_bin"], sub["age_bin"])
    p_we, t_we = _fisher(sub["pay_willing_bin"], sub["edu_bin"])
    p_wi, t_wi = _chi2_yates(sub["pay_willing_bin"], sub["income_bin"])
    p_ae, t_ae = _fisher(sub["age_bin"], sub["edu_bin"])
    p_ai, t_ai = _fisher(sub["age_bin"], sub["income_bin"])
    p_ei, t_ei = _fisher(sub["edu_bin"], sub["income_bin"])

    rows = [
        {"变量": "知识付费意愿-年龄", "检验方法": "校正的卡方检验", "P值": p_wa},
        {"变量": "知识付费意愿-教育背景", "检验方法": "Fisher 精确检验", "P值": p_we},
        {"变量": "知识付费意愿-月可支配收入", "检验方法": "校正的卡方检验", "P值": p_wi},
        {"变量": "年龄-教育背景", "检验方法": "Fisher 精确检验", "P值": p_ae},
        {"变量": "年龄-月可支配收入", "检验方法": "Fisher 精确检验", "P值": p_ai},
        {"变量": "教育背景-月可支配收入", "检验方法": "Fisher 精确检验", "P值": p_ei},
    ]

    for r in rows:
        p = float(r["P值"])
        r["P值_三位"] = _p3(p)
        r["显著性"] = _sig_stars(p)
        r["P值_展示"] = f"{r['P值_三位']}{r['显著性']}"

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    details = {
        "source": str(in_path).replace("\\", "/"),
        "sample_n": int(len(sub)),
        "coding": {
            "知识付费意愿_二分": "C025>=3 记为1（愿意为景区花钱/较高消费意愿），否则0",
            "年龄_二分": "C002<=3 记为1，否则0",
            "教育背景_二分": "C003<=3 记为1，否则0",
            "月可支配收入_二分": "C005<=3 记为1，否则0",
        },
        "methods": {
            "知识付费意愿-年龄": "chi2_contingency(correction=True)",
            "知识付费意愿-教育背景": "fisher_exact",
            "知识付费意愿-月可支配收入": "chi2_contingency(correction=True)",
            "年龄-教育背景": "fisher_exact",
            "年龄-月可支配收入": "fisher_exact",
            "教育背景-月可支配收入": "fisher_exact",
        },
        "p_values": {r["变量"]: float(r["P值"]) for r in rows},
        "tables": {
            "知识付费意愿-年龄": _table_dict(t_wa),
            "知识付费意愿-教育背景": _table_dict(t_we),
            "知识付费意愿-月可支配收入": _table_dict(t_wi),
            "年龄-教育背景": _table_dict(t_ae),
            "年龄-月可支配收入": _table_dict(t_ai),
            "教育背景-月可支配收入": _table_dict(t_ei),
        },
    }
    out_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
