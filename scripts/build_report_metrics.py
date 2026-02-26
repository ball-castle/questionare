#!/usr/bin/env python3
"""Build metrics JSON and age-gender figure for JS docx backfill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from qp_io import numeric_matrix, read_xlsx_first_sheet
from qp_stats import cronbach_alpha, kmo_bartlett

matplotlib.use("Agg")
from matplotlib import pyplot as plt


AGE_ORDER = [1, 3, 2, 4, 5]
AGE_LABELS = {
    1: "18岁以下",
    3: "18-25岁",
    2: "26-45岁",
    4: "46-64岁",
    5: "65岁及以上",
}

MALE_COLOR = "#BFE3C0"
FEMALE_COLOR = "#2E7D32"
CORE_COLOR = "#1f77b4"
IPA_COLOR = "#2ca02c"
AUX_COLOR = "gray"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def alpha_grade(alpha: float) -> str:
    if alpha >= 0.8:
        return "良好"
    if alpha >= 0.7:
        return "可接受"
    return "一般"


def freq_map(num: np.ndarray, col_1b: int, codes: list[int]) -> dict[int, dict[str, float]]:
    v = num[:, col_1b - 1]
    out = {}
    tot = int(np.sum(~np.isnan(v)))
    for c in codes:
        cnt = int(np.sum(v == c))
        pct = 100.0 * cnt / tot if tot > 0 else 0.0
        out[c] = {"count": cnt, "pct": pct}
    return out


def build_age_gender(num: np.ndarray, valid_n: int) -> tuple[list[dict], dict]:
    age_vals = num[:, 1]
    sex_vals = num[:, 0]
    rows = []
    counts = {k: {1: 0, 2: 0} for k in AGE_ORDER}

    for age_code in AGE_ORDER:
        for sex_code in [1, 2]:
            cnt = int(np.sum((age_vals == age_code) & (sex_vals == sex_code)))
            counts[age_code][sex_code] = cnt
        total_n = counts[age_code][1] + counts[age_code][2]
        pct = 100.0 * total_n / valid_n if valid_n > 0 else 0.0
        rows.append(
            {
                "age_code": age_code,
                "age_label": AGE_LABELS[age_code],
                "male": counts[age_code][1],
                "female": counts[age_code][2],
                "total": total_n,
                "pct": round(pct, 2),
            }
        )
    return rows, counts


def plot_age_gender(age_rows: list[dict], out_path: Path) -> None:
    ensure_parent(out_path)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = [r["age_label"] for r in age_rows]
    male = [r["male"] for r in age_rows]
    female = [r["female"] for r in age_rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y, male, color=MALE_COLOR, label="男")
    ax.barh(y, female, left=male, color=FEMALE_COLOR, label="女")
    ax.set_yticks(y, labels)
    ax.set_xlabel("人数")
    ax.set_ylabel("年龄段")
    ax.set_title("各年龄段人数（按性别堆叠）")
    ax.legend(frameon=False)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=AUX_COLOR)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build report metrics for JS docx backfill.")
    p.add_argument("--input-xlsx", required=True, help="Path to 108-column xlsx.")
    p.add_argument("--distributed-n", type=int, default=1000, help="Distributed/recovered questionnaire total.")
    p.add_argument("--output-json", required=True, help="Output metrics JSON path.")
    p.add_argument("--age-figure", required=True, help="Output age-gender figure path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.input_xlsx)
    output_json = Path(args.output_json)
    age_figure = Path(args.age_figure)

    headers, rows_dense = read_xlsx_first_sheet(xlsx_path)
    if len(headers) != 108:
        raise ValueError(f"Expected 108 columns, got {len(headers)} from {xlsx_path}")

    num, _ = numeric_matrix(rows_dense)
    valid_n = len(rows_dense)
    distributed_n = int(args.distributed_n)
    recovered_n = distributed_n
    valid_rate = 100.0 * valid_n / distributed_n if distributed_n > 0 else 0.0

    q1 = freq_map(num, 1, [1, 2])
    q2 = freq_map(num, 2, [1, 2, 3, 4, 5])
    q3 = freq_map(num, 3, [1, 2, 3, 4, 5])
    q4 = freq_map(num, 4, [1, 2, 3, 4, 5, 6, 7, 8])
    q5 = freq_map(num, 5, [1, 2, 3, 4, 5])
    q8 = freq_map(num, 8, [1, 2])

    rel_blocks = [
        ("文化体验维度", [52, 53, 54]),
        ("非遗体验维度", [53, 54]),
        ("产品体验维度", [55, 56, 57]),
        ("配套保障维度", [58, 59, 60, 61]),
        ("宣传策略维度", [62, 63, 65]),
        ("整体量表", list(range(52, 64)) + [65]),
    ]
    rel_rows = []
    for name, cols in rel_blocks:
        alpha, n_complete = cronbach_alpha(num[:, [c - 1 for c in cols]])
        rel_rows.append(
            {
                "dim_name": name,
                "item_count": len(cols),
                "alpha": round(float(alpha), 4) if np.isfinite(alpha) else None,
                "grade": alpha_grade(float(alpha)) if np.isfinite(alpha) else "一般",
                "n_complete": int(n_complete),
            }
        )

    val = kmo_bartlett(num[:, [c - 1 for c in list(range(52, 64)) + [65]]])
    val_p = float(val["bartlett_p"]) if np.isfinite(val["bartlett_p"]) else np.nan
    val_obj = {
        "kmo": round(float(val["kmo"]), 4) if np.isfinite(val["kmo"]) else None,
        "bartlett_chi2": round(float(val["bartlett_chi2"]), 3) if np.isfinite(val["bartlett_chi2"]) else None,
        "bartlett_df": int(val["bartlett_df"]) if np.isfinite(val["bartlett_df"]) else None,
        "bartlett_p": "<0.001" if np.isfinite(val_p) and val_p < 0.001 else (f"{val_p:.3f}" if np.isfinite(val_p) else None),
        "n_complete": int(val["n_complete"]),
    }

    age_rows, age_gender_counts = build_age_gender(num, valid_n)
    plot_age_gender(age_rows, age_figure)

    age_desc = "，".join([f"{AGE_LABELS[c]}占{q2[c]['pct']:.2f}%" for c in AGE_ORDER])
    canonical_alloc = (
        f"正式调查按既定配额实施，累计发放问卷{distributed_n}份，回收问卷{recovered_n}份，"
        f"质控后有效问卷{valid_n}份（到访{q8[1]['count']}份、未到访{q8[2]['count']}份），有效率{valid_rate:.2f}%。"
    )
    canonical_overall = (
        f"本次调查累计发放问卷{distributed_n}份，回收问卷{recovered_n}份，"
        f"质控后有效问卷{valid_n}份（有效率{valid_rate:.2f}%）。以下为有效问卷样本的结构分析。"
    )
    canonical_gender_age = (
        f"本次调查回收有效问卷共{valid_n}份，在所有受访者中，男女比例为{q1[1]['count']}:{q1[2]['count']}"
        f"（{q1[1]['pct']:.2f}%:{q1[2]['pct']:.2f}%）。从年龄分布看，{age_desc}。"
    )

    table5_rows = []

    def add_row(var_name: str, attr: str, count: int, pct: float, total: str) -> None:
        table5_rows.append(
            {
                "var_name": var_name,
                "attr": attr,
                "count": int(count),
                "pct": f"{pct:.2f}%",
                "total": total,
            }
        )

    add_row("文化程度", "初中及以下", q3[1]["count"], q3[1]["pct"], "100%")
    add_row("", "中专/高中", q3[2]["count"], q3[2]["pct"], "")
    add_row("", "大专", q3[3]["count"], q3[3]["pct"], "")
    add_row("", "本科", q3[4]["count"], q3[4]["pct"], "")
    add_row("", "硕士及以上", q3[5]["count"], q3[5]["pct"], "")
    add_row("职业", "学生", q4[1]["count"], q4[1]["pct"], "100%")
    add_row("", "企业/公司职员", q4[2]["count"], q4[2]["pct"], "")
    add_row("", "事业单位人员/公务员", q4[3]["count"], q4[3]["pct"], "")
    add_row("", "自由职业者", q4[4]["count"], q4[4]["pct"], "")
    add_row("", "个体经营者", q4[5]["count"], q4[5]["pct"], "")
    add_row("", "服务业从业者", q4[6]["count"], q4[6]["pct"], "")
    add_row("", "离退休人员", q4[7]["count"], q4[7]["pct"], "")
    add_row("", "其他", q4[8]["count"], q4[8]["pct"], "")
    add_row("月收入", "3000元以下", q5[1]["count"], q5[1]["pct"], "100%")
    add_row("", "3001-5000元", q5[2]["count"], q5[2]["pct"], "")
    add_row("", "5001-8000元", q5[3]["count"], q5[3]["pct"], "")
    add_row("", "8001-15000元", q5[4]["count"], q5[4]["pct"], "")
    add_row("", "15000元以上", q5[5]["count"], q5[5]["pct"], "")

    metrics = {
        "input_xlsx": str(xlsx_path),
        "distributed_n": distributed_n,
        "recovered_n": recovered_n,
        "valid_n": valid_n,
        "valid_rate_pct": round(valid_rate, 2),
        "paragraphs": {
            "formal_allocation": canonical_alloc,
            "overall_summary": canonical_overall,
            "gender_age": canonical_gender_age,
        },
        "tables": {
            "table3_reliability": rel_rows,
            "table4_validity": val_obj,
            "table5_sample_structure": table5_rows,
            "table5_valid_count": valid_n,
            "pretest_policy": "keep_pending",
        },
        "age_gender": {
            "order": AGE_ORDER,
            "labels": {str(k): AGE_LABELS[k] for k in AGE_ORDER},
            "counts": {str(k): age_gender_counts[k] for k in AGE_ORDER},
            "rows": age_rows,
        },
        "artifacts": {
            "age_gender_figure": str(age_figure),
        },
        "style_constants": {
            "male_color": MALE_COLOR,
            "female_color": FEMALE_COLOR,
            "core_color": CORE_COLOR,
            "ipa_color": IPA_COLOR,
            "aux_color": AUX_COLOR,
        },
    }

    ensure_parent(output_json)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics_done: valid_n={valid_n}, distributed_n={distributed_n}, output={output_json}")


if __name__ == "__main__":
    main()
