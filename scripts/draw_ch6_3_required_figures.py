#!/usr/bin/env python3
"""Draw chapter 6.3 required figures and save them under data/figure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = [
    "M1_Q20_visit_intent",
    "M2_Q21_recommend_direct",
    "M3_Q21_recommend_with_q20",
]

MODEL_LABEL = {
    "M1_Q20_visit_intent": "M1",
    "M2_Q21_recommend_direct": "M2",
    "M3_Q21_recommend_with_q20": "M3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw chapter 6.3 required figures.")
    parser.add_argument("--input-dir", default="data/logit", help="Directory for logit csv outputs.")
    parser.add_argument("--output-dir", default="data/figure", help="Directory for generated figures.")
    parser.add_argument("--dpi", type=int, default=320, help="Output dpi.")
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def metric_value(metrics_df: pd.DataFrame, key: str) -> float:
    row = metrics_df.loc[metrics_df["metric"] == key, "value"]
    if row.empty:
        raise ValueError(f"Metric {key!r} not found in cross metrics.")
    return float(row.iloc[0])


def extract_or(or_df: pd.DataFrame, model_id: str, term: str) -> dict[str, float]:
    row = or_df[(or_df["model_id"] == model_id) & (or_df["term"] == term)]
    if row.empty:
        raise ValueError(f"Missing OR row for model={model_id}, term={term}")
    r = row.iloc[0]
    return {
        "or": float(r["odds_ratio"]),
        "low": float(r["or_ci_lower"]),
        "high": float(r["or_ci_upper"]),
    }


def draw_figure_1_cross_distribution(cross_df: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, float]:
    p_high = metric_value(cross_df, "p_q21_high_given_q20_high")
    p_low = metric_value(cross_df, "p_q21_high_given_q20_low")
    diff_pp = (p_high - p_low) * 100.0
    unadjusted_or = metric_value(cross_df, "odds_ratio_q20_to_q21_unadjusted")

    labels = ["高游览意愿组\n(Q20=1)", "低游览意愿组\n(Q20=0)"]
    values = np.array([p_high, p_low], dtype=float)
    colors = ["#2A9D8F", "#F4A261"]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(np.arange(2), values, width=0.56, color=colors, edgecolor="#2F2F2F", linewidth=0.8)

    for i, bar in enumerate(bars):
        h = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018, f"{h * 100:.2f}%", ha="center", va="bottom", fontsize=10)

    x_arrow = 1.25
    ax.annotate(
        "",
        xy=(x_arrow, p_high),
        xytext=(x_arrow, p_low),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.4, "color": "#1D3557"},
    )
    ax.text(
        x_arrow + 0.06,
        (p_high + p_low) / 2,
        f"差值 {diff_pp:.2f}pp",
        fontsize=10,
        va="center",
        color="#1D3557",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#F1FAEE", "edgecolor": "#1D3557", "linewidth": 0.8},
    )

    ax.text(
        0.02,
        0.95,
        f"未控制变量 OR={unadjusted_or:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        color="#3A3A3A",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFF7E6", "edgecolor": "#D6A25E", "linewidth": 0.7},
    )

    ax.set_xticks([0, 1], labels)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlim(-0.5, 1.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("推荐意愿占比（Q21=1）")
    ax.set_title("图1 游览意愿与推荐意愿交叉分布图")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {"p_high": p_high, "p_low": p_low, "diff_pp": diff_pp, "or": unadjusted_or}


def draw_figure_2_forest(or_df: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, dict[str, float]]:
    rows = [
        ("M1 C088", extract_or(or_df, "M1_Q20_visit_intent", "C088"), "#457B9D"),
        ("M2 C088", extract_or(or_df, "M2_Q21_recommend_direct", "C088"), "#457B9D"),
        ("M3 Q20", extract_or(or_df, "M3_Q21_recommend_with_q20", "y_q20_high"), "#C1121F"),
        ("M3 C088", extract_or(or_df, "M3_Q21_recommend_with_q20", "C088"), "#457B9D"),
    ]
    labels = [r[0] for r in rows]
    ors = np.array([r[1]["or"] for r in rows], dtype=float)
    lows = np.array([r[1]["low"] for r in rows], dtype=float)
    highs = np.array([r[1]["high"] for r in rows], dtype=float)
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))

    xmax = float(np.max(highs)) * 2.0
    xmax = max(xmax, 5.0)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.axvline(1.0, color="#6B6B6B", linestyle="--", linewidth=1.2, label="OR=1 基准线")

    for i in range(len(rows)):
        ax.hlines(y[i], lows[i], highs[i], color=colors[i], linewidth=2.0, alpha=0.95)
        ax.plot(ors[i], y[i], marker="o", markersize=7, color=colors[i], markeredgecolor="#202020")
        ax.text(
            highs[i] + 0.08,
            y[i],
            f"{ors[i]:.3f} [{lows[i]:.3f}, {highs[i]:.3f}]",
            va="center",
            fontsize=9,
            color="#303030",
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.8, xmax)
    ax.set_xlabel("优势比 OR（95%CI）")
    ax.set_title("图2 三模型核心OR值对比图（森林图）")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {label: {"or": float(o), "low": float(l), "high": float(h)} for label, o, l, h in zip(labels, ors, lows, highs)}


def draw_figure_3_model_fit(model_df: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, dict[str, float]]:
    metric_cols = [
        ("AUC", "auc_oof", "#355070"),
        ("PR-AUC", "pr_auc_oof", "#6D597A"),
        ("McFadden R²", "pseudo_r2_mcfadden", "#B56576"),
    ]
    x = np.arange(len(MODEL_ORDER))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    figure_metrics: dict[str, dict[str, float]] = {}

    for i, (label, col, color) in enumerate(metric_cols):
        vals = []
        for model_id in MODEL_ORDER:
            row = model_df[model_df["model_id"] == model_id]
            if row.empty:
                raise ValueError(f"Missing model row in model metrics: {model_id}")
            v = float(row.iloc[0][col])
            vals.append(v)
            figure_metrics.setdefault(MODEL_LABEL[model_id], {})[col] = v
        positions = x + (i - 1) * width
        bars = ax.bar(positions, vals, width=width, label=label, color=color, edgecolor="#2F2F2F", linewidth=0.7)
        for bar in bars:
            h = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    brier_vals = []
    for model_id in MODEL_ORDER:
        row = model_df[model_df["model_id"] == model_id]
        v = float(row.iloc[0]["brier_oof"])
        brier_vals.append(v)
        figure_metrics[MODEL_LABEL[model_id]]["brier_oof"] = v

    ax2 = ax.twinx()
    ax2.plot(
        x,
        brier_vals,
        color="#2A9D8F",
        marker="D",
        markersize=6,
        linewidth=2.0,
        label="Brier（越低越好）",
    )
    for xi, val in zip(x, brier_vals):
        ax2.text(xi, val + 0.0025, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#116466")

    auc_m3 = figure_metrics["M3"]["auc_oof"]
    ax.annotate(
        "M3综合表现最佳",
        xy=(2, auc_m3),
        xytext=(1.25, 0.83),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "#264653"},
        fontsize=9,
        color="#264653",
    )

    ax.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylim(0.0, 0.9)
    y2_low = max(0.0, min(brier_vals) - 0.02)
    y2_high = min(1.0, max(brier_vals) + 0.03)
    ax2.set_ylim(y2_low, y2_high)
    ax.set_ylabel("模型拟合指标（越高越好）")
    ax2.set_ylabel("Brier（越低越好）")
    ax.set_title("图3 三模型拟合指标对比图")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return figure_metrics


def draw_figure_4_gradient(grad_df: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, tuple[float, float]]:
    use_df = grad_df.sort_values("c088_level").copy()
    x = use_df["c088_level"].astype(int).to_numpy()
    s1 = use_df["p_q20_high_m1_standardized"].astype(float).to_numpy()
    s2 = use_df["p_q21_high_m3_standardized_q20_0"].astype(float).to_numpy()
    s3 = use_df["p_q21_high_m3_standardized_q20_1"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(x, s1, marker="o", linewidth=2.3, color="#1D3557", label="M1 游览意愿概率")
    ax.plot(x, s2, marker="s", linewidth=2.3, color="#E76F51", label="M3 推荐概率（Q20=0）")
    ax.plot(x, s3, marker="^", linewidth=2.3, color="#2A9D8F", label="M3 推荐概率（Q20=1）")

    end_labels = [
        ("M1", s1[0], s1[-1], "#1D3557", 0.012),
        ("M3(Q20=0)", s2[0], s2[-1], "#E76F51", -0.02),
        ("M3(Q20=1)", s3[0], s3[-1], "#2A9D8F", 0.016),
    ]
    for name, start, end, color, dy in end_labels:
        ax.text(
            x[-1] + 0.08,
            end + dy,
            f"{start:.3f} -> {end:.3f}",
            fontsize=9,
            color=color,
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFFFFF", "edgecolor": color, "linewidth": 0.7},
        )

    ax.set_xlim(1, 5.7)
    ax.set_ylim(0.1, 0.9)
    ax.set_xticks(x)
    ax.set_xlabel("C088 分值（1-5分）")
    ax.set_ylabel("预测概率（0-1）")
    ax.set_title("图4 认知增益梯度效应折线图")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {
        "M1_q20_prob": (float(s1[0]), float(s1[-1])),
        "M3_q21_prob_q20_0": (float(s2[0]), float(s2[-1])),
        "M3_q21_prob_q20_1": (float(s3[0]), float(s3[-1])),
    }


def _draw_node(ax, center: tuple[float, float], text: str, fc: str, ec: str = "#2F2F2F") -> None:
    w, h = 0.22, 0.13
    x0 = center[0] - w / 2
    y0 = center[1] - h / 2
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=11, color="#202020")


def draw_figure_5_path(or_df: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, float]:
    or_c088_to_q20 = extract_or(or_df, "M1_Q20_visit_intent", "C088")["or"]
    or_q20_to_q21 = extract_or(or_df, "M3_Q21_recommend_with_q20", "y_q20_high")["or"]
    or_c088_to_q21_direct = extract_or(or_df, "M3_Q21_recommend_with_q20", "C088")["or"]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    p_c088 = (0.20, 0.72)
    p_q20 = (0.52, 0.55)
    p_q21 = (0.83, 0.55)

    _draw_node(ax, p_c088, "认知增益\n(C088)", "#EAF4F4")
    _draw_node(ax, p_q20, "游览意愿\n(Q20)", "#FFF4E6")
    _draw_node(ax, p_q21, "推荐意愿\n(Q21)", "#FDEDEC")

    ax.annotate(
        "",
        xy=(p_q20[0] - 0.12, p_q20[1] + 0.03),
        xytext=(p_c088[0] + 0.12, p_c088[1] - 0.03),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "#355070"},
    )
    ax.text(
        0.36,
        0.67,
        f"OR={or_c088_to_q20:.3f}",
        fontsize=10,
        color="#355070",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#355070", "linewidth": 0.8},
    )

    ax.annotate(
        "",
        xy=(p_q21[0] - 0.12, p_q21[1]),
        xytext=(p_q20[0] + 0.12, p_q20[1]),
        arrowprops={"arrowstyle": "->", "linewidth": 2.4, "color": "#C1121F"},
    )
    ax.text(
        0.66,
        0.59,
        f"OR={or_q20_to_q21:.3f}",
        fontsize=10,
        color="#C1121F",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#C1121F", "linewidth": 0.8},
    )

    ax.annotate(
        "",
        xy=(p_q21[0] - 0.12, p_q21[1] - 0.05),
        xytext=(p_c088[0] + 0.12, p_c088[1] - 0.09),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "#6D597A", "connectionstyle": "arc3,rad=-0.26"},
    )
    ax.text(
        0.55,
        0.36,
        f"直接效应 OR={or_c088_to_q21_direct:.3f}",
        fontsize=10,
        color="#6D597A",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#6D597A", "linewidth": 0.8},
    )

    ax.text(
        0.5,
        0.20,
        "部分中介结构：认知培育 -> 到访转化 -> 口碑扩散",
        ha="center",
        fontsize=11,
        color="#264653",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F1FAEE", "edgecolor": "#7FB069", "linewidth": 0.9},
    )

    ax.set_title("图5 链路结构示意图", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {
        "or_c088_to_q20": float(or_c088_to_q20),
        "or_q20_to_q21": float(or_q20_to_q21),
        "or_c088_to_q21_direct": float(or_c088_to_q21_direct),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_path = input_dir / "Logit改进5_Q20Q21交叉分析.csv"
    or_path = input_dir / "Logit改进5_系数OR.csv"
    model_path = input_dir / "Logit改进5_模型指标.csv"
    grad_path = input_dir / "Logit改进5_认知增益梯度.csv"

    cross_df = read_csv_required(cross_path)
    or_df = read_csv_required(or_path)
    model_df = read_csv_required(model_path)
    grad_df = read_csv_required(grad_path)

    outputs = {
        "fig1_cross_distribution": output_dir / "图6-3-1_游览意愿与推荐意愿交叉分布图.png",
        "fig2_forest_or": output_dir / "图6-3-2_三模型核心OR值对比图.png",
        "fig3_model_fit": output_dir / "图6-3-3_三模型拟合指标对比图.png",
        "fig4_gradient": output_dir / "图6-3-4_认知增益梯度效应折线图.png",
        "fig5_path": output_dir / "图6-3-5_链路结构示意图.png",
    }

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "source_files": {
            "cross": str(cross_path),
            "or": str(or_path),
            "model_metrics": str(model_path),
            "gradient": str(grad_path),
        },
    }

    audit["fig1"] = draw_figure_1_cross_distribution(cross_df, outputs["fig1_cross_distribution"], args.dpi)
    audit["fig2"] = draw_figure_2_forest(or_df, outputs["fig2_forest_or"], args.dpi)
    audit["fig3"] = draw_figure_3_model_fit(model_df, outputs["fig3_model_fit"], args.dpi)
    audit["fig4"] = draw_figure_4_gradient(grad_df, outputs["fig4_gradient"], args.dpi)
    audit["fig5"] = draw_figure_5_path(or_df, outputs["fig5_path"], args.dpi)

    audit_path = output_dir / "图6-3_缺少图片需求_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"draw_done: output_dir={output_dir} audit={audit_path}")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
