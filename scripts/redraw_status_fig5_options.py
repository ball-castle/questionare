#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = Path("new/现状分析")
BASE_CSV = OUTPUT_DIR / "fig5_数据口径_沿用原图标注值.csv"


def build_data() -> pd.DataFrame:
    # 口径说明：数值沿用 new/现状分析/fig5.png 图中标注值，不更换数据来源。
    rows = [
        ("中医药文化博物馆", 4.21, 4.08),
        ("国医堂诊疗区", 4.15, 4.23),
        ("秘药局产品区", 3.87, 3.76),
        ("互动体验区", 3.64, 3.45),
        ("明清建筑景观区", 4.33, 4.19),
    ]
    df = pd.DataFrame(rows, columns=["功能区", "保护状况均值", "体验满意度均值"])
    df["差值(保护-体验)"] = (df["保护状况均值"] - df["体验满意度均值"]).round(2)
    return df


def draw_dumbbell(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy()
    y = np.arange(len(work))

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("#F7F7F5")
    ax.set_facecolor("#F7F7F5")

    for i, row in work.iterrows():
        x1 = float(row["保护状况均值"])
        x2 = float(row["体验满意度均值"])
        ax.hlines(i, min(x1, x2), max(x1, x2), color="#B8BDC7", linewidth=3.0, alpha=0.9, zorder=1)

    ax.scatter(work["保护状况均值"], y, s=170, color="#2563EB", edgecolor="white", linewidth=1.5, label="保护状况均值", zorder=3)
    ax.scatter(work["体验满意度均值"], y, s=170, color="#F59E0B", edgecolor="white", linewidth=1.5, label="体验满意度均值", zorder=3)

    for i, row in work.iterrows():
        p = float(row["保护状况均值"])
        e = float(row["体验满意度均值"])
        gap = float(row["差值(保护-体验)"])
        ax.text(p + 0.02, i + 0.10, f"{p:.2f}", fontsize=10, color="#1E3A8A")
        ax.text(e + 0.02, i - 0.22, f"{e:.2f}", fontsize=10, color="#92400E")
        ax.text(max(p, e) + 0.08, i - 0.03, f"Δ={gap:+.2f}", fontsize=9, color="#4B5563")

    ax.set_yticks(y)
    ax.set_yticklabels(work["功能区"], fontsize=11)
    ax.set_xlim(3.25, 4.48)
    ax.set_xlabel("均值（1-5分）", fontsize=12)
    ax.set_title("fig5 叶开泰各功能区保护与体验对比（哑铃图）", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle=(0, (2, 3)), alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def draw_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy()
    mat = work[["保护状况均值", "体验满意度均值"]].to_numpy(dtype=float)

    # 绿色主色调：低值浅绿，高值深绿，保持可读对比。
    cmap = LinearSegmentedColormap.from_list(
        "status_green",
        ["#F2FBF3", "#DDF4E1", "#BFE8C7", "#8FD5A2", "#4FAE73", "#1F7A4E"],
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    fig.patch.set_facecolor("#FCFCFA")
    ax.set_facecolor("#FCFCFA")

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=3.3, vmax=4.4)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11, color="#1F2937")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["保护状况均值", "体验满意度均值"], fontsize=11)
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work["功能区"], fontsize=11)
    # 按需去除图题，便于直接贴入正文版式。

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("均值（1-5分）", fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_data()
    df.to_csv(BASE_CSV, index=False, encoding="utf-8-sig")

    draw_dumbbell(df, OUTPUT_DIR / "fig5_optionA_哑铃图.png")
    draw_heatmap(df, OUTPUT_DIR / "fig5_optionB_热力矩阵.png")


if __name__ == "__main__":
    main()
