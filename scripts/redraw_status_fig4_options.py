#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = Path("new/现状分析")
INPUT_CSV = OUTPUT_DIR / "fig4_价值维度占比表.csv"

FIG_BG = "#E6F2E8"
AX_BG = "#E6F2E8"
GRID_COLOR = "#9DC7A3"
PALETTE = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A", "#81C784"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["占比（%）"] = pd.to_numeric(df["占比（%）"], errors="coerce").fillna(0.0)
    df = df.sort_values("占比（%）", ascending=False).reset_index(drop=True)
    return df


def draw_lollipop(df: pd.DataFrame, out_png: Path) -> None:
    work = df.sort_values("占比（%）", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))

    fig, ax = plt.subplots(figsize=(11.5, 6.8), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    for i, row in work.iterrows():
        val = float(row["占比（%）"])
        ax.hlines(i, 0, val, color="#87B88B", linewidth=4.0, zorder=1)

    ax.scatter(work["占比（%）"], y, s=210, color=PALETTE[::-1], edgecolor="#F3FAF4", linewidth=1.4, zorder=2)

    for i, row in work.iterrows():
        val = float(row["占比（%）"])
        ax.text(val + 0.25, i, f"{val:.1f}%", va="center", fontsize=12, color="#173E1A")

    ax.set_yticks(y)
    ax.set_yticklabels(work["价值维度"], fontsize=12)
    ax.set_xlabel("占比（%）", fontsize=13, color="#173E1A")
    ax.set_xlim(0, float(work["占比（%）"].max()) + 3.8)
    ax.grid(axis="x", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.5, color=GRID_COLOR)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2F5F35")
    ax.spines["bottom"].set_color("#2F5F35")
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="x", labelsize=12, colors="#1F4A25")
    ax.tick_params(axis="y", length=0, colors="#1F4A25")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def draw_treemap(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy()
    split_idx = int(np.ceil(len(work) / 2))
    rows = [work.iloc[:split_idx].reset_index(drop=True), work.iloc[split_idx:].reset_index(drop=True)]

    fig, ax = plt.subplots(figsize=(11.5, 6.8), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    total = float(work["占比（%）"].sum())
    current_top = 1.0
    color_idx = 0

    for row in rows:
        if row.empty:
            continue
        row_total = float(row["占比（%）"].sum())
        row_h = row_total / total
        y0 = current_top - row_h
        x0 = 0.0

        for _, rec in row.iterrows():
            val = float(rec["占比（%）"])
            w = 0.0 if row_total == 0 else val / row_total
            c = PALETTE[color_idx % len(PALETTE)]
            rect = Rectangle((x0, y0), w, row_h, facecolor=c, edgecolor="#EAF6EC", linewidth=2.0)
            ax.add_patch(rect)

            text = f"{rec['价值维度']}\n{val:.1f}%"
            font_sz = 12 if w > 0.16 else 10
            ax.text(x0 + w / 2, y0 + row_h / 2, text, ha="center", va="center", fontsize=font_sz, color="#F6FFF7")

            x0 += w
            color_idx += 1

        current_top = y0

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def rounded_percent_cells(values: np.ndarray, total_cells: int = 100) -> np.ndarray:
    floors = np.floor(values).astype(int)
    remainder = values - floors
    need = int(total_cells - floors.sum())
    if need > 0:
        idx = np.argsort(-remainder)[:need]
        floors[idx] += 1
    elif need < 0:
        idx = np.argsort(remainder)[: abs(need)]
        floors[idx] -= 1
    return floors


def draw_waffle(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy()
    values = work["占比（%）"].to_numpy(dtype=float)
    cell_counts = rounded_percent_cells(values, total_cells=100)

    labels_for_cells: list[str] = []
    for label, count in zip(work["价值维度"], cell_counts):
        labels_for_cells.extend([label] * int(count))
    labels_for_cells = labels_for_cells[:100]

    color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(work["价值维度"])}
    cell_colors = [color_map[label] for label in labels_for_cells]

    fig, ax = plt.subplots(figsize=(11.5, 6.8), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    xs, ys = [], []
    for idx in range(100):
        col = idx % 10
        row = 9 - (idx // 10)
        xs.append(col)
        ys.append(row)

    ax.scatter(xs, ys, c=cell_colors, s=560, marker="s", edgecolor="#EDF7EF", linewidth=1.0)

    ax.set_xlim(-0.7, 14.5)
    ax.set_ylim(-0.7, 9.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_x = 10.8
    legend_y_start = 8.8
    gap = 1.45
    for i, rec in work.iterrows():
        y = legend_y_start - i * gap
        ax.scatter([legend_x], [y], s=220, marker="s", color=color_map[rec["价值维度"]], edgecolor="#EDF7EF", linewidth=1.0)
        ax.text(legend_x + 0.5, y, f"{rec['价值维度']}  {float(rec['占比（%）']):.1f}%", va="center", fontsize=11, color="#173E1A")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def draw_polar_rose(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy().reset_index(drop=True)
    n = len(work)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = work["占比（%）"].to_numpy(dtype=float)
    width = (2 * np.pi / n) * 0.86

    fig = plt.figure(figsize=(8.6, 8.6), facecolor=FIG_BG)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(AX_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    ax.bar(theta, radii, width=width, color=colors, edgecolor="#EEF8F0", linewidth=1.4, alpha=0.96)

    r_max = float(np.ceil((radii.max() + 3) / 5) * 5)
    ax.set_ylim(0, r_max + 3.5)
    rticks = np.arange(5, r_max + 0.1, 5)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{int(t)}" for t in rticks], fontsize=9, color="#3A6A40")
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)

    for i, row in work.iterrows():
        t = theta[i]
        v = float(row["占比（%）"])
        label = str(row["价值维度"])
        text_angle = np.degrees(t)
        align = "left"
        if 90 <= text_angle <= 270:
            text_angle += 180
            align = "right"
        ax.text(
            t,
            v * 0.58 + 0.4,
            f"{v:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            color="#EFFFF0",
            fontweight="semibold",
        )
        ax.text(
            t,
            r_max + 2.3,
            label,
            ha=align,
            va="center",
            rotation=text_angle - 90,
            rotation_mode="anchor",
            fontsize=11,
            color="#1E5B2B",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def wrap_cn_label(text: str, max_len: int = 8) -> str:
    if len(text) <= max_len:
        return text
    for sep in ["与", "和", "、"]:
        if sep in text:
            return text.replace(sep, sep + "\n", 1)
    return text[:max_len] + "\n" + text[max_len:]


def draw_liquid_circles(df: pd.DataFrame, out_png: Path) -> None:
    work = df.copy().reset_index(drop=True)
    fig, axs = plt.subplots(2, 3, figsize=(11.8, 7.4), facecolor=FIG_BG)
    axs = axs.flatten()

    x = np.linspace(-1.2, 1.2, 600)
    freq = 3.5 * np.pi
    amp = 0.06

    for i, ax in enumerate(axs):
        ax.set_facecolor(AX_BG)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.38, 1.25)
        ax.axis("off")
        if i >= len(work):
            continue

        row = work.iloc[i]
        label = str(row["价值维度"])
        pct = float(row["占比（%）"])
        color = PALETTE[i % len(PALETTE)]

        # 把百分比映射到圆内液位：0%=-1，100%=+1
        level = 2 * (pct / 100.0) - 1
        phase = i * 0.9
        y_wave = level + amp * np.sin(freq * x + phase)

        clip_circle = Circle((0, 0), 1.0, transform=ax.transData)
        fill = ax.fill_between(x, -1.2, y_wave, color=color, alpha=0.93, zorder=2)
        fill.set_clip_path(clip_circle)

        border = Circle((0, 0), 1.0, facecolor="none", edgecolor=color, linewidth=2.4, zorder=3)
        ax.add_patch(border)

        ax.text(0, 0.0, f"{pct:.1f}%", ha="center", va="center", fontsize=20, color="#234B2B")
        ax.text(0, -1.24, wrap_cn_label(label), ha="center", va="top", fontsize=12, color="#1F5B2A")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_png, dpi=320)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    draw_lollipop(df, OUTPUT_DIR / "fig4_optionA_棒棒糖图.png")
    draw_treemap(df, OUTPUT_DIR / "fig4_optionB_treemap矩形树图.png")
    draw_waffle(df, OUTPUT_DIR / "fig4_optionC_华夫图.png")
    draw_polar_rose(df, OUTPUT_DIR / "fig4_optionD_扇形玫瑰图.png")
    draw_liquid_circles(df, OUTPUT_DIR / "fig4_optionE_液位圆阵列图.png")


if __name__ == "__main__":
    main()
