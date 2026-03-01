#!/usr/bin/env python3
"""Draw Figure 48 payment-willingness MCA chart (reference-style reconstruction)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


POINTS = [
    {"label": "18岁以下", "group": "年龄", "x": -1.66, "y": 1.55},
    {"label": "18-25岁", "group": "年龄", "x": -0.66, "y": -0.83},
    {"label": "26-30岁", "group": "年龄", "x": 0.92, "y": -0.31},
    {"label": "31-40岁", "group": "年龄", "x": 1.02, "y": -0.43},
    {"label": "41-50岁", "group": "年龄", "x": 0.45, "y": 0.30},
    {"label": "51-60岁", "group": "年龄", "x": 0.85, "y": 0.98},
    {"label": "60岁以上", "group": "年龄", "x": 0.38, "y": 1.00},
    {"label": "小学", "group": "教育背景", "x": -0.69, "y": 1.26},
    {"label": "初中", "group": "教育背景", "x": -0.79, "y": 1.48},
    {"label": "高中或中专", "group": "教育背景", "x": 0.73, "y": 0.72},
    {"label": "大学专科", "group": "教育背景", "x": 0.82, "y": -0.10},
    {"label": "大学本科", "group": "教育背景", "x": -0.41, "y": -0.78},
    {"label": "硕士研究生", "group": "教育背景", "x": 0.96, "y": -0.93},
    {"label": "博士研究生", "group": "教育背景", "x": 1.83, "y": 0.04},
    {"label": "2000元及以下", "group": "月可支配收入", "x": -1.08, "y": 0.02},
    {"label": "2001-4000元", "group": "月可支配收入", "x": 0.50, "y": 0.53},
    {"label": "4001-6000元", "group": "月可支配收入", "x": 0.42, "y": -0.26},
    {"label": "6001-8000元", "group": "月可支配收入", "x": 0.46, "y": -0.45},
    {"label": "8001-10000元", "group": "月可支配收入", "x": 0.35, "y": -0.14},
    {"label": "10000元以上", "group": "月可支配收入", "x": 0.74, "y": -0.50},
    {"label": "非常不愿意", "group": "知识付费意愿", "x": 0.10, "y": 0.86},
    {"label": "不愿意", "group": "知识付费意愿", "x": 0.06, "y": 0.15},
    {"label": "一般", "group": "知识付费意愿", "x": -0.38, "y": -0.14},
    {"label": "愿意", "group": "知识付费意愿", "x": 0.02, "y": -0.24},
    {"label": "非常愿意", "group": "知识付费意愿", "x": 0.64, "y": 0.07},
]

# Label offsets in points: (dx, dy).
OFFSETS = {
    "18岁以下": (-8, 6),
    "18-25岁": (-34, -13),
    "26-30岁": (-10, -15),
    "31-40岁": (-8, -16),
    "41-50岁": (-31, 8),
    "51-60岁": (-8, 4),
    "60岁以上": (-10, 4),
    "小学": (2, 2),
    "初中": (2, 3),
    "高中或中专": (-6, 2),
    "大学专科": (-13, -2),
    "大学本科": (-10, 2),
    "硕士研究生": (-40, -12),
    "博士研究生": (-40, 0),
    "2000元及以下": (-47, -11),
    "2001-4000元": (-20, 7),
    "4001-6000元": (-17, -18),
    "6001-8000元": (-34, -13),
    "8001-10000元": (-27, -14),
    "10000元以上": (-16, -16),
    "非常不愿意": (4, 5),
    "不愿意": (-28, -10),
    "一般": (4, 0),
    "愿意": (-21, -11),
    "非常愿意": (-7, 6),
}

ELLIPSES = [
    # left-bottom cluster
    {"xy": (-0.80, -0.43), "width": 1.48, "height": 1.20, "angle": 0},
    # upper-mid cluster
    {"xy": (0.46, 0.73), "width": 1.62, "height": 0.76, "angle": 27},
    # right-bottom cluster
    {"xy": (0.63, -0.25), "width": 1.34, "height": 1.02, "angle": -6},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Figure 48 payment willingness MCA chart.")
    parser.add_argument(
        "--output-path",
        default="new/MCA/图48_知识付费意愿多重对应分析图.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=320, help="Image dpi.")
    return parser.parse_args()


def to_hex(color: tuple[float, float, float, float]) -> str:
    r = int(np.clip(color[0], 0, 1) * 255)
    g = int(np.clip(color[1], 0, 1) * 255)
    b = int(np.clip(color[2], 0, 1) * 255)
    return f"#{r:02X}{g:02X}{b:02X}"


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coord_path = out_path.with_name(out_path.stem + "_坐标.csv")
    audit_path = out_path.with_name(out_path.stem + "_audit.json")

    xs = np.array([p["x"] for p in POINTS], dtype=float)
    ys = np.array([p["y"] for p in POINTS], dtype=float)

    cmap = plt.get_cmap("plasma_r")
    norm = plt.Normalize(vmin=-1.8, vmax=2.0)
    colors = [cmap(norm(x)) for x in xs]

    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    fig.patch.set_facecolor("#F5F5F5")
    ax.set_facecolor("#F2F2F2")

    ax.set_xlim(-1.85, 2.0)
    ax.set_ylim(-1.05, 1.65)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=1.0, color="#C8C8C8", alpha=0.95)
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axvline(0, color="#666666", linewidth=1.0, linestyle=(0, (3, 3)))

    for sp in ax.spines.values():
        sp.set_color("#6A6A6A")
        sp.set_linewidth(1.0)

    for e in ELLIPSES:
        ax.add_patch(
            Ellipse(
                xy=e["xy"],
                width=e["width"],
                height=e["height"],
                angle=e["angle"],
                fill=False,
                linewidth=2.0,
                edgecolor="#5B5AB2",
                alpha=0.95,
                zorder=2,
            )
        )

    ax.scatter(xs, ys, s=150, marker="^", c=colors, edgecolors="none", alpha=0.98, zorder=3)

    for p, color in zip(POINTS, colors):
        dx, dy = OFFSETS.get(p["label"], (4, 4))
        ax.annotate(
            p["label"],
            xy=(p["x"], p["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=16,
            color=to_hex(color),
            ha="left",
            va="center",
            zorder=4,
        )

    ax.set_xlabel("第一维度", fontsize=22)
    ax.set_ylabel("第二维度", fontsize=22)
    ax.tick_params(axis="both", labelsize=15, colors="#444444")

    fig.text(
        0.5,
        0.016,
        "图 48   知识付费意愿多重对应分析图",
        ha="center",
        va="bottom",
        fontsize=30,
        fontweight="bold",
        color="#111111",
    )
    fig.tight_layout(rect=[0.04, 0.07, 0.98, 0.98])
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    with coord_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "group", "dim1", "dim2"])
        for p in POINTS:
            writer.writerow([p["label"], p["group"], p["x"], p["y"]])

    audit = {
        "output_path": str(out_path).replace("\\", "/"),
        "coordinate_csv_path": str(coord_path).replace("\\", "/"),
        "point_count": len(POINTS),
        "axis": {
            "xlim": [-1.85, 2.0],
            "ylim": [-1.05, 1.65],
            "xticks": [-1, 0, 1, 2],
            "yticks": [-1, 0, 1],
        },
        "ellipses": ELLIPSES,
        "note": "reference-style reconstruction from user-provided Figure 48 screenshot",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "figure48_done:",
        f"output={out_path}",
        f"coord={coord_path}",
        f"audit={audit_path}",
    )


if __name__ == "__main__":
    main()
