#!/usr/bin/env python3
"""Draw cultural learning-preparation MCA figure and store it under new/MCA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from qp_io import numeric_matrix, read_xlsx_first_sheet, write_dict_csv
from qp_stats import run_mca

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 108-column questionnaire data: age(C002), education(C003), cognition items(C087/088/089).
AGE_COL = 2
EDU_COL = 3
BEFORE_COL = 87
AFTER_COL = 88
DURING_COL = 89

# Derived column index after appending one new column to 108-column matrix.
LEARNING_MODE_COL = 109

AGE_LABELS = {
    1: "18岁以下",
    2: "26-45岁",
    3: "18-25岁",
    4: "46-64岁",
    5: "65岁及以上",
}

EDU_LABELS = {
    1: "初中及以下",
    2: "中专/高中",
    3: "大专",
    4: "本科",
    5: "硕士及以上",
}

# 6-mode merged taxonomy, aligned to figure semantics:
# - 前-only -> 前中
# - 前后 -> 前中后
LEARNING_MODE_LABELS = {
    1: "不学习",
    2: "游览中学习",
    3: "游览后学习",
    4: "游览中后学习",
    5: "游览前中学习",
    6: "游览前中后学习",
}

# combo = before*4 + during*2 + after
COMBO_TO_MODE_CODE = {
    0: 1,  # 000
    2: 2,  # 010
    1: 3,  # 001
    3: 4,  # 011
    4: 5,  # 100
    6: 5,  # 110
    5: 6,  # 101
    7: 6,  # 111
}

GROUP_BY_COL = {
    AGE_COL: "年龄层",
    EDU_COL: "教育层",
    LEARNING_MODE_COL: "学习准备",
}

GROUP_COLOR = {
    "年龄层": "#8BC34A",
    "教育层": "#4CAF50",
    "学习准备": "#1B5E20",
}

CLUSTER_A_KEYS = {
    (LEARNING_MODE_COL, 6),
    (LEARNING_MODE_COL, 5),
    (EDU_COL, 4),
    (EDU_COL, 5),
    (AGE_COL, 2),
    (AGE_COL, 3),
}
CLUSTER_B_KEYS = {
    (LEARNING_MODE_COL, 4),
    (LEARNING_MODE_COL, 3),
    (LEARNING_MODE_COL, 2),
    (AGE_COL, 4),
}
CLUSTER_C_KEYS = {
    (LEARNING_MODE_COL, 1),
    (EDU_COL, 1),
    (AGE_COL, 1),
    (AGE_COL, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw cultural learning-preparation MCA figure.")
    parser.add_argument("--input-xlsx", default="data/叶开泰问卷数据.xlsx", help="Input questionnaire xlsx path.")
    parser.add_argument("--output-path", default="new/MCA/图47_文化学习准备MCA散点图.png", help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG dpi.")
    parser.add_argument("--learning-threshold", type=int, default=4, help="Likert threshold for positive learning preparation.")
    parser.add_argument("--require-raw-n", type=int, default=863, help="Expected raw sample size; set negative to disable.")
    return parser.parse_args()


def parse_mca_label(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"Q(\d+)=(\d+)", str(raw))
    if not m:
        raise ValueError(f"Unexpected MCA label: {raw}")
    return int(m.group(1)), int(m.group(2))


def human_label(col_idx: int, code: int) -> str:
    if col_idx == AGE_COL:
        return AGE_LABELS.get(code, f"年龄{code}")
    if col_idx == EDU_COL:
        return EDU_LABELS.get(code, f"教育{code}")
    if col_idx == LEARNING_MODE_COL:
        return LEARNING_MODE_LABELS.get(code, f"学习模式{code}")
    return f"Q{col_idx}={code}"


def build_learning_mode(num: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    out = np.full((num.shape[0],), np.nan, dtype=float)

    before = num[:, BEFORE_COL - 1]
    after = num[:, AFTER_COL - 1]
    during = num[:, DURING_COL - 1]
    valid_mask = ~np.isnan(before) & ~np.isnan(after) & ~np.isnan(during)

    b = (before[valid_mask] >= float(threshold)).astype(int)
    d = (during[valid_mask] >= float(threshold)).astype(int)
    a = (after[valid_mask] >= float(threshold)).astype(int)
    combo = b * 4 + d * 2 + a

    mapped = np.array([COMBO_TO_MODE_CODE[int(x)] for x in combo], dtype=float)
    out[valid_mask] = mapped

    counts: dict[int, int] = {}
    for code in sorted(LEARNING_MODE_LABELS):
        counts[code] = int(np.sum(mapped == float(code)))
    return out, valid_mask, counts


def build_points(mca: dict) -> list[dict]:
    col = mca["col"]
    contrib = np.asarray(mca.get("contrib"), dtype=float)
    points = []
    for i, raw in enumerate(mca["labels"]):
        col_idx, code = parse_mca_label(raw)
        c1 = float(contrib[i, 0]) if contrib.ndim == 2 and contrib.shape[0] > i and contrib.shape[1] > 0 else 0.0
        c2 = float(contrib[i, 1]) if contrib.ndim == 2 and contrib.shape[0] > i and contrib.shape[1] > 1 else 0.0
        points.append(
            {
                "x": float(col[i, 0]),
                "y": float(col[i, 1]),
                "col_idx": col_idx,
                "code": code,
                "group": GROUP_BY_COL.get(col_idx, "其他"),
                "label": human_label(col_idx, code),
                "c1": c1,
                "c2": c2,
            }
        )
    return points


def place_labels(points: list[dict]) -> list[dict]:
    xs = np.array([p["x"] for p in points], dtype=float)
    ys = np.array([p["y"] for p in points], dtype=float)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    x_span = max(float(np.ptp(xs)), 1e-6)
    y_span = max(float(np.ptp(ys)), 1e-6)
    max_span = max(x_span, y_span)

    coords = np.column_stack([xs, ys])
    n = len(points)

    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    k = min(4, max(1, n - 1))
    knn_mean = np.sort(dmat, axis=1)[:, :k].mean(axis=1)
    density = np.clip((0.34 * max_span) / np.maximum(knn_mean, 1e-6), 0.7, 2.8)

    anchor = np.zeros((n, 2), dtype=float)
    pos = np.zeros((n, 2), dtype=float)
    width = np.zeros(n, dtype=float)
    height = np.zeros(n, dtype=float)

    for i, p in enumerate(points):
        vx = float(p["x"]) - cx
        vy = float(p["y"]) - cy
        dist_c = float(np.hypot(vx, vy))
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            angle = (2.0 * np.pi * i) / max(1, n)
        else:
            angle = float(np.arctan2(vy, vx))
        center_factor = 1.0 - min(1.0, dist_c / max(0.38 * max_span, 1e-6))
        radius = max_span * (0.07 + 0.045 * density[i] + 0.09 * center_factor)
        lx = float(p["x"]) + radius * float(np.cos(angle))
        ly = float(p["y"]) + radius * float(np.sin(angle))
        anchor[i] = [lx, ly]
        pos[i] = [lx, ly]
        width[i] = max(0.09 * x_span, (0.012 + 0.007 * len(p["label"])) * x_span)
        height[i] = max(0.055 * y_span, 0.058 * y_span)

    for _ in range(360):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                ox = (width[i] + width[j]) * 0.5 - abs(dx)
                oy = (height[i] + height[j]) * 0.5 - abs(dy)
                if ox <= 0 or oy <= 0:
                    continue
                sx = 1.0 if dx >= 0 else -1.0
                sy = 1.0 if dy >= 0 else -1.0
                if abs(dx) < 1e-9:
                    sx = 1.0 if ((i + j) % 2 == 0) else -1.0
                if abs(dy) < 1e-9:
                    sy = 1.0 if ((i - j) % 2 == 0) else -1.0
                px = sx * ox * 0.42
                py = sy * oy * 0.42
                pos[i, 0] -= px
                pos[j, 0] += px
                pos[i, 1] -= py
                pos[j, 1] += py
                moved = True
        pos = pos * 0.99 + anchor * 0.01
        if not moved:
            break

    out = []
    for i, p in enumerate(points):
        lx = float(pos[i, 0])
        ly = float(pos[i, 1])
        vx = lx - float(p["x"])
        vy = ly - float(p["y"])
        deg = float(np.degrees(np.arctan2(vy, vx)))
        out.append(
            {
                **p,
                "lx": lx,
                "ly": ly,
                "ha": "left" if vx >= 0 else "right",
                "va": "bottom" if vy >= 0 else "top",
                "rot": float(np.clip(deg * 0.16, -24.0, 24.0)),
            }
        )
    return out


def draw_cluster(ax: plt.Axes, points: list[dict], keys: set[tuple[int, int]], color: str) -> bool:
    chosen = [p for p in points if (p["col_idx"], p["code"]) in keys]
    if len(chosen) < 2:
        return False
    xs = np.array([p["x"] for p in chosen], dtype=float)
    ys = np.array([p["y"] for p in chosen], dtype=float)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    width = max(0.30, float(xs.max() - xs.min()) * 1.8 + 0.16)
    height = max(0.26, float(ys.max() - ys.min()) * 1.8 + 0.16)
    ell = Ellipse((cx, cy), width=width, height=height, fill=False, edgecolor=color, linewidth=1.3, alpha=0.95)
    ax.add_patch(ell)
    return True


def plot_figure(points: list[dict], dim1_pct: float, dim2_pct: float, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 12.5 / 2.54))
    ax.set_facecolor("#F3F8F3")
    ax.axhline(0.0, color="#4E6A4E", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axvline(0.0, color="#4E6A4E", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.grid(True, linestyle=(0, (3, 4)), linewidth=0.65, color="#C7D8C7", alpha=0.9)

    for group_name, color in GROUP_COLOR.items():
        gp = [p for p in points if p["group"] == group_name]
        if not gp:
            continue
        ax.scatter(
            [p["x"] for p in gp],
            [p["y"] for p in gp],
            s=68,
            marker="^",
            color=color,
            edgecolors="#2E3A2E",
            linewidths=0.35,
            alpha=0.95,
            label=group_name,
            zorder=3,
        )

    laid_out = place_labels(points)
    for p in laid_out:
        ann = ax.annotate(
            p["label"],
            xy=(p["x"], p["y"]),
            xytext=(p["lx"], p["ly"]),
            textcoords="data",
            fontsize=8.0,
            color="#233223",
            ha=p["ha"],
            va=p["va"],
            rotation=p["rot"],
            arrowprops={
                "arrowstyle": "-",
                "color": "#7A8F7A",
                "linewidth": 0.48,
                "alpha": 0.75,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=4,
        )
        ann.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#F3F8F3", alpha=0.95)])

    draw_cluster(ax, points, CLUSTER_A_KEYS, "#2E7D32")
    draw_cluster(ax, points, CLUSTER_B_KEYS, "#2E7D32")
    draw_cluster(ax, points, CLUSTER_C_KEYS, "#2E7D32")

    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["年龄层"], markeredgecolor="#2E3A2E", markersize=8, label="年龄"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["教育层"], markeredgecolor="#2E3A2E", markersize=8, label="学历"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["学习准备"], markeredgecolor="#2E3A2E", markersize=8, label="学习准备模式"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, facecolor="#F8FCF8", edgecolor="#AFC3AF")

    ax.margins(x=0.20, y=0.20)
    ax.set_title("图47 文化学习准备多重对应分析图", fontsize=12)
    ax.set_xlabel(f"第一维度（Dim 1, {dim1_pct:.2f}%）", fontsize=10)
    ax.set_ylabel(f"第二维度（Dim 2, {dim2_pct:.2f}%）", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx)
    out_path = Path(args.output_path)
    audit_path = out_path.with_name(out_path.stem + "_audit.json")
    coord_path = out_path.with_name(out_path.stem + "_坐标.csv")

    headers, rows_dense = read_xlsx_first_sheet(input_xlsx)
    if len(headers) < DURING_COL:
        raise ValueError(f"Expected at least {DURING_COL} columns, got {len(headers)}")

    raw_n = len(rows_dense)
    if int(args.require_raw_n) >= 0 and raw_n != int(args.require_raw_n):
        raise ValueError(f"Raw sample size mismatch: expected {args.require_raw_n}, got {raw_n}")

    num, _ = numeric_matrix(rows_dense)
    learning_mode, learning_valid, mode_counts = build_learning_mode(num, int(args.learning_threshold))
    num_ext = np.column_stack([num, learning_mode])

    mca_cols = [AGE_COL, EDU_COL, LEARNING_MODE_COL]
    valid_mask = ~np.isnan(num_ext[:, [c - 1 for c in mca_cols]]).any(axis=1)
    mca_n = int(valid_mask.sum())

    mca = run_mca(num_ext, mca_cols)
    if mca is None:
        raise RuntimeError("MCA failed: not enough valid rows.")

    eig = np.asarray(mca["eigen"], dtype=float)
    den = float(np.nansum(eig))
    dim1_pct = float(eig[0] / den * 100.0) if den > 0 else 0.0
    dim2_pct = float(eig[1] / den * 100.0) if den > 0 else 0.0

    points = build_points(mca)
    plot_figure(points, dim1_pct, dim2_pct, out_path, int(args.dpi))

    coord_rows = [
        {
            "category": f"Q{p['col_idx']}={p['code']}",
            "label": p["label"],
            "group": p["group"],
            "dim1": p["x"],
            "dim2": p["y"],
            "dim1_contrib": p["c1"],
            "dim2_contrib": p["c2"],
        }
        for p in points
    ]
    write_dict_csv(coord_path, ["category", "label", "group", "dim1", "dim2", "dim1_contrib", "dim2_contrib"], coord_rows)

    audit = {
        "input_xlsx": str(input_xlsx),
        "output_path": str(out_path),
        "raw_n": raw_n,
        "learning_valid_n": int(learning_valid.sum()),
        "mca_n": mca_n,
        "mca_columns": mca_cols,
        "dim1_pct": dim1_pct,
        "dim2_pct": dim2_pct,
        "learning_threshold": int(args.learning_threshold),
        "learning_mode_counts": {LEARNING_MODE_LABELS[k]: int(v) for k, v in mode_counts.items()},
        "mode_merge_rule": {
            "before_only_to": "游览前中学习",
            "before_after_to": "游览前中后学习",
            "positive_threshold": f">={int(args.learning_threshold)}",
        },
        "coordinate_csv_path": str(coord_path),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "cultural_learning_mca_done:",
        f"output={out_path}",
        f"raw_n={raw_n}",
        f"mca_n={mca_n}",
        f"audit={audit_path}",
    )


if __name__ == "__main__":
    main()
