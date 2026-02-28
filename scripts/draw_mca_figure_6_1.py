#!/usr/bin/env python3
"""Draw figure 6-1 MCA scatter and store it under data/figure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe

from qp_io import numeric_matrix, read_xlsx_first_sheet
from qp_stats import run_mca

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MCA_COLS = [1, 2, 4, 5, 6, 7, 8, 90, 91]

GROUP_BY_COL = {
    1: "人口属性",
    2: "人口属性",
    4: "人口属性",
    5: "人口属性",
    6: "习惯与认知",
    7: "习惯与认知",
    8: "习惯与认知",
    90: "行为意愿",
    91: "行为意愿",
}

GROUP_COLOR = {
    "人口属性": "#8D2E2E",
    "习惯与认知": "#2F6B8E",
    "行为意愿": "#D3832F",
}

SHORT_VAR = {
    1: "性",
    2: "龄",
    4: "职",
    5: "收",
    6: "习",
    7: "认",
    8: "访",
    90: "游",
    91: "荐",
}

SHORT_OPTION = {
    1: {1: "男", 2: "女"},
    2: {1: "<18", 2: "26-45", 3: "18-25", 4: "46-64", 5: "65+"},
    4: {1: "学生", 2: "职员", 3: "自由/个体", 4: "退休/其他"},
    5: {1: "低", 2: "中低", 3: "中", 4: "中高", 5: "高"},
    6: {1: "低频", 2: "中频", 3: "高频"},
    7: {1: "低认知", 2: "中认知", 3: "高认知"},
    8: {1: "到访", 2: "未到访"},
    90: {1: "1-2分", 2: "3分", 3: "4-5分"},
    91: {1: "1-2分", 2: "3分", 3: "4-5分"},
}

# Left-upper: student / 18-25 / low income / low cognition.
CLUSTER_1_KEYS = {(4, 1), (2, 3), (5, 1), (7, 1)}

# Right-side: high income / high cognition / visited / high intention.
CLUSTER_2_KEYS = {(5, 5), (7, 3), (8, 1), (90, 3), (91, 3)}
MUST_LABEL_KEYS = CLUSTER_1_KEYS | CLUSTER_2_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw MCA figure 6-1 from questionnaire xlsx.")
    parser.add_argument("--input-xlsx", default="data/叶开泰问卷数据.xlsx", help="Raw questionnaire xlsx path.")
    parser.add_argument("--output-path", default="data/figure/图6-1_MCA散点图.png", help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG dpi.")
    parser.add_argument("--require-raw-n", type=int, default=863, help="Expected raw sample size.")
    parser.add_argument("--require-mca-n", type=int, default=863, help="Expected valid sample size for MCA.")
    parser.add_argument("--label-top-k", type=int, default=9, help="Max number of labels to render (auto-selected).")
    return parser.parse_args()


def _recode_col(vals: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.full(vals.shape, np.nan, dtype=float)
    for src, dst in mapping.items():
        out[np.isclose(vals, float(src), equal_nan=False)] = float(dst)
    return out


def recode_for_figure(num: np.ndarray) -> np.ndarray:
    out = num.copy()

    # C004 职业 -> 4类
    out[:, 3] = _recode_col(
        out[:, 3],
        {
            1: 1,  # 学生
            2: 2,  # 企业/公司职员
            3: 2,  # 公务员/事业单位
            6: 2,  # 服务业从业者
            4: 3,  # 自由职业者
            5: 3,  # 个体经营者
            7: 4,  # 离退休
            8: 4,  # 其他
        },
    )

    # C006 消费频率 -> 低/中/高
    out[:, 5] = _recode_col(
        out[:, 5],
        {
            4: 1,  # 从不
            3: 1,  # 很少
            2: 2,  # 偶尔
            1: 3,  # 经常
        },
    )

    # C007 融合模式认知 -> 低/中/高
    out[:, 6] = _recode_col(
        out[:, 6],
        {
            5: 1,  # 完全不了解
            4: 1,  # 不太了解
            3: 2,  # 一般
            2: 3,  # 比较了解
            1: 3,  # 非常了解
        },
    )

    # C090/C091 意愿 -> 1-2 / 3 / 4-5
    out[:, 89] = _recode_col(out[:, 89], {1: 1, 2: 1, 3: 2, 4: 3, 5: 3})
    out[:, 90] = _recode_col(out[:, 90], {1: 1, 2: 1, 3: 2, 4: 3, 5: 3})
    return out


def parse_mca_label(raw: str) -> tuple[int, int]:
    m = re.fullmatch(r"Q(\d+)=(\d+)", str(raw))
    if not m:
        raise ValueError(f"Unexpected MCA label: {raw}")
    return int(m.group(1)), int(m.group(2))


def human_label(col_idx: int, code: int) -> str:
    var = SHORT_VAR.get(col_idx, f"Q{col_idx}")
    opt = SHORT_OPTION.get(col_idx, {}).get(code, str(code))
    return f"{var}:{opt}"


def build_points(mca: dict) -> list[dict]:
    col = mca["col"]
    contrib = np.asarray(mca.get("contrib"), dtype=float)
    pts = []
    for i, raw in enumerate(mca["labels"]):
        col_idx, code = parse_mca_label(raw)
        group = GROUP_BY_COL.get(col_idx, "其他")
        c1 = float(contrib[i, 0]) if contrib.ndim == 2 and contrib.shape[0] > i and contrib.shape[1] > 0 else 0.0
        c2 = float(contrib[i, 1]) if contrib.ndim == 2 and contrib.shape[0] > i and contrib.shape[1] > 1 else 0.0
        pts.append(
            {
                "x": float(col[i, 0]),
                "y": float(col[i, 1]),
                "col_idx": col_idx,
                "code": code,
                "group": group,
                "label": human_label(col_idx, code),
                "c1": c1,
                "c2": c2,
            }
        )
    return pts


def select_label_points(points: list[dict], top_k: int) -> list[dict]:
    if not points:
        return []
    top_k = max(1, int(top_k))
    dists = [float(np.hypot(p["x"], p["y"])) for p in points]
    dmax = max(dists) if dists else 1.0
    dmax = max(dmax, 1e-6)
    scored = []
    for p, d in zip(points, dists):
        key = (p["col_idx"], p["code"])
        dist_norm = d / dmax
        imp = max(0.0, p.get("c1", 0.0)) + max(0.0, p.get("c2", 0.0))
        bonus = 0.0
        if key in MUST_LABEL_KEYS:
            bonus += 10.0
        if p["col_idx"] in (90, 91):
            bonus += 1.0
        score = imp * 5.0 + dist_norm * 1.7 + bonus
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    picked = set()
    for _, p in scored:
        key = (p["col_idx"], p["code"])
        if key in picked:
            continue
        chosen.append(p)
        picked.add(key)
        if len(chosen) >= top_k:
            break
    return chosen


def draw_cluster(ax, points: list[dict], keys: set[tuple[int, int]], title: str, color: str) -> bool:
    chosen = [p for p in points if (p["col_idx"], p["code"]) in keys]
    if len(chosen) < 2:
        return False
    xs = np.array([p["x"] for p in chosen], dtype=float)
    ys = np.array([p["y"] for p in chosen], dtype=float)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    width = max(0.30, float(xs.max() - xs.min()) * 1.7 + 0.16)
    height = max(0.24, float(ys.max() - ys.min()) * 1.7 + 0.16)
    ell = Ellipse(
        (cx, cy),
        width=width,
        height=height,
        fill=False,
        edgecolor=color,
        linewidth=1.2,
        linestyle=(0, (5, 3)),
        alpha=0.95,
    )
    ax.add_patch(ell)
    ax.text(
        cx + width * 0.52,
        cy + height * 0.45,
        title,
        fontsize=8,
        color=color,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFF8EF", "edgecolor": color, "linewidth": 0.6},
    )
    return True


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
        ax = float(p["x"]) + radius * float(np.cos(angle))
        ay = float(p["y"]) + radius * float(np.sin(angle))
        anchor[i] = [ax, ay]
        pos[i] = [ax, ay]
        width[i] = max(0.095 * x_span, (0.013 + 0.0075 * len(p["label"])) * x_span)
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
        # Weak spring so labels keep pointing away from the dense center.
        pos = pos * 0.99 + anchor * 0.01
        if not moved:
            break

    out = []
    for i, p in enumerate(points):
        lx = float(pos[i, 0])
        ly = float(pos[i, 1])
        vx = lx - float(p["x"])
        vy = ly - float(p["y"])
        d = float(np.hypot(vx, vy))
        min_d = 0.10 * max_span
        if d < min_d:
            scale = min_d / max(d, 1e-6)
            lx = float(p["x"]) + vx * scale
            ly = float(p["y"]) + vy * scale
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
                "rot": float(np.clip(deg * 0.18, -30.0, 30.0)),
            }
        )
    return out


def plot_figure(points: list[dict], dim1_pct: float, dim2_pct: float, out_path: Path, dpi: int, label_top_k: int) -> int:
    fig, ax = plt.subplots(figsize=(14.0 / 2.54, 11.0 / 2.54))
    ax.set_facecolor("#FBF6ED")
    ax.axhline(0.0, color="#AAAAAA", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#AAAAAA", linewidth=1.0, linestyle="--")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#DDCFBF", alpha=0.55)

    for group_name, color in GROUP_COLOR.items():
        gp = [p for p in points if p["group"] == group_name]
        if not gp:
            continue
        ax.scatter(
            [p["x"] for p in gp],
            [p["y"] for p in gp],
            s=62,
            marker="^",
            color=color,
            edgecolors="#3A2E2A",
            linewidths=0.35,
            alpha=0.95,
            label=group_name,
        )
    label_points = select_label_points(points, label_top_k)
    laid_out = place_labels(label_points)
    for p in laid_out:
        ann = ax.annotate(
            p["label"],
            xy=(p["x"], p["y"]),
            xytext=(p["lx"], p["ly"]),
            textcoords="data",
            fontsize=7.6,
            color="#3F2F2F",
            ha=p["ha"],
            va=p["va"],
            rotation=p["rot"],
            arrowprops={
                "arrowstyle": "-",
                "color": "#8D7F73",
                "linewidth": 0.45,
                "alpha": 0.7,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        ann.set_path_effects([pe.withStroke(linewidth=2.1, foreground="#FBF6ED", alpha=0.95)])

    draw_cluster(ax, points, CLUSTER_1_KEYS, "潜在客群", "#6D2B2B")
    draw_cluster(ax, points, CLUSTER_2_KEYS, "核心活跃客群", "#8A4A1F")

    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["人口属性"], markeredgecolor="#3A2E2A", markersize=8, label="人口属性（Q1/Q2/Q4/Q5）"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["习惯与认知"], markeredgecolor="#3A2E2A", markersize=8, label="习惯与认知（Q6/Q7/Q8）"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GROUP_COLOR["行为意愿"], markeredgecolor="#3A2E2A", markersize=8, label="行为意愿（Q20/Q21）"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, facecolor="#FFF8EF", edgecolor="#C9B69C")

    ax.margins(x=0.22, y=0.20)
    ax.set_title("图6-1 游客文化认知MCA二维散点图", fontsize=12, color="#4A2B2B")
    ax.set_xlabel(f"认知-习惯分化轴（Dim 1, {dim1_pct:.2f}%）", fontsize=10)
    ax.set_ylabel(f"行为意愿极化轴（Dim 2, {dim2_pct:.2f}%）", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return len(laid_out)


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx)
    output_path = Path(args.output_path)
    audit_path = output_path.with_name(output_path.stem + "_audit.json")

    headers, rows_dense = read_xlsx_first_sheet(input_xlsx)
    if len(headers) != 108:
        raise ValueError(f"Expected 108 columns, got {len(headers)} from {input_xlsx}")

    raw_n = len(rows_dense)
    if args.require_raw_n is not None and raw_n != int(args.require_raw_n):
        raise ValueError(f"Raw sample size mismatch: expected {args.require_raw_n}, got {raw_n}")

    num, _ = numeric_matrix(rows_dense)
    num_recoded = recode_for_figure(num)
    valid_mask = ~np.isnan(num_recoded[:, [c - 1 for c in MCA_COLS]]).any(axis=1)
    mca_n = int(valid_mask.sum())
    if args.require_mca_n is not None and mca_n != int(args.require_mca_n):
        raise ValueError(f"MCA valid sample size mismatch: expected {args.require_mca_n}, got {mca_n}")

    mca = run_mca(num_recoded, MCA_COLS)
    if mca is None:
        raise RuntimeError("MCA failed: not enough valid rows.")

    eig = np.asarray(mca["eigen"], dtype=float)
    den = float(np.nansum(eig))
    dim1_pct = float(eig[0] / den * 100.0) if den > 0 else 0.0
    dim2_pct = float(eig[1] / den * 100.0) if den > 0 else 0.0

    points = build_points(mca)
    labeled_count = plot_figure(points, dim1_pct, dim2_pct, output_path, args.dpi, args.label_top_k)

    audit = {
        "input_xlsx": str(input_xlsx),
        "output_path": str(output_path),
        "raw_n": raw_n,
        "mca_n": mca_n,
        "mca_columns": MCA_COLS,
        "dim1_pct": dim1_pct,
        "dim2_pct": dim2_pct,
        "label_count_all_points": len(points),
        "label_count_rendered": labeled_count,
        "label_top_k": int(args.label_top_k),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mca_figure_done: output={output_path} raw_n={raw_n} mca_n={mca_n} audit={audit_path}")


if __name__ == "__main__":
    main()
