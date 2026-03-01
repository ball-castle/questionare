#!/usr/bin/env python3
"""Draw chapter 7 clustering figures from data/clustering1 outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw clustering required figures.")
    parser.add_argument("--cluster-dir", default="data/clustering1", help="Directory of clustering outputs.")
    parser.add_argument("--output-dir", default="data/figure/聚类", help="Directory for figure outputs.")
    parser.add_argument("--dpi", type=int, default=320, help="Output dpi.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str | None) -> float:
    s = "" if v is None else str(v).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def to_int(v: str | None) -> int:
    x = to_float(v)
    if not np.isfinite(x):
        return 0
    return int(round(x))


def draw_radar(profile_rows: list[dict[str, str]], out_path: Path, dpi: int) -> None:
    metrics = [
        ("motive_count", "到访动机数"),
        ("new_project_pref_count", "新增项目偏好"),
        ("promo_pref_count", "优惠偏好"),
        ("importance_mean", "重要度均值"),
        ("performance_mean", "表现度均值"),
        ("cognition_mean", "认知均值"),
    ]
    rows = sorted(profile_rows, key=lambda r: to_int(r.get("cluster")))
    if not rows:
        raise RuntimeError("empty profile rows")

    raw = np.array([[to_float(r.get(col)) for col, _ in metrics] for r in rows], dtype=float)
    col_min = np.nanmin(raw, axis=0)
    col_max = np.nanmax(raw, axis=0)
    denom = np.where((col_max - col_min) < 1e-12, 1.0, col_max - col_min)
    norm = (raw - col_min) / denom
    same_col = (col_max - col_min) < 1e-12
    if np.any(same_col):
        norm[:, same_col] = 0.5

    labels = [name for _, name in metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.6, 7.5))
    ax = plt.subplot(111, polar=True)
    colors = ["#2E7D32", "#AD1457", "#1565C0", "#EF6C00", "#6A1B9A"]

    for i, r in enumerate(rows):
        vals = norm[i].tolist()
        vals += vals[:1]
        c = colors[i % len(colors)]
        cid = to_int(r.get("cluster"))
        name = str(r.get("cluster_name", f"类型{cid}"))
        share = to_float(r.get("share_pct"))
        txt = f"C{cid} {name} ({share:.1f}%)"
        ax.plot(angles, vals, color=c, linewidth=2.0, label=txt)
        ax.fill(angles, vals, color=c, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666666")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.35)
    ax.set_title("图7-1 聚类变量雷达图（簇内均值按指标标准化）", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_k_line(model_rows: list[dict[str, str]], selected: dict[str, str], out_path: Path, dpi: int) -> None:
    feature = str(selected.get("feature_set", "enhanced"))
    target = [r for r in model_rows if str(r.get("feature_set")) == feature]
    if not target:
        raise RuntimeError("no model rows for selected feature set")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    style = {
        "robust": dict(color="#1B5E20", marker="o", label="robust"),
        "zscore": dict(color="#0D47A1", marker="s", label="zscore"),
    }

    for prep in ("robust", "zscore"):
        rows = [r for r in target if str(r.get("preprocess")) == prep]
        rows.sort(key=lambda r: to_int(r.get("k")))
        if not rows:
            continue
        xs = [to_int(r.get("k")) for r in rows]
        ys = [to_float(r.get("silhouette")) for r in rows]
        st = style[prep]
        ax.plot(xs, ys, linewidth=2.2, markersize=7, **st)

    sel_k = to_int(selected.get("k"))
    sel_s = to_float(selected.get("silhouette"))
    sel_p = str(selected.get("preprocess", "robust"))
    ax.scatter([sel_k], [sel_s], s=180, marker="*", color="#C62828", edgecolor="white", linewidth=1.0, zorder=5)
    ax.text(sel_k + 0.04, sel_s + 0.003, f"主口径: {sel_p}, K={sel_k}", color="#C62828", fontsize=9)

    k_vals = sorted({to_int(r.get("k")) for r in target})
    ax.set_xticks(k_vals)
    ax.set_xlabel("聚类数 K")
    ax.set_ylabel("Silhouette")
    ax.set_title(f"图7-2 K值选择折线图（feature_set={feature}）")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_soft_funnel(soft_rows: list[dict[str, str]], out_path: Path, dpi: int) -> None:
    rows = sorted(soft_rows, key=lambda r: to_float(r.get("share_pct")), reverse=True)
    if not rows:
        raise RuntimeError("empty soft segment rows")

    labels = [str(r.get("soft_segment", "")) for r in rows]
    shares = np.array([to_float(r.get("share_pct")) for r in rows], dtype=float)
    counts = [to_int(r.get("n")) for r in rows]
    max_share = float(np.nanmax(shares))
    widths = shares / max_share * 100.0
    lefts = (100.0 - widths) / 2.0
    ypos = np.arange(len(rows))
    colors = ["#2E7D32", "#1565C0", "#EF6C00", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for i, (y, w, lft, lab, n, sp) in enumerate(zip(ypos, widths, lefts, labels, counts, shares)):
        c = colors[i % len(colors)]
        ax.barh(y, w, left=lft, height=0.62, color=c, alpha=0.86)
        ax.text(50, y, f"{lab}", ha="center", va="center", color="white", fontsize=10, weight="bold")
        ax.text(101.2, y, f"n={n} ({sp:.1f}%)", va="center", fontsize=9, color="#333333")

    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("图7-3 软分层漏斗图")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_mca_scatter(mca_rows: list[dict[str, str]], out_path: Path, dpi: int) -> None:
    if not mca_rows:
        raise RuntimeError("empty mca rows")

    xs = np.array([to_float(r.get("mca_dim1")) for r in mca_rows], dtype=float)
    ys = np.array([to_float(r.get("mca_dim2")) for r in mca_rows], dtype=float)
    clusters = np.array([to_int(r.get("hard_cluster")) for r in mca_rows], dtype=int)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[keep]
    ys = ys[keep]
    clusters = clusters[keep]

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    palette = {
        1: "#1B5E20",
        2: "#AD1457",
        3: "#1565C0",
        4: "#EF6C00",
        5: "#6A1B9A",
    }
    uniq = sorted(set(clusters.tolist()))
    for cid in uniq:
        mk = clusters == cid
        ax.scatter(
            xs[mk],
            ys[mk],
            s=18,
            alpha=0.36,
            c=palette.get(cid, "#455A64"),
            edgecolors="none",
            label=f"C{cid} (n={int(mk.sum())})",
        )

    qx_min, qx_max = np.nanpercentile(xs, [1, 99])
    qy_min, qy_max = np.nanpercentile(ys, [1, 99])
    dx = (qx_max - qx_min) * 0.12 + 1e-8
    dy = (qy_max - qy_min) * 0.12 + 1e-8
    ax.set_xlim(qx_min - dx, qx_max + dx)
    ax.set_ylim(qy_min - dy, qy_max + dy)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.1)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.1)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x0 + 0.02 * (x1 - x0), y1 - 0.06 * (y1 - y0), "低认知-高意愿", color="#555555", fontsize=9)
    ax.text(x1 - 0.30 * (x1 - x0), y1 - 0.06 * (y1 - y0), "高认知-高意愿", color="#555555", fontsize=9)
    ax.text(x0 + 0.02 * (x1 - x0), y0 + 0.04 * (y1 - y0), "低认知-低意愿", color="#555555", fontsize=9)
    ax.text(x1 - 0.30 * (x1 - x0), y0 + 0.04 * (y1 - y0), "高认知-低意愿", color="#555555", fontsize=9)

    ax.set_xlabel("MCA 维度1")
    ax.set_ylabel("MCA 维度2")
    ax.set_title("图7-4 MCA四象限散点图（按硬聚类着色）")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_rows = read_csv_rows(cluster_dir / "聚类优化_画像卡.csv")
    model_rows = read_csv_rows(cluster_dir / "聚类优化_模型对比.csv")
    soft_rows = read_csv_rows(cluster_dir / "聚类优化_软分层汇总.csv")
    mca_rows = read_csv_rows(cluster_dir / "聚类优化_MCA四象限个体.csv")
    selected_rows = read_csv_rows(cluster_dir / "聚类优化_关键结论.csv")
    selected = {}
    for row in selected_rows:
        selected[str(row.get("item", ""))] = row.get("value", "")
    selected_model = {
        "feature_set": selected.get("selected_feature_set", "enhanced"),
        "preprocess": selected.get("selected_preprocess", "robust"),
        "k": selected.get("selected_k", "2"),
        "silhouette": selected.get("selected_silhouette", "nan"),
    }

    draw_radar(profile_rows, output_dir / "图7-1_聚类变量雷达图.png", args.dpi)
    draw_k_line(model_rows, selected_model, output_dir / "图7-2_K值选择折线图.png", args.dpi)
    draw_soft_funnel(soft_rows, output_dir / "图7-3_软分层漏斗图.png", args.dpi)
    draw_mca_scatter(mca_rows, output_dir / "图7-4_MCA四象限散点图.png", args.dpi)
    print(f"clustering_figures_drawn: cluster_dir={cluster_dir} out_dir={output_dir}")


if __name__ == "__main__":
    main()

