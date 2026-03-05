#!/usr/bin/env python3
"""Generate an elbow plot for clustering quality check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


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
    parser = argparse.ArgumentParser(description="Draw elbow plot for clustering quality check.")
    parser.add_argument(
        "--features-file",
        default=r"C:\Users\TSOU\Desktop\聚类数据捏造\features_encoded.csv",
        help="Input features file (csv/xlsx).",
    )
    parser.add_argument(
        "--output-path",
        default="new/聚类/图57_聚类质量检验_手肘图.png",
        help="Output PNG path.",
    )
    parser.add_argument("--k-min", type=int, default=1, help="Minimum k value.")
    parser.add_argument("--k-max", type=int, default=7, help="Maximum k value.")
    parser.add_argument("--n-init", type=int, default=20, help="KMeans n_init.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for KMeans.")
    parser.add_argument("--dpi", type=int, default=320, help="Output image dpi.")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def pick_continuous_columns(columns: list[str]) -> list[str]:
    continuous = [c for c in columns if "连续值" in str(c)]
    if not continuous:
        raise RuntimeError("No continuous feature columns found (expected columns containing '连续值').")
    return continuous


def draw_elbow(ks: list[int], inertias: list[float], out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    bg = "#FFFFFF"
    fg = "#111111"
    green = "#2FB47C"
    green_soft = "#8EE6C5"
    grid = "#DDE7E2"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(
        ks,
        inertias,
        color=green,
        marker="o",
        linewidth=2.2,
        markersize=6.2,
        markerfacecolor=bg,
        markeredgecolor=green_soft,
        markeredgewidth=1.3,
        label="聚类质量",
        zorder=3,
    )

    ax.set_xlabel("聚类数", fontsize=13, color=fg, labelpad=6)
    ax.set_ylabel("SSE值", fontsize=13, color=fg, labelpad=8)
    ax.set_xticks(ks)
    ax.tick_params(axis="x", colors=fg, labelsize=11)
    ax.tick_params(axis="y", colors=fg, labelsize=11)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(fg)
        ax.spines[side].set_linewidth(1.0)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=grid, alpha=0.65, zorder=0)
    ax.legend(loc="upper right", frameon=True, facecolor=bg, edgecolor="#C6C6C6", labelcolor=fg)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    features_path = Path(args.features_file)
    out_path = Path(args.output_path)

    df = read_table(features_path)
    cont_cols = pick_continuous_columns(df.columns.tolist())

    X_raw = df[cont_cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = RobustScaler().fit_transform(X_raw.to_numpy())

    k_min = int(args.k_min)
    k_max = int(args.k_max)
    if k_min < 1 or k_max < k_min:
        raise ValueError(f"Invalid k range: [{k_min}, {k_max}]")

    ks = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=int(args.n_init), random_state=int(args.random_state))
        km.fit(X)
        inertias.append(float(km.inertia_))

    draw_elbow(ks, inertias, out_path, int(args.dpi))

    audit = {
        "features_file": str(features_path),
        "sample_size": int(X_raw.shape[0]),
        "continuous_columns": cont_cols,
        "preprocess": "RobustScaler",
        "k_range": [k_min, k_max],
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "sse_by_k": {str(k): v for k, v in zip(ks, inertias)},
        "output_path": str(out_path),
    }
    audit_path = out_path.with_name(out_path.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "clustering_elbow_figure_done:",
        f"output={out_path}",
        f"audit={audit_path}",
    )


if __name__ == "__main__":
    main()
