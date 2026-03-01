#!/usr/bin/env python3
"""Draw clustering quality figure for a target K (default: 4)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
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
    parser = argparse.ArgumentParser(description="Draw clustering quality figure.")
    parser.add_argument("--cluster-dir", default="data/clustering1", help="Directory of clustering outputs.")
    parser.add_argument("--output-path", default="new/聚类/图48_聚类质量图.png", help="Output PNG path.")
    parser.add_argument("--k", type=int, default=4, help="Target cluster number K.")
    parser.add_argument(
        "--feature-set",
        default="",
        help="Target feature_set. Empty means use selected_feature_set from key conclusion.",
    )
    parser.add_argument(
        "--preprocess",
        default="auto",
        help="Target preprocess, one of auto/zscore/robust. auto uses appendix_k{K}_preferred_preprocess first.",
    )
    parser.add_argument(
        "--manual-silhouette",
        type=float,
        default=None,
        help="If provided, draw figure directly with this silhouette value (skip model row selection).",
    )
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
    if x != x:
        return 0
    return int(round(x))


def key_values(rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        k = str(row.get("item", "")).strip()
        if not k:
            continue
        out[k] = str(row.get("value", "")).strip()
    return out


def _sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        to_float(row.get("stratified_external_score")),
        to_float(row.get("silhouette")),
        -to_float(row.get("davies_bouldin")),
    )


def select_model_row(
    model_rows: list[dict[str, str]],
    summary: dict[str, str],
    k: int,
    feature_set_arg: str,
    preprocess_arg: str,
) -> tuple[dict[str, str], str]:
    feature_set = feature_set_arg.strip() or summary.get("selected_feature_set", "enhanced")
    rows = [r for r in model_rows if to_int(r.get("k")) == int(k) and str(r.get("feature_set", "")) == feature_set]
    if not rows:
        rows = [r for r in model_rows if to_int(r.get("k")) == int(k)]
    if not rows:
        raise RuntimeError(f"No model rows found for k={k}.")

    if preprocess_arg.strip().lower() != "auto":
        prep = preprocess_arg.strip().lower()
        hit = [r for r in rows if str(r.get("preprocess", "")).strip().lower() == prep]
        if not hit:
            raise RuntimeError(f"No row found for k={k}, preprocess={prep}.")
        return max(hit, key=_sort_key), f"explicit preprocess={prep}"

    preferred_key = f"appendix_k{k}_preferred_preprocess"
    preferred = summary.get(preferred_key, "").strip()
    if preferred:
        hit = [r for r in rows if str(r.get("preprocess", "")).strip() == preferred]
        if hit:
            return max(hit, key=_sort_key), f"key conclusion {preferred_key}={preferred}"

    selected_k = to_int(summary.get("selected_k"))
    selected_prep = summary.get("selected_preprocess", "").strip()
    if selected_k == int(k) and selected_prep:
        hit = [r for r in rows if str(r.get("preprocess", "")).strip() == selected_prep]
        if hit:
            return max(hit, key=_sort_key), f"selected_k={selected_k} matched, selected_preprocess={selected_prep}"

    return max(rows, key=_sort_key), "fallback: max stratified_external_score -> silhouette -> -davies_bouldin"


def draw_quality(score: float, k: int, feature_set: str, preprocess: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    bands = [
        (-1.0, 0.2, "#e9c8c8", "差"),
        (0.2, 0.5, "#ebe8c6", "一般"),
        (0.5, 1.0, "#cfe1cf", "良好"),
    ]
    for left, right, color, label in bands:
        ax.axvspan(left, right, color=color, alpha=0.95, zorder=0)
        ax.text((left + right) / 2, -0.34, label, ha="center", va="center", fontsize=14, color="#6b5d5d")

    width = max(0.0, min(2.0, score + 1.0))
    ax.barh(
        y=0.0,
        width=width,
        left=-1.0,
        height=0.52,
        color="#8d90d8",
        edgecolor="#3d3d67",
        linewidth=1.6,
        zorder=2,
    )
    ax.axvline(score, color="#3d3d67", linewidth=1.1, alpha=0.55, linestyle=":")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.55, 0.45)
    ax.set_yticks([])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("凝聚和分离的轮廓测量", fontsize=18, labelpad=10, color="#1f1f1f")
    ax.set_title("聚类质量", fontsize=26, pad=14)
    ax.text(
        1.0,
        0.33,
        f"K={k} | feature_set={feature_set} | preprocess={preprocess} | silhouette={score:.4f}",
        ha="right",
        va="center",
        fontsize=10,
        color="#3a3a3a",
    )

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color("#262626")
        ax.spines[side].set_linewidth(1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cluster_dir = Path(args.cluster_dir)
    out_path = Path(args.output_path)
    audit_path = out_path.with_name(out_path.stem + "_audit.json")

    model_path = cluster_dir / "聚类优化_模型对比.csv"
    key_path = cluster_dir / "聚类优化_关键结论.csv"

    model_rows = read_csv_rows(model_path)
    key_rows = read_csv_rows(key_path)
    summary = key_values(key_rows)

    row: dict[str, str] = {}
    if args.manual_silhouette is not None:
        silhouette = float(args.manual_silhouette)
        feature_set = str(args.feature_set).strip() or summary.get("selected_feature_set", "enhanced")
        preprocess = str(args.preprocess).strip()
        if not preprocess or preprocess.lower() == "auto":
            preprocess = summary.get("selected_preprocess", "robust")
        reason = "manual silhouette argument"
    else:
        row, reason = select_model_row(model_rows, summary, args.k, args.feature_set, args.preprocess)
        silhouette = to_float(row.get("silhouette"))
        if silhouette != silhouette:
            raise RuntimeError(f"Silhouette is NaN for chosen row: {row}")
        feature_set = str(row.get("feature_set", ""))
        preprocess = str(row.get("preprocess", ""))

    draw_quality(silhouette, int(args.k), feature_set, preprocess, out_path, int(args.dpi))

    audit = {
        "cluster_dir": str(cluster_dir),
        "model_csv_path": str(model_path),
        "key_conclusion_csv_path": str(key_path),
        "target_k": int(args.k),
        "target_feature_set_arg": str(args.feature_set),
        "target_preprocess_arg": str(args.preprocess),
        "picked_feature_set": feature_set,
        "picked_preprocess": preprocess,
        "picked_silhouette": float(silhouette),
        "picked_reason": reason,
        "manual_silhouette_arg": args.manual_silhouette,
        "output_path": str(out_path),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"clustering_quality_figure_done: k={args.k} preprocess={preprocess} "
        f"silhouette={silhouette:.4f} output={out_path} audit={audit_path}"
    )


if __name__ == "__main__":
    main()
