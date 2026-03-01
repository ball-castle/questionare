from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Point:
    item_no: int
    item_text: str
    importance: float
    performance: float
    quadrant: str


SHORT_LABELS = {
    1: "中医药文化展示/非遗体验",
    2: "环境舒适度与卫生状况",
    3: "交通便捷与停车",
    4: "亲友推荐/口碑",
    5: "特色美食与文创",
    6: "个性化体质辨识/养生咨询",
    7: "服务专业度与态度",
    8: "产品价格与优惠",
    9: "中医药服务专业度",
    10: "线上线下宣传推广",
}


OFFSETS = {
    1: (0.0018, 0.0040),
    2: (-0.0330, 0.0030),
    3: (-0.0180, -0.0100),
    4: (0.0100, 0.0008),
    5: (-0.0240, 0.0002),
    6: (0.0020, 0.0035),
    7: (0.0020, 0.0035),
    8: (0.0018, 0.0030),
    9: (0.0020, 0.0033),
    10: (0.0018, -0.0012),
}


COLORS = {
    "Q1_保持优势": "#2e7d32",
    "Q2_优先改进": "#c62828",
    "Q3_低优先级": "#616161",
    "Q4_可能过度投入": "#ef6c00",
}


def load_points(csv_path: Path) -> list[Point]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    pts: list[Point] = []
    for r in rows:
        pts.append(
            Point(
                item_no=int(r["item_no"]),
                item_text=r["item_text"],
                importance=float(r["importance_mean"]),
                performance=float(r["performance_mean"]),
                quadrant=r["quadrant"],
            )
        )
    return pts


def mean_threshold(points: list[Point]) -> tuple[float, float]:
    imp = sum(p.importance for p in points) / len(points)
    perf = sum(p.performance for p in points) / len(points)
    return imp, perf


def classify(importance: float, performance: float, imp_th: float, perf_th: float) -> str:
    if importance >= imp_th and performance >= perf_th:
        return "Q1_保持优势"
    if importance >= imp_th and performance < perf_th:
        return "Q2_优先改进"
    if importance < imp_th and performance < perf_th:
        return "Q3_低优先级"
    return "Q4_可能过度投入"


def draw_scatter(points: list[Point], imp_th: float, perf_th: float, out_png: Path, dpi: int = 320) -> None:
    x_values = [p.importance for p in points]
    y_values = [p.performance for p in points]
    x_min = min(x_values) - 0.03
    x_max = max(x_values) + 0.03
    y_min = min(y_values) - 0.03
    y_max = max(y_values) + 0.03

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.add_patch(Rectangle((x_min, perf_th), imp_th - x_min, y_max - perf_th, facecolor="#fbe7e8", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((imp_th, perf_th), x_max - imp_th, y_max - perf_th, facecolor="#e6f4ea", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_min, y_min), imp_th - x_min, perf_th - y_min, facecolor="#edf0f2", alpha=0.60, zorder=0))
    ax.add_patch(Rectangle((imp_th, y_min), x_max - imp_th, perf_th - y_min, facecolor="#fff2df", alpha=0.55, zorder=0))

    ax.axvline(imp_th, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(perf_th, color="#555555", linestyle="--", linewidth=1.2, zorder=1)

    for p in points:
        q = classify(p.importance, p.performance, imp_th, perf_th)
        c = COLORS[q]
        ax.scatter(p.importance, p.performance, s=68, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = OFFSETS.get(p.item_no, (0.002, 0.003))
        label = SHORT_LABELS.get(p.item_no, p.item_text)
        ax.text(
            p.importance + dx,
            p.performance + dy,
            f"{p.item_no} {label}",
            fontsize=8.3,
            color=c,
            weight="bold",
            zorder=4,
        )

    ax.text(x_min + 0.008, y_max - 0.007, "Q4 可能过度投入区", fontsize=11, color="#ad5200", va="top")
    ax.text(x_max - 0.008, y_max - 0.007, "Q1 保持优势区", fontsize=11, color="#1b5e20", va="top", ha="right")
    ax.text(x_min + 0.008, y_min + 0.003, "Q3 低优先级区", fontsize=11, color="#37474f", va="bottom")
    ax.text(x_max - 0.008, y_min + 0.003, "Q2 优先改进区", fontsize=11, color="#8b1e1e", va="bottom", ha="right")

    ax.text(
        x_min + 0.002,
        perf_th + 0.001,
        f"表现度均值线 = {perf_th:.4f}",
        fontsize=9,
        color="#424242",
        va="bottom",
    )
    ax.text(
        imp_th + 0.001,
        y_min + 0.001,
        f"重要度均值线 = {imp_th:.4f}",
        fontsize=9,
        color="#424242",
        va="bottom",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("重要度评分")
    ax.set_ylabel("表现度评分")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def draw_bar(points: list[Point], imp_th: float, perf_th: float, out_png: Path, dpi: int = 320) -> None:
    points = sorted(points, key=lambda p: p.item_no)
    x = list(range(len(points)))
    importance = [p.importance for p in points]
    performance = [p.performance for p in points]
    labels = [f"{p.item_no}" for p in points]
    short_names = [SHORT_LABELS.get(p.item_no, p.item_text) for p in points]

    width = 0.37
    y_min = min(min(importance), min(performance)) - 0.04
    y_max = max(max(importance), max(performance)) + 0.04

    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.bar([i - width / 2 for i in x], importance, width=width, color="#3b82f6", label="重要度均值")
    ax.bar([i + width / 2 for i in x], performance, width=width, color="#f59e0b", label="表现度均值")

    ax.axhline(imp_th, color="#1d4ed8", linestyle="--", linewidth=1.1, label=f"重要度基准 {imp_th:.4f}")
    ax.axhline(perf_th, color="#b45309", linestyle="--", linewidth=1.1, label=f"表现度基准 {perf_th:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx}\n{name}" for idx, name in zip(labels, short_names)], fontsize=8.5)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("均值分数")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def write_audit(
    points: list[Point],
    imp_th: float,
    perf_th: float,
    out_scatter_png: Path,
    out_bar_png: Path,
    out_json: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": "data/data_analysis/_source_analysis/tables/IPA结果表.csv",
        "outputs": {
            "scatter_png": str(out_scatter_png).replace("/", "\\"),
            "bar_png": str(out_bar_png).replace("/", "\\"),
        },
        "thresholds": {
            "importance_mean": imp_th,
            "performance_mean": perf_th,
        },
        "points": [
            {
                "item_no": p.item_no,
                "item_text": p.item_text,
                "importance": p.importance,
                "performance": p.performance,
                "quadrant_from_csv": p.quadrant,
                "quadrant_recomputed": classify(p.importance, p.performance, imp_th, perf_th),
            }
            for p in points
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    src = Path("data/data_analysis/_source_analysis/tables/IPA结果表.csv")
    out_dir = Path("new/ipa")
    out_scatter_png = out_dir / "图7-3_IPA四象限散点图.png"
    out_bar_png = out_dir / "图7-2_各文旅属性重要度与表现度均值对比.png"
    out_json = out_dir / "IPA_当前数据图片生成_audit.json"

    points = load_points(src)
    imp_th, perf_th = mean_threshold(points)
    draw_scatter(points, imp_th, perf_th, out_png=out_scatter_png, dpi=320)
    draw_bar(points, imp_th, perf_th, out_png=out_bar_png, dpi=320)
    write_audit(
        points,
        imp_th,
        perf_th,
        out_scatter_png=out_scatter_png,
        out_bar_png=out_bar_png,
        out_json=out_json,
    )

    print(f"done: {out_scatter_png}")
    print(f"done: {out_bar_png}")
    print(f"audit: {out_json}")


if __name__ == "__main__":
    main()
