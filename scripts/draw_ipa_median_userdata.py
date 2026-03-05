from __future__ import annotations

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


@dataclass(frozen=True)
class Point:
    item_no: int
    item_text: str
    importance: float
    satisfaction: float


POINTS = [
    Point(1, "丰富的中医药文化展示和非遗体验项目", 3.7561, 3.7189),
    Point(2, "环境舒适度与卫生状况", 3.7410, 3.6612),
    Point(3, "便捷的交通、充足的停车位", 3.7224, 3.6516),
    Point(4, "亲友推荐/正面评价多", 3.6887, 3.7085),
    Point(5, "特色美食、文创产品的种类及品质", 3.7398, 3.5900),
    Point(6, "提供个性化中医体质辨识、养生咨询等服务", 3.7642, 3.6820),
    Point(7, "服务专业度与态度", 3.8269, 3.6850),
    Point(8, "产品价格与优惠力度", 3.7340, 3.6934),
    Point(9, "中医药服务专业度", 3.7933, 3.6957),
    Point(10, "线上线下宣传推广", 3.6887, 3.7398),
]


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
    2: (0.0020, 0.0030),
    3: (-0.0180, -0.0100),
    4: (0.0020, 0.0020),
    5: (-0.0240, 0.0002),
    6: (0.0020, 0.0035),
    7: (0.0020, 0.0035),
    8: (0.0018, 0.0030),
    9: (0.0020, 0.0033),
    10: (0.0018, -0.0012),
}


COLORS = {
    "Q1_优势区": "#2e7d32",
    "Q2_维持区": "#1565c0",
    "Q3_机会区": "#616161",
    "Q4_改进区": "#c62828",
}


def median_threshold(points: list[Point]) -> tuple[float, float]:
    imp_vals = sorted(p.importance for p in points)
    sat_vals = sorted(p.satisfaction for p in points)
    n = len(points)
    mid = n // 2
    if n % 2 == 1:
        return imp_vals[mid], sat_vals[mid]
    return (imp_vals[mid - 1] + imp_vals[mid]) / 2.0, (sat_vals[mid - 1] + sat_vals[mid]) / 2.0


def classify(importance: float, satisfaction: float, imp_th: float, sat_th: float) -> str:
    if importance >= imp_th and satisfaction >= sat_th:
        return "Q1_优势区"
    if importance < imp_th and satisfaction >= sat_th:
        return "Q2_维持区"
    if importance < imp_th and satisfaction < sat_th:
        return "Q3_机会区"
    return "Q4_改进区"


def draw_scatter(points: list[Point], imp_th: float, sat_th: float, output: Path, dpi: int = 320) -> None:
    x_values = [p.importance for p in points]
    y_values = [p.satisfaction for p in points]
    x_min = min(x_values) - 0.03
    x_max = max(x_values) + 0.03
    y_min = min(y_values) - 0.03
    y_max = max(y_values) + 0.03

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.add_patch(Rectangle((x_min, sat_th), imp_th - x_min, y_max - sat_th, facecolor="#fbe7e8", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((imp_th, sat_th), x_max - imp_th, y_max - sat_th, facecolor="#e6f4ea", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_min, y_min), imp_th - x_min, sat_th - y_min, facecolor="#edf0f2", alpha=0.60, zorder=0))
    ax.add_patch(Rectangle((imp_th, y_min), x_max - imp_th, sat_th - y_min, facecolor="#fff2df", alpha=0.55, zorder=0))

    ax.axvline(imp_th, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(sat_th, color="#555555", linestyle="--", linewidth=1.2, zorder=1)

    for p in points:
        quadrant = classify(p.importance, p.satisfaction, imp_th, sat_th)
        color = COLORS[quadrant]
        ax.scatter(p.importance, p.satisfaction, s=68, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = OFFSETS.get(p.item_no, (0.002, 0.003))
        label = SHORT_LABELS.get(p.item_no, p.item_text)
        ax.text(
            p.importance + dx,
            p.satisfaction + dy,
            f"{p.item_no} {label}",
            fontsize=8.3,
            color=color,
            weight="bold",
            zorder=4,
        )

    ax.text(x_min + 0.008, y_max - 0.007, "Q2 维持区", fontsize=10.8, color="#0d47a1", va="top")
    ax.text(x_max - 0.008, y_max - 0.007, "Q1 优势区", fontsize=10.8, color="#1b5e20", va="top", ha="right")
    ax.text(x_min + 0.008, y_min + 0.003, "Q3 机会区", fontsize=10.8, color="#37474f", va="bottom")
    ax.text(x_max - 0.008, y_min + 0.003, "Q4 改进区", fontsize=10.8, color="#8b1e1e", va="bottom", ha="right")

    ax.text(
        x_min + 0.002,
        sat_th + 0.001,
        f"满意度中位数线 = {sat_th:.4f}",
        fontsize=8.8,
        color="#424242",
        va="bottom",
    )
    ax.text(
        imp_th + 0.001,
        y_min + 0.001,
        f"重要度中位数线 = {imp_th:.4f}",
        fontsize=8.8,
        color="#424242",
        va="bottom",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("重要度评分")
    ax.set_ylabel("满意度评分")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def draw_bar(points: list[Point], imp_th: float, sat_th: float, output: Path, dpi: int = 320) -> None:
    points = sorted(points, key=lambda p: p.item_no)
    x = list(range(len(points)))
    importance = [p.importance for p in points]
    satisfaction = [p.satisfaction for p in points]
    labels = [str(p.item_no) for p in points]
    short_names = [SHORT_LABELS.get(p.item_no, p.item_text) for p in points]
    tick_labels = [f"{idx} {name}" for idx, name in zip(labels, short_names)]

    width = 0.37
    y_min = min(min(importance), min(satisfaction)) - 0.04
    y_max = max(max(importance), max(satisfaction)) + 0.04

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    ax.bar([i - width / 2 for i in x], importance, width=width, color="#3b82f6", label="重要度均值")
    ax.bar([i + width / 2 for i in x], satisfaction, width=width, color="#f59e0b", label="满意度均值")

    ax.axhline(imp_th, color="#1d4ed8", linestyle="--", linewidth=1.0, label=f"重要度中位数线 {imp_th:.4f}")
    ax.axhline(sat_th, color="#b45309", linestyle="--", linewidth=1.0, label=f"满意度中位数线 {sat_th:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.1, rotation=28, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=6)
    ax.margins(x=0.01)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("均值分数")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def to_win_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def write_audit(
    points: list[Point],
    imp_th: float,
    sat_th: float,
    scatter_png: Path,
    bar_png: Path,
    audit_json: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "用户手工提供均值数据（继续修正版）",
        "threshold_method": "中位数阈值（重要度中位数 + 满意度中位数）",
        "thresholds": {
            "importance_median": imp_th,
            "satisfaction_median": sat_th,
        },
        "outputs": {
            "scatter_png": to_win_path(scatter_png),
            "bar_png": to_win_path(bar_png),
        },
        "points": [
            {
                "item_no": p.item_no,
                "item_text": p.item_text,
                "importance": p.importance,
                "satisfaction": p.satisfaction,
                "quadrant": classify(p.importance, p.satisfaction, imp_th, sat_th),
            }
            for p in points
        ],
    }
    audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out_dir = Path("new/ipa")
    out_scatter = out_dir / "图7-3_IPA四象限散点图_中位数阈值_用户继续数据.png"
    out_bar = out_dir / "图7-2_各文旅属性重要度与表现度均值对比_中位数阈值_用户继续数据.png"
    out_audit = out_dir / "IPA_中位数阈值_用户继续数据_audit.json"

    imp_th, sat_th = median_threshold(POINTS)
    draw_scatter(POINTS, imp_th, sat_th, out_scatter, dpi=320)
    draw_bar(POINTS, imp_th, sat_th, out_bar, dpi=320)
    write_audit(
        points=POINTS,
        imp_th=imp_th,
        sat_th=sat_th,
        scatter_png=out_scatter,
        bar_png=out_bar,
        audit_json=out_audit,
    )

    print(f"done: {out_scatter}")
    print(f"done: {out_bar}")
    print(f"audit: {out_audit}")


if __name__ == "__main__":
    main()
