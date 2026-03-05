from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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
    quadrant_from_csv: str


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
    "Q1_优势区": "#2e7d32",
    "Q2_维持区": "#1565c0",
    "Q3_机会区": "#616161",
    "Q4_改进区": "#c62828",
}


def load_points(csv_path: Path) -> list[Point]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    points: list[Point] = []
    for r in rows:
        points.append(
            Point(
                item_no=int(r["item_no"]),
                item_text=str(r["item_text"]).strip(),
                importance=float(r["importance_mean"]),
                satisfaction=float(r["performance_mean"]),
                quadrant_from_csv=str(r.get("quadrant", "")).strip(),
            )
        )
    return points


def target_oriented_threshold(points: list[Point]) -> tuple[float, float]:
    imp_vals = sorted(p.importance for p in points)
    sat_vals = sorted(p.satisfaction for p in points)
    n_imp = len(imp_vals)
    n_sat = len(sat_vals)

    if n_imp % 2:
        imp = imp_vals[n_imp // 2]
    else:
        imp = (imp_vals[n_imp // 2 - 1] + imp_vals[n_imp // 2]) / 2.0

    if n_sat == 1:
        sat = sat_vals[0]
    else:
        h = (n_sat - 1) * 0.75
        low = int(h)
        high = low if h.is_integer() else low + 1
        if low == high:
            sat = sat_vals[low]
        else:
            sat = sat_vals[low] * (high - h) + sat_vals[high] * (h - low)
    return imp, sat


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
        f"满意度75分位线 = {sat_th:.4f}",
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

    ax.axhline(imp_th, color="#1d4ed8", linestyle="--", linewidth=1.0, label=f"重要度基准 {imp_th:.4f}")
    ax.axhline(sat_th, color="#b45309", linestyle="--", linewidth=1.0, label=f"满意度基准 {sat_th:.4f}")

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


def draw_flow(path: Path, dpi: int = 320) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.07, 0.88, "短期", fontsize=11, color="#1f2937", weight="bold")
    ax.text(0.39, 0.88, "中期", fontsize=11, color="#1f2937", weight="bold")
    ax.text(0.73, 0.88, "持续优化", fontsize=11, color="#1f2937", weight="bold")

    nodes = [
        (0.05, 0.45, 0.20, 0.30, "#fee2e2", "P1（立即）", "药膳食疗定制服务试点扩容", "1-4个月"),
        (0.29, 0.45, 0.20, 0.30, "#ffedd5", "P2（同步）", "文化节/展演活动开发", "1-4个月"),
        (0.53, 0.45, 0.20, 0.30, "#fef3c7", "P3（同步）", "非遗炮制技艺体验课开发", "1-4个月"),
        (0.77, 0.45, 0.20, 0.30, "#dcfce7", "P4（后续）", "团购套票/折扣券绑定促销", "0-2个月"),
    ]

    for x, y, w, h, color, priority, action, month in nodes:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.02, y + h - 0.07, priority, fontsize=10, weight="bold", color="#0f172a")
        ax.text(x + 0.02, y + h - 0.15, action, fontsize=9.3, color="#0f172a")
        ax.text(x + 0.02, y + 0.05, f"时间节点：{month}", fontsize=9, color="#334155")

    arrows = [
        ((0.25, 0.60), (0.29, 0.60)),
        ((0.49, 0.60), (0.53, 0.60)),
        ((0.73, 0.60), (0.77, 0.60)),
    ]
    for p1, p2 in arrows:
        arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#334155")
        ax.add_patch(arr)

    ax.text(0.5, 0.28, "行动路径：P1  →  P2  →  P3  →  P4", fontsize=11, ha="center", weight="bold", color="#111827")
    ax.text(
        0.5,
        0.19,
        "逻辑：先做高需求新项目试点，再扩展文化体验内容，最后将优惠机制与核心产品长期绑定。",
        fontsize=9.3,
        ha="center",
        color="#374151",
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def to_win_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def write_audit(
    audit_path: Path,
    source_csv: Path,
    points: list[Point],
    imp_th: float,
    sat_th: float,
    outputs: dict[str, Path],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": to_win_path(source_csv),
        "output_dir": to_win_path(audit_path.parent),
        "threshold_method": "提升导向阈值（重要度中位数 + 满意度75分位）",
        "thresholds": {
            "importance_median_line": imp_th,
            "satisfaction_q75_line": sat_th,
        },
        "point_count": len(points),
        "points": [
            {
                "item_no": p.item_no,
                "item_text": p.item_text,
                "importance": p.importance,
                "satisfaction": p.satisfaction,
                "quadrant_from_csv": p.quadrant_from_csv,
                "quadrant_recomputed": classify(p.importance, p.satisfaction, imp_th, sat_th),
            }
            for p in points
        ],
        "outputs": {k: to_win_path(v) for k, v in outputs.items()},
    }
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    source_csv = Path("data/data_analysis/_source_analysis/tables/IPA结果表.csv")
    out_dir = Path("data/figure/IPA")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scatter = out_dir / "图7-1_IPA四象限散点图.png"
    out_bar = out_dir / "图7-2_各文旅属性重要度与表现度均值对比.png"
    out_flow = out_dir / "图7-3_IPA整改行动优先级流程示意图.png"
    out_audit = out_dir / "IPA_图片生成_audit.json"

    points = load_points(source_csv)
    imp_th, sat_th = target_oriented_threshold(points)

    draw_scatter(points, imp_th, sat_th, out_scatter, dpi=320)
    draw_bar(points, imp_th, sat_th, out_bar, dpi=320)
    draw_flow(out_flow, dpi=320)
    write_audit(
        out_audit,
        source_csv=source_csv,
        points=points,
        imp_th=imp_th,
        sat_th=sat_th,
        outputs={
            "fig1_scatter": out_scatter,
            "fig2_bar": out_bar,
            "fig3_flow": out_flow,
        },
    )
    print(f"done: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
