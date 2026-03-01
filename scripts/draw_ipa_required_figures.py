from __future__ import annotations

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


THRESH_IMPORTANCE = 3.7672
THRESH_PERFORMANCE = 3.7061


@dataclass(frozen=True)
class IpaPoint:
    idx: str
    name: str
    importance: float
    performance: float
    reference_quadrant: str


POINTS = [
    IpaPoint("①", "中医药文化展示与非遗体验", 3.812, 3.751, "Q1"),
    IpaPoint("②", "环境舒适度与卫生状况", 3.738, 3.734, "Q2"),
    IpaPoint("③", "交通便捷与停车", 3.719, 3.668, "Q3"),
    IpaPoint("④", "亲友推荐/口碑", 3.789, 3.694, "Q4"),
    IpaPoint("⑤", "特色美食与文创", 3.731, 3.642, "Q3"),
    IpaPoint("⑥", "个性化体质辨识/养生咨询", 3.805, 3.744, "Q1"),
    IpaPoint("⑦", "服务专业度与态度", 3.836, 3.756, "Q1"),
    IpaPoint("⑧", "产品价格与优惠", 3.768, 3.704, "Q3/Q4"),
    IpaPoint("⑨", "中医药服务专业度", 3.824, 3.737, "Q1"),
    IpaPoint("⑩", "线上线下宣传推广", 3.781, 3.683, "Q4"),
]


def quadrant_by_threshold(importance: float, performance: float) -> str:
    if importance >= THRESH_IMPORTANCE and performance >= THRESH_PERFORMANCE:
        return "Q1"
    if importance < THRESH_IMPORTANCE and performance >= THRESH_PERFORMANCE:
        return "Q2"
    if importance < THRESH_IMPORTANCE and performance < THRESH_PERFORMANCE:
        return "Q3"
    return "Q4"


def draw_scatter(path: Path, dpi: int = 320) -> None:
    x_min, x_max = 3.66, 3.86
    y_min, y_max = 3.60, 3.78

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.add_patch(
        Rectangle(
            (x_min, THRESH_PERFORMANCE),
            THRESH_IMPORTANCE - x_min,
            y_max - THRESH_PERFORMANCE,
            facecolor="#fbe7e8",
            alpha=0.45,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (THRESH_IMPORTANCE, THRESH_PERFORMANCE),
            x_max - THRESH_IMPORTANCE,
            y_max - THRESH_PERFORMANCE,
            facecolor="#e6f4ea",
            alpha=0.45,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            THRESH_IMPORTANCE - x_min,
            THRESH_PERFORMANCE - y_min,
            facecolor="#edf0f2",
            alpha=0.60,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (THRESH_IMPORTANCE, y_min),
            x_max - THRESH_IMPORTANCE,
            THRESH_PERFORMANCE - y_min,
            facecolor="#fff2df",
            alpha=0.55,
            zorder=0,
        )
    )

    ax.axvline(THRESH_IMPORTANCE, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(THRESH_PERFORMANCE, color="#555555", linestyle="--", linewidth=1.2, zorder=1)

    point_colors = {"Q1": "#2e7d32", "Q2": "#c62828", "Q3": "#616161", "Q4": "#ef6c00"}
    offsets = {
        "①": (0.0018, 0.0038),
        "②": (-0.035, 0.0030),
        "③": (-0.018, -0.010),
        "④": (0.0020, -0.010),
        "⑤": (-0.030, -0.004),
        "⑥": (0.0020, 0.004),
        "⑦": (0.0020, 0.004),
        "⑧": (0.0025, -0.010),
        "⑨": (0.0020, 0.004),
        "⑩": (0.0020, -0.010),
    }

    for p in POINTS:
        quad = quadrant_by_threshold(p.importance, p.performance)
        color = point_colors[quad]
        ax.scatter(p.importance, p.performance, s=68, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = offsets[p.idx]
        ax.text(
            p.importance + dx,
            p.performance + dy,
            f"{p.idx} {p.name}",
            fontsize=8.3,
            color=color,
            weight="bold",
            zorder=4,
        )

    ax.text(3.668, 3.773, "Q2 优先改进区", fontsize=10, color="#8b1e1e", va="top")
    ax.text(3.857, 3.773, "Q1 保持优势区", fontsize=10, color="#1b5e20", va="top", ha="right")
    ax.text(3.668, 3.603, "Q3 低优先级区", fontsize=10, color="#37474f", va="bottom")
    ax.text(3.857, 3.603, "Q4 过度投入区", fontsize=10, color="#ad5200", va="bottom", ha="right")

    ax.text(
        3.662,
        THRESH_PERFORMANCE + 0.001,
        f"表现度均值线 = {THRESH_PERFORMANCE:.4f}",
        fontsize=8.6,
        color="#424242",
        va="bottom",
    )
    ax.text(
        THRESH_IMPORTANCE + 0.001,
        3.601,
        f"重要度均值线 = {THRESH_IMPORTANCE:.4f}",
        fontsize=8.6,
        color="#424242",
        va="bottom",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("重要度评分")
    ax.set_ylabel("表现度评分")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def draw_bar(path: Path, dpi: int = 320) -> None:
    labels = [p.idx for p in POINTS]
    short_names = [
        "中医药文化/非遗",
        "环境舒适/卫生",
        "交通便捷/停车",
        "亲友推荐/口碑",
        "特色美食/文创",
        "体质辨识/养生",
        "服务专业/态度",
        "价格/优惠",
        "中医药服务专业",
        "线上线下宣传",
    ]
    importance = [p.importance for p in POINTS]
    performance = [p.performance for p in POINTS]

    x = list(range(len(labels)))
    width = 0.37

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.bar([i - width / 2 for i in x], importance, width=width, color="#3b82f6", label="重要度均值")
    ax.bar([i + width / 2 for i in x], performance, width=width, color="#f59e0b", label="表现度均值")

    ax.axhline(THRESH_IMPORTANCE, color="#1d4ed8", linestyle="--", linewidth=1.0, label=f"重要度基准 {THRESH_IMPORTANCE:.4f}")
    ax.axhline(THRESH_PERFORMANCE, color="#b45309", linestyle="--", linewidth=1.0, label=f"表现度基准 {THRESH_PERFORMANCE:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx}\n{name}" for idx, name in zip(labels, short_names)], fontsize=8.5)
    ax.set_ylim(3.55, 3.88)
    ax.set_ylabel("均值分数")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
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

    for x, y, w, h, color, p, action, month in nodes:
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
        ax.text(x + 0.02, y + h - 0.07, p, fontsize=10, weight="bold", color="#0f172a")
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
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def to_win_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def write_audit(audit_path: Path, outputs: dict[str, Path]) -> None:
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_spec": to_win_path(Path("C:/Users/TSOU/Desktop/docx/IPA/图片需求.txt")),
        "output_dir": to_win_path(audit_path.parent),
        "thresholds": {
            "importance_mean_line": THRESH_IMPORTANCE,
            "performance_mean_line": THRESH_PERFORMANCE,
        },
        "point_count": len(POINTS),
        "points": [
            {
                "idx": p.idx,
                "name": p.name,
                "importance": p.importance,
                "performance": p.performance,
                "reference_quadrant": p.reference_quadrant,
                "computed_quadrant": quadrant_by_threshold(p.importance, p.performance),
            }
            for p in POINTS
        ],
        "outputs": {k: to_win_path(v) for k, v in outputs.items()},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out_dir = Path("data/figure/IPA")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scatter = out_dir / "图7-1_IPA四象限散点图.png"
    out_bar = out_dir / "图7-2_各文旅属性重要度与表现度均值对比.png"
    out_flow = out_dir / "图7-3_IPA整改行动优先级流程示意图.png"

    draw_scatter(out_scatter)
    draw_bar(out_bar)
    draw_flow(out_flow)

    write_audit(
        out_dir / "IPA_图片生成_audit.json",
        {
            "fig1_scatter": out_scatter,
            "fig2_bar": out_bar,
            "fig3_flow": out_flow,
        },
    )
    print(f"done: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
