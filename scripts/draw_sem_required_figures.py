from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle


# Use common CJK-capable fonts in priority order. Matplotlib will fallback if absent.
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
    x: float
    y: float


def draw_latent(ax, center: Point, text: str) -> None:
    node = Ellipse(
        (center.x, center.y),
        width=0.16,
        height=0.09,
        facecolor="#eef5ff",
        edgecolor="black",
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(node)
    ax.text(center.x, center.y, text, ha="center", va="center", fontsize=10, zorder=5)


def draw_observed(ax, center: Point, text: str) -> None:
    width = 0.19
    height = 0.05
    rect = Rectangle(
        (center.x - width / 2, center.y - height / 2),
        width=width,
        height=height,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(center.x, center.y, text, ha="center", va="center", fontsize=8.2, zorder=3)


def draw_error(ax, center: Point, label: str, latent_center: Point) -> None:
    err = Circle((center.x, center.y), radius=0.022, facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
    ax.add_patch(err)
    ax.text(center.x, center.y, label, ha="center", va="center", fontsize=8, zorder=4)
    arrow = FancyArrowPatch(
        (center.x, center.y - 0.01),
        (latent_center.x + 0.02, latent_center.y + 0.01),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="black",
        connectionstyle="arc3,rad=-0.1",
        zorder=2,
    )
    ax.add_patch(arrow)


def add_arrow(ax, p1: Point, p2: Point, label: str | None = None, rad: float = 0.0, label_xy: Point | None = None) -> None:
    arrow = FancyArrowPatch(
        (p1.x, p1.y),
        (p2.x, p2.y),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color="black",
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy.x,
            label_xy.y,
            label,
            fontsize=8.8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            zorder=6,
        )


def draw_sem_path_figure(output_path: Path, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    latent = {
        "S_env": Point(0.20, 0.80),
        "S_service": Point(0.20, 0.54),
        "S_activity": Point(0.20, 0.28),
        "O_cognition": Point(0.52, 0.54),
        "R_intent": Point(0.82, 0.54),
    }

    draw_latent(ax, latent["S_env"], "S_env\n环境氛围")
    draw_latent(ax, latent["S_service"], "S_service\n服务质量")
    draw_latent(ax, latent["S_activity"], "S_activity\n活动内容")
    draw_latent(ax, latent["O_cognition"], "O_cognition\n文化认知")
    draw_latent(ax, latent["R_intent"], "R_intent\n行为意愿")

    s_env_obs = [
        ("C058\n交通/停车便利", Point(0.06, 0.91)),
        ("C059\n环境舒适与氛围", Point(0.06, 0.84)),
        ("C060\n配套设施完善", Point(0.06, 0.77)),
        ("C061\n周边配套联动", Point(0.06, 0.70)),
    ]
    s_service_obs = [
        ("C052\n文化体验专业深度", Point(0.06, 0.64)),
        ("C053\n非遗项目独特参与", Point(0.06, 0.57)),
        ("C054\n非遗传承互动机会", Point(0.06, 0.50)),
        ("C062\n宣传真实性", Point(0.06, 0.43)),
        ("C063\n品牌知名度口碑", Point(0.06, 0.36)),
        ("C065\n线上宣传种草", Point(0.06, 0.29)),
    ]
    s_activity_obs = [
        ("C055\n特色药膳体验", Point(0.06, 0.22)),
        ("C056\n产品丰富独特", Point(0.06, 0.15)),
        ("C057\n价格合理性", Point(0.06, 0.08)),
    ]
    o_obs = [
        ("C086\n文化兴趣", Point(0.55, 0.86)),
        ("C087\n模式了解", Point(0.68, 0.70)),
        ("C088\n理解提升", Point(0.68, 0.38)),
        ("C089\n学习意愿", Point(0.55, 0.22)),
    ]
    r_obs = [
        ("C090\n游览意愿", Point(0.95, 0.61)),
        ("C091\n推荐意愿", Point(0.95, 0.47)),
    ]

    for text, pos in s_env_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_env"], pos)
    for text, pos in s_service_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_service"], pos)
    for text, pos in s_activity_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_activity"], pos)
    for text, pos in o_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["O_cognition"], pos)
    for text, pos in r_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["R_intent"], pos)

    add_arrow(
        ax,
        latent["S_env"],
        latent["O_cognition"],
        label="β = 0.3055 ***",
        label_xy=Point(0.36, 0.73),
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["O_cognition"],
        label="β = 0.2433 ***",
        label_xy=Point(0.36, 0.56),
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["O_cognition"],
        label="β = 0.2525 ***",
        label_xy=Point(0.36, 0.39),
    )
    add_arrow(
        ax,
        latent["O_cognition"],
        latent["R_intent"],
        label="β = 0.1468 **",
        label_xy=Point(0.67, 0.58),
    )
    add_arrow(
        ax,
        latent["S_env"],
        latent["R_intent"],
        label="β = 0.1567 **",
        rad=-0.28,
        label_xy=Point(0.53, 0.86),
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["R_intent"],
        label="β = 0.1841 ***",
        rad=0.22,
        label_xy=Point(0.50, 0.44),
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["R_intent"],
        label="β = 0.2058 ***",
        rad=0.36,
        label_xy=Point(0.53, 0.19),
    )

    draw_error(ax, Point(0.34, 0.89), "e1", latent["S_env"])
    draw_error(ax, Point(0.34, 0.62), "e2", latent["S_service"])
    draw_error(ax, Point(0.34, 0.35), "e3", latent["S_activity"])
    draw_error(ax, Point(0.62, 0.66), "e4", latent["O_cognition"])
    draw_error(ax, Point(0.91, 0.67), "e5", latent["R_intent"])

    ax.text(0.50, 0.02, "*** p<0.001, ** p<0.01, * p<0.05", fontsize=9, ha="center", va="bottom")
    ax.text(0.50, 0.98, "图7-1  SEM路径分析图（intent_partial_v1）", fontsize=12, ha="center", va="top")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def draw_ipa_figure(output_path: Path, dpi: int = 300) -> None:
    x_mean = 3.7061
    y_mean = 3.7672

    points = [
        ("①", "丰富的中医药文化展示和非遗体验项目", 3.742, 3.791, "Q1"),
        ("②", "环境舒适度与卫生状况", 3.684, 3.779, "Q2"),
        ("③", "便捷交通与停车位", 3.666, 3.735, "Q3"),
        ("④", "亲友推荐/正面评价多", 3.742, 3.701, "Q4"),
        ("⑤", "特色美食与文创品质", 3.616, 3.742, "Q3"),
        ("⑥", "个性化体质辨识与养生咨询", 3.732, 3.804, "Q1"),
        ("⑦", "服务专业度与态度", 3.722, 3.836, "Q1"),
        ("⑧", "产品价格与优惠力度（边界）", 3.708, 3.755, "Q4"),
        ("⑨", "中医药服务专业度", 3.712, 3.813, "Q1"),
        ("⑩", "线上线下宣传推广", 3.753, 3.692, "Q4"),
    ]

    x_min, x_max = 3.58, 3.80
    y_min, y_max = 3.66, 3.86

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.add_patch(Rectangle((x_min, y_mean), x_mean - x_min, y_max - y_mean, facecolor="#fde7e9", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_mean, y_mean), x_max - x_mean, y_max - y_mean, facecolor="#e8f6ec", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_min, y_min), x_mean - x_min, y_mean - y_min, facecolor="#f1f3f5", alpha=0.65, zorder=0))
    ax.add_patch(Rectangle((x_mean, y_min), x_max - x_mean, y_mean - y_min, facecolor="#e7f0fd", alpha=0.45, zorder=0))

    ax.axvline(x_mean, color="#555555", linestyle="--", linewidth=1.2)
    ax.axhline(y_mean, color="#555555", linestyle="--", linewidth=1.2)

    color_map = {"Q1": "#2e7d32", "Q2": "#c62828", "Q3": "#616161", "Q4": "#1565c0"}
    for no, _, x, y, q in points:
        ax.scatter(x, y, s=60, color=color_map[q], zorder=3)
        ax.text(x + 0.0015, y + 0.0015, no, fontsize=10, color=color_map[q], weight="bold")

    ax.text(3.595, 3.848, "Q2: 优先改进\nConcentrate Here", fontsize=9, va="top", color="#7f1d1d")
    ax.text(3.792, 3.848, "Q1: 保持优势\nKeep Up the Good Work", fontsize=9, va="top", ha="right", color="#1b5e20")
    ax.text(3.595, 3.665, "Q3: 低优先级\nLow Priority", fontsize=9, va="bottom", color="#37474f")
    ax.text(3.792, 3.665, "Q4: 过度投入\nPossible Overkill", fontsize=9, va="bottom", ha="right", color="#0d47a1")

    ax.text(3.582, 3.7715, f"重要度均值 = {y_mean:.4f}", fontsize=8.8, color="#444444", va="bottom")
    ax.text(3.7085, 3.662, f"表现度均值 = {x_mean:.4f}", fontsize=8.8, color="#444444", va="bottom")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("表现度（Performance）")
    ax.set_ylabel("重要度（Importance）")
    ax.set_title("图7-2  IPA四象限散点图", pad=10)
    ax.grid(alpha=0.2, linestyle=":")

    item_lines = [f"{no} {name}" for no, name, _, _, _ in points]
    fig.text(
        0.02,
        0.01,
        "属性编号：  " + "  |  ".join(item_lines),
        fontsize=8.1,
        ha="left",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    out_dir = Path("data/figure/SEM")
    draw_sem_path_figure(out_dir / "图7-1_SEM路径分析图.png", dpi=320)
    draw_ipa_figure(out_dir / "图7-2_IPA四象限散点图.png", dpi=320)
    print(f"draw_done: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
