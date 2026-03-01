from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle


# Use common CJK-capable fonts in priority order. Matplotlib will fallback if absent.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "SimSun",
    "STSong",
    "Songti SC",
    "Times New Roman",
    "DejaVu Serif",
]
plt.rcParams["font.sans-serif"] = [
    "SimSun",
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


SEM_BG = "#ececec"
SEM_LINE = "#333333"
LATENT_FACE = "#edd8c1"
LATENT_EDGE = "#ce8f45"
OBS_FACE = "#ece8f6"
OBS_EDGE = "#6f75ad"
ERR_FACE = "#ebf2de"
ERR_EDGE = "#8ea56f"


def draw_latent(ax, center: Point, text: str) -> None:
    node = Ellipse(
        (center.x, center.y),
        width=0.145,
        height=0.082,
        facecolor=LATENT_FACE,
        edgecolor=LATENT_EDGE,
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(node)
    ax.text(center.x, center.y, text, ha="center", va="center", fontsize=11, color="#2b2b2b", zorder=5)


def draw_observed(ax, center: Point, text: str) -> None:
    width = 0.15
    height = 0.046
    rect = Rectangle(
        (center.x - width / 2, center.y - height / 2),
        width=width,
        height=height,
        facecolor=OBS_FACE,
        edgecolor=OBS_EDGE,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(center.x, center.y, text, ha="center", va="center", fontsize=9.2, color="#2f2f2f", zorder=3)


def draw_error(ax, center: Point, label: str, latent_center: Point, rad: float = 0.0) -> None:
    err = Circle((center.x, center.y), radius=0.015, facecolor=ERR_FACE, edgecolor=ERR_EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(err)
    ax.text(center.x, center.y, label, ha="center", va="center", fontsize=8.5, color="#536845", zorder=4)
    arrow = FancyArrowPatch(
        (center.x, center.y - 0.006),
        (latent_center.x, latent_center.y + 0.025),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=0.95,
        color=SEM_LINE,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=13,
        zorder=2,
    )
    ax.add_patch(arrow)


def add_arrow(
    ax,
    p1: Point,
    p2: Point,
    label: str | None = None,
    rad: float = 0.0,
    label_xy: Point | None = None,
    shrink_a: float = 10.0,
    shrink_b: float = 12.0,
    mutation_scale: float = 10.0,
    linewidth: float = 1.0,
) -> None:
    arrow = FancyArrowPatch(
        (p1.x, p1.y),
        (p2.x, p2.y),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=SEM_LINE,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            label_xy.x,
            label_xy.y,
            label,
            fontsize=10,
            ha="center",
            va="center",
            color="#3a3a3a",
            zorder=6,
        )


def draw_sem_path_figure(output_path: Path, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(11.0, 7.2), facecolor=SEM_BG)
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.86], facecolor=SEM_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    latent = {
        "S_env": Point(0.22, 0.77),
        "S_service": Point(0.22, 0.55),
        "S_activity": Point(0.22, 0.33),
        "O_cognition": Point(0.55, 0.55),
        "R_intent": Point(0.84, 0.55),
    }

    draw_latent(ax, latent["S_env"], "环境氛围")
    draw_latent(ax, latent["S_service"], "服务质量")
    draw_latent(ax, latent["S_activity"], "活动内容")
    draw_latent(ax, latent["O_cognition"], "文化认知")
    draw_latent(ax, latent["R_intent"], "行为意愿")

    s_env_obs = [
        ("交通/停车便利", Point(0.09, 0.88)),
        ("环境舒适与氛围", Point(0.09, 0.82)),
        ("配套设施完善", Point(0.09, 0.76)),
        ("周边配套联动", Point(0.09, 0.70)),
    ]
    s_service_obs = [
        ("文化体验专业深度", Point(0.09, 0.64)),
        ("非遗项目独特参与", Point(0.09, 0.58)),
        ("非遗传承互动机会", Point(0.09, 0.52)),
        ("宣传真实性", Point(0.09, 0.46)),
        ("品牌知名度口碑", Point(0.09, 0.40)),
        ("线上宣传种草", Point(0.09, 0.34)),
    ]
    s_activity_obs = [
        ("特色药膳体验", Point(0.09, 0.26)),
        ("产品丰富独特", Point(0.09, 0.20)),
        ("价格合理性", Point(0.09, 0.14)),
    ]
    o_obs = [
        ("文化兴趣", Point(0.56, 0.89)),
        ("模式了解", Point(0.71, 0.71)),
        ("理解提升", Point(0.71, 0.39)),
        ("学习意愿", Point(0.56, 0.17)),
    ]
    r_obs = [
        ("游览意愿", Point(0.92, 0.60)),
        ("推荐意愿", Point(0.92, 0.50)),
    ]

    for idx, (text, pos) in enumerate(s_env_obs):
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_env"], pos, rad=-0.05 + 0.03 * idx)
    for idx, (text, pos) in enumerate(s_service_obs):
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_service"], pos, rad=-0.06 + 0.025 * idx)
    for idx, (text, pos) in enumerate(s_activity_obs):
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["S_activity"], pos, rad=-0.06 + 0.06 * idx)
    for idx, (text, pos) in enumerate(o_obs):
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["O_cognition"], pos, rad=(0.08, 0.03, -0.03, -0.08)[idx])
    for text, pos in r_obs:
        draw_observed(ax, pos, text)
        add_arrow(ax, latent["R_intent"], pos, rad=0.0)

    add_arrow(
        ax,
        latent["S_env"],
        latent["O_cognition"],
        label=".305***",
        label_xy=Point(0.42, 0.68),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["O_cognition"],
        label=".243***",
        label_xy=Point(0.42, 0.56),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["O_cognition"],
        label=".253***",
        label_xy=Point(0.42, 0.43),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["O_cognition"],
        latent["R_intent"],
        label=".147**",
        label_xy=Point(0.69, 0.57),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["S_env"],
        latent["R_intent"],
        label=".157**",
        rad=-0.22,
        label_xy=Point(0.58, 0.76),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["R_intent"],
        label=".184***",
        rad=0.14,
        label_xy=Point(0.57, 0.47),
        linewidth=1.05,
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["R_intent"],
        label=".206***",
        rad=0.23,
        label_xy=Point(0.58, 0.29),
        linewidth=1.05,
    )

    draw_error(ax, Point(0.36, 0.92), "e1", latent["S_env"], rad=-0.08)
    draw_error(ax, Point(0.36, 0.66), "e2", latent["S_service"], rad=-0.08)
    draw_error(ax, Point(0.36, 0.40), "e3", latent["S_activity"], rad=-0.08)
    draw_error(ax, Point(0.65, 0.73), "e4", latent["O_cognition"], rad=-0.06)
    draw_error(ax, Point(0.93, 0.71), "e5", latent["R_intent"], rad=-0.05)

    ax.text(0.50, 0.06, "注：*** p<0.001，** p<0.01，* p<0.05", fontsize=10.5, ha="center", va="bottom", color="#3e3e3e")
    ax.text(0.50, 0.01, "图7-1  路径分析图", fontsize=17, weight="bold", ha="center", va="bottom", color="#151515")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
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
    out_dir_new = Path("new/ESM")
    sem_name = "图7-1_SEM路径分析图.png"

    draw_sem_path_figure(out_dir / sem_name, dpi=360)
    draw_sem_path_figure(out_dir_new / sem_name, dpi=360)
    draw_ipa_figure(out_dir / "图7-2_IPA四象限散点图.png", dpi=320)
    print(f"draw_done: {out_dir.resolve()} | {out_dir_new.resolve()}")


if __name__ == "__main__":
    main()
