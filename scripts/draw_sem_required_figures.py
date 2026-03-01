from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Rectangle


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
MEASURE_LINE = "#676767"
STRUCT_LINE = "#303030"


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


def draw_observed(ax, center: Point, text: str, width: float = 0.15, height: float = 0.046, fontsize: float = 9.2) -> None:
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
    ax.text(center.x, center.y, text, ha="center", va="center", fontsize=fontsize, color="#2f2f2f", zorder=3)


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
    color: str = SEM_LINE,
    alpha: float = 1.0,
    linestyle: str = "-",
    label_bbox: bool = False,
) -> None:
    arrow = FancyArrowPatch(
        (p1.x, p1.y),
        (p2.x, p2.y),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
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
            bbox={"facecolor": SEM_BG, "edgecolor": "none", "boxstyle": "round,pad=0.15", "alpha": 0.94}
            if label_bbox
            else None,
            zorder=6,
        )


def draw_sem_path_figure(output_path: Path, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(13.2, 8.0), facecolor=SEM_BG)
    ax = fig.add_axes([0.02, 0.11, 0.96, 0.83], facecolor=SEM_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    latent = {
        "S_env": Point(0.25, 0.77),
        "S_service": Point(0.25, 0.55),
        "S_activity": Point(0.25, 0.33),
        "O_cognition": Point(0.58, 0.55),
        "R_intent": Point(0.85, 0.55),
    }

    draw_latent(ax, latent["S_env"], "环境氛围")
    draw_latent(ax, latent["S_service"], "服务质量")
    draw_latent(ax, latent["S_activity"], "活动内容")
    draw_latent(ax, latent["O_cognition"], "文化认知")
    draw_latent(ax, latent["R_intent"], "行为意愿")

    s_env_obs = [
        ("交通/停车便利", Point(0.09, 0.89)),
        ("环境舒适与氛围", Point(0.09, 0.83)),
        ("配套设施完善", Point(0.09, 0.77)),
        ("周边配套联动", Point(0.09, 0.71)),
    ]
    s_service_obs = [
        ("文化体验专业深度", Point(0.09, 0.65)),
        ("非遗项目独特参与", Point(0.09, 0.59)),
        ("非遗传承互动机会", Point(0.09, 0.53)),
        ("宣传真实性", Point(0.09, 0.47)),
        ("品牌知名度口碑", Point(0.09, 0.41)),
        ("线上宣传种草", Point(0.09, 0.35)),
    ]
    s_activity_obs = [
        ("特色药膳体验", Point(0.09, 0.27)),
        ("产品丰富独特", Point(0.09, 0.21)),
        ("价格合理性", Point(0.09, 0.15)),
    ]
    o_obs = [
        ("文化兴趣", Point(0.58, 0.90)),
        ("模式了解", Point(0.75, 0.72)),
        ("理解提升", Point(0.75, 0.38)),
        ("学习意愿", Point(0.58, 0.20)),
    ]
    r_obs = [
        ("游览意愿", Point(0.95, 0.62)),
        ("推荐意愿", Point(0.95, 0.48)),
    ]

    for idx, (text, pos) in enumerate(s_env_obs):
        draw_observed(ax, pos, text)
        add_arrow(
            ax,
            latent["S_env"],
            pos,
            rad=-0.06 + 0.03 * idx,
            linewidth=0.95,
            color=MEASURE_LINE,
            alpha=0.78,
        )
    for idx, (text, pos) in enumerate(s_service_obs):
        draw_observed(ax, pos, text)
        add_arrow(
            ax,
            latent["S_service"],
            pos,
            rad=-0.07 + 0.022 * idx,
            linewidth=0.95,
            color=MEASURE_LINE,
            alpha=0.78,
        )
    for idx, (text, pos) in enumerate(s_activity_obs):
        draw_observed(ax, pos, text)
        add_arrow(
            ax,
            latent["S_activity"],
            pos,
            rad=-0.07 + 0.07 * idx,
            linewidth=0.95,
            color=MEASURE_LINE,
            alpha=0.78,
        )
    for idx, (text, pos) in enumerate(o_obs):
        draw_observed(ax, pos, text)
        add_arrow(
            ax,
            latent["O_cognition"],
            pos,
            rad=(0.08, 0.02, -0.02, -0.08)[idx],
            linewidth=0.95,
            color=MEASURE_LINE,
            alpha=0.78,
        )
    for text, pos in r_obs:
        draw_observed(ax, pos, text)
        add_arrow(
            ax,
            latent["R_intent"],
            pos,
            rad=0.0,
            linewidth=0.95,
            color=MEASURE_LINE,
            alpha=0.78,
        )

    add_arrow(
        ax,
        latent["S_env"],
        latent["O_cognition"],
        label=".305***",
        rad=0.03,
        label_xy=Point(0.47, 0.69),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["O_cognition"],
        label=".243***",
        rad=0.00,
        label_xy=Point(0.47, 0.56),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["O_cognition"],
        label=".253***",
        rad=-0.03,
        label_xy=Point(0.47, 0.44),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["O_cognition"],
        latent["R_intent"],
        label=".147**",
        label_xy=Point(0.72, 0.575),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["S_env"],
        latent["R_intent"],
        label=".157**",
        rad=-0.34,
        label_xy=Point(0.68, 0.79),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["S_service"],
        latent["R_intent"],
        label=".184***",
        rad=-0.16,
        label_xy=Point(0.69, 0.665),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )
    add_arrow(
        ax,
        latent["S_activity"],
        latent["R_intent"],
        label=".206***",
        rad=0.30,
        label_xy=Point(0.69, 0.31),
        linewidth=1.15,
        color=STRUCT_LINE,
        label_bbox=True,
    )

    draw_error(ax, Point(0.38, 0.92), "e1", latent["S_env"], rad=-0.08)
    draw_error(ax, Point(0.38, 0.67), "e2", latent["S_service"], rad=-0.08)
    draw_error(ax, Point(0.38, 0.41), "e3", latent["S_activity"], rad=-0.08)
    draw_error(ax, Point(0.68, 0.74), "e4", latent["O_cognition"], rad=-0.07)
    draw_error(ax, Point(0.94, 0.72), "e5", latent["R_intent"], rad=-0.06)

    legend_handles = [
        Patch(facecolor=LATENT_FACE, edgecolor=LATENT_EDGE, label="椭圆：潜变量"),
        Patch(facecolor=OBS_FACE, edgecolor=OBS_EDGE, label="矩形：观测变量"),
        Line2D([0], [0], color=STRUCT_LINE, lw=1.2, label="深色箭线：结构路径（标注β）"),
        Line2D([0], [0], color=MEASURE_LINE, lw=1.0, alpha=0.78, label="浅色箭线：测量路径"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ERR_FACE,
            markeredgecolor=ERR_EDGE,
            markersize=7,
            label="圆形：残差项",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        framealpha=0.96,
        title="图例",
        fontsize=8.7,
        title_fontsize=9.2,
    )
    legend.get_frame().set_facecolor("#f5f5f5")
    legend.get_frame().set_edgecolor("#d0d0d0")

    ax.text(0.50, 0.055, "注：*** p<0.001，** p<0.01，* p<0.05", fontsize=10.5, ha="center", va="bottom", color="#3e3e3e")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_ipa_figure(output_path: Path, dpi: int = 300) -> None:
    source_csv = Path("data/data_analysis/_source_analysis/tables/IPA结果表.csv")
    rows = list(csv.DictReader(source_csv.open("r", encoding="utf-8-sig")))

    points = []
    for r in rows:
        points.append(
            (
                int(r["item_no"]),
                str(r["item_text"]).strip(),
                float(r["importance_mean"]),
                float(r["performance_mean"]),
            )
        )
    points.sort(key=lambda x: x[0])

    x_mean = sum(p[2] for p in points) / len(points)
    y_mean = sum(p[3] for p in points) / len(points)
    x_min = min(p[2] for p in points) - 0.03
    x_max = max(p[2] for p in points) + 0.03
    y_min = min(p[3] for p in points) - 0.03
    y_max = max(p[3] for p in points) + 0.03

    def quadrant(imp: float, sat: float) -> str:
        if imp >= x_mean and sat >= y_mean:
            return "Q1"
        if imp < x_mean and sat >= y_mean:
            return "Q2"
        if imp < x_mean and sat < y_mean:
            return "Q3"
        return "Q4"

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.add_patch(Rectangle((x_min, y_mean), x_mean - x_min, y_max - y_mean, facecolor="#fde7e9", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_mean, y_mean), x_max - x_mean, y_max - y_mean, facecolor="#e8f6ec", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((x_min, y_min), x_mean - x_min, y_mean - y_min, facecolor="#f1f3f5", alpha=0.65, zorder=0))
    ax.add_patch(Rectangle((x_mean, y_min), x_max - x_mean, y_mean - y_min, facecolor="#e7f0fd", alpha=0.45, zorder=0))

    ax.axvline(x_mean, color="#555555", linestyle="--", linewidth=1.2)
    ax.axhline(y_mean, color="#555555", linestyle="--", linewidth=1.2)

    color_map = {"Q1": "#2e7d32", "Q2": "#1565c0", "Q3": "#616161", "Q4": "#c62828"}
    for no, _, imp, sat in points:
        q = quadrant(imp, sat)
        ax.scatter(imp, sat, s=60, color=color_map[q], zorder=3)
        ax.text(imp + 0.0015, sat + 0.0015, str(no), fontsize=10, color=color_map[q], weight="bold")

    ax.text(x_min + 0.01, y_max - 0.008, "Q2 维持区", fontsize=9, va="top", color="#0d47a1")
    ax.text(x_max - 0.01, y_max - 0.008, "Q1 优势区", fontsize=9, va="top", ha="right", color="#1b5e20")
    ax.text(x_min + 0.01, y_min + 0.003, "Q3 机会区", fontsize=9, va="bottom", color="#37474f")
    ax.text(x_max - 0.01, y_min + 0.003, "Q4 改进区", fontsize=9, va="bottom", ha="right", color="#8b1e1e")

    ax.text(x_min + 0.002, y_mean + 0.001, f"满意度均值 = {y_mean:.4f}", fontsize=8.8, color="#444444", va="bottom")
    ax.text(x_mean + 0.001, y_min + 0.001, f"重要度均值 = {x_mean:.4f}", fontsize=8.8, color="#444444", va="bottom")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("重要度（Importance）")
    ax.set_ylabel("满意度（Satisfaction）")
    ax.set_title("图7-2  IPA四象限散点图", pad=10)
    ax.grid(alpha=0.2, linestyle=":")

    item_lines = [f"{no} {name}" for no, name, _, _ in points]
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
