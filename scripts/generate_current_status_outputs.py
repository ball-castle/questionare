#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


INPUT_CSV = Path("data/data_analysis/_source_analysis/tables/survey_clean.csv")
OUTPUT_DIR = Path("new/现状")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def count_any_selected(df: pd.DataFrame, cols: list[str]) -> int:
    arr = df[cols].apply(pd.to_numeric, errors="coerce")
    return int((arr.eq(1)).any(axis=1).sum())


def fig3_channel_distribution(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, Path]:
    channels = OrderedDict(
        [
            ("C009", "社交媒体（抖音、小红书、微信、微博等）"),
            ("C010", "旅游平台（携程、美团等）"),
            ("C011", "亲友/同事推荐"),
            ("C012", "线下门店/街区路过发现"),
            ("C013", "新闻/公众号推文"),
            ("C014", "电视/广播"),
            ("C015", "其他"),
        ]
    )

    rows = []
    for code, label in channels.items():
        count = int(pd.to_numeric(df[code], errors="coerce").eq(1).sum())
        rows.append(
            {
                "渠道": label,
                "对应变量": code,
                "提及频次（次）": count,
                "占比（%）": round(count * 100.0 / n, 2),
            }
        )
    out_df = pd.DataFrame(rows).sort_values("提及频次（次）", ascending=False).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "图3_游客了解叶开泰的信息渠道分布_数据.csv"
    save_csv(out_df, csv_path)

    plot_df = out_df.sort_values("提及频次（次）", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#D1D1D1")
    ax.set_facecolor("#D1D1D1")

    n_bars = len(plot_df)
    shades = np.linspace(0.30, 0.70, n_bars) if n_bars > 1 else np.array([0.55])
    colors = [plt.cm.Greens(s) for s in shades]
    bars = ax.barh(
        plot_df["渠道"],
        plot_df["占比（%）"],
        color=colors,
        edgecolor="#F0F0F0",
        linewidth=1.0,
        height=0.56,
    )

    for bar, pct in zip(bars, plot_df["占比（%）"]):
        ax.text(bar.get_width() + 0.45, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=14, color="#333333")

    ax.set_xlabel("占比 (%)", fontsize=13)
    ax.set_ylabel("")
    ax.set_xlim(0, max(50, float(plot_df["占比（%）"].max()) + 8))
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.tick_params(axis="both", labelsize=12, length=0, colors="#333333")
    fig.tight_layout()

    png_path = OUTPUT_DIR / "图3_游客了解叶开泰的信息渠道分布.png"
    fig.savefig(png_path, dpi=320)
    plt.close(fig)
    return out_df, png_path


def fig4_value_perception(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, Path]:
    # 说明：原问卷中无“价值维度提及”单题，以下为可复核的代理口径映射。
    configs = [
        {
            "价值维度": "建筑风貌与历史氛围",
            "method": "likert_threshold",
            "vars": ["C059"],
            "rule": "C059（环境舒适度与文化氛围）>=3 计为有提及",
        },
        {
            "价值维度": "中医药文化展示体验",
            "method": "any_selected",
            "vars": ["C016", "C027"],
            "rule": "C016（文化学习动机）或 C027（文化参观/研学）任一勾选",
        },
        {
            "价值维度": "近四百年品牌历史",
            "method": "any_selected",
            "vars": ["C020"],
            "rule": "C020（慕名而来/品牌吸引力）勾选",
        },
        {
            "价值维度": "国医堂诊疗服务",
            "method": "any_selected",
            "vars": ["C021", "C026"],
            "rule": "C021（体验调理服务）或 C026（诊疗/理疗服务）任一勾选",
        },
        {
            "价值维度": "景区养生互动项目",
            "method": "any_selected",
            "vars": ["C028", "C031"],
            "rule": "C028（药膳体验）或 C031（主题活动/课程）任一勾选",
        },
        {
            "价值维度": "配套服务与购物体验",
            "method": "any_selected",
            "vars": ["C029", "C030"],
            "rule": "C029（药食同源产品购买）或 C030（中药材/文创购买）任一勾选",
        },
    ]

    rows = []
    for conf in configs:
        if conf["method"] == "likert_threshold":
            s = pd.to_numeric(df[conf["vars"][0]], errors="coerce")
            count = int((s >= 3).sum())
        else:
            count = count_any_selected(df, conf["vars"])

        rows.append(
            {
                "价值维度": conf["价值维度"],
                "提及频次（次）": count,
                "占比（%）": round(count * 100.0 / n, 2),
                "代理口径": conf["rule"],
                "对应变量": ",".join(conf["vars"]),
            }
        )

    out_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "图4_游客对叶开泰街区价值感知分布_数据_代理口径.csv"
    save_csv(out_df, csv_path)

    plot_df = out_df.sort_values("提及频次（次）", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    bars = ax.barh(plot_df["价值维度"], plot_df["提及频次（次）"], color="#F28E2B")
    for bar, cnt, pct in zip(bars, plot_df["提及频次（次）"], plot_df["占比（%）"]):
        ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2, f"{int(cnt)} ({pct:.2f}%)", va="center", fontsize=9)
    ax.set_title("图4 游客对叶开泰街区价值感知分布（代理口径）")
    ax.set_xlabel("提及频次（次）")
    ax.set_ylabel("价值维度")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = OUTPUT_DIR / "图4_游客对叶开泰街区价值感知分布.png"
    fig.savefig(png_path, dpi=320)
    plt.close(fig)
    return out_df, png_path


def fig5_zone_status(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    # 说明：原问卷无“保护状况”直接题项；此处以 IPA 重要度均值近似“保护关注度”。
    configs = [
        ("中医药文化博物馆", "C066", "C076", "文化展示与非遗体验项目"),
        ("国医堂诊疗区", "C074", "C084", "中医药服务专业度"),
        ("秘药局产品区", "C070", "C080", "美食/文创产品种类与品质"),
        ("互动体验区", "C071", "C081", "个性化体质辨识与养生咨询"),
        ("明清建筑景观区", "C067", "C077", "环境舒适度与卫生状况"),
    ]

    rows = []
    for zone, protect_col, exp_col, item in configs:
        protect = float(pd.to_numeric(df[protect_col], errors="coerce").mean())
        exp = float(pd.to_numeric(df[exp_col], errors="coerce").mean())
        rows.append(
            {
                "功能区": zone,
                "保护状况均值（1-5）": round(protect, 3),
                "体验满意度均值（1-5）": round(exp, 3),
                "代理映射题项": item,
                "保护代理变量": protect_col,
                "体验代理变量": exp_col,
            }
        )

    out_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "图5_叶开泰各功能区保护与体验情况_数据_代理口径.csv"
    save_csv(out_df, csv_path)

    x = np.arange(len(out_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    b1 = ax.bar(x - width / 2, out_df["保护状况均值（1-5）"], width=width, color="#59A14F", label="保护状况均值（代理）")
    b2 = ax.bar(x + width / 2, out_df["体验满意度均值（1-5）"], width=width, color="#E15759", label="体验满意度均值")
    for bars in (b1, b2):
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(out_df["功能区"])
    ax.set_ylim(1, 5.2)
    ax.set_ylabel("均值（1-5）")
    ax.set_title("图5 叶开泰各功能区保护与体验情况（代理口径）")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = OUTPUT_DIR / "图5_叶开泰各功能区保护与体验情况.png"
    fig.savefig(png_path, dpi=320)
    plt.close(fig)
    return out_df, png_path


def build_feasibility_list() -> pd.DataFrame:
    rows = [
        {
            "序号": 1,
            "内容模块": "知名度（图3）",
            "可生成性": "可直接生成",
            "说明": "问卷存在知晓渠道多选题（C009-C015）。",
        },
        {
            "序号": 2,
            "内容模块": "价值感知（图4）",
            "可生成性": "可生成（代理口径）",
            "说明": "问卷无“价值维度提及”直题，基于现有题项映射统计。",
        },
        {
            "序号": 3,
            "内容模块": "功能区保护开发（图5）",
            "可生成性": "可生成（代理口径）",
            "说明": "问卷无“保护状况”直题，以IPA重要度均值近似保护关注度。",
        },
        {
            "序号": 4,
            "内容模块": "政策环境影响（图6桑基图）",
            "可生成性": "当前不可直接生成",
            "说明": "现有108题无政策环境影响题项，缺失关键变量。",
        },
        {
            "序号": 5,
            "内容模块": "与同类目的地差异（图7）",
            "可生成性": "当前不可直接生成",
            "说明": "现有问卷无同类目的地差异评分题项。",
        },
        {
            "序号": 6,
            "内容模块": "与老字号景区横向比较（图8雷达图）",
            "可生成性": "当前不可直接生成",
            "说明": "现有问卷无同仁堂/胡庆余堂等横向对比题项。",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    n = len(df)

    fig3_df, fig3_png = fig3_channel_distribution(df, n)
    fig4_df, fig4_png = fig4_value_perception(df, n)
    fig5_df, fig5_png = fig5_zone_status(df)

    feas_df = build_feasibility_list()
    save_csv(feas_df, OUTPUT_DIR / "可生成性清单.csv")

    notes = [
        "# 现状模块口径说明",
        "",
        f"- 样本口径：`{INPUT_CSV.as_posix()}`，样本量 n={n}。",
        "- 图3为原题直连统计。",
        "- 图4与图5均为代理口径，详见对应CSV中的“代理口径/代理映射题项”字段。",
        "- 图6-图8因缺少原始题项，当前仅给出不可直出说明，未生成伪造数据。",
    ]
    (OUTPUT_DIR / "口径说明.md").write_text("\n".join(notes), encoding="utf-8")

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(INPUT_CSV.as_posix()),
        "sample_size": int(n),
        "outputs": {
            "fig3_data_csv": str((OUTPUT_DIR / "图3_游客了解叶开泰的信息渠道分布_数据.csv").as_posix()),
            "fig3_png": str(fig3_png.as_posix()),
            "fig4_data_csv": str((OUTPUT_DIR / "图4_游客对叶开泰街区价值感知分布_数据_代理口径.csv").as_posix()),
            "fig4_png": str(fig4_png.as_posix()),
            "fig5_data_csv": str((OUTPUT_DIR / "图5_叶开泰各功能区保护与体验情况_数据_代理口径.csv").as_posix()),
            "fig5_png": str(fig5_png.as_posix()),
            "feasibility_csv": str((OUTPUT_DIR / "可生成性清单.csv").as_posix()),
            "notes_md": str((OUTPUT_DIR / "口径说明.md").as_posix()),
        },
        "data_preview": {
            "fig3_top_channel": fig3_df.iloc[0].to_dict() if not fig3_df.empty else {},
            "fig4_top_dimension": fig4_df.sort_values("提及频次（次）", ascending=False).iloc[0].to_dict() if not fig4_df.empty else {},
            "fig5_zone_count": int(len(fig5_df)),
        },
    }
    (OUTPUT_DIR / "现状生成_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
