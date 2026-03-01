#!/usr/bin/env python3
"""按 docx 文本需求生成聚类图片（全中文标注）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import FancyBboxPatch


matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 report (1).docx 所需聚类图片。")
    parser.add_argument(
        "--base-dir",
        default=r"C:\Users\TSOU\Desktop\聚类数据捏造",
        help="输入文件所在目录（默认：C:\\Users\\TSOU\\Desktop\\聚类数据捏造）",
    )
    parser.add_argument("--metrics-file", default="kmeans_metrics.csv", help="聚类指标文件（csv/xlsx）")
    parser.add_argument("--raw-file", default="raw_survey_data.xlsx", help="原始问卷文件（csv/xlsx）")
    parser.add_argument("--feature-file", default="features_encoded.xlsx", help="特征编码文件（csv/xlsx）")
    parser.add_argument("--summary-file", default="cluster_summary.xlsx", help="聚类汇总文件（csv/xlsx）")
    parser.add_argument("--output-dir", default="new/聚类", help="图片输出目录")
    parser.add_argument("--dpi", type=int, default=320, help="图片分辨率")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return base_dir / p


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def find_col(columns: list[str], include: list[str], exclude: list[str] | None = None) -> str:
    exclude = exclude or []
    for c in columns:
        if all(k in c for k in include) and not any(x in c for x in exclude):
            return c
    raise KeyError(f"未找到列，包含关键字: {include}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def draw_quality_figure(metrics: pd.DataFrame, out_path: Path, dpi: int) -> dict[str, float]:
    cols = metrics.columns.tolist()
    k_col = "k" if "k" in cols else find_col(cols, ["k"])
    s_col = "silhouette" if "silhouette" in cols else find_col(cols, ["sil"])
    ch_col = "CH" if "CH" in cols else find_col(cols, ["CH"])

    df = metrics[[k_col, s_col, ch_col]].copy()
    df[k_col] = to_numeric(df[k_col]).astype("Int64")
    df[s_col] = to_numeric(df[s_col])
    df[ch_col] = to_numeric(df[ch_col])
    df = df.dropna().sort_values(k_col)

    ks = df[k_col].astype(int).to_list()
    sils = df[s_col].to_list()
    chs = df[ch_col].to_list()

    labels = [f"{int(k)}类" for k in ks]
    if 4 in ks:
        labels[ks.index(4)] = "4类（最优）"

    bar_colors = ["#8EA0AA" if k != 4 else "#E74C3C" for k in ks]
    line_color = "#2C7FB8"

    # 合并为单图双轴，提升插入 docx 后的可读性
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    x = np.arange(len(labels))

    bars = ax.bar(x, sils, color=bar_colors, width=0.52, edgecolor="white", label="轮廓系数")
    ax.axhline(0.62, color="#2E9F62", linestyle="--", linewidth=1.8, label="良好阈值（0.62）")
    ax.axhline(0.68, color="#E7A73D", linestyle="--", linewidth=1.8, label="优秀阈值（0.68）")

    for i, (b, v) in enumerate(zip(bars, sils)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.016,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
            color="#263238" if ks[i] != 4 else "#C62828",
        )

    ax.set_ylim(0, max(0.92, max(sils) + 0.2))
    ax.set_ylabel("轮廓系数", fontsize=13)
    ax.set_xlabel("聚类方案", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, chs, color=line_color, marker="o", linewidth=2.6, markersize=8, label="卡林斯基-哈拉巴斯指数")
    for i, v in enumerate(chs):
        ax2.text(
            x[i],
            v + 20,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
            color=line_color,
        )
    y2_min = min(chs) * 0.78
    y2_max = max(chs) * 1.15
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("卡林斯基-哈拉巴斯指数", fontsize=13, color=line_color)
    ax2.tick_params(axis="y", labelsize=12, colors=line_color)

    if 4 in ks:
        idx = ks.index(4)
        ax.annotate(
            "四类方案最优",
            xy=(x[idx], sils[idx]),
            xytext=(x[idx] - 0.95, sils[idx] + 0.12),
            arrowprops=dict(arrowstyle="->", color="#C0392B", linewidth=1.2),
            color="#C0392B",
            fontsize=12,
            weight="bold",
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "轮廓系数_2类": float(df.loc[df[k_col] == 2, s_col].iloc[0]) if (df[k_col] == 2).any() else float("nan"),
        "轮廓系数_3类": float(df.loc[df[k_col] == 3, s_col].iloc[0]) if (df[k_col] == 3).any() else float("nan"),
        "轮廓系数_4类": float(df.loc[df[k_col] == 4, s_col].iloc[0]) if (df[k_col] == 4).any() else float("nan"),
        "CH_2类": float(df.loc[df[k_col] == 2, ch_col].iloc[0]) if (df[k_col] == 2).any() else float("nan"),
        "CH_3类": float(df.loc[df[k_col] == 3, ch_col].iloc[0]) if (df[k_col] == 3).any() else float("nan"),
        "CH_4类": float(df.loc[df[k_col] == 4, ch_col].iloc[0]) if (df[k_col] == 4).any() else float("nan"),
    }


def draw_cluster_matrix_figure(features: pd.DataFrame, summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    cols = features.columns.tolist()
    cid_col = "Cluster_ID" if "Cluster_ID" in cols else find_col(cols, ["Cluster", "ID"])

    factors = [
        {
            "name": "年龄",
            "col": find_col(cols, ["年龄_编码"]),
            "codes": [0, 1, 2, 3, 4],
            "labels": ["18岁以下", "18-25", "26-45", "46-64", "65岁以上"],
            "weight": 1.00,
        },
        {
            "name": "教育",
            "col": find_col(cols, ["教育程度_编码"]),
            "codes": [0, 1, 2, 3, 4],
            "labels": ["初中及以下", "中专/高中", "大专", "本科", "硕士及以上"],
            "weight": 0.48,
        },
        {
            "name": "收入",
            "col": find_col(cols, ["月收入_编码"]),
            "codes": [0, 1, 2, 3, 4],
            "labels": ["3千以下", "3-5千", "5-8千", "8-15千", "15千以上"],
            "weight": 0.33,
        },
        {
            "name": "数字导览",
            "col": find_col(cols, ["数字化导览重要性"]),
            "codes": [1, 2, 3, 4, 5],
            "labels": ["1分", "2分", "3分", "4分", "5分"],
            "weight": 0.32,
        },
        {
            "name": "文化体验",
            "col": find_col(cols, ["中医药文化重要性"]),
            "codes": [1, 2, 3, 4, 5],
            "labels": ["1分", "2分", "3分", "4分", "5分"],
            "weight": 0.17,
        },
        {
            "name": "交通便利",
            "col": find_col(cols, ["交通便利重要性"]),
            "codes": [1, 2, 3, 4, 5],
            "labels": ["1分", "2分", "3分", "4分", "5分"],
            "weight": 0.13,
        },
        {
            "name": "职业",
            "col": find_col(cols, ["职业_编码"]),
            "codes": [0, 1, 2, 3, 4, 5, 6, 7],
            "labels": ["学生", "企业职员", "公职人员", "自由职业", "服务从业", "离退休", "个体经营", "其他"],
            "weight": 0.07,
        },
    ]

    sid_col = "聚类编号" if "聚类编号" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "编号"])
    sname_col = "聚类名称" if "聚类名称" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "名称"])
    sn_col = "样本量" if "样本量" in summary.columns else find_col(summary.columns.tolist(), ["样本量"])
    sshare_col = "占比(%)" if "占比(%)" in summary.columns else find_col(summary.columns.tolist(), ["占比"])

    meta = summary[[sid_col, sname_col, sn_col, sshare_col]].copy()
    meta[sid_col] = to_numeric(meta[sid_col]).astype(int)
    meta = meta.sort_values(sid_col)

    row_count = meta.shape[0]
    col_count = len(factors)
    # 控制画布宽度，避免插入 docx 后整体缩放过大导致标题过小
    fig, axes = plt.subplots(row_count, col_count, figsize=(18.5, 12.5), sharey=True)

    if row_count == 1:
        axes = np.array([axes])

    cluster_colors = {
        1: "#53C788",
        2: "#6AB0DD",
        3: "#E4A564",
        4: "#B494C4",
    }
    class_label_map = {
        1: "第一类",
        2: "第二类",
        3: "第三类",
        4: "第四类",
    }

    for i, (_, row) in enumerate(meta.iterrows()):
        cid = int(row[sid_col])
        cname = str(row[sname_col])
        cshare = float(row[sshare_col])
        subset = features[to_numeric(features[cid_col]) == cid].copy()
        for j, factor in enumerate(factors):
            ax = axes[i, j]
            series = to_numeric(subset[factor["col"]]).round().astype("Int64")
            dist = (
                series.value_counts(normalize=True)
                .reindex(factor["codes"], fill_value=0.0)
                .astype(float)
                * 100.0
            )

            weight = float(factor["weight"])
            shade = matplotlib.cm.YlOrBr(0.08 + 0.47 * weight)
            ax.set_facecolor(to_hex(shade))

            x = np.arange(len(factor["codes"]))
            bars = ax.bar(
                x,
                dist.to_numpy(),
                color=cluster_colors.get(cid, "#6C7A89"),
                width=0.82,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.88,
            )
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.15)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xticks(x)

            if i == row_count - 1:
                ax.set_xticklabels(factor["labels"], fontsize=7)
            else:
                ax.set_xticklabels([])

            if i == 0:
                ax.set_title(
                    f"{factor['name']}\n权重{weight:.2f}",
                    fontsize=18,
                    pad=10,
                    weight="bold",
                )

            if j == 0:
                class_name = class_label_map.get(cid, f"第{cid}类")
                ax.set_ylabel(
                    f"{class_name}\n{cshare:.1f}%",
                    fontsize=20,
                    rotation=0,
                    labelpad=54,
                    va="center",
                    weight="bold",
                )

            for bar, val in zip(bars, dist.to_numpy()):
                if val >= 10.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val * 0.5,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                        weight="bold",
                    )

    fig.tight_layout(rect=[0.06, 0.03, 1.0, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def guess_preference_label(col_name: str) -> str:
    t = col_name
    t = t.replace("Q17_", "").replace("Q18_", "")
    t = t.replace("兴趣均值", "").replace("期望均值", "")
    t = t.replace("中医药体验", "中医药体验")
    t = t.replace("_", "")
    return t


def draw_persona_figure(features: pd.DataFrame, summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    cols = features.columns.tolist()
    cid_col = "Cluster_ID" if "Cluster_ID" in cols else find_col(cols, ["Cluster", "ID"])
    sid_col = "聚类编号" if "聚类编号" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "编号"])
    sname_col = "聚类名称" if "聚类名称" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "名称"])
    sshare_col = "占比(%)" if "占比(%)" in summary.columns else find_col(summary.columns.tolist(), ["占比"])

    age_col = find_col(cols, ["年龄_编码"])
    edu_col = find_col(cols, ["教育程度_编码"])
    income_col = find_col(cols, ["月收入_编码"])
    occ_col = find_col(cols, ["职业_编码"])

    age_map = {0: "18岁以下", 1: "18-25岁", 2: "26-45岁", 3: "46-64岁", 4: "65岁及以上"}
    edu_map = {0: "初中及以下", 1: "中专/高中", 2: "大专", 3: "本科", 4: "硕士及以上"}
    income_map = {0: "3千元以下", 1: "3-5千元", 2: "5-8千元", 3: "8-15千元", 4: "15千元以上"}
    occ_map = {
        0: "学生",
        1: "企业职员",
        2: "公职人员",
        3: "自由职业",
        4: "服务从业",
        5: "离退休人员",
        6: "个体经营",
        7: "其他",
    }

    story = {
        1: "年轻中高教育群体，偏好自由参观与轻量体验，主题活动兴趣较高。",
        2: "规模最大且付费意愿更强，偏好智慧导览、非遗展演与深度文化体验。",
        3: "高龄与低龄两端并存，整体需求较低，更关注省心便捷与交通接驳。",
        4: "以年轻低收入群体为主，重视高性价比服务与高效率参观支持。",
    }
    action = {
        1: ["增加自助路线与打卡点", "强化节庆活动传播", "提供轻互动体验包"],
        2: ["打造深度体验套餐", "升级数字化导览系统", "建设会员分层权益"],
        3: ["开通一键接驳服务", "设计低体力游览路线", "配置志愿讲解支持"],
        4: ["扩展公益讲解时段", "推广智慧导览免费版", "推出经济型一日游产品"],
    }

    pref_cols = [c for c in summary.columns if (str(c).startswith("Q17_") or str(c).startswith("Q18_")) and "均值" in str(c)]
    meta = summary.copy()
    meta[sid_col] = to_numeric(meta[sid_col]).astype(int)
    meta = meta.sort_values(sid_col)

    card_count = int(meta.shape[0])
    col_count = 1 if card_count == 1 else 2
    row_count = int(np.ceil(card_count / col_count))
    fig_width = 9.6 if col_count == 1 else 13.2
    fig_height = 6.2 * row_count
    fig, axes = plt.subplots(row_count, col_count, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    if isinstance(axes, np.ndarray):
        axes_flat = axes.reshape(-1)
    else:
        axes_flat = np.array([axes])

    card_colors = {
        1: "#2FAE62",
        2: "#3484B8",
        3: "#E67E22",
        4: "#8E44AD",
    }

    for ax, (_, row) in zip(axes_flat, meta.iterrows()):
        cid = int(row[sid_col])
        cname = str(row[sname_col])
        share = float(row[sshare_col])
        c = card_colors.get(cid, "#607D8B")
        subset = features[to_numeric(features[cid_col]) == cid].copy()

        def mode_label(col: str, mapper: dict[int, str]) -> str:
            vals = to_numeric(subset[col]).dropna().round().astype(int)
            if vals.empty:
                return "无"
            mode_code = int(vals.mode().iloc[0])
            return mapper.get(mode_code, str(mode_code))

        age_txt = mode_label(age_col, age_map)
        edu_txt = mode_label(edu_col, edu_map)
        income_txt = mode_label(income_col, income_map)
        occ_txt = mode_label(occ_col, occ_map)

        pref_scores = []
        for pc in pref_cols:
            v = to_numeric(pd.Series([row[pc]])).iloc[0]
            if pd.isna(v):
                continue
            pref_scores.append((pc, float(v)))
        pref_scores.sort(key=lambda x: x[1], reverse=True)
        tags = [guess_preference_label(x[0]) for x in pref_scores[:4]]

        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("white")

        card_box = FancyBboxPatch(
            (0.03, 0.03),
            0.94,
            0.94,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#D8DEE6",
            facecolor="#FFFFFF",
            alpha=1.0,
        )
        ax.add_patch(card_box)

        top_box = FancyBboxPatch(
            (0.06, 0.79),
            0.88,
            0.16,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0,
            facecolor=c,
            alpha=0.97,
        )
        ax.add_patch(top_box)
        ax.text(0.50, 0.90, f"第{cid}类", ha="center", va="center", fontsize=23, color="white", weight="bold")
        ax.text(0.50, 0.85, cname, ha="center", va="center", fontsize=18, color="white", weight="bold")
        ax.text(0.50, 0.81, f"占比 {share:.1f}%", ha="center", va="center", fontsize=14, color="white", weight="bold")

        ax.text(0.10, 0.72, f"主要年龄：{age_txt}", fontsize=12.5, color="#333333", weight="bold")
        ax.text(0.56, 0.72, f"主要学历：{edu_txt}", fontsize=12.5, color="#333333", weight="bold")
        ax.text(0.10, 0.66, f"主要收入：{income_txt}", fontsize=12.5, color="#333333", weight="bold")
        ax.text(0.56, 0.66, f"主要职业：{occ_txt}", fontsize=12.5, color="#333333", weight="bold")

        ax.text(0.10, 0.59, "高偏好项目", fontsize=13.5, color="#4F5965", weight="bold")
        for i, tag in enumerate(tags):
            x = 0.10 + (i % 2) * 0.43
            y = 0.52 - (i // 2) * 0.06
            tag_box = FancyBboxPatch(
                (x, y),
                0.36,
                0.047,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                linewidth=0.6,
                edgecolor=c,
                facecolor=c,
                alpha=0.20,
            )
            ax.add_patch(tag_box)
            ax.text(x + 0.18, y + 0.0235, tag, ha="center", va="center", fontsize=11.5, color=c, weight="bold")

        ax.text(0.10, 0.37, "画像概述", fontsize=13.5, color="#4F5965", weight="bold")
        ax.text(0.10, 0.325, fill(story.get(cid, ""), width=18), fontsize=11.5, color="#4F5965", va="top")

        ax.text(0.10, 0.205, "运营建议", fontsize=13.5, color="#4F5965", weight="bold")
        for j, line in enumerate(action.get(cid, [])):
            ax.text(0.10, 0.155 - 0.047 * j, f"- {line}", fontsize=11.5, color=c, weight="bold")

    for ax in axes_flat[card_count:]:
        ax.axis("off")

    fig.tight_layout(pad=0.9, w_pad=1.1, h_pad=1.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_preference_figure(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    sid_col = "聚类编号" if "聚类编号" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "编号"])
    sname_col = "聚类名称" if "聚类名称" in summary.columns else find_col(summary.columns.tolist(), ["聚类", "名称"])

    group_defs = {
        "讲解活动": [
            ("Q17_公益讲解兴趣均值", "公益讲解"),
            ("Q17_智慧导览兴趣均值", "智慧导览"),
            ("Q17_研学活动兴趣均值", "研学活动"),
        ],
        "文娱体验": [
            ("Q17_数字互动展区兴趣均值", "数字互动展区"),
            ("Q17_非遗展演兴趣均值", "非遗展演"),
            ("Q17_中医药体验兴趣均值", "中医药体验"),
            ("Q17_主题活动兴趣均值", "主题活动"),
            ("Q17_养生知识付费兴趣均值", "养生知识付费"),
        ],
        "街区规划": [
            ("Q18_商业布局期望均值", "商业布局"),
            ("Q18_主题线路期望均值", "主题线路"),
            ("Q18_交通接驳期望均值", "交通接驳"),
        ],
    }

    cluster_colors = {
        1: "#4FB87A",
        2: "#4C90BD",
        3: "#E38B3A",
        4: "#9762B8",
    }

    meta = summary.copy()
    meta[sid_col] = to_numeric(meta[sid_col]).astype(int)
    meta = meta.sort_values(sid_col)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    width = 0.18
    for ax, (gname, items) in zip(axes, group_defs.items()):
        valid_items = [(c, l) for c, l in items if c in meta.columns]
        x = np.arange(len(valid_items))
        for i, (_, row) in enumerate(meta.iterrows()):
            cid = int(row[sid_col])
            cname = str(row[sname_col])
            ys = [float(row[c]) for c, _ in valid_items]
            ax.bar(
                x + (i - 1.5) * width,
                ys,
                width=width,
                color=cluster_colors.get(cid, "#6C7A89"),
                edgecolor="white",
                label=f"第{cid}类（{cname}）",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in valid_items], rotation=15, ha="right", fontsize=10)
        ax.set_ylim(2.5, 5.0)
        ax.set_yticks([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        ax.axhline(4.0, color="#999999", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(gname, fontsize=16, pad=8, weight="bold")
        ax.set_ylabel("平均分（1-5分）", fontsize=11)
        ax.grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=10)
    fig.suptitle("各类受访者对街区体验项目偏好评分对比（基于表23）", fontsize=18, y=1.04, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = resolve_path(base_dir, args.metrics_file)
    raw_path = resolve_path(base_dir, args.raw_file)
    feature_path = resolve_path(base_dir, args.feature_file)
    summary_path = resolve_path(base_dir, args.summary_file)

    metrics = read_table(metrics_path)
    raw = read_table(raw_path)
    features = read_table(feature_path)
    summary = read_table(summary_path)

    out_quality = output_dir / "图54_聚类质量评估_二三四类对比.png"
    out_matrix = output_dir / "图55_叶开泰文化街区受访者聚类结果可视化.png"
    out_persona = output_dir / "图56_叶开泰文化街区四类受访者画像.png"
    out_pref = output_dir / "图_各类受访者对街区体验项目偏好评分对比.png"

    quality_stats = draw_quality_figure(metrics, out_quality, args.dpi)
    draw_cluster_matrix_figure(features, summary, out_matrix, args.dpi)
    draw_persona_figure(features, summary, out_persona, args.dpi)
    draw_preference_figure(summary, out_pref, args.dpi)

    cid_col = "Cluster_ID" if "Cluster_ID" in raw.columns else find_col(raw.columns.tolist(), ["Cluster", "ID"])
    cluster_counts = (
        to_numeric(raw[cid_col])
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    audit = {
        "任务": "根据 report (1).docx 生成聚类章节图片",
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "输入文件": {
            "聚类指标": str(metrics_path),
            "原始问卷": str(raw_path),
            "特征编码": str(feature_path),
            "聚类汇总": str(summary_path),
        },
        "样本量": int(len(raw)),
        "聚类样本量": {str(k): int(v) for k, v in cluster_counts.items()},
        "聚类质量指标": quality_stats,
        "输出文件": [
            str(out_quality),
            str(out_matrix),
            str(out_persona),
            str(out_pref),
        ],
    }
    audit_path = output_dir / "docx聚类图片生成_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "docx_cluster_figures_done:",
        f"out_quality={out_quality}",
        f"out_matrix={out_matrix}",
        f"out_persona={out_persona}",
        f"out_pref={out_pref}",
        f"audit={audit_path}",
    )


if __name__ == "__main__":
    main()
