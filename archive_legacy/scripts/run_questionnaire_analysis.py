#!/usr/bin/env python3
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qp_io import read_xlsx_first_sheet, numeric_matrix, write_dict_csv, write_rows_csv, fmt
from qp_stats import cronbach_alpha, kmo_bartlett, freq_table, crosstab, run_mca, logistic_fit, two_stage_cluster, assign_cluster_names

# Global plotting config for Chinese rendering.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

FIELD_CN_MAPPING = [
    {"english_field": "Q2_age_code", "chinese_label": "年龄段（Q2编码）", "business_meaning": "受访者年龄序位编码", "model_scope": "Logit"},
    {"english_field": "Q6_habit_code", "chinese_label": "中医药消费习惯（Q6编码）", "business_meaning": "是否有中医药相关消费习惯", "model_scope": "Logit"},
    {"english_field": "Q7_knowledge_code", "chinese_label": "融合模式了解程度（Q7编码）", "business_meaning": "对中医药+文旅了解程度", "model_scope": "Logit"},
    {"english_field": "Q8_visit_status_code", "chinese_label": "街区认知/到访状态（Q8编码）", "business_meaning": "是否听说或到访叶开泰街区", "model_scope": "Logit"},
    {"english_field": "perception_mean", "chinese_label": "感知评价均值", "business_meaning": "感知维度（52-63,65）平均得分", "model_scope": "Logit/聚类"},
    {"english_field": "importance_mean", "chinese_label": "重要度均值", "business_meaning": "重要度维度（66-75）平均得分", "model_scope": "IPA/聚类"},
    {"english_field": "performance_mean", "chinese_label": "表现度均值", "business_meaning": "表现维度（76-85）平均得分", "model_scope": "IPA/Logit/聚类"},
    {"english_field": "cognition_mean", "chinese_label": "文化认知均值", "business_meaning": "认知维度（86-89）平均得分", "model_scope": "Logit/聚类"},
    {"english_field": "motive_count", "chinese_label": "到访动机数量", "business_meaning": "到访动机多选项勾选总数（16-23）", "model_scope": "Logit/聚类"},
    {"english_field": "new_project_pref_count", "chinese_label": "新增项目偏好数量", "business_meaning": "新增项目偏好多选勾选总数（92-100）", "model_scope": "聚类"},
    {"english_field": "promo_pref_count", "chinese_label": "优惠活动偏好数量", "business_meaning": "优惠活动偏好多选勾选总数（101-107）", "model_scope": "聚类"},
]
TERM_CN_MAP = {r["english_field"]: r["chinese_label"] for r in FIELD_CN_MAPPING}


def block_name(c):
    if 1 <= c <= 8:
        return "基础特征与认知"
    if 9 <= c <= 15:
        return "知晓渠道（多选）"
    if 16 <= c <= 23:
        return "到访动机（多选）"
    if c == 24:
        return "停留时长"
    if c == 25:
        return "消费金额"
    if 26 <= c <= 32:
        return "体验/消费项目（多选）"
    if 33 <= c <= 42:
        return "到访问题感知（分支多选）"
    if 43 <= c <= 51:
        return "未到访原因（分支多选）"
    if 52 <= c <= 65:
        return "街区感知评价（Likert）"
    if 66 <= c <= 75:
        return "重要度评价（Likert）"
    if 76 <= c <= 85:
        return "满意度/表现评价（Likert）"
    if 86 <= c <= 89:
        return "文化认知与学习意愿（Likert）"
    if c == 90:
        return "到访意愿"
    if c == 91:
        return "推荐意愿"
    if 92 <= c <= 100:
        return "新增项目偏好（多选）"
    if 101 <= c <= 107:
        return "优惠活动偏好（多选）"
    return "开放题"


def item_type(c):
    if c in {1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 90, 91}:
        return "单选"
    if 52 <= c <= 89:
        return "Likert(1-5)"
    if c == 108:
        return "文本"
    return "多选/二元"


def model_use(c):
    u = []
    if c in {1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 90, 91}:
        u += ["描述统计", "交叉分析"]
    if 52 <= c <= 65:
        u += ["信效度", "Logit"]
    if 66 <= c <= 75:
        u += ["信效度", "IPA-重要度", "聚类"]
    if 76 <= c <= 85:
        u += ["信效度", "IPA-表现", "聚类"]
    if 86 <= c <= 89:
        u += ["信效度", "MCA", "Logit"]
    if 16 <= c <= 23 or 92 <= c <= 107:
        u += ["描述统计", "聚类"]
    if c == 64:
        u += ["质控-注意力题"]
    if c == 108:
        u += ["仅记录(本批次无有效文本)"]
    out = []
    for x in u:
        if x not in out:
            out.append(x)
    return "；".join(out) if out else "描述统计"


def missing_rule(c):
    if 33 <= c <= 42 or 43 <= c <= 51:
        return "-3为结构缺失（Q8分支不适用）"
    if c == 108:
        return "文本题；本批数据均为“无”"
    return "空值/异常编码记缺失"


def nanmean_cols(mat, cols_1b):
    return np.nanmean(mat[:, [c - 1 for c in cols_1b]], axis=1)


def nansum_cols(mat, cols_1b):
    return np.nansum(mat[:, [c - 1 for c in cols_1b]], axis=1)


def plot_core_profile(num, out):
    cs = [1, 2, 6, 8]
    ts = ["Q1 性别", "Q2 年龄段", "Q6 中医药消费习惯", "Q8 听说/到访状态"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c, t in zip(axes.flatten(), cs, ts):
        v = num[:, c - 1]
        cnt = Counter(int(x) for x in v[~np.isnan(v)])
        k = sorted(cnt)
        ax.bar([str(x) for x in k], [cnt[x] for x in k], color="#1f77b4")
        ax.set_title(t)
        ax.set_xlabel("编码")
        ax.set_ylabel("人数")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_mca(mca, out):
    c = mca["col"]
    lbs = mca["labels"]
    plt.figure(figsize=(10, 8))
    plt.axhline(0, c="gray", lw=0.8)
    plt.axvline(0, c="gray", lw=0.8)
    plt.scatter(c[:, 0], c[:, 1], s=18, c="#1f77b4")
    for i, t in enumerate(lbs):
        plt.text(c[i, 0], c[i, 1], t, fontsize=7)
    plt.title("MCA 类别关系图")
    plt.xlabel("维度1")
    plt.ylabel("维度2")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def plot_cluster(rows, out):
    ms = ["motive_count", "new_project_pref_count", "promo_pref_count", "importance_mean", "performance_mean", "cognition_mean"]
    ls = ["到访动机数", "新增项目偏好数", "优惠偏好数", "重要度均值", "表现度均值", "认知均值"]
    x = np.arange(len(ms))
    w = 0.8 / len(rows)
    plt.figure(figsize=(10, 6))
    for i, r in enumerate(rows):
        plt.bar(x + i * w, [r[m] for m in ms], width=w, label=f"C{r['cluster']}-{r['cluster_name']}")
    plt.xticks(x + w * (len(rows) - 1) / 2, ls)
    plt.legend(fontsize=8)
    plt.ylabel("均值")
    plt.title("游客聚类画像特征图")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def plot_ipa(ipa_rows, xm, ym, out):
    x = np.array([r["performance_mean"] for r in ipa_rows])
    y = np.array([r["importance_mean"] for r in ipa_rows])
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, c="#2ca02c")
    for i in range(len(ipa_rows)):
        plt.text(x[i], y[i], str(i + 1), fontsize=9)
    plt.axvline(xm, c="gray", ls="--")
    plt.axhline(ym, c="gray", ls="--")
    plt.title("IPA 重要度-表现度矩阵")
    plt.xlabel("表现度均值")
    plt.ylabel("重要度均值")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    inp = Path("原始数据_Amethyst.xlsx")
    out = Path("output")
    tdir = out / "tables"
    fdir = out / "figures"

    headers, rows_dense = read_xlsx_first_sheet(inp)
    n, m = len(rows_dense), len(headers)
    num_raw, markers = numeric_matrix(rows_dense)

    vdict = []
    for i, h in enumerate(headers, 1):
        vdict.append(
            {"col_idx": i, "var_code": f"C{i:03d}", "item_text": h, "question_block": block_name(i), "item_type": item_type(i), "model_use": model_use(i), "missing_rule": missing_rule(i)}
        )
    write_dict_csv(tdir / "变量字典.csv", list(vdict[0].keys()), vdict)

    raw_head = ["respondent_id"] + [f"C{i:03d}" for i in range(1, m + 1)]
    raw_rows = [[i + 1] + rows_dense[i] for i in range(n)]
    write_rows_csv(tdir / "survey_raw.csv", raw_head, raw_rows)

    if markers:
        write_dict_csv(tdir / "特殊编码文本标记.csv", ["respondent_id", "col_idx", "marker_text"], [{"respondent_id": a, "col_idx": b, "marker_text": c} for a, b, c in markers])

    q8 = num_raw[:, 7]
    attention_flag = np.where(np.isnan(num_raw[:, 63]), 1, (num_raw[:, 63] != 1).astype(int))
    v33_42, v43_51 = num_raw[:, 32:42], num_raw[:, 42:51]
    vb, nb = q8 == 1, q8 == 2
    logic_flag = (
        (vb & np.any(v33_42 == -3, axis=1))
        | (vb & np.any((v43_51 != -3) & (~np.isnan(v43_51)), axis=1))
        | (nb & np.any((v33_42 != -3) & (~np.isnan(v33_42)), axis=1))
        | (nb & np.any(v43_51 == -3, axis=1))
    ).astype(int)

    tup = [tuple(r) for r in rows_dense]
    cc, seen = Counter(tup), defaultdict(int)
    dup = np.zeros(n, dtype=int)
    for i, t in enumerate(tup):
        seen[t] += 1
        if cc[t] > 1 and seen[t] > 1:
            dup[i] = 1

    b1, b2 = num_raw[:, 51:63], num_raw[:, 65:85]
    s1 = np.array([int((not np.isnan(r).any()) and len(set(r.tolist())) == 1) for r in b1])
    s2 = np.array([int((not np.isnan(r).any()) and len(set(r.tolist())) == 1) for r in b2])
    straight = ((s1 == 1) & (s2 == 1)).astype(int)

    num = num_raw.copy()
    for c in range(32, 51):
        num[:, c] = np.where(num[:, c] == -3, np.nan, num[:, c])

    visit_depth = np.where((num[:, 23] >= 3) & (num[:, 24] >= 3), 1, 0).astype(float)
    perception = nanmean_cols(num, list(range(52, 64)) + [65])
    importance = nanmean_cols(num, list(range(66, 76)))
    performance = nanmean_cols(num, list(range(76, 86)))
    cognition = nanmean_cols(num, list(range(86, 90)))
    motive_cnt = nansum_cols(num, list(range(16, 24)))
    new_cnt = nansum_cols(num, list(range(92, 101)))
    pro_cnt = nansum_cols(num, list(range(101, 108)))

    chead = ["respondent_id"] + [f"C{i:03d}" for i in range(1, m + 1)] + [
        "attention_flag",
        "logic_flag",
        "duplicate_flag",
        "straightline_flag",
        "visit_depth_bin",
        "perception_mean",
        "importance_mean",
        "performance_mean",
        "cognition_mean",
        "motive_count",
        "new_project_pref_count",
        "promo_pref_count",
    ]
    crows = []
    for i in range(n):
        row = [i + 1] + [fmt(num[i, j]) for j in range(m)]
        row += [
            int(attention_flag[i]),
            int(logic_flag[i]),
            int(dup[i]),
            int(straight[i]),
            int(visit_depth[i]),
            fmt(perception[i]),
            fmt(importance[i]),
            fmt(performance[i]),
            fmt(cognition[i]),
            fmt(motive_cnt[i]),
            fmt(new_cnt[i]),
            fmt(pro_cnt[i]),
        ]
        crows.append(row)
    write_rows_csv(tdir / "survey_clean.csv", chead, crows)

    bad = []
    for i in range(n):
        if attention_flag[i] or logic_flag[i] or dup[i] or straight[i]:
            bad.append({"respondent_id": i + 1, "attention_flag": int(attention_flag[i]), "logic_flag": int(logic_flag[i]), "duplicate_flag": int(dup[i]), "straightline_flag": int(straight[i])})
    write_dict_csv(tdir / "异常样本清单.csv", ["respondent_id", "attention_flag", "logic_flag", "duplicate_flag", "straightline_flag"], bad)
    write_dict_csv(
        tdir / "样本流转表.csv",
        ["step", "n", "rule"],
        [
            {"step": "raw_input", "n": n, "rule": "原始样本"},
            {"step": "main_analysis", "n": n, "rule": "平衡口径：主分析不硬剔除，仅标记异常"},
            {"step": "sensitivity", "n": int(np.sum((attention_flag == 0) & (logic_flag == 0))), "rule": "敏感性：剔除注意力题异常与逻辑异常"},
        ],
    )
    (tdir / "清洗规则表.txt").write_text(
        textwrap.dedent(
            """
            问卷数据清洗规则（平衡口径）
            1) 原始读取：按xlsx单工作表读取，保留108列。
            2) 编码统一：含“^”的编码按“左侧数值编码+右侧文本标记”拆分。
            3) 结构缺失：33-51列中-3统一转为缺失NaN，不纳入均值与回归。
            4) 注意力题：C064按“应选编码=1”构建attention_flag，主分析仅标记。
            5) 逻辑校验：依据Q8分支对33-42与43-51列做一致性检查，生成logic_flag。
            6) 重复样本：全行完全重复且非首次出现者标记duplicate_flag=1。
            7) 直线作答：52-63与66-85同时同值者标记straightline_flag=1。
            8) 开放题：C108全为“无”，不进入文本挖掘与主题分析。
            """
        ).strip(),
        encoding="utf-8",
    )

    blocks = {
        "感知维度(52-63,65)": list(range(52, 64)) + [65],
        "重要度维度(66-75)": list(range(66, 76)),
        "表现维度(76-85)": list(range(76, 86)),
        "认知维度(86-89)": list(range(86, 90)),
        "综合量表(52-63,65,66-85,86-89)": list(range(52, 64)) + [65] + list(range(66, 86)) + list(range(86, 90)),
    }
    rel = []
    for k, cols in blocks.items():
        a, nn = cronbach_alpha(num[:, [c - 1 for c in cols]])
        rel.append({"block": k, "alpha": a, "n_complete": nn})
    write_dict_csv(tdir / "信度分析表.csv", ["block", "alpha", "n_complete"], rel)
    val = kmo_bartlett(num[:, [c - 1 for c in (list(range(52, 64)) + [65] + list(range(66, 86)) + list(range(86, 90)))]] )
    write_dict_csv(tdir / "效度分析表.csv", ["n_complete", "kmo", "bartlett_chi2", "bartlett_df", "bartlett_p"], [val])

    single = [1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 90, 91]
    fr = []
    for c in single:
        for r in freq_table(num[:, c - 1]):
            fr.append({"col_idx": c, "item_text": headers[c - 1], "code": r["code"], "count": r["count"], "pct": r["pct"]})
    write_dict_csv(tdir / "单选题频数百分比表.csv", ["col_idx", "item_text", "code", "count", "pct"], fr)

    mranges = [(9, 15), (16, 23), (26, 32), (33, 42), (43, 51), (92, 100), (101, 107)]
    mrows = []
    for a, b in mranges:
        for c in range(a, b + 1):
            v = num[:, c - 1]
            mk = ~np.isnan(v)
            if mk.sum() == 0:
                continue
            cnt = int(np.sum(v[mk] == 1))
            mrows.append({"col_idx": c, "item_text": headers[c - 1], "selected_count": cnt, "valid_n": int(mk.sum()), "selected_pct": 100.0 * cnt / int(mk.sum())})
    write_dict_csv(tdir / "多选题选择率表.csv", ["col_idx", "item_text", "selected_count", "valid_n", "selected_pct"], mrows)

    cpairs = [(8, 6), (8, 7), (8, 90), (8, 91), (8, 24), (8, 25)]
    csum, cdet = [], []
    for a, b in cpairs:
        z = crosstab(num[:, a - 1], num[:, b - 1])
        if z is None:
            continue
        csum.append({"var1": f"C{a:03d}", "var2": f"C{b:03d}", "n": z["n"], "chi2": z["chi2"], "dof": z["dof"], "p_value": z["p"]})
        for i, x in enumerate(z["xa"]):
            for j, y in enumerate(z["ya"]):
                cdet.append({"var1": f"C{a:03d}", "var2": f"C{b:03d}", "var1_code": x, "var2_code": y, "count": int(z["mat"][i, j])})
    write_dict_csv(tdir / "交叉分析卡方汇总.csv", ["var1", "var2", "n", "chi2", "dof", "p_value"], csum)
    write_dict_csv(tdir / "交叉分析列联明细.csv", ["var1", "var2", "var1_code", "var2_code", "count"], cdet)
    plot_core_profile(num, fdir / "核心画像图_core_profile.png")

    mca = run_mca(num, [1, 2, 3, 4, 5, 6, 7, 8, 90, 91])
    if mca is not None:
        mca_rows = []
        for i, lb in enumerate(mca["labels"]):
            mca_rows.append({"category": lb, "dim1": float(mca["col"][i, 0]), "dim2": float(mca["col"][i, 1]), "dim1_contrib": float(mca["contrib"][i, 0]), "dim2_contrib": float(mca["contrib"][i, 1])})
        write_dict_csv(tdir / "MCA类别坐标与贡献.csv", ["category", "dim1", "dim2", "dim1_contrib", "dim2_contrib"], mca_rows)
        write_dict_csv(tdir / "MCA特征值.csv", ["dimension", "eigenvalue"], [{"dimension": i + 1, "eigenvalue": float(v)} for i, v in enumerate(mca["eigen"][:5])])
        top1 = sorted(mca_rows, key=lambda r: abs(r["dim1"]), reverse=True)[:8]
        top2 = sorted(mca_rows, key=lambda r: abs(r["dim2"]), reverse=True)[:8]
        txt = ["MCA群体解释卡", "", "维度1绝对载荷Top8："] + [f"- {r['category']}: 维度1={r['dim1']:.4f}, 贡献={r['dim1_contrib']:.4f}" for r in top1] + ["", "维度2绝对载荷Top8："] + [f"- {r['category']}: 维度2={r['dim2']:.4f}, 贡献={r['dim2_contrib']:.4f}" for r in top2]
        (tdir / "MCA群体解释卡.txt").write_text("\n".join(txt), encoding="utf-8")
        plot_mca(mca, fdir / "MCA二维图.png")

    feats = np.column_stack([num[:, 1], num[:, 5], num[:, 6], num[:, 7], perception, performance, cognition, motive_cnt])
    fn = ["Q2_age_code", "Q6_habit_code", "Q7_knowledge_code", "Q8_visit_status_code", "perception_mean", "performance_mean", "cognition_mean", "motive_count"]
    write_dict_csv(
        tdir / "变量中英映射表.csv",
        ["english_field", "chinese_label", "business_meaning", "model_scope"],
        FIELD_CN_MAPPING,
    )

    lm = logistic_fit(feats, visit_depth, fn)
    metrics = []
    if lm is not None:
        write_dict_csv(tdir / "Logit回归结果_主样本.csv", ["term", "coef", "std_err", "z", "p_value", "odds_ratio", "marginal_effect"], lm["rows"])
        metrics.append({"sample": "main", "n": lm["n"], "events": lm["events"], "accuracy": lm["accuracy"], "auc": lm["auc"], "pseudo_r2": lm["pseudo_r2"]})
    sens = (attention_flag == 0) & (logic_flag == 0)
    ls = logistic_fit(feats[sens], visit_depth[sens], fn)
    if ls is not None:
        write_dict_csv(tdir / "Logit回归结果_敏感性样本.csv", ["term", "coef", "std_err", "z", "p_value", "odds_ratio", "marginal_effect"], ls["rows"])
        metrics.append({"sample": "sensitivity", "n": ls["n"], "events": ls["events"], "accuracy": ls["accuracy"], "auc": ls["auc"], "pseudo_r2": ls["pseudo_r2"]})
        if lm is not None:
            rb = [{"feature": f, "main_sign": lm["sign"].get(f, np.nan), "sensitivity_sign": ls["sign"].get(f, np.nan), "reversed": int(lm["sign"].get(f, 0) * ls["sign"].get(f, 0) < 0)} for f in fn]
            write_dict_csv(tdir / "Logit稳健性方向对比.csv", ["feature", "main_sign", "sensitivity_sign", "reversed"], rb)
    if metrics:
        write_dict_csv(tdir / "Logit模型指标.csv", ["sample", "n", "events", "accuracy", "auc", "pseudo_r2"], metrics)

    def compare_direction(main_fit, sub_fit):
        if main_fit is None or sub_fit is None:
            return {"consistent_direction_n": np.nan, "reversed_direction_n": np.nan, "unknown_direction_n": np.nan}
        same, rev, unk = 0, 0, 0
        for f in fn:
            s1 = main_fit["sign"].get(f, np.nan)
            s2 = sub_fit["sign"].get(f, np.nan)
            if np.isnan(s1) or np.isnan(s2) or s1 == 0 or s2 == 0:
                unk += 1
            elif s1 * s2 > 0:
                same += 1
            else:
                rev += 1
        return {"consistent_direction_n": same, "reversed_direction_n": rev, "unknown_direction_n": unk}

    attention_flag_code5 = np.where(np.isnan(num_raw[:, 63]), 1, (num_raw[:, 63] != 5).astype(int))
    sens5 = (attention_flag_code5 == 0) & (logic_flag == 0)
    ls5 = logistic_fit(feats[sens5], visit_depth[sens5], fn)

    dual_rows = []
    for name, exp_code, af, fit in [
        ("口径A_应选1", 1, attention_flag, ls),
        ("口径B_应选5", 5, attention_flag_code5, ls5),
    ]:
        base = {
            "calibration": name,
            "expected_code": exp_code,
            "attention_abnormal_n": int(np.sum(af == 1)),
            "sensitivity_n": int(np.sum((af == 0) & (logic_flag == 0))),
        }
        if fit is None:
            base.update({"model_n": np.nan, "events": np.nan, "accuracy": np.nan, "auc": np.nan, "pseudo_r2": np.nan})
            base.update({"consistent_direction_n": np.nan, "reversed_direction_n": np.nan, "unknown_direction_n": np.nan})
        else:
            base.update({"model_n": fit["n"], "events": fit["events"], "accuracy": fit["accuracy"], "auc": fit["auc"], "pseudo_r2": fit["pseudo_r2"]})
            base.update(compare_direction(lm, fit))
        dual_rows.append(base)
    write_dict_csv(
        tdir / "注意力题双口径对比.csv",
        [
            "calibration",
            "expected_code",
            "attention_abnormal_n",
            "sensitivity_n",
            "model_n",
            "events",
            "accuracy",
            "auc",
            "pseudo_r2",
            "consistent_direction_n",
            "reversed_direction_n",
            "unknown_direction_n",
        ],
        dual_rows,
    )

    xc = num[:, [c - 1 for c in (list(range(16, 24)) + list(range(92, 101)) + list(range(101, 108)))]]
    xc = np.nan_to_num(xc, nan=0.0)
    xc = np.column_stack([xc, importance, performance, cognition])
    xc = np.nan_to_num(xc, nan=np.nanmean(xc, axis=0))
    mu, sd = xc.mean(axis=0), xc.std(axis=0)
    sd[sd < 1e-10] = 1.0
    xz = (xc - mu) / sd
    best, cand = two_stage_cluster(xz, ks=(2, 3, 4), seed=42)
    labels = best["labels"]
    write_dict_csv(tdir / "二阶聚类候选K评估.csv", ["k", "silhouette"], [{"k": c["k"], "silhouette": c["silhouette"]} for c in cand])
    prof = []
    for c in sorted(np.unique(labels)):
        mk = labels == c
        prof.append({"cluster": int(c), "n": int(mk.sum()), "share_pct": 100.0 * mk.sum() / n, "motive_count": float(np.nanmean(motive_cnt[mk])), "new_project_pref_count": float(np.nanmean(new_cnt[mk])), "promo_pref_count": float(np.nanmean(pro_cnt[mk])), "importance_mean": float(np.nanmean(importance[mk])), "performance_mean": float(np.nanmean(performance[mk])), "cognition_mean": float(np.nanmean(cognition[mk]))})
    nm = assign_cluster_names(prof)
    for r in prof:
        r["cluster_name"] = nm.get(r["cluster"], f"类型{r['cluster']}")
    write_dict_csv(tdir / "二阶聚类画像卡.csv", ["cluster", "cluster_name", "n", "share_pct", "motive_count", "new_project_pref_count", "promo_pref_count", "importance_mean", "performance_mean", "cognition_mean"], prof)
    plot_cluster(prof, fdir / "二阶聚类画像图.png")

    mhead = ["respondent_id", "visit_group", "visit_depth_bin", "attention_flag", "logic_flag", "cluster_label", "perception_mean", "importance_mean", "performance_mean", "cognition_mean", "motive_count", "new_project_pref_count", "promo_pref_count"]
    model_rows = [[i + 1, fmt(q8[i]), int(visit_depth[i]), int(attention_flag[i]), int(logic_flag[i]), int(labels[i]), fmt(perception[i]), fmt(importance[i]), fmt(performance[i]), fmt(cognition[i]), fmt(motive_cnt[i]), fmt(new_cnt[i]), fmt(pro_cnt[i])] for i in range(n)]
    write_rows_csv(tdir / "survey_model.csv", mhead, model_rows)

    ipa = []
    oi, op = float(np.nanmean(importance)), float(np.nanmean(performance))
    for i in range(10):
        ci, cp = 66 + i, 76 + i
        im, pm = float(np.nanmean(num[:, ci - 1])), float(np.nanmean(num[:, cp - 1]))
        if im >= oi and pm < op:
            q = "Q2_优先改进"
        elif im >= oi and pm >= op:
            q = "Q1_保持优势"
        elif im < oi and pm < op:
            q = "Q3_低优先级"
        else:
            q = "Q4_可能过度投入"
        ipa.append({"item_no": i + 1, "importance_col": ci, "performance_col": cp, "item_text": headers[ci - 1], "importance_mean": im, "performance_mean": pm, "gap_perf_minus_imp": pm - im, "quadrant": q})
    write_dict_csv(tdir / "IPA结果表.csv", ["item_no", "importance_col", "performance_col", "item_text", "importance_mean", "performance_mean", "gap_perf_minus_imp", "quadrant"], ipa)
    ip = sorted([r for r in ipa if r["quadrant"] == "Q2_优先改进"], key=lambda x: x["gap_perf_minus_imp"])
    write_dict_csv(
        tdir / "IPA整改优先级表.csv",
        ["item_no", "importance_col", "performance_col", "item_text", "importance_mean", "performance_mean", "gap_perf_minus_imp", "quadrant"],
        ip,
    )
    plot_ipa(ipa, op, oi, fdir / "IPA象限图.png")

    top_new = sorted([r for r in mrows if 92 <= r["col_idx"] <= 100], key=lambda x: x["selected_pct"], reverse=True)[:3]
    top_pro = sorted([r for r in mrows if 101 <= r["col_idx"] <= 107], key=lambda x: x["selected_pct"], reverse=True)[:3]
    sug, rk = [], 1
    for r in ip[:3]:
        sug.append({"priority": rk, "problem": f"高重要度低表现：{r['item_text']}", "evidence": f"重要度均值{r['importance_mean']:.3f}，表现均值{r['performance_mean']:.3f}，差值{r['gap_perf_minus_imp']:.3f}", "suggestion": "纳入季度优先整改清单，配置专项预算和KPI跟踪。"})
        rk += 1
    for r in top_new:
        sug.append({"priority": rk, "problem": f"新增项目需求集中于：{r['item_text']}", "evidence": f"选择率{r['selected_pct']:.2f}%（{r['selected_count']}/{r['valid_n']}）", "suggestion": "优先开发该类中医药文旅产品，做小规模A/B试点后扩容。"})
        rk += 1
    for r in top_pro:
        sug.append({"priority": rk, "problem": f"优惠机制偏好集中于：{r['item_text']}", "evidence": f"选择率{r['selected_pct']:.2f}%（{r['selected_count']}/{r['valid_n']}）", "suggestion": "将高偏好优惠形式与核心产品绑定，提升转化和复游。"})
        rk += 1
    if lm is not None:
        sig = sorted([r for r in lm["rows"] if r["term"] != "Intercept" and r["p_value"] < 0.05], key=lambda x: abs(x["coef"]), reverse=True)[:3]
        for r in sig:
            d = "正向" if r["coef"] > 0 else "负向"
            term_cn = TERM_CN_MAP.get(r["term"], r["term"])
            sug.append({"priority": rk, "problem": f"深入游览关键影响因子：{term_cn}（{d}）", "evidence": f"coef={r['coef']:.3f}, OR={r['odds_ratio']:.3f}, p={r['p_value']:.4f}", "suggestion": "围绕该因子制定分层触达与产品优化策略，作为转化抓手。"})
            rk += 1
    write_dict_csv(tdir / "问题-证据-建议对照表.csv", ["priority", "problem", "evidence", "suggestion"], sug)

    sum_lines = [
        "叶开泰项目：问卷数据处理与分析结果摘要",
        "=" * 60,
        f"样本规模：{n}（原始=主分析）",
        f"异常标记：attention={int(attention_flag.sum())}，logic={int(logic_flag.sum())}，duplicate={int(dup.sum())}，straightline={int(straight.sum())}",
        "",
        "一、信效度",
    ]
    for r in rel:
        sum_lines.append(f"- {r['block']}：alpha={r['alpha']:.4f}（n={r['n_complete']}）")
    sum_lines.append(f"- KMO={val['kmo']:.4f}；Bartlett chi2={val['bartlett_chi2']:.2f}, p={val['bartlett_p']:.4g}")
    sum_lines += ["", "二、二元Logit"]
    if lm is None:
        sum_lines.append("- 主样本Logit未成功拟合。")
    else:
        sum_lines.append(f"- 主样本：n={lm['n']}, events={lm['events']}, accuracy={lm['accuracy']:.3f}, auc={lm['auc']:.3f}, pseudo_r2={lm['pseudo_r2']:.3f}")
    if ls is not None:
        sum_lines.append(f"- 敏感性样本：n={ls['n']}, accuracy={ls['accuracy']:.3f}, auc={ls['auc']:.3f}, pseudo_r2={ls['pseudo_r2']:.3f}")
    sum_lines += ["", "三、二阶聚类", f"- 最优簇数K={best['k']}，silhouette={best['silhouette']:.4f}"]
    for r in prof:
        sum_lines.append(
            f"  * C{r['cluster']} {r['cluster_name']}：占比{r['share_pct']:.2f}%，动机均值={r['motive_count']:.2f}，表现度均值={r['performance_mean']:.2f}，认知均值={r['cognition_mean']:.2f}"
        )
    sum_lines += ["", "四、IPA优先改进"]
    if ip:
        for r in ip[:6]:
            sum_lines.append(
                f"- [{r['item_no']}] {r['item_text']} | 重要度={r['importance_mean']:.3f}, 表现度={r['performance_mean']:.3f}, 差值={r['gap_perf_minus_imp']:.3f}"
            )
    else:
        sum_lines.append("- 当前无落入Q2（高重要度低表现）的条目。")
    sum_lines += ["", "五、注意力题双口径敏感性对比"]
    for row in dual_rows:
        sum_lines.append(
            f"- {row['calibration']}：异常样本={row['attention_abnormal_n']}，敏感性样本={row['sensitivity_n']}，accuracy={row['accuracy']}, auc={row['auc']}, pseudo_r2={row['pseudo_r2']}，方向反转变量数={row['reversed_direction_n']}"
        )
    (out / "问卷数据处理与分析_结果摘要.txt").write_text("\n".join(sum_lines), encoding="utf-8")

    outline = [
        "问卷数据处理与分析_执行大纲",
        "=" * 50,
        "本文件仅覆盖问卷数据处理与分析，不包含评论挖掘与其他章节。",
        "",
        "一、数据处理与质量控制",
        "1) 输入：原始数据_Amethyst.xlsx",
        "2) 处理：编码统一、结构缺失处理、异常标记（attention/logic/duplicate/straightline）",
        "3) 产出：survey_raw/survey_clean/survey_model、清洗规则、异常样本、样本流转",
        "",
        "二、统计分析模块",
        "1) 信效度：Cronbach alpha、KMO、Bartlett",
        "2) 描述统计与交叉：单选/多选分布、Q8关键交叉卡方",
        "3) MCA：游客特征-认知-行为偏好的二维结构",
        "4) 二元Logit：visit_depth_bin=1(Q11>=3且Q12>=3)",
        "5) 二阶聚类：2-4类游客画像及策略",
        "6) IPA：66-75(重要度) vs 76-85(表现) 的四象限优先级",
        "",
        "三、证据到建议闭环",
        "1) 问题-证据-建议对照表（可直接入报告结论章节）",
        "2) 每条建议绑定统计证据（模型参数/均值差/选择率）",
        "3) 注意力题双口径敏感性结果（应选1 vs 应选5）用于稳健性说明",
        "",
        "四、当前批次执行状态",
        f"1) 已执行样本量：{n}",
        f"2) attention_flag=1 数量：{int(attention_flag.sum())}",
        f"3) logic_flag=1 数量：{int(logic_flag.sum())}",
        f"4) 聚类最优K：{best['k']}（silhouette={best['silhouette']:.4f}）",
        f"5) IPA优先改进条目数：{len(ip)}",
        "",
        "五、成果目录",
        "- output/tables/*.csv",
        "- output/tables/变量中英映射表.csv",
        "- output/tables/注意力题双口径对比.csv",
        "- output/figures/*.png",
        "- output/问卷数据处理与分析_结果摘要.txt",
    ]
    Path("问卷数据处理与分析_执行大纲.txt").write_text("\n".join(outline), encoding="utf-8")

    meta = {
        "input": str(inp),
        "n_samples": n,
        "n_columns": m,
        "attention_flag_n": int(attention_flag.sum()),
        "logic_flag_n": int(logic_flag.sum()),
        "duplicate_flag_n": int(dup.sum()),
        "straightline_flag_n": int(straight.sum()),
        "cluster_best_k": int(best["k"]),
        "cluster_best_silhouette": float(best["silhouette"]),
        "ipa_priority_n": len(ip),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Analysis pipeline completed.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
