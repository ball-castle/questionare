#!/usr/bin/env python3
"""Generate 'national-award style' booster artifacts for questionnaire analysis."""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


BASE = Path(".")
TABLES = BASE / "output" / "tables"
OUT = BASE / "output" / "tables"
OUT_TEXT = BASE / "output"


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def to_float(s: str):
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def logistic_fit_with_prob(x: np.ndarray, y: np.ndarray, feature_names: List[str]):
    mask = (~np.isnan(y)) & (~np.isnan(x).any(axis=1))
    x2 = x[mask]
    y2 = y[mask]
    if x2.shape[0] < 30 or len(np.unique(y2)) < 2:
        return None

    mu = x2.mean(axis=0)
    sd = x2.std(axis=0)
    sd[sd < 1e-10] = 1.0
    xs = (x2 - mu) / sd
    n, p = xs.shape
    X = np.column_stack([np.ones(n), xs])

    def nll(beta):
        z = np.clip(X @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        return -np.sum(y2 * np.log(pr + 1e-12) + (1 - y2) * np.log(1 - pr + 1e-12))

    def grad(beta):
        z = np.clip(X @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        return X.T @ (pr - y2)

    res = minimize(nll, np.zeros(p + 1), jac=grad, method="BFGS")
    beta = res.x
    cov = np.asarray(res.hess_inv)
    if cov.shape != (p + 1, p + 1):
        return None

    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    zval = beta / se
    pval = 2.0 * stats.norm.sf(np.abs(zval))

    z = np.clip(X @ beta, -30, 30)
    prob = 1.0 / (1.0 + np.exp(-z))
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y2).mean())

    p0 = y2.mean()
    llm = -nll(beta)
    lln = np.sum(y2 * np.log(p0 + 1e-12) + (1 - y2) * np.log(1 - p0 + 1e-12))
    pseudo_r2 = float(1.0 - llm / lln) if abs(lln) > 1e-12 else np.nan

    pos = prob[y2 == 1]
    neg = prob[y2 == 0]
    if len(pos) > 0 and len(neg) > 0:
        ranks = stats.rankdata(np.concatenate([pos, neg]))
        auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    else:
        auc = np.nan

    rows = []
    rows.append({"term": "Intercept", "coef": beta[0], "std_err": se[0], "z": zval[0], "p_value": pval[0], "odds_ratio": math.exp(beta[0])})
    for i, fn in enumerate(feature_names):
        rows.append(
            {
                "term": fn,
                "coef": beta[i + 1],
                "std_err": se[i + 1],
                "z": zval[i + 1],
                "p_value": pval[i + 1],
                "odds_ratio": math.exp(beta[i + 1]),
            }
        )

    return {
        "n": int(n),
        "events": int(y2.sum()),
        "accuracy": acc,
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "pseudo_r2": pseudo_r2,
        "rows": rows,
        "prob": prob,
        "y": y2,
        "sign": {feature_names[i]: float(np.sign(beta[i + 1])) for i in range(len(feature_names))},
    }


def confusion(y, prob, th):
    pred = (prob >= th).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / len(y) if len(y) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else np.nan
    return tp, fp, tn, fn, acc, prec, rec, f1


def compute_vif(x: np.ndarray, names: List[str]):
    out = []
    for i, nm in enumerate(names):
        y = x[:, i]
        X = np.delete(x, i, axis=1)
        X = np.column_stack([np.ones(X.shape[0]), X])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        vif = 1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 0.999999 else np.inf
        out.append({"feature": nm, "r2": r2, "vif": vif})
    return out


def silhouette_manual(x: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    d = cdist(x, x, metric="euclidean")
    vals = []
    for i in range(x.shape[0]):
        same = labels == labels[i]
        same[i] = False
        a = d[i, same].mean() if same.sum() else 0.0
        b = np.inf
        for u in uniq:
            if u == labels[i]:
                continue
            m = labels == u
            b = min(b, d[i, m].mean())
        den = max(a, b)
        vals.append((b - a) / den if den > 0 else 0.0)
    return float(np.mean(vals))


def ch_index(x: np.ndarray, labels: np.ndarray) -> float:
    n = x.shape[0]
    uniq = np.unique(labels)
    k = len(uniq)
    if k <= 1 or n <= k:
        return np.nan
    overall = x.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for u in uniq:
        pts = x[labels == u]
        c = pts.mean(axis=0)
        sw += np.sum((pts - c) ** 2)
        sb += pts.shape[0] * np.sum((c - overall) ** 2)
    return (sb / (k - 1)) / (sw / (n - k)) if sw > 1e-12 else np.nan


def db_index(x: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    k = len(uniq)
    if k <= 1:
        return np.nan
    cent = []
    scat = []
    for u in uniq:
        pts = x[labels == u]
        c = pts.mean(axis=0)
        cent.append(c)
        scat.append(np.mean(np.linalg.norm(pts - c, axis=1)))
    cent = np.array(cent)
    scat = np.array(scat)
    M = cdist(cent, cent)
    R = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if M[i, j] <= 1e-12:
                R[i, j] = np.inf
            else:
                R[i, j] = (scat[i] + scat[j]) / M[i, j]
    Di = np.max(R, axis=1)
    return float(np.mean(Di))


def run_cluster_for_k(x: np.ndarray, k: int):
    z = linkage(x, method="ward")
    h = fcluster(z, k, criterion="maxclust")
    cent = np.array([x[h == c].mean(axis=0) for c in sorted(np.unique(h))])
    try:
        _, lk = kmeans2(x, cent, minit="matrix", iter=80)
        labels = lk + 1
    except Exception:
        labels = h
    return labels


def parse_survey_clean():
    rows = read_csv(TABLES / "survey_clean.csv")
    data = {}
    for col in rows[0].keys():
        data[col] = np.array([to_float(r[col]) for r in rows], dtype=float)
    return data


def read_run_metadata():
    p = OUT_TEXT / "run_metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def make_pretest_statement(n_samples: int, quality_profile: str):
    if quality_profile == "balanced_v20260221":
        qc_line = "2) 主分析采用重筛平衡口径（balanced_v20260221），对高风险异常样本执行硬剔除；并保留敏感性分析说明稳健性边界。"
    else:
        qc_line = "2) 主分析保留旧平衡口径，不做大规模硬剔除；敏感性分析用于说明结论稳定性边界。"
    txt = """国奖补强口径声明（问卷部分）
================================
一、预调查口径
1) 预调查在本项目中定位为“方法预演+专家评审”，用于优化问卷结构、题项表述与逻辑跳转。
2) 由于当前工作目录未包含独立预调查原始数据及其统计输出，不进行预调查信效度数值伪造回填。
3) 正式报告中应将预调查统计位标注为“待补（需原始预调查数据）”，并单列数据需求清单。

二、正式调查口径
1) 本轮问卷分析以原始文件《原始数据_Amethyst.xlsx》有效样本{n_samples}为统一口径。
2) 关键模型与表格均基于同一口径输出，保证可追溯与复现。

三、质量控制说明
1) 注意力题采用“双口径并行”（应选1与应选5）做稳健性对比。
{qc_line}
"""
    (OUT_TEXT / "国奖补强_口径声明.txt").write_text(txt.format(n_samples=n_samples, qc_line=qc_line), encoding="utf-8")


def make_pretest_formal_diff(n_samples: int):
    rows = [
        {
            "对比维度": "研究目的",
            "预调查口径": "方法预演（题项可理解性、流程可执行性）",
            "正式调查口径": "统计推断与模型估计",
            "差异说明": "预调查重在修订，正式调查重在实证",
            "当前状态": "已落地",
        },
        {
            "对比维度": "样本来源",
            "预调查口径": "待补（缺少独立原始数据）",
            "正式调查口径": "原始数据_Amethyst.xlsx",
            "差异说明": "预调查数据暂缺，正式样本完整",
            "当前状态": "预调查待补",
        },
        {
            "对比维度": "样本量",
            "预调查口径": "待补",
            "正式调查口径": str(n_samples),
            "差异说明": "仅正式样本可复现",
            "当前状态": "预调查待补",
        },
        {
            "对比维度": "信度/效度",
            "预调查口径": "待补",
            "正式调查口径": "已输出（alpha/KMO/Bartlett）",
            "差异说明": "避免伪造预调查统计值",
            "当前状态": "预调查待补",
        },
        {
            "对比维度": "是否入模",
            "预调查口径": "不入模",
            "正式调查口径": "入模（MCA/Logit/聚类/IPA）",
            "差异说明": "符合国奖常见方法链",
            "当前状态": "已落地",
        },
    ]
    write_csv(OUT / "预调查-正式调查差异表.csv", list(rows[0].keys()), rows)


def make_sample_size_and_sampling_bias(data):
    # sample size table
    N = 13_000_000
    z = 1.96
    p = 0.5
    q = 1 - p
    d = 0.045
    deff = 1.5
    rr = 0.85
    n0 = (z ** 2) * p * q / (d ** 2)
    n_fpc = n0 / (1 + (n0 - 1) / N)
    n_deff = n_fpc * deff
    n_issue = n_deff / rr
    n_final = int(np.sum(~np.isnan(data.get("C001", np.array([])))))

    rows = [
        {"步骤": "参数设定", "公式": "N=1300万, z=1.96, p=0.5, d=0.045, Deff=1.5, rr=0.85", "结果": "", "说明": "保守口径，便于答辩说明"},
        {"步骤": "初始样本量", "公式": "n0=z^2*p*(1-p)/d^2", "结果": f"{n0:.3f}", "说明": "无限总体近似"},
        {"步骤": "有限总体修正", "公式": "n=n0/(1+(n0-1)/N)", "结果": f"{n_fpc:.3f}", "说明": "有限总体修正后"},
        {"步骤": "设计效应修正", "公式": "n_deff=n*Deff", "结果": f"{n_deff:.3f}", "说明": "考虑复杂抽样"},
        {"步骤": "发放量估计", "公式": "n_issue=n_deff/rr", "结果": f"{n_issue:.3f}", "说明": "按回收率倒推"},
        {"步骤": "实际有效样本", "公式": "n_final", "结果": f"{n_final}", "说明": "当前数据有效样本"},
    ]
    write_csv(OUT / "样本量计算表.csv", list(rows[0].keys()), rows)

    # sampling bias by visit strata
    q8 = data["C008"]
    target_total = 1000
    target_visit = 600
    target_unvisit = 400
    actual_visit = int(np.nansum(q8 == 1))
    actual_unvisit = int(np.nansum(q8 == 2))
    actual_total = actual_visit + actual_unvisit

    def row(name, tn, an):
        t_share = tn / target_total
        a_share = an / actual_total if actual_total else np.nan
        bias = a_share - t_share
        weight = t_share / a_share if a_share > 0 else np.nan
        return {
            "分层": name,
            "目标样本量": tn,
            "实际样本量": an,
            "目标占比": f"{t_share*100:.2f}%",
            "实际占比": f"{a_share*100:.2f}%",
            "占比偏差(百分点)": f"{bias*100:.2f}",
            "实际/目标": f"{(an/tn):.3f}" if tn > 0 else "",
            "建议校正权重": f"{weight:.3f}" if np.isfinite(weight) else "",
        }

    rows2 = [
        row("到访层", target_visit, actual_visit),
        row("未到访层", target_unvisit, actual_unvisit),
        {
            "分层": "合计",
            "目标样本量": target_total,
            "实际样本量": actual_total,
            "目标占比": "100.00%",
            "实际占比": "100.00%",
            "占比偏差(百分点)": "0.00",
            "实际/目标": f"{(actual_total/target_total):.3f}",
            "建议校正权重": "",
        },
    ]
    write_csv(OUT / "抽样分配与回收偏差表.csv", list(rows2[0].keys()), rows2)


def make_logit_extended(data):
    feature_names = [
        "Q2_age_code",
        "Q6_habit_code",
        "Q7_knowledge_code",
        "Q8_visit_status_code",
        "perception_mean",
        "performance_mean",
        "cognition_mean",
        "motive_count",
    ]
    X = np.column_stack(
        [
            data["C002"],
            data["C006"],
            data["C007"],
            data["C008"],
            data["perception_mean"],
            data["performance_mean"],
            data["cognition_mean"],
            data["motive_count"],
        ]
    )
    y_base = data["visit_depth_bin"]
    y_stay = np.where(data["C024"] >= 3, 1.0, 0.0)
    y_spend = np.where(data["C025"] >= 3, 1.0, 0.0)

    fit_base = logistic_fit_with_prob(X, y_base, feature_names)
    fit_stay = logistic_fit_with_prob(X, y_stay, feature_names)
    fit_spend = logistic_fit_with_prob(X, y_spend, feature_names)

    rows = []
    if fit_base is not None:
        # threshold sensitivity
        for th in [0.40, 0.50, 0.60]:
            tp, fp, tn, fn, acc, prec, rec, f1 = confusion(fit_base["y"], fit_base["prob"], th)
            rows.append(
                {
                    "模块": "阈值敏感性",
                    "模型": "base(Q11>=3 & Q12>=3)",
                    "阈值": th,
                    "n": fit_base["n"],
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": fit_base["auc"],
                    "pseudo_r2": fit_base["pseudo_r2"],
                    "说明": "基线模型阈值比较",
                }
            )

        # VIF rows
        mask = (~np.isnan(y_base)) & (~np.isnan(X).any(axis=1))
        vif_rows = compute_vif(X[mask], feature_names)
        for v in vif_rows:
            rows.append(
                {
                    "模块": "共线性检查",
                    "模型": "base(Q11>=3 & Q12>=3)",
                    "阈值": "",
                    "n": int(mask.sum()),
                    "TP": "",
                    "FP": "",
                    "TN": "",
                    "FN": "",
                    "accuracy": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "auc": "",
                    "pseudo_r2": "",
                    "说明": f"{v['feature']}: VIF={v['vif']:.3f}, R2={v['r2']:.3f}",
                }
            )

    def direction_consistency(base, other):
        if base is None or other is None:
            return np.nan, np.nan
        same, rev = 0, 0
        for f in feature_names:
            s1 = base["sign"].get(f, np.nan)
            s2 = other["sign"].get(f, np.nan)
            if np.isnan(s1) or np.isnan(s2) or s1 == 0 or s2 == 0:
                continue
            if s1 * s2 > 0:
                same += 1
            else:
                rev += 1
        return same, rev

    for name, fit in [("alt_stay(Q11>=3)", fit_stay), ("alt_spend(Q12>=3)", fit_spend)]:
        if fit is None:
            continue
        same, rev = direction_consistency(fit_base, fit)
        rows.append(
            {
                "模块": "替代因变量稳健性",
                "模型": name,
                "阈值": 0.50,
                "n": fit["n"],
                "TP": "",
                "FP": "",
                "TN": "",
                "FN": "",
                "accuracy": fit["accuracy"],
                "precision": "",
                "recall": "",
                "f1": "",
                "auc": fit["auc"],
                "pseudo_r2": fit["pseudo_r2"],
                "说明": f"与基线方向一致={same}，方向反转={rev}",
            }
        )

    write_csv(
        OUT / "Logit稳健性扩展表.csv",
        ["模块", "模型", "阈值", "n", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1", "auc", "pseudo_r2", "说明"],
        rows,
    )


def make_cluster_stability(data):
    # Same feature design as existing pipeline
    cols_binary = [f"C{i:03d}" for i in list(range(16, 24)) + list(range(92, 101)) + list(range(101, 108))]
    Xb = np.column_stack([data[c] for c in cols_binary])
    Xb = np.nan_to_num(Xb, nan=0.0)
    X = np.column_stack([Xb, data["importance_mean"], data["performance_mean"], data["cognition_mean"]])
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd

    rows = []
    recs = []
    for k in [2, 3, 4]:
        labels = run_cluster_for_k(Z, k)
        sil = silhouette_manual(Z, labels)
        ch = ch_index(Z, labels)
        db = db_index(Z, labels)
        counts = [int(np.sum(labels == c)) for c in sorted(np.unique(labels))]
        recs.append((k, sil, ch, db))
        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "cluster_size_distribution": "|".join(str(x) for x in counts),
                "稳定性说明": "",
            }
        )

    # best by silhouette (primary) + db secondary
    best_k = sorted(recs, key=lambda x: (-x[1], x[3]))[0][0]
    for r in rows:
        if r["k"] == best_k:
            r["稳定性说明"] = "优先方案（轮廓系数相对最优）"
        else:
            r["稳定性说明"] = "备选方案"

    write_csv(
        OUT / "聚类稳定性对比表.csv",
        ["k", "silhouette", "calinski_harabasz", "davies_bouldin", "cluster_size_distribution", "稳定性说明"],
        rows,
    )


def make_ipa_sensitivity(data):
    items = []
    for i in range(10):
        c_imp = f"C{66+i:03d}"
        c_perf = f"C{76+i:03d}"
        imp = float(np.nanmean(data[c_imp]))
        perf = float(np.nanmean(data[c_perf]))
        items.append((i + 1, imp, perf))

    imp_vals = np.array([x[1] for x in items], dtype=float)
    perf_vals = np.array([x[2] for x in items], dtype=float)

    methods = [
        ("均值阈值", float(np.mean(imp_vals)), float(np.mean(perf_vals))),
        ("中位数阈值", float(np.median(imp_vals)), float(np.median(perf_vals))),
    ]

    var_dict = read_csv(TABLES / "变量字典.csv")
    text_map = {}
    for row in var_dict:
        if row["col_idx"].isdigit():
            text_map[int(row["col_idx"])] = row["item_text"]

    rows = []
    for method, imp_th, perf_th in methods:
        for item_no, imp, perf in items:
            if imp >= imp_th and perf >= perf_th:
                quad = "Q1_优势区"
            elif imp < imp_th and perf >= perf_th:
                quad = "Q2_维持区"
            elif imp < imp_th and perf < perf_th:
                quad = "Q3_机会区"
            else:
                quad = "Q4_改进区"
            col_idx = 66 + item_no - 1
            rows.append(
                {
                    "阈值方法": method,
                    "item_no": item_no,
                    "item_text": text_map.get(col_idx, f"Q{col_idx}"),
                    "importance_mean": imp,
                    "performance_mean": perf,
                    "importance_threshold": imp_th,
                    "performance_threshold": perf_th,
                    "quadrant": quad,
                    "is_priority": 1 if quad == "Q4_改进区" else 0,
                }
            )
    write_csv(
        OUT / "IPA阈值敏感性表.csv",
        [
            "阈值方法",
            "item_no",
            "item_text",
            "importance_mean",
            "performance_mean",
            "importance_threshold",
            "performance_threshold",
            "quadrant",
            "is_priority",
        ],
        rows,
    )


def make_hypothesis_model_mapping():
    # Pull logit signs/p-values
    logit_rows = read_csv(TABLES / "Logit回归结果_主样本.csv")
    m = {r["term"]: r for r in logit_rows}

    def logit_evidence(term):
        if term not in m:
            return "无"
        r = m[term]
        return f"coef={float(r['coef']):.3f}, p={float(r['p_value']):.4f}, OR={float(r['odds_ratio']):.3f}"

    ipa_priority_n = 0
    ipa_path = TABLES / "IPA整改优先级表.csv"
    if ipa_path.exists():
        try:
            ipa_priority_n = len(read_csv(ipa_path))
        except Exception:
            ipa_priority_n = 0

    rows = [
        {
            "假设编号": "H1",
            "研究假设": "年龄结构会影响深入游览行为",
            "变量": "Q2_age_code",
            "模型": "二元Logit",
            "预期方向": "双向（不预设）",
            "证据": logit_evidence("Q2_age_code"),
            "结论": "需结合分组解释",
        },
        {
            "假设编号": "H2",
            "研究假设": "中医药消费习惯越强，深入游览概率越高",
            "变量": "Q6_habit_code",
            "模型": "二元Logit",
            "预期方向": "正向",
            "证据": logit_evidence("Q6_habit_code"),
            "结论": "以实证结果为准",
        },
        {
            "假设编号": "H3",
            "研究假设": "融合模式认知越高，深入游览概率越高",
            "变量": "Q7_knowledge_code/cognition_mean",
            "模型": "二元Logit + MCA",
            "预期方向": "正向",
            "证据": f"{logit_evidence('Q7_knowledge_code')}；{logit_evidence('cognition_mean')}",
            "结论": "认知变量需与其他因子联判",
        },
        {
            "假设编号": "H4",
            "研究假设": "到访状态与深入游览显著相关",
            "变量": "Q8_visit_status_code",
            "模型": "二元Logit + 交叉卡方",
            "预期方向": "正向",
            "证据": logit_evidence("Q8_visit_status_code"),
            "结论": "方向与显著性按结果解释",
        },
        {
            "假设编号": "H5",
            "研究假设": "整体感知评价越高，深入游览概率越高",
            "变量": "perception_mean",
            "模型": "二元Logit",
            "预期方向": "正向",
            "证据": logit_evidence("perception_mean"),
            "结论": "需与表现度区分讨论",
        },
        {
            "假设编号": "H6",
            "研究假设": "存在显著的游客细分画像",
            "变量": "motive_count/new_project_pref_count/promo_pref_count",
            "模型": "二阶聚类",
            "预期方向": "存在差异",
            "证据": "见聚类稳定性与画像卡",
            "结论": "成立（按当前K=2方案）",
        },
        {
            "假设编号": "H7",
            "研究假设": "高重要度低满意条目构成优先改进区",
            "变量": "Q66-75 vs Q76-85",
            "模型": "IPA",
            "预期方向": "存在Q4条目",
            "证据": "见IPA结果表与阈值敏感性表",
            "结论": f"成立（当前识别{ipa_priority_n}项优先改进）",
        },
    ]
    write_csv(
        OUT / "假设变量模型映射表.csv",
        ["假设编号", "研究假设", "变量", "模型", "预期方向", "证据", "结论"],
        rows,
    )


def make_action_matrix():
    src = read_csv(TABLES / "问题-证据-建议对照表.csv")
    rows = []
    for r in src:
        problem = r["problem"]
        action = r["suggestion"]
        owner = "街区运营中心"
        window = "0-3个月"
        kpi = "相关指标较基线提升>=10%"
        if "环境舒适度" in problem:
            owner = "运营保障部"
            window = "0-2个月"
            kpi = "IPA对应条目表现度提升>=0.15"
        elif "美食" in problem or "文创" in problem:
            owner = "产品与商户管理部"
            window = "1-3个月"
            kpi = "相关品类满意度提升>=0.12"
        elif "新增项目需求" in problem:
            owner = "活动策划部"
            window = "1-4个月"
            kpi = "新项目参与率>=30%"
        elif "优惠机制" in problem:
            owner = "品牌营销中心"
            window = "0-2个月"
            kpi = "优惠活动核销率>=20%"
        elif "深入游览关键影响因子" in problem:
            owner = "数据运营与用户增长组"
            window = "2-5个月"
            kpi = "深度游览转化率提升>=5%"

        rows.append(
            {
                "优先级": r["priority"],
                "问题": problem,
                "证据": r["evidence"],
                "动作": action,
                "责任方": owner,
                "时间窗": window,
                "KPI": kpi,
                "依赖条件": "预算审批+跨部门协同",
            }
        )
    write_csv(
        OUT / "建议落地行动矩阵.csv",
        ["优先级", "问题", "证据", "动作", "责任方", "时间窗", "KPI", "依赖条件"],
        rows,
    )


def main():
    data = parse_survey_clean()
    n_samples = int(np.sum(~np.isnan(data.get("C001", np.array([])))))
    meta = read_run_metadata()
    quality_profile = str(meta.get("quality_profile", "legacy_balanced"))

    make_pretest_statement(n_samples, quality_profile)
    make_pretest_formal_diff(n_samples)
    make_sample_size_and_sampling_bias(data)
    make_logit_extended(data)
    make_cluster_stability(data)
    make_ipa_sensitivity(data)
    make_hypothesis_model_mapping()
    make_action_matrix()
    print("Generated national-award booster artifacts.")


if __name__ == "__main__":
    main()
