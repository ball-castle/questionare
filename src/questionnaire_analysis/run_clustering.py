#!/usr/bin/env python3
"""执行聚类分析并输出画像与评估结果。"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .qp_io import write_dict_csv, write_rows_csv


INPUT_CLEAN = Path("data") / "data_analysis" / "_source_analysis" / "tables" / "survey_clean.csv"
OUT_DIR = Path("data") / "clustering1"

SEED = 42
K_CANDIDATES = (2, 3, 4)
PREPROCESS_CANDIDATES = ("zscore", "robust")
FINAL_FEATURE_SET = "enhanced"
FINAL_PREPROCESS = "robust"
FINAL_K_MAIN = 2
FINAL_K_APPENDIX = 4
VISITED_CODE = 1
UNVISITED_CODE = 2


def _load_dependencies() -> None:
    global np, fcluster, linkage, kmeans2, linear_sum_assignment, cdist, chi2_contingency, kruskal

    import numpy as np_module
    from scipy.cluster.hierarchy import fcluster as fcluster_fn, linkage as linkage_fn
    from scipy.cluster.vq import kmeans2 as kmeans2_fn
    from scipy.optimize import linear_sum_assignment as linear_sum_assignment_fn
    from scipy.spatial.distance import cdist as cdist_fn
    from scipy.stats import chi2_contingency as chi2_contingency_fn, kruskal as kruskal_fn

    np = np_module
    fcluster = fcluster_fn
    linkage = linkage_fn
    kmeans2 = kmeans2_fn
    linear_sum_assignment = linear_sum_assignment_fn
    cdist = cdist_fn
    chi2_contingency = chi2_contingency_fn
    kruskal = kruskal_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering analysis.")
    parser.add_argument(
        "--input-csv",
        default=str(INPUT_CLEAN),
        help="Path to cleaned survey csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Directory for clustering outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for clustering routines.",
    )
    return parser.parse_args()


def to_float(v):
    s = str(v).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def read_numeric_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")
    data = {}
    for col in rows[0].keys():
        if col == "respondent_id":
            data[col] = np.array([int(float(r[col])) if str(r[col]).strip() else -1 for r in rows], dtype=int)
            continue
        vals = np.array([to_float(r[col]) for r in rows], dtype=float)
        if col.startswith("C"):
            vals[vals < 0] = np.nan
        data[col] = vals
    return data


def row_nanmean(parts):
    arr = np.column_stack(parts)
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=1)


def safe_col(data, col):
    if col not in data:
        return np.full(len(next(iter(data.values()))), np.nan, dtype=float)
    return data[col]


def impute_col_mean(x):
    z = x.copy()
    means = np.nanmean(z, axis=0)
    inds = np.where(np.isnan(z))
    z[inds] = np.take(means, inds[1])
    return z


def preprocess_matrix(x, method):
    z = x.copy()
    if method == "winsor_z":
        q1 = np.nanpercentile(z, 1, axis=0)
        q99 = np.nanpercentile(z, 99, axis=0)
        z = np.clip(z, q1, q99)
    elif method == "log1p_z":
        nonneg = np.nanmin(z, axis=0) >= 0
        z[:, nonneg] = np.log1p(z[:, nonneg])
    z = impute_col_mean(z)
    if method == "robust":
        med = np.nanmedian(z, axis=0)
        iqr = np.nanpercentile(z, 75, axis=0) - np.nanpercentile(z, 25, axis=0)
        iqr[iqr < 1e-10] = 1.0
        z = (z - med) / iqr
    else:
        mu = np.nanmean(z, axis=0)
        sd = np.nanstd(z, axis=0)
        sd[sd < 1e-10] = 1.0
        z = (z - mu) / sd
    return z


def silhouette_manual(x, labels):
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    if len(uniq) < 2:
        return np.nan
    d = cdist(x, x, metric="euclidean")
    vals = []
    for i in range(x.shape[0]):
        same = labs == labs[i]
        same[i] = False
        a = d[i, same].mean() if same.sum() else 0.0
        b = np.inf
        for c in uniq:
            if c == labs[i]:
                continue
            m = labs == c
            b = min(b, d[i, m].mean())
        den = max(a, b)
        vals.append((b - a) / den if den > 0 else 0.0)
    return float(np.mean(vals))


def ch_index(x, labels):
    n = x.shape[0]
    uniq = np.unique(labels)
    k = len(uniq)
    if k <= 1 or n <= k:
        return np.nan
    overall = x.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for c in uniq:
        pts = x[labels == c]
        cen = pts.mean(axis=0)
        sw += np.sum((pts - cen) ** 2)
        sb += pts.shape[0] * np.sum((cen - overall) ** 2)
    if sw <= 1e-12:
        return np.nan
    return float((sb / (k - 1)) / (sw / (n - k)))


def db_index(x, labels):
    uniq = np.unique(labels)
    k = len(uniq)
    if k <= 1:
        return np.nan
    cent = []
    scat = []
    for c in uniq:
        pts = x[labels == c]
        cen = pts.mean(axis=0)
        cent.append(cen)
        scat.append(np.mean(np.linalg.norm(pts - cen, axis=1)))
    cent = np.array(cent)
    scat = np.array(scat)
    d = cdist(cent, cent, metric="euclidean")
    r = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if d[i, j] <= 1e-12:
                r[i, j] = np.inf
            else:
                r[i, j] = (scat[i] + scat[j]) / d[i, j]
    return float(np.mean(np.max(r, axis=1)))


def run_cluster_for_k(x, k, seed=None):
    seed = SEED if seed is None else seed
    np.random.seed(seed)
    z = linkage(x, method="ward")
    h = fcluster(z, k, criterion="maxclust")
    cent = np.array([x[h == c].mean(axis=0) for c in sorted(np.unique(h))])
    try:
        _, lk = kmeans2(x, cent, minit="matrix", iter=80)
        return lk + 1
    except Exception:
        return h


def align_labels_to_ref(ref_labels, new_labels):
    ref_vals = np.array(sorted(np.unique(ref_labels)))
    new_vals = np.array(sorted(np.unique(new_labels)))
    cnt = np.zeros((len(ref_vals), len(new_vals)), dtype=int)
    rmap = {v: i for i, v in enumerate(ref_vals)}
    nmap = {v: i for i, v in enumerate(new_vals)}
    for r, n in zip(ref_labels, new_labels):
        cnt[rmap[r], nmap[n]] += 1
    rr, cc = linear_sum_assignment(-cnt)
    mapping = {}
    for i, j in zip(rr, cc):
        mapping[new_vals[j]] = ref_vals[i]
    aligned = np.array([mapping.get(v, v) for v in new_labels], dtype=int)
    return aligned


def assign_cluster_names(profile_rows):
    if not profile_rows:
        return {}
    ids = [r["cluster"] for r in profile_rows]
    cog = np.array([r["cognition_mean"] for r in profile_rows], dtype=float)
    mot = np.array([r["motive_count"] for r in profile_rows], dtype=float)
    prm = np.array([r["promo_pref_count"] for r in profile_rows], dtype=float)
    perf = np.array([r["performance_mean"] for r in profile_rows], dtype=float)
    out = {}
    out[ids[int(np.argmax(cog + mot))]] = "文化深度体验型"
    rem = [c for c in ids if c not in out]
    if rem:
        idx = [ids.index(c) for c in rem]
        out[rem[int(np.argmax(prm[idx]))]] = "价格优惠敏感型"
    rem = [c for c in ids if c not in out]
    if rem:
        idx = [ids.index(c) for c in rem]
        out[rem[int(np.argmax(perf[idx]))]] = "品质稳定复游型"
    for c in ids:
        out.setdefault(c, "轻度打卡观光型")
    return out


def build_feature_matrix(data, feature_set):
    bin_cols = [f"C{i:03d}" for i in list(range(16, 24)) + list(range(92, 101)) + list(range(101, 108))]
    xb = np.column_stack([safe_col(data, c) for c in bin_cols])
    xb = np.nan_to_num(xb, nan=0.0)

    imp = safe_col(data, "importance_mean")
    perf = safe_col(data, "performance_mean")
    cog = safe_col(data, "cognition_mean")

    if feature_set == "baseline":
        x = np.column_stack([xb, imp, perf, cog])
        names = bin_cols + ["importance_mean", "performance_mean", "cognition_mean"]
        return x, names

    likert_cols = [f"C{i:03d}" for i in list(range(66, 76)) + list(range(76, 86)) + list(range(86, 90))]
    xl = np.column_stack([safe_col(data, c) for c in likert_cols])

    behavior_cols = ["C024", "C025", "C090", "C091"]
    xbvr = np.column_stack([safe_col(data, c) for c in behavior_cols])

    c016 = safe_col(data, "C016")
    c017 = safe_col(data, "C017")
    c021 = safe_col(data, "C021")
    c026 = safe_col(data, "C026")
    c027 = safe_col(data, "C027")
    c031 = safe_col(data, "C031")
    c038 = safe_col(data, "C038")
    c057 = safe_col(data, "C057")
    c073 = safe_col(data, "C073")
    c083 = safe_col(data, "C083")
    c086 = safe_col(data, "C086")
    c089 = safe_col(data, "C089")
    c092 = safe_col(data, "C092")
    c094 = safe_col(data, "C094")
    c097 = safe_col(data, "C097")
    c098 = safe_col(data, "C098")
    c099 = safe_col(data, "C099")

    price_sensitivity = row_nanmean([c073, 6 - c083, 6 - c057, np.where(np.isnan(c038), np.nan, c038 * 5.0)])
    culture_interest = row_nanmean([c086, c089, c016 * 5.0, c017 * 5.0, c027 * 5.0])
    deep_experience = row_nanmean([c092 * 5.0, c094 * 5.0, c097 * 5.0, c098 * 5.0, c099 * 5.0, c021 * 5.0, c026 * 5.0, c031 * 5.0])

    x = np.column_stack(
        [
            xb,
            xl,
            xbvr,
            imp,
            perf,
            cog,
            safe_col(data, "motive_count"),
            safe_col(data, "new_project_pref_count"),
            safe_col(data, "promo_pref_count"),
            price_sensitivity,
            culture_interest,
            deep_experience,
        ]
    )
    names = (
        bin_cols
        + likert_cols
        + behavior_cols
        + [
            "importance_mean",
            "performance_mean",
            "cognition_mean",
            "motive_count",
            "new_project_pref_count",
            "promo_pref_count",
            "price_sensitivity_index",
            "culture_interest_index",
            "deep_experience_index",
        ]
    )
    return x, names


def epsilon_squared_kruskal(h_stat, n, k):
    if n <= k:
        return np.nan
    val = (h_stat - k + 1.0) / (n - k)
    return float(max(0.0, val))


def cramers_v(chi2, table):
    n = table.sum()
    if n <= 0:
        return np.nan
    r, c = table.shape
    denom = n * max(1, min(r - 1, c - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max(chi2, 0.0) / denom))


def describe_by_cluster(values, labels, mode):
    out = []
    for c in sorted(np.unique(labels)):
        v = values[labels == c]
        if mode == "binary":
            out.append(f"C{c}={np.nanmean(v == 1):.3f}")
        else:
            out.append(f"C{c}={np.nanmean(v):.3f}")
    return "|".join(out)


def _single_external_test(vals, ll, kind):
    uniq_l = np.unique(ll)
    if len(uniq_l) < 2:
        return None
    if kind == "binary":
        levels = sorted([int(x) for x in np.unique(vals) if np.isfinite(x)])
        if len(levels) < 2:
            return None
        mat = np.zeros((len(uniq_l), len(levels)), dtype=int)
        lmap = {v: i for i, v in enumerate(uniq_l)}
        vmap = {v: i for i, v in enumerate(levels)}
        for a, b in zip(ll, vals):
            bi = int(round(b))
            if bi not in vmap:
                continue
            mat[lmap[a], vmap[bi]] += 1
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            return None
        stat, p, dof, _ = chi2_contingency(mat, correction=False)
        effect = cramers_v(stat, mat)
        test_name = "chi2_cramers_v"
    else:
        groups = [vals[ll == c] for c in uniq_l]
        groups = [g[np.isfinite(g)] for g in groups]
        if any(len(g) < 5 for g in groups):
            return None
        stat, p = kruskal(*groups)
        dof = len(groups) - 1
        effect = epsilon_squared_kruskal(stat, int(np.isfinite(vals).sum()), len(groups))
        test_name = "kruskal_epsilon2"
    return test_name, float(stat), int(dof), float(p), float(effect) if np.isfinite(effect) else np.nan


def external_validity_rows_stratified(data, labels, feature_set, preprocess, k):
    common_vars = {"C090", "C091"}
    visited_only_vars = {"C036", "C037", "C038", "C040"}
    unvisited_only_vars = {"C044", "C047", "C048"}

    def include_in_score(layer_id, var):
        # Mutually exclusive scoring: each variable contributes once.
        if var in common_vars:
            return layer_id == "all"
        if var in visited_only_vars:
            return layer_id == "visited"
        if var in unvisited_only_vars:
            return layer_id == "unvisited"
        return False

    layer_cfg = [
        {
            "layer_id": "all",
            "layer_name": "全样本对照",
            "mask": np.isfinite(safe_col(data, "C008")),
            "vars": [
                ("C090", "到访意愿", "ordinal"),
                ("C091", "推荐意愿", "ordinal"),
            ],
        },
        {
            "layer_id": "visited",
            "layer_name": "到访层(C008=1)",
            "mask": safe_col(data, "C008") == VISITED_CODE,
            "vars": [
                ("C036", "到访问题-配套设施不完善", "binary"),
                ("C037", "到访问题-文化展示不够详细", "binary"),
                ("C038", "到访问题-价格偏高", "binary"),
                ("C040", "到访问题-指示标识不清晰", "binary"),
                ("C090", "到访意愿", "ordinal"),
                ("C091", "推荐意愿", "ordinal"),
            ],
        },
        {
            "layer_id": "unvisited",
            "layer_name": "未到访层(C008=2)",
            "mask": safe_col(data, "C008") == UNVISITED_CODE,
            "vars": [
                ("C044", "未到访原因-兴趣不大", "binary"),
                ("C047", "未到访原因-缺乏吸引力", "binary"),
                ("C048", "未到访原因-预算考虑", "binary"),
                ("C090", "到访意愿", "ordinal"),
                ("C091", "推荐意愿", "ordinal"),
            ],
        },
    ]

    detail_rows = []
    summary_rows = []
    all_effects = []
    score_effects = []
    total_sig_n = 0
    layer_sig_map = {}
    layer_scored_sig_map = {}

    for layer in layer_cfg:
        lm = layer["mask"]
        layer_effects = []
        layer_sig_n = 0
        layer_scored_sig_n = 0
        tested_n = 0
        for var, label, kind in layer["vars"]:
            vals0 = safe_col(data, var)
            m = lm & np.isfinite(vals0)
            if m.sum() < 30:
                continue
            vv = vals0[m]
            ll = labels[m]
            tested_n += 1
            res = _single_external_test(vv, ll, kind)
            if res is None:
                continue
            test_name, stat, dof, p, effect = res
            sig = int((p < 0.05) and np.isfinite(effect) and (effect >= 0.03))
            layer_sig_n += sig
            score_included = int(include_in_score(layer["layer_id"], var))
            if score_included:
                layer_scored_sig_n += sig
                if np.isfinite(effect):
                    score_effects.append(effect)
                total_sig_n += sig
            layer_effects.append(effect if np.isfinite(effect) else np.nan)
            all_effects.append(effect if np.isfinite(effect) else np.nan)
            detail_rows.append(
                {
                    "feature_set": feature_set,
                    "preprocess": preprocess,
                    "k": int(k),
                    "layer_id": layer["layer_id"],
                    "layer_name": layer["layer_name"],
                    "variable": var,
                    "variable_label": label,
                    "test": test_name,
                    "statistic": stat,
                    "df": dof,
                    "p_value": p,
                    "effect_size": effect,
                    "cluster_stats": describe_by_cluster(vv, ll, "binary" if kind == "binary" else "ordinal"),
                    "is_sig_with_effect": sig,
                    "score_included": score_included,
                }
            )
        layer_mean_effect = float(np.nanmean(np.array(layer_effects, dtype=float))) if layer_effects else np.nan
        layer_score = float(layer_sig_n + (0.0 if np.isnan(layer_mean_effect) else layer_mean_effect * 10.0))
        summary_rows.append(
            {
                "feature_set": feature_set,
                "preprocess": preprocess,
                "k": int(k),
                "layer_id": layer["layer_id"],
                "layer_name": layer["layer_name"],
                "layer_n": int(np.sum(lm)),
                "tested_vars_n": int(tested_n),
                "sig_vars_n": int(layer_sig_n),
                "scored_sig_vars_n": int(layer_scored_sig_n),
                "mean_effect": layer_mean_effect,
                "layer_score": layer_score,
            }
        )
        layer_sig_map[layer["layer_id"]] = layer_sig_n
        layer_scored_sig_map[layer["layer_id"]] = layer_scored_sig_n

    mean_effect = float(np.nanmean(np.array(score_effects, dtype=float))) if score_effects else np.nan
    all_mean_effect = float(np.nanmean(np.array(all_effects, dtype=float))) if all_effects else np.nan
    stratified_score = float(total_sig_n + (0.0 if np.isnan(mean_effect) else mean_effect * 10.0))
    summary = {
        "stratified_sig_n": int(total_sig_n),
        "stratified_mean_effect": mean_effect,
        "all_layers_mean_effect": all_mean_effect,
        "stratified_external_score": stratified_score,
        "visited_sig_n": int(layer_sig_map.get("visited", 0)),
        "unvisited_sig_n": int(layer_sig_map.get("unvisited", 0)),
        "all_sig_n": int(layer_sig_map.get("all", 0)),
        "visited_scored_sig_n": int(layer_scored_sig_map.get("visited", 0)),
        "unvisited_scored_sig_n": int(layer_scored_sig_map.get("unvisited", 0)),
        "all_scored_sig_n": int(layer_scored_sig_map.get("all", 0)),
    }
    return detail_rows, summary_rows, summary


def cluster_profiles(data, labels):
    rows = []
    n = len(labels)
    for c in sorted(np.unique(labels)):
        mk = labels == c
        rows.append(
            {
                "cluster": int(c),
                "n": int(mk.sum()),
                "share_pct": float(100.0 * mk.sum() / n),
                "motive_count": float(np.nanmean(safe_col(data, "motive_count")[mk])),
                "new_project_pref_count": float(np.nanmean(safe_col(data, "new_project_pref_count")[mk])),
                "promo_pref_count": float(np.nanmean(safe_col(data, "promo_pref_count")[mk])),
                "importance_mean": float(np.nanmean(safe_col(data, "importance_mean")[mk])),
                "performance_mean": float(np.nanmean(safe_col(data, "performance_mean")[mk])),
                "cognition_mean": float(np.nanmean(safe_col(data, "cognition_mean")[mk])),
            }
        )
    names = assign_cluster_names(rows)
    for r in rows:
        r["cluster_name"] = names.get(r["cluster"], f"类型{r['cluster']}")
    return rows


def choose_model(model_rows):
    if not model_rows:
        raise RuntimeError("no model candidates")
    main_model = None
    appendix_z = None
    appendix_r = None
    for r in model_rows:
        if r["feature_set"] == FINAL_FEATURE_SET and r["preprocess"] == FINAL_PREPROCESS and r["k"] == FINAL_K_MAIN:
            main_model = r
        if r["feature_set"] == FINAL_FEATURE_SET and r["preprocess"] == "zscore" and r["k"] == FINAL_K_APPENDIX:
            appendix_z = r
        if r["feature_set"] == FINAL_FEATURE_SET and r["preprocess"] == "robust" and r["k"] == FINAL_K_APPENDIX:
            appendix_r = r
    if main_model is None:
        raise RuntimeError("cannot find configured main model")
    if appendix_z is None:
        appendix_z = main_model
    if appendix_r is None:
        appendix_r = main_model
    appendix_preferred = sorted(
        [appendix_z, appendix_r],
        key=lambda r: (r["stratified_external_score"], r["silhouette"], -r["davies_bouldin"]),
        reverse=True,
    )[0]
    reason = (
        "按改进方案3收口：正文主口径固定为 enhanced + robust + K=2；"
        "附录并列展示 K=4+zscore（统计优先）与 K=4+robust（结构优先），并标注综合优先版本。"
    )
    return main_model, appendix_z, appendix_r, appendix_preferred, reason


def soft_segment(z, labels, profiles, respondent_ids):
    uniq = sorted(np.unique(labels))
    centers = np.array([z[labels == c].mean(axis=0) for c in uniq], dtype=float)
    dist = cdist(z, centers, metric="euclidean")
    score = np.exp(-dist)
    prob = score / np.sum(score, axis=1, keepdims=True)

    culture_cluster = sorted(profiles, key=lambda r: r["cognition_mean"] + r["motive_count"], reverse=True)[0]["cluster"]
    idx_map = {c: i for i, c in enumerate(uniq)}
    culture_affinity = prob[:, idx_map[culture_cluster]]
    # Use adaptive quantile split to avoid extreme group imbalance.
    q_low = float(np.quantile(culture_affinity, 0.30))
    q_high = float(np.quantile(culture_affinity, 0.70))
    sorted_prob = np.sort(prob, axis=1)
    certainty = sorted_prob[:, -1] - sorted_prob[:, -2] if prob.shape[1] >= 2 else np.ones(len(labels))

    rows = []
    for i in range(len(labels)):
        if culture_affinity[i] >= q_high:
            seg = "高亲和C2_文化深度型"
        elif culture_affinity[i] <= q_low:
            seg = "高亲和C1_价格权益型"
        else:
            seg = "灰区_待探索"
        rows.append(
            {
                "respondent_id": int(respondent_ids[i]),
                "hard_cluster": int(labels[i]),
                "hard_cluster_name": next(r["cluster_name"] for r in profiles if r["cluster"] == int(labels[i])),
                "culture_affinity_score": float(culture_affinity[i]),
                "segment_certainty": float(certainty[i]),
                "soft_segment": seg,
            }
        )

    summary = []
    total = len(rows)
    for seg in ["高亲和C1_价格权益型", "高亲和C2_文化深度型", "灰区_待探索"]:
        vals = [r["culture_affinity_score"] for r in rows if r["soft_segment"] == seg]
        cert_vals = [r["segment_certainty"] for r in rows if r["soft_segment"] == seg]
        n = len(vals)
        summary.append(
            {
                "soft_segment": seg,
                "n": n,
                "share_pct": float(100.0 * n / total),
                "mean_culture_affinity_score": float(np.mean(vals)) if vals else np.nan,
                "mean_segment_certainty": float(np.mean(cert_vals)) if cert_vals else np.nan,
            }
        )
    return rows, summary, int(culture_cluster), q_low, q_high


def mca_row_scores(data, cols):
    mat = np.column_stack([safe_col(data, c) for c in cols])
    valid = ~np.isnan(mat).any(axis=1)
    x = mat[valid]
    if len(x) < 30:
        return np.full(len(mat), np.nan), np.full(len(mat), np.nan), np.array([])
    levels = [sorted(set(int(v) for v in x[:, j])) for j in range(x.shape[1])]
    d = sum(len(c) for c in levels)
    z = np.zeros((x.shape[0], d), dtype=float)
    t = 0
    for j, cats in enumerate(levels):
        for c in cats:
            z[:, t] = (x[:, j] == c).astype(float)
            t += 1
    p = z / z.sum()
    r = p.sum(axis=1)
    c = p.sum(axis=0)
    s = (p - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    u, sg, _ = np.linalg.svd(s, full_matrices=False)
    eig = sg**2
    f = (u[:, :2] * sg[:2]) / np.sqrt(r[:, None])
    dim1 = np.full(len(mat), np.nan, dtype=float)
    dim2 = np.full(len(mat), np.nan, dtype=float)
    dim1[valid] = f[:, 0]
    dim2[valid] = f[:, 1]
    return dim1, dim2, eig


def safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def mca_quadrant_rows(data, respondent_ids, labels):
    cols = [f"C{i:03d}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 90, 91]]
    dim1, dim2, eig = mca_row_scores(data, cols)

    anchor1 = safe_col(data, "C007") + safe_col(data, "C006") + safe_col(data, "C008")
    anchor2 = safe_col(data, "C090") + safe_col(data, "C091")
    if np.isfinite(safe_corr(dim1, anchor1)) and safe_corr(dim1, anchor1) < 0:
        dim1 = -dim1
    if np.isfinite(safe_corr(dim2, anchor2)) and safe_corr(dim2, anchor2) < 0:
        dim2 = -dim2

    rows = []
    for i in range(len(respondent_ids)):
        if not (np.isfinite(dim1[i]) and np.isfinite(dim2[i])):
            quad = "NA"
        elif dim1[i] >= 0 and dim2[i] >= 0:
            quad = "高认知_高意愿"
        elif dim1[i] < 0 and dim2[i] >= 0:
            quad = "低认知_高意愿"
        elif dim1[i] < 0 and dim2[i] < 0:
            quad = "低认知_低意愿"
        else:
            quad = "高认知_低意愿"
        rows.append(
            {
                "respondent_id": int(respondent_ids[i]),
                "mca_dim1": float(dim1[i]) if np.isfinite(dim1[i]) else np.nan,
                "mca_dim2": float(dim2[i]) if np.isfinite(dim2[i]) else np.nan,
                "mca_quadrant": quad,
                "hard_cluster": int(labels[i]),
            }
        )
    summary = []
    total = sum(1 for r in rows if r["mca_quadrant"] != "NA")
    for q in ["低认知_低意愿", "低认知_高意愿", "高认知_低意愿", "高认知_高意愿"]:
        n = sum(1 for r in rows if r["mca_quadrant"] == q)
        summary.append({"mca_quadrant": q, "n": n, "share_pct": float(100.0 * n / total) if total else np.nan})
    return rows, summary, eig


def run_resample_stability(x_raw, labels_ref, preprocess):
    n = len(labels_ref)
    b = 120
    sample_n = max(50, int(round(0.7 * n)))
    rng = np.random.default_rng(SEED)
    run_rows = []
    hit = np.zeros(n, dtype=float)
    seen = np.zeros(n, dtype=float)
    k = len(np.unique(labels_ref))
    for i in range(b):
        idx = np.sort(rng.choice(n, size=sample_n, replace=False))
        z_sub = preprocess_matrix(x_raw[idx], preprocess)
        lb_sub = run_cluster_for_k(z_sub, k, seed=SEED + i + 1)
        lb_aligned = align_labels_to_ref(labels_ref[idx], lb_sub)
        cons = float(np.mean(lb_aligned == labels_ref[idx]))
        sil = float(silhouette_manual(z_sub, lb_sub))
        run_rows.append({"run_id": i + 1, "sample_n": int(sample_n), "consistency_rate": cons, "silhouette": sil})
        seen[idx] += 1.0
        hit[idx] += (lb_aligned == labels_ref[idx]).astype(float)
    person_rows = []
    for i in range(n):
        if seen[i] <= 0:
            continue
        person_rows.append({"row_index": i, "included_runs": int(seen[i]), "consistency_rate": float(hit[i] / seen[i])})
    summary = {
        "resample_runs": b,
        "sample_ratio": 0.70,
        "consistency_mean": float(np.mean([r["consistency_rate"] for r in run_rows])),
        "consistency_p10": float(np.percentile([r["consistency_rate"] for r in run_rows], 10)),
        "consistency_p50": float(np.percentile([r["consistency_rate"] for r in run_rows], 50)),
        "consistency_min": float(np.min([r["consistency_rate"] for r in run_rows])),
    }
    return run_rows, person_rows, summary


def preprocess_compare_rows(model_rows, model_states, feature_set, ks=(2, 4)):
    rows = []
    for k in ks:
        z_row = next((r for r in model_rows if r["feature_set"] == feature_set and r["preprocess"] == "zscore" and r["k"] == k), None)
        r_row = next((r for r in model_rows if r["feature_set"] == feature_set and r["preprocess"] == "robust" and r["k"] == k), None)
        if z_row is None or r_row is None:
            continue
        z_state = model_states[(feature_set, "zscore", k)]
        r_state = model_states[(feature_set, "robust", k)]
        r_aligned = align_labels_to_ref(z_state["labels"], r_state["labels"])
        cons = float(np.mean(r_aligned == z_state["labels"]))
        rows.append(
            {
                "feature_set": feature_set,
                "k": int(k),
                "zscore_silhouette": z_row["silhouette"],
                "robust_silhouette": r_row["silhouette"],
                "zscore_stratified_score": z_row["stratified_external_score"],
                "robust_stratified_score": r_row["stratified_external_score"],
                "zscore_visited_sig_n": z_row["visited_sig_n"],
                "robust_visited_sig_n": r_row["visited_sig_n"],
                "zscore_unvisited_sig_n": z_row["unvisited_sig_n"],
                "robust_unvisited_sig_n": r_row["unvisited_sig_n"],
                "cluster_consistency_robust_vs_zscore": cons,
                "changed_share_robust_vs_zscore": float(1.0 - cons),
                "preferred_preprocess": "robust"
                if (r_row["stratified_external_score"] >= z_row["stratified_external_score"] and r_row["silhouette"] >= z_row["silhouette"] - 0.02)
                else "zscore",
            }
        )
    return rows


def imbalance_action_rows(labels, soft_summary):
    rows = []
    n = len(labels)
    uniq = sorted(np.unique(labels))
    for c in uniq:
        cnt = int(np.sum(labels == c))
        share = float(cnt / n)
        if share >= 0.70:
            risk = "偏高"
            action = "该簇执行分层触达上限，避免资源过度集中"
        elif share <= 0.20:
            risk = "偏低"
            action = "该簇增加探索预算与素材覆盖，防止样本稀疏"
        else:
            risk = "正常"
            action = "按常规节奏执行"
        rows.append(
            {
                "type": "hard_cluster",
                "segment": f"C{int(c)}",
                "n": cnt,
                "share_pct": float(share * 100.0),
                "imbalance_risk": risk,
                "action_hint": action,
            }
        )
    for r in soft_summary:
        share = float(r["share_pct"] / 100.0)
        if share >= 0.70:
            risk = "偏高"
            action = "该软分层占比高，建议按渠道或人群再细分后投放"
        elif share <= 0.15:
            risk = "偏低"
            action = "该软分层样本偏少，建议滚动累积样本并延长实验周期"
        else:
            risk = "正常"
            action = "可直接用于AB分层投放"
        rows.append(
            {
                "type": "soft_segment",
                "segment": r["soft_segment"],
                "n": int(r["n"]),
                "share_pct": float(r["share_pct"]),
                "imbalance_risk": risk,
                "action_hint": action,
            }
        )
    return rows


def main():
    global INPUT_CLEAN, OUT_DIR, SEED

    args = parse_args()
    _load_dependencies()
    INPUT_CLEAN = Path(args.input_csv)
    OUT_DIR = Path(args.output_dir)
    SEED = int(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_all = read_numeric_csv(INPUT_CLEAN)
    ids_all = data_all["respondent_id"]

    valid = np.isfinite(data_all["invalid_union_flag"]) & (data_all["invalid_union_flag"] == 0)
    data = {k: v[valid] for k, v in data_all.items()}
    respondent_ids = ids_all[valid]

    model_rows = []
    model_states = {}
    external_rows_all = []
    external_summary_all = []

    for feature_set in ["baseline", "enhanced"]:
        x_raw, feature_names = build_feature_matrix(data, feature_set)
        for preprocess in PREPROCESS_CANDIDATES:
            z = preprocess_matrix(x_raw, preprocess)
            for k in K_CANDIDATES:
                labels = run_cluster_for_k(z, k, seed=SEED)
                sil = silhouette_manual(z, labels)
                ch = ch_index(z, labels)
                db = db_index(z, labels)
                ext_rows, ext_layer_rows, ext_summary = external_validity_rows_stratified(data, labels, feature_set, preprocess, k)
                external_rows_all.extend(ext_rows)
                external_summary_all.extend(ext_layer_rows)
                sizes = [int(np.sum(labels == c)) for c in sorted(np.unique(labels))]
                rec = {
                    "feature_set": feature_set,
                    "preprocess": preprocess,
                    "k": int(k),
                    "silhouette": float(sil),
                    "calinski_harabasz": float(ch),
                    "davies_bouldin": float(db),
                    "cluster_size_distribution": "|".join(str(x) for x in sizes),
                    "stratified_sig_n": int(ext_summary["stratified_sig_n"]),
                    "stratified_mean_effect": float(ext_summary["stratified_mean_effect"]),
                    "stratified_external_score": float(ext_summary["stratified_external_score"]),
                    "visited_sig_n": int(ext_summary["visited_sig_n"]),
                    "unvisited_sig_n": int(ext_summary["unvisited_sig_n"]),
                    "all_sig_n": int(ext_summary["all_sig_n"]),
                    "visited_scored_sig_n": int(ext_summary["visited_scored_sig_n"]),
                    "unvisited_scored_sig_n": int(ext_summary["unvisited_scored_sig_n"]),
                    "all_scored_sig_n": int(ext_summary["all_scored_sig_n"]),
                }
                model_rows.append(rec)
                model_states[(feature_set, preprocess, int(k))] = {
                    "x_raw": x_raw,
                    "z": z,
                    "labels": labels,
                    "feature_names": feature_names,
                }

    write_dict_csv(
        OUT_DIR / "聚类优化_模型对比.csv",
        [
            "feature_set",
            "preprocess",
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cluster_size_distribution",
            "stratified_sig_n",
            "stratified_mean_effect",
            "stratified_external_score",
            "visited_sig_n",
            "unvisited_sig_n",
            "all_sig_n",
            "visited_scored_sig_n",
            "unvisited_scored_sig_n",
            "all_scored_sig_n",
        ],
        model_rows,
    )
    write_dict_csv(
        OUT_DIR / "聚类优化_分层外部效度明细.csv",
        [
            "feature_set",
            "preprocess",
            "k",
            "layer_id",
            "layer_name",
            "variable",
            "variable_label",
            "test",
            "statistic",
            "df",
            "p_value",
            "effect_size",
            "cluster_stats",
            "is_sig_with_effect",
            "score_included",
        ],
        external_rows_all,
    )
    write_dict_csv(
        OUT_DIR / "聚类优化_分层外部效度汇总.csv",
        ["feature_set", "preprocess", "k", "layer_id", "layer_name", "layer_n", "tested_vars_n", "sig_vars_n", "scored_sig_vars_n", "mean_effect", "layer_score"],
        external_summary_all,
    )

    selected, appendix_k4_z, appendix_k4_r, appendix_k4_preferred, selection_reason = choose_model(model_rows)
    state = model_states[(selected["feature_set"], selected["preprocess"], selected["k"])]
    labels = state["labels"]
    x_raw = state["x_raw"]
    z = state["z"]

    profiles = cluster_profiles(data, labels)
    write_dict_csv(
        OUT_DIR / "聚类优化_画像卡.csv",
        [
            "cluster",
            "cluster_name",
            "n",
            "share_pct",
            "motive_count",
            "new_project_pref_count",
            "promo_pref_count",
            "importance_mean",
            "performance_mean",
            "cognition_mean",
        ],
        profiles,
    )

    # Appendix profiles for K=4 dual schemes.
    appendix_state_z = model_states[(appendix_k4_z["feature_set"], appendix_k4_z["preprocess"], appendix_k4_z["k"])]
    appendix_profiles_z = cluster_profiles(data, appendix_state_z["labels"])
    appendix_state_r = model_states[(appendix_k4_r["feature_set"], appendix_k4_r["preprocess"], appendix_k4_r["k"])]
    appendix_profiles_r = cluster_profiles(data, appendix_state_r["labels"])
    write_dict_csv(
        OUT_DIR / "聚类优化_画像卡_K4_zscore附录.csv",
        [
            "cluster",
            "cluster_name",
            "n",
            "share_pct",
            "motive_count",
            "new_project_pref_count",
            "promo_pref_count",
            "importance_mean",
            "performance_mean",
            "cognition_mean",
        ],
        appendix_profiles_z,
    )
    write_dict_csv(
        OUT_DIR / "聚类优化_画像卡_K4_robust附录.csv",
        [
            "cluster",
            "cluster_name",
            "n",
            "share_pct",
            "motive_count",
            "new_project_pref_count",
            "promo_pref_count",
            "importance_mean",
            "performance_mean",
            "cognition_mean",
        ],
        appendix_profiles_r,
    )

    k_robust_rows = [r for r in model_rows if r["feature_set"] == FINAL_FEATURE_SET and r["preprocess"] == FINAL_PREPROCESS]
    extra_z_k4 = next(
        (r for r in model_rows if r["feature_set"] == FINAL_FEATURE_SET and r["preprocess"] == "zscore" and r["k"] == FINAL_K_APPENDIX),
        None,
    )
    if extra_z_k4 is not None:
        k_robust_rows.append(extra_z_k4)
    k_robust_rows = sorted(k_robust_rows, key=lambda x: x["k"])
    for r in k_robust_rows:
        if r["k"] == selected["k"]:
            r["k_robust_note"] = "正文主口径"
        elif r["k"] == FINAL_K_APPENDIX and r["preprocess"] == "zscore":
            r["k_robust_note"] = "附录统计优先版（zscore）"
        elif r["k"] == FINAL_K_APPENDIX and r["preprocess"] == "robust":
            r["k_robust_note"] = "附录结构优先版（robust）"
        else:
            r["k_robust_note"] = "备选"
    write_dict_csv(
        OUT_DIR / "聚类优化_K鲁棒性.csv",
        [
            "feature_set",
            "preprocess",
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cluster_size_distribution",
            "stratified_sig_n",
            "stratified_mean_effect",
            "stratified_external_score",
            "visited_sig_n",
            "unvisited_sig_n",
            "all_sig_n",
            "visited_scored_sig_n",
            "unvisited_scored_sig_n",
            "all_scored_sig_n",
            "k_robust_note",
        ],
        k_robust_rows,
    )
    k4_dual_rows = [dict(appendix_k4_z), dict(appendix_k4_r)]
    for rr in k4_dual_rows:
        if rr.get("preprocess") == "zscore":
            rr["k_robust_note"] = "统计优先版"
        elif rr.get("preprocess") == "robust":
            rr["k_robust_note"] = "结构优先版"
        else:
            rr["k_robust_note"] = "附录版"
    write_dict_csv(
        OUT_DIR / "聚类优化_K4双方案对照.csv",
        [
            "feature_set",
            "preprocess",
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cluster_size_distribution",
            "stratified_sig_n",
            "stratified_mean_effect",
            "stratified_external_score",
            "visited_sig_n",
            "unvisited_sig_n",
            "all_sig_n",
            "visited_scored_sig_n",
            "unvisited_scored_sig_n",
            "all_scored_sig_n",
            "k_robust_note",
        ],
        k4_dual_rows,
    )

    # Robust vs zscore comparison under stratified external validity for final candidate Ks.
    preprocess_rows = preprocess_compare_rows(model_rows, model_states, FINAL_FEATURE_SET, ks=(FINAL_K_MAIN, FINAL_K_APPENDIX))
    write_dict_csv(
        OUT_DIR / "聚类优化_预处理对照_分层效度.csv",
        [
            "feature_set",
            "k",
            "zscore_silhouette",
            "robust_silhouette",
            "zscore_stratified_score",
            "robust_stratified_score",
            "zscore_visited_sig_n",
            "robust_visited_sig_n",
            "zscore_unvisited_sig_n",
            "robust_unvisited_sig_n",
            "cluster_consistency_robust_vs_zscore",
            "changed_share_robust_vs_zscore",
            "preferred_preprocess",
        ],
        preprocess_rows,
    )

    run_rows, person_rows, stable_summary = run_resample_stability(x_raw, labels, selected["preprocess"])
    write_dict_csv(OUT_DIR / "聚类优化_稳定性重抽样.csv", ["run_id", "sample_n", "consistency_rate", "silhouette"], run_rows)
    write_dict_csv(OUT_DIR / "聚类优化_稳定性个体一致性.csv", ["row_index", "included_runs", "consistency_rate"], person_rows)
    write_dict_csv(OUT_DIR / "聚类优化_稳定性汇总.csv", ["metric", "value"], [{"metric": k, "value": v} for k, v in stable_summary.items()])

    soft_rows, soft_summary, culture_cluster, soft_q_low, soft_q_high = soft_segment(z, labels, profiles, respondent_ids)
    write_dict_csv(
        OUT_DIR / "聚类优化_软分层个体得分.csv",
        ["respondent_id", "hard_cluster", "hard_cluster_name", "culture_affinity_score", "segment_certainty", "soft_segment"],
        soft_rows,
    )
    write_dict_csv(
        OUT_DIR / "聚类优化_软分层汇总.csv",
        ["soft_segment", "n", "share_pct", "mean_culture_affinity_score", "mean_segment_certainty"],
        soft_summary,
    )
    imbalance_rows = imbalance_action_rows(labels, soft_summary)
    write_dict_csv(
        OUT_DIR / "聚类优化_簇不均衡处置建议.csv",
        ["type", "segment", "n", "share_pct", "imbalance_risk", "action_hint"],
        imbalance_rows,
    )

    mca_rows, mca_summary, mca_eig = mca_quadrant_rows(data, respondent_ids, labels)
    write_dict_csv(OUT_DIR / "聚类优化_MCA四象限个体.csv", ["respondent_id", "mca_dim1", "mca_dim2", "mca_quadrant", "hard_cluster"], mca_rows)
    write_dict_csv(OUT_DIR / "聚类优化_MCA四象限汇总.csv", ["mca_quadrant", "n", "share_pct"], mca_summary)
    write_dict_csv(OUT_DIR / "聚类优化_MCA特征值.csv", ["dimension", "eigenvalue"], [{"dimension": i + 1, "eigenvalue": float(v)} for i, v in enumerate(mca_eig[:5])])

    exp_rows = [
        {
            "group": "高概率C1_价格权益型",
            "material_strategy": "权益型素材（折扣/套票/团购）",
            "landing_strategy": "优惠聚合页与快速核销入口",
            "primary_metric": "点击率->到访转化率",
            "secondary_metric": "客单价/复访意愿",
            "run_weeks": "1-4",
        },
        {
            "group": "高概率C2_文化深度型",
            "material_strategy": "内容型素材（非遗体验/深度导览）",
            "landing_strategy": "活动报名页与内容专题页",
            "primary_metric": "活动报名率->到访转化率",
            "secondary_metric": "推荐意愿/内容互动率",
            "run_weeks": "1-4",
        },
        {
            "group": "灰区_待探索",
            "material_strategy": "低成本AB探索（权益vs内容）",
            "landing_strategy": "双落地页并行",
            "primary_metric": "策略差异Lift",
            "secondary_metric": "后续归因转群率",
            "run_weeks": "1-4",
        },
    ]
    write_dict_csv(
        OUT_DIR / "聚类优化_实验设计模板.csv",
        ["group", "material_strategy", "landing_strategy", "primary_metric", "secondary_metric", "run_weeks"],
        exp_rows,
    )

    lines = []
    lines.append("聚类优化说明（按改进方案3执行）")
    lines.append("")
    lines.append(f"数据源: {INPUT_CLEAN.as_posix()}")
    lines.append(f"分析样本量: {int(len(labels))}（invalid_union_flag=0）")
    lines.append("")
    lines.append("一、模型选择")
    lines.append(
        f"- 正文主口径: feature_set={selected['feature_set']} | preprocess={selected['preprocess']} | "
        f"K={selected['k']} | silhouette={selected['silhouette']:.4f} | stratified_score={selected['stratified_external_score']:.4f}"
    )
    lines.append(
        f"- 附录统计优先版: feature_set={appendix_k4_z['feature_set']} | preprocess={appendix_k4_z['preprocess']} | "
        f"K={appendix_k4_z['k']} | silhouette={appendix_k4_z['silhouette']:.4f} | stratified_score={appendix_k4_z['stratified_external_score']:.4f}"
    )
    lines.append(
        f"- 附录结构优先版: feature_set={appendix_k4_r['feature_set']} | preprocess={appendix_k4_r['preprocess']} | "
        f"K={appendix_k4_r['k']} | silhouette={appendix_k4_r['silhouette']:.4f} | stratified_score={appendix_k4_r['stratified_external_score']:.4f}"
    )
    lines.append(
        f"- K=4 综合优先: preprocess={appendix_k4_preferred['preprocess']} "
        f"(stratified_score={appendix_k4_preferred['stratified_external_score']:.4f}, silhouette={appendix_k4_preferred['silhouette']:.4f})"
    )
    lines.append(f"- 选择理由: {selection_reason}")
    lines.append("")
    lines.append("二、分层外部效度（C008分层）")
    mean_eff = selected["stratified_mean_effect"]
    mean_eff_txt = f"{mean_eff:.4f}" if np.isfinite(mean_eff) else "nan"
    lines.append(
        f"- 主口径显著变量数(含效应量阈值): {selected['stratified_sig_n']}；平均效应量: {mean_eff_txt}；"
        f"Common(all)计分显著数={selected['all_scored_sig_n']}；到访层计分显著数={selected['visited_scored_sig_n']}；"
        f"未到访层计分显著数={selected['unvisited_scored_sig_n']}（互斥计分，避免重复）"
    )
    lines.append("- 明细见: 聚类优化_分层外部效度明细.csv")
    lines.append("- 汇总见: 聚类优化_分层外部效度汇总.csv")
    lines.append("")
    lines.append("三、预处理收口")
    lines.append("- robust vs zscore（K=2/K=4）见: 聚类优化_预处理对照_分层效度.csv")
    lines.append("- K=4双方案并列见: 聚类优化_K4双方案对照.csv")
    lines.append(f"- 软分层阈值采用分位数切分：q30={soft_q_low:.4f}, q70={soft_q_high:.4f}")
    lines.append("")
    lines.append("四、稳定性")
    lines.append(
        f"- 70%重抽样120次: 平均一致率={stable_summary['consistency_mean']:.4f}, "
        f"P10={stable_summary['consistency_p10']:.4f}, 最低={stable_summary['consistency_min']:.4f}"
    )
    lines.append("")
    lines.append("五、软分层与MCA")
    lines.append(f"- 文化深度型簇ID: C{culture_cluster}")
    lines.append("- 软分层输出: 聚类优化_软分层个体得分.csv / 聚类优化_软分层汇总.csv（亲和度分数，非校准概率）")
    lines.append("- 簇不均衡处置: 聚类优化_簇不均衡处置建议.csv")
    lines.append("- MCA四象限输出: 聚类优化_MCA四象限个体.csv / 聚类优化_MCA四象限汇总.csv")
    lines.append("")
    lines.append("六、实验化落地")
    lines.append("- 最小AB实验模板已生成: 聚类优化_实验设计模板.csv")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    (OUT_DIR / "聚类优化_模型说明.txt").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_clean": str(INPUT_CLEAN),
        "output_dir": str(OUT_DIR),
        "valid_n": int(len(labels)),
        "selected_model": selected,
        "appendix_k4_zscore_model": appendix_k4_z,
        "appendix_k4_robust_model": appendix_k4_r,
        "appendix_k4_preferred_model": appendix_k4_preferred,
        "soft_segment_quantile_thresholds": {"q30": soft_q_low, "q70": soft_q_high},
        "selection_reason": selection_reason,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    write_rows_csv(
        OUT_DIR / "聚类优化_关键结论.csv",
        ["item", "value"],
        [
            ["selected_feature_set", selected["feature_set"]],
            ["selected_preprocess", selected["preprocess"]],
            ["selected_k", selected["k"]],
            ["selected_silhouette", selected["silhouette"]],
            ["selected_stratified_external_score", selected["stratified_external_score"]],
            ["selected_visited_sig_n", selected["visited_sig_n"]],
            ["selected_unvisited_sig_n", selected["unvisited_sig_n"]],
            ["selected_all_scored_sig_n", selected["all_scored_sig_n"]],
            ["selected_visited_scored_sig_n", selected["visited_scored_sig_n"]],
            ["selected_unvisited_scored_sig_n", selected["unvisited_scored_sig_n"]],
            ["appendix_k4_zscore_silhouette", appendix_k4_z["silhouette"]],
            ["appendix_k4_zscore_stratified_external_score", appendix_k4_z["stratified_external_score"]],
            ["appendix_k4_robust_silhouette", appendix_k4_r["silhouette"]],
            ["appendix_k4_robust_stratified_external_score", appendix_k4_r["stratified_external_score"]],
            ["appendix_k4_preferred_preprocess", appendix_k4_preferred["preprocess"]],
            ["soft_q30", soft_q_low],
            ["soft_q70", soft_q_high],
            ["resample_consistency_mean", stable_summary["consistency_mean"]],
            ["resample_consistency_p10", stable_summary["consistency_p10"]],
        ],
    )


if __name__ == "__main__":
    main()

