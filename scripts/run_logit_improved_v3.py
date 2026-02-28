#!/usr/bin/env python3
"""Improved Logit workflow aligned with 改进方案3.

Main ideas:
1) Treat high-depth task as ranking/screening (Top-K and Recall-constrained threshold).
2) Cost-sensitive ElasticNet-Logit with PR-AUC-based tuning.
3) Trust-weighted training instead of aggressive sample deletion.
4) Upper-bound check with tree model (prefer XGBoost/LightGBM; fallback to sklearn HGB).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from qp_io import read_xlsx_first_sheet


def ccols(start: int, end: int) -> list[str]:
    return [f"C{i:03d}" for i in range(start, end + 1)]


DEMOGRAPHIC_COLS = ["C002", "C006", "C007"]
MOTIVE_COLS = ccols(16, 23)
PERCEPTION_COLS = ccols(52, 63) + ["C065"]
IMPORTANCE_COLS = ccols(66, 75)
PERFORMANCE_COLS = ccols(76, 85)
COGNITION_COLS = ccols(86, 89)
FEATURE_COLS_BASE = DEMOGRAPHIC_COLS + MOTIVE_COLS + PERCEPTION_COLS + IMPORTANCE_COLS + PERFORMANCE_COLS + COGNITION_COLS
TOP_KS = (5, 10, 20)


def maybe_import(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def get_header_map(source_xlsx: Path) -> dict[str, str]:
    headers, _ = read_xlsx_first_sheet(source_xlsx)
    return {f"C{i:03d}": headers[i - 1] for i in range(1, len(headers) + 1)}


def feature_group(feature: str) -> str:
    if feature.startswith("VISIT_X_"):
        return "到访×表现度交互"
    if feature in DEMOGRAPHIC_COLS:
        return "基础画像"
    if feature == "C008":
        return "到访状态"
    if not feature.startswith("C"):
        return "其他"
    try:
        idx = int(feature[1:])
    except Exception:
        return "其他"
    if 16 <= idx <= 23:
        return "动机哑变量"
    if 52 <= idx <= 65:
        return "感知条目"
    if 66 <= idx <= 75:
        return "重要度条目"
    if 76 <= idx <= 85:
        return "表现度条目"
    if 86 <= idx <= 89:
        return "文化认知条目"
    return "其他"


def compute_trust_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"respondent_id": df["respondent_id"].to_numpy()})
    out["visited_flag"] = (pd.to_numeric(df["C008"], errors="coerce") == 1).astype(int)

    for c in [
        "duration_lt90_flag",
        "severe_straightline_flag",
        "attention_conditional_flag",
        "logic_branch_flag",
        "key_missing_flag",
        "open_gibberish_flag",
        "invalid_union_flag",
    ]:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c in df.columns else 0

    visit_issue_cnt = df[ccols(33, 42)].apply(pd.to_numeric, errors="coerce").fillna(0.0).eq(1).sum(axis=1)
    not_visit_cnt = df[ccols(43, 51)].apply(pd.to_numeric, errors="coerce").fillna(0.0).eq(1).sum(axis=1)
    branch_conflict = np.where(
        out["visited_flag"].to_numpy() == 1,
        (not_visit_cnt.to_numpy() > 0).astype(int),
        (visit_issue_cnt.to_numpy() > 0).astype(int),
    )
    out["branch_conflict_flag"] = branch_conflict.astype(int)

    likert = df[PERCEPTION_COLS + IMPORTANCE_COLS + PERFORMANCE_COLS + COGNITION_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    unique_levels = np.array([len(np.unique(r[~np.isnan(r)])) for r in likert], dtype=int)

    max_run = []
    extreme_ratio = []
    for r in likert:
        vals = r[~np.isnan(r)]
        if len(vals) == 0:
            max_run.append(0)
            extreme_ratio.append(0.0)
            continue
        best = 0
        cur = 0
        prev = None
        for v in vals:
            if prev is None or v != prev:
                cur = 1
                prev = v
            else:
                cur += 1
            if cur > best:
                best = cur
        max_run.append(best)
        ext = np.mean((vals == 1) | (vals == 5))
        extreme_ratio.append(float(ext))

    out["unique_levels_52_89"] = unique_levels
    out["max_same_run_52_89"] = np.array(max_run, dtype=int)
    out["extreme_ratio_52_89"] = np.array(extreme_ratio, dtype=float)
    out["low_info_pattern_flag"] = (out["unique_levels_52_89"] <= 2).astype(int)
    out["long_run_pattern_flag"] = (out["max_same_run_52_89"] >= 14).astype(int)
    out["extreme_pattern_flag"] = ((out["extreme_ratio_52_89"] >= 0.8) & (out["unique_levels_52_89"] <= 3)).astype(int)

    penalty = (
        0.25 * out["invalid_union_flag"]
        + 0.20 * out["logic_branch_flag"]
        + 0.15 * out["attention_conditional_flag"]
        + 0.15 * out["duration_lt90_flag"]
        + 0.15 * out["severe_straightline_flag"]
        + 0.20 * out["branch_conflict_flag"]
        + 0.10 * out["open_gibberish_flag"]
        + 0.10 * out["key_missing_flag"]
        + 0.15 * out["low_info_pattern_flag"]
        + 0.10 * out["long_run_pattern_flag"]
        + 0.10 * out["extreme_pattern_flag"]
    ).to_numpy(dtype=float)
    trust = np.clip(1.0 - penalty, 0.2, 1.0)
    out["trust_score"] = trust
    out["high_trust_flag"] = (trust >= 0.9).astype(int)
    out["sample_weight_base"] = trust
    return out


def build_features(df: pd.DataFrame, include_visit_status: bool, include_interactions: bool) -> pd.DataFrame:
    cols = FEATURE_COLS_BASE.copy()
    if include_visit_status:
        cols.append("C008")
    feats = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    if include_interactions:
        visit = (pd.to_numeric(df["C008"], errors="coerce") == 1).astype(float)
        for c in PERFORMANCE_COLS:
            feats[f"VISIT_X_{c}"] = visit * pd.to_numeric(df[c], errors="coerce")
    return feats


def build_target_high(df: pd.DataFrame) -> np.ndarray:
    stay = pd.to_numeric(df["C024"], errors="coerce")
    spend = pd.to_numeric(df["C025"], errors="coerce")
    return ((stay >= 3) & (spend >= 3)).astype(int).to_numpy()


def make_splits(y: np.ndarray, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    use_splits = max(2, min(n_splits, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=use_splits, shuffle=True, random_state=random_state)
    return list(cv.split(np.zeros(len(y)), y))


def preprocess_fold(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    xtr = np.where(np.isnan(x_train), med, x_train)
    xte = np.where(np.isnan(x_test), med, x_test)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (xtr - mu) / sd, (xte - mu) / sd


def oof_prob_logit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
    pos_weight: float,
    l1_ratio: float,
    random_state: int,
) -> np.ndarray:
    p = np.zeros(len(y), dtype=float)
    for i, (tr, te) in enumerate(splits, start=1):
        xtr, xte = preprocess_fold(x[tr], x[te])
        ytr = y[tr]
        wtr = weights[tr]
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=float(c_value),
            class_weight={0: 1.0, 1: float(pos_weight)},
            max_iter=8000,
            random_state=random_state + i,
        )
        model.fit(xtr, ytr, sample_weight=wtr)
        p[te] = model.predict_proba(xte)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def tune_logit_by_pr_auc(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    c_grid: list[float],
    pos_weight_grid: list[float],
    l1_ratio: float,
    random_state: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best = None
    for c in c_grid:
        for pw in pos_weight_grid:
            p = oof_prob_logit(
                x=x,
                y=y,
                weights=weights,
                splits=splits,
                c_value=c,
                pos_weight=pw,
                l1_ratio=l1_ratio,
                random_state=random_state + 17,
            )
            pr_auc = float(average_precision_score(y, p))
            auc = float(roc_auc_score(y, p))
            brier = float(brier_score_loss(y, p))
            rows.append({"C": float(c), "pos_weight": float(pw), "pr_auc": pr_auc, "auc": auc, "brier": brier})
            if best is None or pr_auc > best["pr_auc"] or (abs(pr_auc - best["pr_auc"]) < 1e-12 and brier < best["brier"]):
                best = {"C": float(c), "pos_weight": float(pw), "pr_auc": pr_auc, "auc": auc, "brier": brier}
    assert best is not None
    return best["C"], best["pos_weight"], pd.DataFrame(rows).sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)


def fit_final_logit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    c_value: float,
    pos_weight: float,
    l1_ratio: float,
    random_state: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    xf = np.where(np.isnan(x), med, x)
    mu = xf.mean(axis=0)
    sd = xf.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xz = (xf - mu) / sd
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=float(c_value),
        class_weight={0: 1.0, 1: float(pos_weight)},
        max_iter=10000,
        random_state=random_state,
    )
    model.fit(xz, y, sample_weight=weights)
    return model, mu, sd


def fit_sigmoid_calibrator(p_raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    m = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    m.fit(p_raw.reshape(-1, 1), y)
    return m


def apply_sigmoid(m: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    return np.clip(m.predict_proba(p_raw.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def fit_iso_calibrator(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    m = IsotonicRegression(out_of_bounds="clip")
    m.fit(p_raw, y)
    return m


def apply_iso(m: IsotonicRegression, p_raw: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(m.transform(p_raw), dtype=float), 1e-6, 1 - 1e-6)


def crossfit_calibration_scores(
    p_raw: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    p_sig = np.zeros(len(y), dtype=float)
    p_iso = np.zeros(len(y), dtype=float)
    for tr, te in splits:
        sig = fit_sigmoid_calibrator(p_raw[tr], y[tr])
        iso = fit_iso_calibrator(p_raw[tr], y[tr])
        p_sig[te] = apply_sigmoid(sig, p_raw[te])
        p_iso[te] = apply_iso(iso, p_raw[te])
    return np.clip(p_sig, 1e-6, 1 - 1e-6), np.clip(p_iso, 1e-6, 1 - 1e-6)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    cal = calibration_table(y, p, bins=bins)
    n = cal["n"].to_numpy(dtype=float)
    if n.sum() == 0:
        return np.nan
    gap = np.abs(cal["predicted_mean"].to_numpy() - cal["observed_rate"].to_numpy())
    return float((gap * n).sum() / n.sum())


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    q = min(bins, int(d["p"].nunique()))
    if q < 2:
        d["bin"] = 0
    else:
        d["bin"] = pd.qcut(d["p"], q=q, labels=False, duplicates="drop")
    g = d.groupby("bin", dropna=False).agg(n=("y", "size"), predicted_mean=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    g["bin"] = g["bin"].astype(str)
    return g


def hosmer_lemeshow_p(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float, int]:
    cal = calibration_table(y, p, bins=bins)
    obs = cal["observed_rate"].to_numpy() * cal["n"].to_numpy()
    exp = cal["predicted_mean"].to_numpy() * cal["n"].to_numpy()
    n = cal["n"].to_numpy(dtype=float)
    eps = 1e-12
    stat = np.sum(((obs - exp) ** 2) / (exp + eps) + (((n - obs) - (n - exp)) ** 2) / ((n - exp) + eps))
    dof = max(int(len(cal) - 2), 1)
    pval = float(1.0 - chi2.cdf(stat, dof))
    return float(stat), pval, int(dof)


def topk_metrics(y: np.ndarray, score: np.ndarray, ks: tuple[int, ...] = TOP_KS) -> list[dict]:
    rows = []
    order = np.argsort(-score)
    n = len(y)
    base = float(y.mean()) if n else np.nan
    for k in ks:
        kk = max(1, int(np.ceil(n * (k / 100.0))))
        sel = order[:kk]
        hit = float(y[sel].mean()) if kk else np.nan
        rec = float(y[sel].sum() / max(y.sum(), 1))
        lift = float(hit / base) if base and base > 0 else np.nan
        rows.append({"top_k_pct": int(k), "top_n": int(kk), "precision_at_k": hit, "recall_at_k": rec, "lift_at_k": lift, "base_rate": base})
    return rows


def threshold_for_target_recall(y: np.ndarray, score: np.ndarray, target_recall: float) -> tuple[float, float, float]:
    order = np.argsort(-score)
    y_sorted = y[order]
    s_sorted = score[order]
    tp_cum = np.cumsum(y_sorted)
    pos = max(int(y.sum()), 1)
    recall_cum = tp_cum / pos
    idx = int(np.searchsorted(recall_cum, target_recall, side="left"))
    if idx >= len(s_sorted):
        idx = len(s_sorted) - 1
    thr = float(s_sorted[idx])
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    return thr, prec, rec


def best_fbeta_threshold(y: np.ndarray, score: np.ndarray, beta: float = 2.0) -> tuple[float, float, float, float]:
    beta2 = beta * beta
    best = None
    for thr in np.unique(score):
        pred = (score >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = float(tp / max(tp + fp, 1))
        r = float(tp / max(tp + fn, 1))
        den = beta2 * p + r
        fbeta = float(((1 + beta2) * p * r / den) if den > 0 else 0.0)
        if best is None or fbeta > best["fbeta"]:
            best = {"thr": float(thr), "precision": p, "recall": r, "fbeta": fbeta}
    assert best is not None
    return best["thr"], best["precision"], best["recall"], best["fbeta"]


def run_tree_upper_bound(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> tuple[np.ndarray, str]:
    xgb = maybe_import("xgboost")
    lgb = maybe_import("lightgbm")

    p = np.zeros(len(y), dtype=float)
    if xgb is not None:
        engine = "xgboost"
        for i, (tr, te) in enumerate(splits, start=1):
            xtr, xte = preprocess_fold(x[tr], x[te])
            ytr = y[tr]
            wtr = weights[tr] * np.where(ytr == 1, 3.0, 1.0)
            model = xgb.XGBClassifier(
                n_estimators=450,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=random_state + i,
                n_jobs=1,
            )
            model.fit(xtr, ytr, sample_weight=wtr)
            p[te] = model.predict_proba(xte)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6), engine

    if lgb is not None:
        engine = "lightgbm"
        for i, (tr, te) in enumerate(splits, start=1):
            xtr, xte = preprocess_fold(x[tr], x[te])
            ytr = y[tr]
            wtr = weights[tr] * np.where(ytr == 1, 3.0, 1.0)
            model = lgb.LGBMClassifier(
                n_estimators=450,
                learning_rate=0.04,
                max_depth=-1,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state + i,
            )
            model.fit(xtr, ytr, sample_weight=wtr)
            p[te] = model.predict_proba(xte)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6), engine

    engine = "sklearn_hgb_fallback"
    for i, (tr, te) in enumerate(splits, start=1):
        xtr, xte = preprocess_fold(x[tr], x[te])
        ytr = y[tr]
        wtr = weights[tr] * np.where(ytr == 1, 3.0, 1.0)
        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=20,
            random_state=random_state + i,
        )
        model.fit(xtr, ytr, sample_weight=wtr)
        p[te] = model.predict_proba(xte)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6), engine


def run_scope(
    df: pd.DataFrame,
    scope_name: str,
    mask: pd.Series,
    include_visit_status: bool,
    include_interactions: bool,
    c_grid: list[float],
    pos_weight_grid: list[float],
    l1_ratio: float,
    cv_folds: int,
    target_recall: float,
    random_state: int,
) -> dict:
    sub = df.loc[mask].copy().reset_index(drop=True)
    y = build_target_high(sub).astype(int)
    x_df = build_features(sub, include_visit_status=include_visit_status, include_interactions=include_interactions)
    x = x_df.to_numpy(dtype=float)
    weights = pd.to_numeric(sub["sample_weight_base"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    weights = np.clip(weights, 0.2, 2.0)

    splits = make_splits(y, n_splits=cv_folds, random_state=random_state + 7)
    best_c, best_pw, tuning_table = tune_logit_by_pr_auc(
        x=x,
        y=y,
        weights=weights,
        splits=splits,
        c_grid=c_grid,
        pos_weight_grid=pos_weight_grid,
        l1_ratio=l1_ratio,
        random_state=random_state + 11,
    )

    p_raw = oof_prob_logit(
        x=x,
        y=y,
        weights=weights,
        splits=splits,
        c_value=best_c,
        pos_weight=best_pw,
        l1_ratio=l1_ratio,
        random_state=random_state + 23,
    )

    cal_splits = make_splits(y, n_splits=max(3, min(5, cv_folds)), random_state=random_state + 51)
    p_sig, p_iso = crossfit_calibration_scores(p_raw=p_raw, y=y, splits=cal_splits)

    brier_raw = float(brier_score_loss(y, p_raw))
    brier_sig = float(brier_score_loss(y, p_sig))
    brier_iso = float(brier_score_loss(y, p_iso))
    auc_raw = float(roc_auc_score(y, p_raw))
    auc_sig = float(roc_auc_score(y, p_sig))
    auc_iso = float(roc_auc_score(y, p_iso))
    if brier_sig <= brier_iso:
        cal_method = "sigmoid"
        p_cal = p_sig
        brier_cal = brier_sig
        auc_cal = auc_sig
    else:
        cal_method = "isotonic"
        p_cal = p_iso
        brier_cal = brier_iso
        auc_cal = auc_iso
    if auc_cal + 0.001 < auc_raw and cal_method == "isotonic":
        cal_method = "sigmoid"
        p_cal = p_sig
        brier_cal = brier_sig
        auc_cal = auc_sig

    pr_auc_raw = float(average_precision_score(y, p_raw))
    pr_auc_cal = float(average_precision_score(y, p_cal))
    hl_raw_stat, hl_raw_p, hl_raw_dof = hosmer_lemeshow_p(y, p_raw, bins=10)
    hl_cal_stat, hl_cal_p, hl_cal_dof = hosmer_lemeshow_p(y, p_cal, bins=10)
    ece_raw = ece_score(y, p_raw, bins=10)
    ece_cal = ece_score(y, p_cal, bins=10)

    topk_raw = topk_metrics(y, p_raw, ks=TOP_KS)
    topk_cal = topk_metrics(y, p_cal, ks=TOP_KS)
    thr_target, p_target, r_target = threshold_for_target_recall(y, p_cal, target_recall=target_recall)
    f2_thr, f2_p, f2_r, f2 = best_fbeta_threshold(y, p_cal, beta=2.0)

    model_final, mu, sd = fit_final_logit(
        x=x,
        y=y,
        weights=weights,
        c_value=best_c,
        pos_weight=best_pw,
        l1_ratio=l1_ratio,
        random_state=random_state + 31,
    )
    coef_rows = [
        {
            "sample_scope": scope_name,
            "feature": "Intercept",
            "coef": float(model_final.intercept_[0]),
            "odds_ratio": float(np.exp(model_final.intercept_[0])),
            "abs_coef": float(abs(model_final.intercept_[0])),
        }
    ]
    for f, c in zip(x_df.columns.tolist(), model_final.coef_.ravel()):
        coef_rows.append(
            {
                "sample_scope": scope_name,
                "feature": f,
                "coef": float(c),
                "odds_ratio": float(np.exp(c)),
                "abs_coef": float(abs(c)),
            }
        )

    cal_rows = []
    for score_name, score in [("raw", p_raw), ("calibrated", p_cal)]:
        ctab = calibration_table(y, score, bins=10)
        ctab["sample_scope"] = scope_name
        ctab["score_type"] = score_name
        cal_rows.append(ctab)

    topk_rows = []
    for score_name, rows in [("raw", topk_raw), ("calibrated", topk_cal)]:
        for r in rows:
            rec = {"sample_scope": scope_name, "score_type": score_name}
            rec.update(r)
            topk_rows.append(rec)

    tuning_table = tuning_table.copy()
    tuning_table["sample_scope"] = scope_name
    tuning_table = tuning_table[["sample_scope", "C", "pos_weight", "pr_auc", "auc", "brier"]]

    oof_pred = pd.DataFrame(
        {
            "respondent_id": sub["respondent_id"].to_numpy(),
            "sample_scope": scope_name,
            "y_true_high": y.astype(int),
            "prob_raw": p_raw,
            "prob_calibrated": p_cal,
            "threshold_target_recall": float(thr_target),
            "pred_target_recall": (p_cal >= thr_target).astype(int),
            "threshold_f2": float(f2_thr),
            "pred_f2": (p_cal >= f2_thr).astype(int),
            "trust_score": sub["trust_score"].to_numpy(),
            "sample_weight_base": sub["sample_weight_base"].to_numpy(),
        }
    )

    upper_prob, upper_engine = run_tree_upper_bound(
        x=x,
        y=y,
        weights=weights,
        splits=splits,
        random_state=random_state + 41,
    )
    upper_row = {
        "sample_scope": scope_name,
        "model_family": "tree_upper_bound",
        "engine": upper_engine,
        "auc": float(roc_auc_score(y, upper_prob)),
        "pr_auc": float(average_precision_score(y, upper_prob)),
        "brier": float(brier_score_loss(y, upper_prob)),
    }
    ref_row = {
        "sample_scope": scope_name,
        "model_family": "cost_sensitive_logit",
        "engine": "elasticnet_logit",
        "auc": auc_cal,
        "pr_auc": pr_auc_cal,
        "brier": brier_cal,
    }

    metric_row = {
        "sample_scope": scope_name,
        "n": int(len(sub)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "best_c": float(best_c),
        "best_pos_weight": float(best_pw),
        "calibration_method": cal_method,
        "auc_raw": auc_raw,
        "pr_auc_raw": pr_auc_raw,
        "brier_raw": brier_raw,
        "ece_raw": ece_raw,
        "hl_raw_stat": hl_raw_stat,
        "hl_raw_dof": hl_raw_dof,
        "hl_raw_p_value": hl_raw_p,
        "auc_calibrated": auc_cal,
        "pr_auc_calibrated": pr_auc_cal,
        "brier_calibrated": brier_cal,
        "ece_calibrated": ece_cal,
        "hl_cal_stat": hl_cal_stat,
        "hl_cal_dof": hl_cal_dof,
        "hl_cal_p_value": hl_cal_p,
        "target_recall": float(target_recall),
        "threshold_target_recall": float(thr_target),
        "precision_at_target_recall": float(p_target),
        "recall_at_target_recall": float(r_target),
        "threshold_f2": float(f2_thr),
        "precision_at_f2": float(f2_p),
        "recall_at_f2": float(f2_r),
        "f2": float(f2),
    }

    return {
        "metric_row": metric_row,
        "topk_rows": topk_rows,
        "cal_rows": pd.concat(cal_rows, ignore_index=True),
        "coef_rows": coef_rows,
        "oof_pred": oof_pred,
        "tuning_rows": tuning_table,
        "upper_rows": [ref_row, upper_row],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved Logit v3 and export to data/data_logit2.")
    parser.add_argument("--input-csv", default="data/data_analysis/_source_analysis/tables/survey_clean.csv", help="Model-ready sample file.")
    parser.add_argument("--source-xlsx", default="data/叶开泰问卷数据.xlsx", help="Raw xlsx for header mapping.")
    parser.add_argument("--output-dir", default="data/data_logit2", help="Output directory.")
    parser.add_argument("--cv-folds", type=int, default=5, help="OOF CV folds.")
    parser.add_argument("--target-recall", type=float, default=0.30, help="Recall target for threshold selection.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--l1-ratio", type=float, default=0.2, help="ElasticNet l1_ratio.")
    parser.add_argument("--c-grid", default="0.05,0.1,0.2,0.5,1,2", help="Comma-separated C grid.")
    parser.add_argument("--pos-weight-grid", default="1,2,3,5,8", help="Comma-separated positive-class weights.")
    return parser.parse_args()


def write_model_note(out_dir: Path, metric_df: pd.DataFrame, upper_df: pd.DataFrame, target_recall: float) -> None:
    lines = [
        "Logit改进3模型说明（按改进方案3）",
        "1) 任务改写为高深度筛选：核心看Top-K与Recall约束，而非固定0.5阈值。",
        "2) 训练为代价敏感ElasticNet-Logit，超参以PR-AUC选择（偏召回）。",
        "3) 样本可信度采用一致性规则构建trust_score，并作为sample_weight参与训练。",
        "4) 校准流程为固定OOF分数后做单调校准（sigmoid/isotonic），重心放在Brier/ECE。",
        "5) 追加树模型上限测试，用于判断当前信息上限。",
        "",
    ]
    for _, r in metric_df.iterrows():
        lines.append(
            f"{r['sample_scope']}: auc_cal={r['auc_calibrated']:.3f}, pr_auc_cal={r['pr_auc_calibrated']:.3f}, "
            f"brier_cal={r['brier_calibrated']:.3f}, ece_cal={r['ece_calibrated']:.3f}, "
            f"recall@target({target_recall:.0%})={r['recall_at_target_recall']:.3f}, "
            f"precision@target={r['precision_at_target_recall']:.3f}"
        )
    lines.append("")
    lines.append("上限测试（同口径OOF）：")
    for _, r in upper_df.iterrows():
        lines.append(
            f"{r['sample_scope']} | {r['model_family']}({r['engine']}): auc={r['auc']:.3f}, pr_auc={r['pr_auc']:.3f}, brier={r['brier']:.3f}"
        )
    (out_dir / "Logit改进3_模型说明.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*'penalty' was deprecated.*")

        c_grid = [float(x) for x in str(args.c_grid).split(",") if str(x).strip()]
        pos_weight_grid = [float(x) for x in str(args.pos_weight_grid).split(",") if str(x).strip()]

        df = pd.read_csv(args.input_csv)
        required = {"respondent_id", "C008", "C024", "C025"}
        miss = sorted(c for c in required if c not in df.columns)
        if miss:
            raise ValueError(f"输入缺少字段：{miss}")

        trust = compute_trust_table(df)
        df = df.merge(trust, on="respondent_id", how="left")

        sample_defs = [
            {
                "sample_scope": "A_到访口径主模型",
                "mask": pd.to_numeric(df["C008"], errors="coerce") == 1,
                "include_visit_status": False,
                "include_interactions": False,
            },
            {
                "sample_scope": "B_全样本对照",
                "mask": pd.to_numeric(df["C008"], errors="coerce").isin([1, 2]),
                "include_visit_status": True,
                "include_interactions": True,
            },
        ]

        metric_rows = []
        topk_rows = []
        cal_tables = []
        coef_rows = []
        pred_tables = []
        tuning_tables = []
        upper_rows = []

        for i, s in enumerate(sample_defs, start=1):
            res = run_scope(
                df=df,
                scope_name=s["sample_scope"],
                mask=s["mask"],
                include_visit_status=s["include_visit_status"],
                include_interactions=s["include_interactions"],
                c_grid=c_grid,
                pos_weight_grid=pos_weight_grid,
                l1_ratio=args.l1_ratio,
                cv_folds=args.cv_folds,
                target_recall=args.target_recall,
                random_state=args.random_state + i * 100,
            )
            metric_rows.append(res["metric_row"])
            topk_rows.extend(res["topk_rows"])
            cal_tables.append(res["cal_rows"])
            coef_rows.extend(res["coef_rows"])
            pred_tables.append(res["oof_pred"])
            tuning_tables.append(res["tuning_rows"])
            upper_rows.extend(res["upper_rows"])

    metric_df = pd.DataFrame(metric_rows).sort_values("sample_scope").reset_index(drop=True)
    topk_df = pd.DataFrame(topk_rows).sort_values(["sample_scope", "score_type", "top_k_pct"]).reset_index(drop=True)
    cal_df = pd.concat(cal_tables, ignore_index=True)
    coef_df = pd.DataFrame(coef_rows).sort_values(["sample_scope", "abs_coef"], ascending=[True, False]).reset_index(drop=True)
    pred_df = pd.concat(pred_tables, ignore_index=True)
    tuning_df = pd.concat(tuning_tables, ignore_index=True).sort_values(["sample_scope", "pr_auc", "brier"], ascending=[True, False, True]).reset_index(drop=True)
    upper_df = pd.DataFrame(upper_rows).sort_values(["sample_scope", "model_family"]).reset_index(drop=True)
    trust_out = trust.sort_values("respondent_id").reset_index(drop=True)

    metric_df.to_csv(out_dir / "Logit改进3_模型指标.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(out_dir / "Logit改进3_TopK指标.csv", index=False, encoding="utf-8-sig")
    cal_df.to_csv(out_dir / "Logit改进3_校准分箱.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(out_dir / "Logit改进3_系数OR.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "Logit改进3_OOF预测.csv", index=False, encoding="utf-8-sig")
    tuning_df.to_csv(out_dir / "Logit改进3_调参结果.csv", index=False, encoding="utf-8-sig")
    upper_df.to_csv(out_dir / "Logit改进3_上限测试.csv", index=False, encoding="utf-8-sig")
    trust_out.to_csv(out_dir / "Logit改进3_样本可信度.csv", index=False, encoding="utf-8-sig")

    header_map = get_header_map(Path(args.source_xlsx))
    fmap_rows = []
    for f in sorted(set(coef_df["feature"].tolist())):
        if f == "Intercept":
            fmap_rows.append({"feature": f, "feature_group": "截距", "question_text": "截距", "is_interaction": 0})
            continue
        if f.startswith("VISIT_X_"):
            base = f.replace("VISIT_X_", "")
            fmap_rows.append(
                {
                    "feature": f,
                    "feature_group": feature_group(f),
                    "question_text": f"到访状态 × {header_map.get(base, base)}",
                    "is_interaction": 1,
                }
            )
        else:
            fmap_rows.append(
                {
                    "feature": f,
                    "feature_group": feature_group(f),
                    "question_text": header_map.get(f, f),
                    "is_interaction": 0,
                }
            )
    pd.DataFrame(fmap_rows).to_csv(out_dir / "Logit改进3_特征映射.csv", index=False, encoding="utf-8-sig")

    write_model_note(out_dir=out_dir, metric_df=metric_df, upper_df=upper_df, target_recall=args.target_recall)

    meta = {
        "input_csv": str(args.input_csv),
        "source_xlsx": str(args.source_xlsx),
        "output_dir": str(out_dir),
        "target_task": "high_depth_ranking_and_screening",
        "target_recall": float(args.target_recall),
        "cv_folds": int(args.cv_folds),
        "l1_ratio": float(args.l1_ratio),
        "c_grid": c_grid,
        "pos_weight_grid": pos_weight_grid,
        "sample_defs": ["A_到访口径主模型", "B_全样本对照"],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"logit_improved_v3_done: {out_dir}")


if __name__ == "__main__":
    main()
