#!/usr/bin/env python3
"""Improved Logit workflow aligned with 改进方案2.

Enhancements:
1) Ordered depth modeling via two-stage cumulative logits (low/mid/high).
2) Nested CV + calibration (Platt/Isotonic) for high-depth screening probability.
3) High-information visited sample as an extra modeling scope.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from qp_io import read_xlsx_first_sheet


def ccols(start: int, end: int) -> list[str]:
    return [f"C{i:03d}" for i in range(start, end + 1)]


DEMOGRAPHIC_COLS = ["C002", "C006", "C007"]
MOTIVE_COLS = ccols(16, 23)
PERCEPTION_COLS = ccols(52, 63) + ["C065"]
IMPORTANCE_COLS = ccols(66, 75)
PERFORMANCE_COLS = ccols(76, 85)
COGNITION_COLS = ccols(86, 89)
LIKERT_CORE_COLS = PERCEPTION_COLS + IMPORTANCE_COLS + PERFORMANCE_COLS + COGNITION_COLS
TOP_CUTS = (0.10, 0.20)


def choose_cv_splits(y: np.ndarray, max_splits: int) -> int:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return max(2, min(max_splits, n_pos, n_neg))


def make_binary_logit(c_value: float, l1_ratio: float, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=float(c_value),
        max_iter=10000,
        random_state=random_state,
    )


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


def build_features(df: pd.DataFrame, include_visit_status: bool, include_interactions: bool) -> pd.DataFrame:
    cols = DEMOGRAPHIC_COLS + PERCEPTION_COLS + IMPORTANCE_COLS + PERFORMANCE_COLS + COGNITION_COLS + MOTIVE_COLS
    if include_visit_status:
        cols.append("C008")
    feats = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    if include_interactions:
        visit = (pd.to_numeric(df["C008"], errors="coerce") == 1).astype(float)
        for c in PERFORMANCE_COLS:
            feats[f"VISIT_X_{c}"] = visit * pd.to_numeric(df[c], errors="coerce")
    return feats


def build_depth_class(df: pd.DataFrame) -> np.ndarray:
    stay = pd.to_numeric(df["C024"], errors="coerce")
    spend = pd.to_numeric(df["C025"], errors="coerce")
    return np.where((stay >= 3) & (spend >= 3), 2, np.where((stay >= 3) | (spend >= 3), 1, 0)).astype(int)


def compute_high_info_table(df: pd.DataFrame) -> pd.DataFrame:
    likert = df[LIKERT_CORE_COLS].apply(pd.to_numeric, errors="coerce")
    arr = likert.to_numpy(dtype=float)
    unique_levels = np.array([len(np.unique(r[~np.isnan(r)])) for r in arr], dtype=int)
    sd = np.nanstd(arr, axis=1)
    motive = df[MOTIVE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy()
    flag = (unique_levels >= 4) & (sd >= 0.55) & (motive >= 1)
    out = pd.DataFrame(
        {
            "respondent_id": df["respondent_id"].to_numpy() if "respondent_id" in df.columns else np.arange(1, len(df) + 1),
            "unique_levels_52_89": unique_levels,
            "likert_sd_52_89": sd,
            "motive_count_16_23": motive,
            "high_info_flag": flag.astype(int),
        }
    )
    return out


def select_best_c_auc(
    x: np.ndarray,
    y: np.ndarray,
    c_grid: list[float],
    cv_folds: int,
    random_state: int,
    l1_ratio: float,
) -> tuple[float, float]:
    n_splits = choose_cv_splits(y, cv_folds)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_c = float(c_grid[0])
    best_auc = -np.inf
    for c in c_grid:
        est = make_binary_logit(c_value=c, l1_ratio=l1_ratio, random_state=random_state)
        auc = float(np.mean(cross_val_score(est, x, y, cv=cv, scoring="roc_auc")))
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_c = float(c)
    return best_c, float(best_auc)


def fit_sigmoid_calibrator(raw_prob: np.ndarray, y: np.ndarray) -> LogisticRegression:
    cal = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    cal.fit(raw_prob.reshape(-1, 1), y)
    return cal


def apply_sigmoid_calibrator(cal: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    return cal.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def fit_isotonic_calibrator(raw_prob: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_prob, y)
    return cal


def apply_isotonic_calibrator(cal: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    return np.asarray(cal.transform(raw_prob), dtype=float)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    q = min(bins, int(d["p"].nunique()))
    if q < 2:
        d["bin"] = 0
    else:
        d["bin"] = pd.qcut(d["p"], q=q, labels=False, duplicates="drop")
    g = (
        d.groupby("bin", dropna=False)
        .agg(n=("y", "size"), predicted_mean=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    g["bin"] = g["bin"].astype(str)
    return g


def hosmer_lemeshow_p(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float, int]:
    cal = calibration_table(y, p, bins=bins)
    obs = cal["observed_rate"].to_numpy() * cal["n"].to_numpy()
    exp = cal["predicted_mean"].to_numpy() * cal["n"].to_numpy()
    n = cal["n"].to_numpy().astype(float)
    eps = 1e-12
    hl_stat = np.sum(((obs - exp) ** 2) / (exp + eps) + (((n - obs) - (n - exp)) ** 2) / ((n - exp) + eps))
    dof = max(int(len(cal) - 2), 1)
    p_value = float(1.0 - chi2.cdf(hl_stat, dof))
    return float(hl_stat), p_value, dof


def lift_rows(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    base_rate = float(np.mean(y)) if len(y) else np.nan
    if len(y) == 0:
        return rows
    order = np.argsort(-p)
    for cut in TOP_CUTS:
        k = max(1, int(np.ceil(len(y) * cut)))
        top_rate = float(np.mean(y[order[:k]]))
        lift = float(top_rate / base_rate) if base_rate > 0 else np.nan
        rows.append(
            {
                "top_cut_pct": int(round(cut * 100)),
                "top_n": int(k),
                "hit_rate": top_rate,
                "base_rate": base_rate,
                "lift": lift,
            }
        )
    return rows


def nested_cv_binary_with_calibration(
    x: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    c_grid: list[float],
    l1_ratio: float,
    random_state: int,
) -> dict:
    if len(y) < 80 or int(y.sum()) < 10 or int((1 - y).sum()) < 10:
        raise ValueError(f"样本过少或类别过稀：n={len(y)}, pos={int(y.sum())}, neg={int((1-y).sum())}")

    outer_splits = choose_cv_splits(y, outer_folds)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    prob_cal = np.zeros(len(y), dtype=float)
    prob_raw = np.zeros(len(y), dtype=float)
    fold_rows: list[dict] = []

    for fold_no, (tr_idx, te_idx) in enumerate(outer_cv.split(x, y), start=1):
        x_tr, x_te = x[tr_idx], x[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        best_c, inner_auc = select_best_c_auc(
            x=x_tr,
            y=y_tr,
            c_grid=c_grid,
            cv_folds=inner_folds,
            random_state=random_state + fold_no,
            l1_ratio=l1_ratio,
        )

        base_for_oof = make_binary_logit(c_value=best_c, l1_ratio=l1_ratio, random_state=random_state + fold_no)
        cal_splits = choose_cv_splits(y_tr, min(inner_folds, 4))
        cal_cv = StratifiedKFold(n_splits=cal_splits, shuffle=True, random_state=random_state + 100 + fold_no)
        raw_oof_tr = cross_val_predict(base_for_oof, x_tr, y_tr, cv=cal_cv, method="predict_proba")[:, 1]

        # Calibration method selection on training OOF probabilities.
        sigmoid = fit_sigmoid_calibrator(raw_oof_tr, y_tr)
        p_sig = np.clip(apply_sigmoid_calibrator(sigmoid, raw_oof_tr), 1e-6, 1 - 1e-6)
        brier_sig = float(brier_score_loss(y_tr, p_sig))

        isotonic = fit_isotonic_calibrator(raw_oof_tr, y_tr)
        p_iso = np.clip(apply_isotonic_calibrator(isotonic, raw_oof_tr), 1e-6, 1 - 1e-6)
        brier_iso = float(brier_score_loss(y_tr, p_iso))

        if brier_iso < brier_sig:
            cal_method = "isotonic"
            cal_obj = isotonic
            cal_brier = brier_iso
        else:
            cal_method = "sigmoid"
            cal_obj = sigmoid
            cal_brier = brier_sig

        base_final = make_binary_logit(c_value=best_c, l1_ratio=l1_ratio, random_state=random_state + 200 + fold_no)
        base_final.fit(x_tr, y_tr)
        raw_te = base_final.predict_proba(x_te)[:, 1]
        if cal_method == "isotonic":
            cal_te = apply_isotonic_calibrator(cal_obj, raw_te)
        else:
            cal_te = apply_sigmoid_calibrator(cal_obj, raw_te)

        prob_raw[te_idx] = np.clip(raw_te, 1e-6, 1 - 1e-6)
        prob_cal[te_idx] = np.clip(cal_te, 1e-6, 1 - 1e-6)
        fold_rows.append(
            {
                "fold": fold_no,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "events_train": int(y_tr.sum()),
                "best_c": float(best_c),
                "inner_auc": float(inner_auc),
                "calibration_method": cal_method,
                "calibration_brier_train_oof": float(cal_brier),
                "test_auc_raw": float(roc_auc_score(y_te, prob_raw[te_idx])),
                "test_auc_calibrated": float(roc_auc_score(y_te, prob_cal[te_idx])),
            }
        )

    best_c_full, full_auc = select_best_c_auc(
        x=x,
        y=y,
        c_grid=c_grid,
        cv_folds=inner_folds,
        random_state=random_state + 999,
        l1_ratio=l1_ratio,
    )
    final_model = make_binary_logit(c_value=best_c_full, l1_ratio=l1_ratio, random_state=random_state + 777)
    final_model.fit(x, y)

    return {
        "prob_raw_oof": np.clip(prob_raw, 1e-6, 1 - 1e-6),
        "prob_cal_oof": np.clip(prob_cal, 1e-6, 1 - 1e-6),
        "fold_rows": fold_rows,
        "final_model": final_model,
        "best_c_full": float(best_c_full),
        "cv_auc_mean_full": float(full_auc),
        "outer_folds": int(outer_splits),
    }


def multiclass_brier(y: np.ndarray, prob3: np.ndarray) -> float:
    onehot = np.eye(3, dtype=float)[y]
    return float(np.mean(np.sum((onehot - prob3) ** 2, axis=1)))


def run_scope(
    df: pd.DataFrame,
    sample_scope: str,
    sample_mask: pd.Series,
    include_visit_status: bool,
    include_interactions: bool,
    outer_folds: int,
    inner_folds: int,
    c_grid: list[float],
    l1_ratio: float,
    random_state: int,
) -> dict:
    sub = df.loc[sample_mask].copy()
    y_cls = build_depth_class(sub)
    y_ge1 = (y_cls >= 1).astype(int)
    y_ge2 = (y_cls >= 2).astype(int)

    x_df = build_features(sub, include_visit_status=include_visit_status, include_interactions=include_interactions)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy(dtype=float)

    stage1 = nested_cv_binary_with_calibration(
        x=x,
        y=y_ge1,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        c_grid=c_grid,
        l1_ratio=l1_ratio,
        random_state=random_state + 10,
    )
    stage2 = nested_cv_binary_with_calibration(
        x=x,
        y=y_ge2,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        c_grid=c_grid,
        l1_ratio=l1_ratio,
        random_state=random_state + 20,
    )

    p_ge1 = stage1["prob_cal_oof"].copy()
    p_ge2 = stage2["prob_cal_oof"].copy()
    p_ge1 = np.clip(p_ge1, 1e-6, 1 - 1e-6)
    p_ge2 = np.clip(p_ge2, 1e-6, 1 - 1e-6)
    p_ge2 = np.minimum(p_ge2, p_ge1)

    p_low = 1.0 - p_ge1
    p_mid = p_ge1 - p_ge2
    p_high = p_ge2

    prob3 = np.column_stack([p_low, p_mid, p_high])
    prob3 = np.clip(prob3, 1e-6, None)
    prob3 = prob3 / prob3.sum(axis=1, keepdims=True)

    pred_cls = np.argmax(prob3, axis=1)

    high_y = (y_cls == 2).astype(int)
    hl_stat, hl_p, hl_dof = hosmer_lemeshow_p(high_y, prob3[:, 2], bins=10)
    auc_macro = float(roc_auc_score(y_cls, prob3, multi_class="ovr", average="macro"))
    high_auc = float(roc_auc_score(high_y, prob3[:, 2]))
    high_pr_auc = float(average_precision_score(high_y, prob3[:, 2]))
    high_brier = float(brier_score_loss(high_y, prob3[:, 2]))
    high_base_brier = float(brier_score_loss(high_y, np.repeat(high_y.mean(), len(high_y))))
    high_brier_skill = float(1.0 - high_brier / high_base_brier) if high_base_brier > 0 else np.nan

    metric_row = {
        "sample_scope": sample_scope,
        "n": int(len(sub)),
        "low_n": int(np.sum(y_cls == 0)),
        "mid_n": int(np.sum(y_cls == 1)),
        "high_n": int(np.sum(y_cls == 2)),
        "high_rate": float(np.mean(high_y)),
        "accuracy_3class": float(accuracy_score(y_cls, pred_cls)),
        "f1_macro_3class": float(f1_score(y_cls, pred_cls, average="macro")),
        "auc_macro_ovr_3class": auc_macro,
        "log_loss_3class": float(log_loss(y_cls, prob3, labels=[0, 1, 2])),
        "brier_3class": multiclass_brier(y_cls, prob3),
        "auc_high": high_auc,
        "pr_auc_high": high_pr_auc,
        "brier_high": high_brier,
        "brier_high_baseline": high_base_brier,
        "brier_high_skill": high_brier_skill,
        "hl_high_stat": hl_stat,
        "hl_high_dof": int(hl_dof),
        "hl_high_p_value": float(hl_p),
        "ge1_best_c_full": stage1["best_c_full"],
        "ge1_cv_auc_mean_full": stage1["cv_auc_mean_full"],
        "ge2_best_c_full": stage2["best_c_full"],
        "ge2_cv_auc_mean_full": stage2["cv_auc_mean_full"],
        "outer_folds": int(stage1["outer_folds"]),
    }

    lift = []
    for r in lift_rows(high_y, prob3[:, 2]):
        rec = {"sample_scope": sample_scope}
        rec.update(r)
        lift.append(rec)

    cal = calibration_table(high_y, prob3[:, 2], bins=10)
    cal["sample_scope"] = sample_scope

    fold_rows = []
    for r in stage1["fold_rows"]:
        rr = dict(r)
        rr["sample_scope"] = sample_scope
        rr["stage"] = "GE1(y>=1)"
        fold_rows.append(rr)
    for r in stage2["fold_rows"]:
        rr = dict(r)
        rr["sample_scope"] = sample_scope
        rr["stage"] = "GE2(y>=2)"
        fold_rows.append(rr)

    coef_rows = []
    coef_rows.append(
        {
            "sample_scope": sample_scope,
            "stage": "GE1(y>=1)",
            "feature": "Intercept",
            "coef": float(stage1["final_model"].intercept_[0]),
            "odds_ratio": float(np.exp(stage1["final_model"].intercept_[0])),
            "abs_coef": float(abs(stage1["final_model"].intercept_[0])),
        }
    )
    for f, c in zip(x_df.columns.tolist(), stage1["final_model"].coef_.ravel()):
        coef_rows.append(
            {
                "sample_scope": sample_scope,
                "stage": "GE1(y>=1)",
                "feature": f,
                "coef": float(c),
                "odds_ratio": float(np.exp(c)),
                "abs_coef": float(abs(c)),
            }
        )
    coef_rows.append(
        {
            "sample_scope": sample_scope,
            "stage": "GE2(y>=2)",
            "feature": "Intercept",
            "coef": float(stage2["final_model"].intercept_[0]),
            "odds_ratio": float(np.exp(stage2["final_model"].intercept_[0])),
            "abs_coef": float(abs(stage2["final_model"].intercept_[0])),
        }
    )
    for f, c in zip(x_df.columns.tolist(), stage2["final_model"].coef_.ravel()):
        coef_rows.append(
            {
                "sample_scope": sample_scope,
                "stage": "GE2(y>=2)",
                "feature": f,
                "coef": float(c),
                "odds_ratio": float(np.exp(c)),
                "abs_coef": float(abs(c)),
            }
        )

    pred = pd.DataFrame(
        {
            "respondent_id": sub["respondent_id"].to_numpy() if "respondent_id" in sub.columns else np.arange(1, len(sub) + 1),
            "sample_scope": sample_scope,
            "depth_class_true": y_cls.astype(int),
            "depth_class_pred": pred_cls.astype(int),
            "high_true": high_y.astype(int),
            "prob_low": prob3[:, 0],
            "prob_mid": prob3[:, 1],
            "prob_high": prob3[:, 2],
            "prob_ge1_cal": p_ge1,
            "prob_ge2_cal": p_ge2,
            "prob_ge1_raw": stage1["prob_raw_oof"],
            "prob_ge2_raw": stage2["prob_raw_oof"],
        }
    )

    return {
        "metric_row": metric_row,
        "lift_rows": lift,
        "calibration": cal,
        "coef_rows": coef_rows,
        "predictions": pred,
        "fold_rows": fold_rows,
    }


def write_model_note(out_dir: Path, metrics: pd.DataFrame, lift: pd.DataFrame) -> None:
    lines = [
        "Logit改进2模型说明（按改进方案2）",
        "1) 模型形态：有序三分类（低/中/高）两阶段累积Logit：GE1(y>=1), GE2(y>=2)。",
        "2) 口径：A到访主模型、A1高信息到访、B全样本对照。",
        "3) 训练评估：外层CV做OOF预测，内层CV调参C，随后做Platt/Isotonic校准并按Brier择优。",
        "4) 核心指标：三分类AUC/LogLoss/Brier；高深度筛选AUC/PR-AUC/Lift/Brier/H-L/校准分箱。",
        "5) 高信息定义：Likert(52-89)唯一值>=4 且 SD>=0.55，且动机条目和>=1。",
        "",
    ]
    for _, r in metrics.iterrows():
        lines.append(
            f"{r['sample_scope']}: n={int(r['n'])}, high_rate={r['high_rate']:.3f}, "
            f"auc_macro={r['auc_macro_ovr_3class']:.3f}, auc_high={r['auc_high']:.3f}, "
            f"brier_high_skill={r['brier_high_skill']:.3f}, hl_high_p={r['hl_high_p_value']:.4g}"
        )
    lines.append("")
    for _, r in lift[lift["top_cut_pct"] == 20].iterrows():
        lines.append(
            f"{r['sample_scope']} Top20: hit={r['hit_rate']:.3f}, base={r['base_rate']:.3f}, lift={r['lift']:.3f}"
        )
    (out_dir / "Logit改进2_模型说明.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved Logit v2 and export to data/data_logit1.")
    parser.add_argument(
        "--input-csv",
        default="data/data_analysis/_source_analysis/tables/survey_clean.csv",
        help="Model-ready sample file with C001-C108 and respondent_id.",
    )
    parser.add_argument(
        "--source-xlsx",
        default="data/叶开泰问卷数据.xlsx",
        help="Raw xlsx used for question text mapping.",
    )
    parser.add_argument("--output-dir", default="data/data_logit1", help="Output directory.")
    parser.add_argument("--outer-folds", type=int, default=5, help="Nested CV outer folds.")
    parser.add_argument("--inner-folds", type=int, default=4, help="Nested CV inner folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--l1-ratio", type=float, default=0.2, help="ElasticNet l1_ratio.")
    parser.add_argument("--c-grid", default="0.03,0.05,0.1,0.2,0.5,1,2,5", help="Comma-separated C grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*'penalty' was deprecated.*")

        c_grid = [float(x) for x in str(args.c_grid).split(",") if str(x).strip()]
        df = pd.read_csv(args.input_csv)

        required = {"respondent_id", "C002", "C006", "C007", "C008", "C024", "C025"}
        missing = sorted(c for c in required if c not in df.columns)
        if missing:
            raise ValueError(f"输入数据缺少字段：{missing}")

        hi = compute_high_info_table(df)
        df = df.merge(hi[["respondent_id", "high_info_flag"]], on="respondent_id", how="left")
        df["high_info_flag"] = df["high_info_flag"].fillna(0).astype(int)

        sample_defs = [
            {
                "sample_scope": "A_到访口径主模型",
                "mask": pd.to_numeric(df["C008"], errors="coerce") == 1,
                "include_visit_status": False,
                "include_interactions": False,
            },
            {
                "sample_scope": "A1_高信息到访",
                "mask": (pd.to_numeric(df["C008"], errors="coerce") == 1) & (df["high_info_flag"] == 1),
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

        metric_rows: list[dict] = []
        lift_rows_all: list[dict] = []
        calibration_tables: list[pd.DataFrame] = []
        coef_rows_all: list[dict] = []
        pred_tables: list[pd.DataFrame] = []
        fold_rows_all: list[dict] = []

        for s in sample_defs:
            res = run_scope(
                df=df,
                sample_scope=s["sample_scope"],
                sample_mask=s["mask"],
                include_visit_status=s["include_visit_status"],
                include_interactions=s["include_interactions"],
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                c_grid=c_grid,
                l1_ratio=args.l1_ratio,
                random_state=args.random_state,
            )
            metric_rows.append(res["metric_row"])
            lift_rows_all.extend(res["lift_rows"])
            calibration_tables.append(res["calibration"])
            coef_rows_all.extend(res["coef_rows"])
            pred_tables.append(res["predictions"])
            fold_rows_all.extend(res["fold_rows"])

    metrics = pd.DataFrame(metric_rows).sort_values(["sample_scope"]).reset_index(drop=True)
    lift = pd.DataFrame(lift_rows_all).sort_values(["sample_scope", "top_cut_pct"]).reset_index(drop=True)
    calibration = pd.concat(calibration_tables, ignore_index=True)
    coefs = (
        pd.DataFrame(coef_rows_all)
        .sort_values(["sample_scope", "stage", "abs_coef"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    preds = pd.concat(pred_tables, ignore_index=True)
    folds = pd.DataFrame(fold_rows_all).sort_values(["sample_scope", "stage", "fold"]).reset_index(drop=True)

    coefs_ge1 = coefs[coefs["stage"] == "GE1(y>=1)"].reset_index(drop=True)
    coefs_ge2 = coefs[coefs["stage"] == "GE2(y>=2)"].reset_index(drop=True)

    metrics.to_csv(out_dir / "Logit改进2_模型指标.csv", index=False, encoding="utf-8-sig")
    lift.to_csv(out_dir / "Logit改进2_Lift.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(out_dir / "Logit改进2_校准分箱_高深度.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(out_dir / "Logit改进2_系数OR_全部.csv", index=False, encoding="utf-8-sig")
    coefs_ge1.to_csv(out_dir / "Logit改进2_系数OR_GE1.csv", index=False, encoding="utf-8-sig")
    coefs_ge2.to_csv(out_dir / "Logit改进2_系数OR_GE2.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "Logit改进2_预测概率_OOF.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out_dir / "Logit改进2_嵌套CV折结果.csv", index=False, encoding="utf-8-sig")
    hi.to_csv(out_dir / "Logit改进2_高信息样本口径.csv", index=False, encoding="utf-8-sig")

    header_map = get_header_map(Path(args.source_xlsx))
    feature_rows = []
    for f in sorted(set(coefs["feature"].tolist())):
        if f == "Intercept":
            feature_rows.append(
                {
                    "feature": f,
                    "feature_group": "截距",
                    "question_text": "截距",
                    "is_interaction": 0,
                }
            )
            continue
        if f.startswith("VISIT_X_"):
            base = f.replace("VISIT_X_", "")
            feature_rows.append(
                {
                    "feature": f,
                    "feature_group": feature_group(f),
                    "question_text": f"到访状态 × {header_map.get(base, base)}",
                    "is_interaction": 1,
                }
            )
            continue
        feature_rows.append(
            {
                "feature": f,
                "feature_group": feature_group(f),
                "question_text": header_map.get(f, f),
                "is_interaction": 0,
            }
        )
    pd.DataFrame(feature_rows).to_csv(out_dir / "Logit改进2_特征映射.csv", index=False, encoding="utf-8-sig")

    write_model_note(out_dir=out_dir, metrics=metrics, lift=lift)

    meta = {
        "input_csv": str(args.input_csv),
        "source_xlsx": str(args.source_xlsx),
        "output_dir": str(out_dir),
        "model_type": "ordered_two_stage_cumulative_logit",
        "sample_defs": ["A_到访口径主模型", "A1_高信息到访", "B_全样本对照"],
        "high_info_rule": "unique_levels_52_89>=4 and likert_sd_52_89>=0.55 and motive_count_16_23>=1",
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "random_state": int(args.random_state),
        "l1_ratio": float(args.l1_ratio),
        "c_grid": [float(x) for x in str(args.c_grid).split(",") if str(x).strip()],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"logit_improved_v2_done: {out_dir}")


if __name__ == "__main__":
    main()
