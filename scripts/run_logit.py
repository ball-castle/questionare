#!/usr/bin/env python3
"""执行 Logit 建模并输出稳健性结果。"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import date
from pathlib import Path

from qp_io import read_xlsx_first_sheet


CORE_COL = "C088"
Q20_COL = "C090"
Q21_COL = "C091"
CONTROL_COLS = ["C001", "C002", "C003", "C004", "C005", "C008"]
TARGET_Q20 = "y_q20_high"
TARGET_Q21 = "y_q21_high"


def _load_dependencies() -> None:
    global np, pd, smf
    global ColumnTransformer, SimpleImputer, LogisticRegression
    global accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
    global StratifiedKFold, cross_val_predict, Pipeline, OneHotEncoder, ConvergenceWarning

    import numpy as np_module
    import pandas as pd_module
    import statsmodels.formula.api as smf_module
    from sklearn.compose import ColumnTransformer as ColumnTransformer_cls
    from sklearn.impute import SimpleImputer as SimpleImputer_cls
    from sklearn.linear_model import LogisticRegression as LogisticRegression_cls
    from sklearn.metrics import (
        accuracy_score as accuracy_score_fn,
        average_precision_score as average_precision_score_fn,
        brier_score_loss as brier_score_loss_fn,
        log_loss as log_loss_fn,
        roc_auc_score as roc_auc_score_fn,
    )
    from sklearn.model_selection import StratifiedKFold as StratifiedKFold_cls, cross_val_predict as cross_val_predict_fn
    from sklearn.pipeline import Pipeline as Pipeline_cls
    from sklearn.preprocessing import OneHotEncoder as OneHotEncoder_cls
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as ConvergenceWarning_cls

    np = np_module
    pd = pd_module
    smf = smf_module
    ColumnTransformer = ColumnTransformer_cls
    SimpleImputer = SimpleImputer_cls
    LogisticRegression = LogisticRegression_cls
    accuracy_score = accuracy_score_fn
    average_precision_score = average_precision_score_fn
    brier_score_loss = brier_score_loss_fn
    log_loss = log_loss_fn
    roc_auc_score = roc_auc_score_fn
    StratifiedKFold = StratifiedKFold_cls
    cross_val_predict = cross_val_predict_fn
    Pipeline = Pipeline_cls
    OneHotEncoder = OneHotEncoder_cls
    ConvergenceWarning = ConvergenceWarning_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Logit analysis and export artifacts.")
    parser.add_argument("--input-csv", default="data/data_analysis/_source_analysis/tables/survey_clean.csv", help="Model-ready sample file.")
    parser.add_argument("--source-xlsx", default="data/叶开泰问卷数据.xlsx", help="Raw xlsx for header mapping.")
    parser.add_argument("--output-dir", default="data/data_logit3", help="Output directory.")
    parser.add_argument("--cv-folds", type=int, default=5, help="OOF CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=500, help="Max iteration for statsmodels Logit.")
    return parser.parse_args()


def get_header_map(source_xlsx: Path) -> dict[str, str]:
    headers, _ = read_xlsx_first_sheet(source_xlsx)
    return {f"C{i:03d}": headers[i - 1] for i in range(1, len(headers) + 1)}


def to_binary_high(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.isin([1, 2])] = 1.0
    out[s.isin([3, 4, 5])] = 0.0
    return out


def choose_cv_splits(y: np.ndarray, max_splits: int) -> int:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    use = min(max_splits, n_pos, n_neg)
    return int(use) if use >= 2 else 0


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def build_model_df(base_df: pd.DataFrame, target: str, include_q20: bool) -> pd.DataFrame:
    need = [target, CORE_COL] + CONTROL_COLS
    if include_q20:
        need.append(TARGET_Q20)
    cols = ["respondent_id"] + need
    df = base_df[cols].copy()
    df = df.dropna(subset=need).reset_index(drop=True)
    df[target] = pd.to_numeric(df[target], errors="coerce").astype(int)
    df[CORE_COL] = pd.to_numeric(df[CORE_COL], errors="coerce").astype(float)
    if include_q20:
        df[TARGET_Q20] = pd.to_numeric(df[TARGET_Q20], errors="coerce").astype(int)
    for c in CONTROL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    return df


def fit_statsmodels_logit(model_df: pd.DataFrame, formula: str, max_iter: int):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        res = smf.logit(formula=formula, data=model_df).fit(disp=0, maxiter=max_iter)
    try:
        robust = res.get_robustcov_results(cov_type="HC3")
    except Exception:
        robust = res
    return res, robust


def build_oof_probs(
    model_df: pd.DataFrame,
    target: str,
    include_q20: bool,
    cv_folds: int,
    random_state: int,
) -> tuple[np.ndarray, str]:
    numeric_cols = [CORE_COL]
    if include_q20:
        numeric_cols.append(TARGET_Q20)
    categorical_cols = CONTROL_COLS.copy()

    x = model_df[numeric_cols + categorical_cols].copy()
    y = model_df[target].to_numpy(dtype=int)

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=random_state)
    pipe = Pipeline([("prep", prep), ("clf", clf)])

    n_splits = choose_cv_splits(y=y, max_splits=cv_folds)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        p = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6), f"stratified_{n_splits}fold_oof"

    pipe.fit(x, y)
    p = pipe.predict_proba(x)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6), "insample_fallback"


def extract_coef_rows(robust_res, model_id: str, sample_scope: str) -> list[dict]:
    params = pd.Series(robust_res.params)
    bse = pd.Series(robust_res.bse, index=params.index)
    zvals = pd.Series(robust_res.tvalues, index=params.index) if hasattr(robust_res, "tvalues") else params / bse.replace(0, np.nan)
    pvals = pd.Series(robust_res.pvalues, index=params.index)

    ci = robust_res.conf_int()
    if isinstance(ci, pd.DataFrame):
        ci_df = ci
        ci_df.columns = ["ci_lower", "ci_upper"]
    else:
        ci_df = pd.DataFrame(ci, index=params.index, columns=["ci_lower", "ci_upper"])

    mfx_map: dict[str, float] = {}
    try:
        mfx = robust_res.get_margeff(at="overall")
        mfx_df = mfx.summary_frame()
        dy_col = "dy/dx" if "dy/dx" in mfx_df.columns else mfx_df.columns[0]
        mfx_map = mfx_df[dy_col].to_dict()
    except Exception:
        mfx_map = {}

    rows = []
    for term in params.index.tolist():
        coef = float(params[term])
        rows.append(
            {
                "model_id": model_id,
                "sample_scope": sample_scope,
                "term": term,
                "coef": coef,
                "std_err": float(bse.get(term, np.nan)),
                "z": float(zvals.get(term, np.nan)),
                "p_value": float(pvals.get(term, np.nan)),
                "ci_lower": float(ci_df.loc[term, "ci_lower"]) if term in ci_df.index else np.nan,
                "ci_upper": float(ci_df.loc[term, "ci_upper"]) if term in ci_df.index else np.nan,
                "odds_ratio": float(np.exp(coef)),
                "or_ci_lower": float(np.exp(ci_df.loc[term, "ci_lower"])) if term in ci_df.index else np.nan,
                "or_ci_upper": float(np.exp(ci_df.loc[term, "ci_upper"])) if term in ci_df.index else np.nan,
                "marginal_effect": float(mfx_map.get(term, np.nan)),
                "abs_coef": float(abs(coef)),
            }
        )
    return rows


def evaluate_model(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    fit_res,
    model_id: str,
    sample_scope: str,
    target_name: str,
    oof_method: str,
) -> dict:
    y_pred = (p_pred >= 0.5).astype(int)
    return {
        "model_id": model_id,
        "sample_scope": sample_scope,
        "target_name": target_name,
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "oof_method": oof_method,
        "accuracy_oof": safe_metric(accuracy_score, y_true, y_pred),
        "auc_oof": safe_metric(roc_auc_score, y_true, p_pred),
        "pr_auc_oof": safe_metric(average_precision_score, y_true, p_pred),
        "brier_oof": safe_metric(brier_score_loss, y_true, p_pred),
        "log_loss_oof": safe_metric(log_loss, y_true, p_pred, labels=[0, 1]),
        "pseudo_r2_mcfadden": float(getattr(fit_res, "prsquared", np.nan)),
        "llf": float(getattr(fit_res, "llf", np.nan)),
        "llnull": float(getattr(fit_res, "llnull", np.nan)),
        "aic": float(getattr(fit_res, "aic", np.nan)),
        "bic": float(getattr(fit_res, "bic", np.nan)),
    }


def make_cross_table(base_df: pd.DataFrame) -> pd.DataFrame:
    sub = base_df[["respondent_id", TARGET_Q20, TARGET_Q21]].dropna().copy()
    sub[TARGET_Q20] = sub[TARGET_Q20].astype(int)
    sub[TARGET_Q21] = sub[TARGET_Q21].astype(int)

    ctab = pd.crosstab(sub[TARGET_Q20], sub[TARGET_Q21])
    a = int(ctab.loc[1, 1]) if (1 in ctab.index and 1 in ctab.columns) else 0
    b = int(ctab.loc[1, 0]) if (1 in ctab.index and 0 in ctab.columns) else 0
    c = int(ctab.loc[0, 1]) if (0 in ctab.index and 1 in ctab.columns) else 0
    d = int(ctab.loc[0, 0]) if (0 in ctab.index and 0 in ctab.columns) else 0
    eps = 1e-9
    odds_ratio = (a * d + eps) / (b * c + eps)

    p_rec_given_q20_1 = float(a / max(a + b, 1))
    p_rec_given_q20_0 = float(c / max(c + d, 1))
    diff = p_rec_given_q20_1 - p_rec_given_q20_0

    rows = [
        {
            "metric": "n_total",
            "value": int(len(sub)),
            "note": "Q20与Q21同时有效样本",
        },
        {
            "metric": "p_q21_high_given_q20_high",
            "value": p_rec_given_q20_1,
            "note": "P(Q21=1 | Q20=1)",
        },
        {
            "metric": "p_q21_high_given_q20_low",
            "value": p_rec_given_q20_0,
            "note": "P(Q21=1 | Q20=0)",
        },
        {
            "metric": "rate_diff_q21_by_q20",
            "value": diff,
            "note": "两组推荐率差值",
        },
        {
            "metric": "odds_ratio_q20_to_q21_unadjusted",
            "value": odds_ratio,
            "note": "未控制变量的2x2优势比",
        },
        {
            "metric": "cell_q20_1_q21_1",
            "value": a,
            "note": "计数",
        },
        {
            "metric": "cell_q20_1_q21_0",
            "value": b,
            "note": "计数",
        },
        {
            "metric": "cell_q20_0_q21_1",
            "value": c,
            "note": "计数",
        },
        {
            "metric": "cell_q20_0_q21_0",
            "value": d,
            "note": "计数",
        },
    ]
    return pd.DataFrame(rows)


def make_gradient_table(
    m1_res,
    m1_df: pd.DataFrame,
    m2_res,
    m2_df: pd.DataFrame,
    m3_res,
    m3_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for level in [1, 2, 3, 4, 5]:
        t1 = m1_df.copy()
        t1[CORE_COL] = float(level)
        p1 = float(np.mean(m1_res.predict(t1)))

        t2 = m2_df.copy()
        t2[CORE_COL] = float(level)
        p2 = float(np.mean(m2_res.predict(t2)))

        t3a = m3_df.copy()
        t3a[CORE_COL] = float(level)
        t3a[TARGET_Q20] = 0
        p3_q20_0 = float(np.mean(m3_res.predict(t3a)))

        t3b = m3_df.copy()
        t3b[CORE_COL] = float(level)
        t3b[TARGET_Q20] = 1
        p3_q20_1 = float(np.mean(m3_res.predict(t3b)))

        rows.append(
            {
                "c088_level": int(level),
                "p_q20_high_m1_standardized": p1,
                "p_q21_high_m2_standardized": p2,
                "p_q21_high_m3_standardized_q20_0": p3_q20_0,
                "p_q21_high_m3_standardized_q20_1": p3_q20_1,
            }
        )
    return pd.DataFrame(rows)


def term_to_feature_map(term: str, header_map: dict[str, str]) -> dict[str, str | int]:
    if term == "Intercept":
        return {"feature": term, "feature_group": "截距", "question_text": "截距", "is_dummy": 0}
    if term == CORE_COL:
        return {"feature": term, "feature_group": "核心假设变量", "question_text": header_map.get(CORE_COL, CORE_COL), "is_dummy": 0}
    if term == TARGET_Q20:
        return {"feature": term, "feature_group": "机制变量", "question_text": "高游览意愿（Q20二元）", "is_dummy": 0}

    m = re.match(r"^C\((C\d{3})\)\[T\.(.+)\]$", term)
    if m:
        var = m.group(1)
        lvl = m.group(2).replace(".0", "")
        base_text = header_map.get(var, var)
        return {
            "feature": term,
            "feature_group": "控制变量哑变量",
            "question_text": f"{base_text}（类别={lvl}，对比基准=1）",
            "is_dummy": 1,
        }
    return {"feature": term, "feature_group": "其他", "question_text": term, "is_dummy": 0}


def write_model_note(
    out_path: Path,
    metric_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    gradient_df: pd.DataFrame,
) -> None:
    def find_coef(model_id: str, term: str) -> dict:
        q = coef_df[(coef_df["model_id"] == model_id) & (coef_df["term"] == term)]
        if q.empty:
            return {}
        return q.iloc[0].to_dict()

    def get_metric(name: str) -> float:
        q = cross_df[cross_df["metric"] == name]
        return float(q.iloc[0]["value"]) if not q.empty else np.nan

    c_m1 = find_coef("M1_Q20_visit_intent", CORE_COL)
    c_m2 = find_coef("M2_Q21_recommend_direct", CORE_COL)
    c_m3_c088 = find_coef("M3_Q21_recommend_with_q20", CORE_COL)
    c_m3_q20 = find_coef("M3_Q21_recommend_with_q20", TARGET_Q20)

    g1 = gradient_df[gradient_df["c088_level"] == 1]
    g5 = gradient_df[gradient_df["c088_level"] == 5]
    p1 = float(g1.iloc[0]["p_q20_high_m1_standardized"]) if not g1.empty else np.nan
    p5 = float(g5.iloc[0]["p_q20_high_m1_standardized"]) if not g5.empty else np.nan

    lines = [
        "Logit改进5模型说明（按改进方案5：更换假设）",
        "核心假设链路：C088（认知增益） -> Q20高游览意愿 -> Q21高推荐意愿。",
        "二元化口径：Q20/Q21均按(1,2)=1，(3,4,5)=0。",
        "",
        f"Q20高意愿占比：{get_metric('cell_q20_1_q21_1') + get_metric('cell_q20_1_q21_0'):.0f}/{get_metric('n_total'):.0f}"
        f" = {(get_metric('cell_q20_1_q21_1') + get_metric('cell_q20_1_q21_0')) / max(get_metric('n_total'), 1):.3f}",
        f"Q21高推荐在Q20=1组：{get_metric('p_q21_high_given_q20_high'):.3f}；在Q20=0组：{get_metric('p_q21_high_given_q20_low'):.3f}",
        f"Q20->Q21未调整优势比：OR={get_metric('odds_ratio_q20_to_q21_unadjusted'):.2f}",
        "",
        "关键系数（控制人口学与到访状态后）：",
        f"M1(Q20) C088: OR={c_m1.get('odds_ratio', np.nan):.3f}, 95%CI[{c_m1.get('or_ci_lower', np.nan):.3f}, {c_m1.get('or_ci_upper', np.nan):.3f}], p={c_m1.get('p_value', np.nan):.3g}",
        f"M2(Q21) C088: OR={c_m2.get('odds_ratio', np.nan):.3f}, 95%CI[{c_m2.get('or_ci_lower', np.nan):.3f}, {c_m2.get('or_ci_upper', np.nan):.3f}], p={c_m2.get('p_value', np.nan):.3g}",
        f"M3(Q21|含Q20) Q20: OR={c_m3_q20.get('odds_ratio', np.nan):.3f}, 95%CI[{c_m3_q20.get('or_ci_lower', np.nan):.3f}, {c_m3_q20.get('or_ci_upper', np.nan):.3f}], p={c_m3_q20.get('p_value', np.nan):.3g}",
        f"M3(Q21|含Q20) C088: OR={c_m3_c088.get('odds_ratio', np.nan):.3f}, 95%CI[{c_m3_c088.get('or_ci_lower', np.nan):.3f}, {c_m3_c088.get('or_ci_upper', np.nan):.3f}], p={c_m3_c088.get('p_value', np.nan):.3g}",
        "",
        f"认知增益梯度（标准化预测，M1）：C088=1时Q20高意愿约{p1:.3f}，C088=5时约{p5:.3f}。",
        "",
    ]

    if not metric_df.empty:
        lines.append("模型OOF指标：")
        for _, r in metric_df.iterrows():
            lines.append(
                f"{r['model_id']}: auc_oof={r['auc_oof']:.3f}, pr_auc_oof={r['pr_auc_oof']:.3f}, "
                f"brier_oof={r['brier_oof']:.3f}, pseudo_r2={r['pseudo_r2_mcfadden']:.3f}"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _load_dependencies()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    required = {"respondent_id", CORE_COL, Q20_COL, Q21_COL}.union(CONTROL_COLS)
    miss = sorted(c for c in required if c not in raw_df.columns)
    if miss:
        raise ValueError(f"输入缺少字段：{miss}")

    base_df = raw_df.copy()
    base_df[TARGET_Q20] = to_binary_high(base_df[Q20_COL])
    base_df[TARGET_Q21] = to_binary_high(base_df[Q21_COL])
    base_df[CORE_COL] = pd.to_numeric(base_df[CORE_COL], errors="coerce")
    for c in CONTROL_COLS:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce")

    control_formula = " + ".join(f"C({c})" for c in CONTROL_COLS)
    model_defs = [
        {
            "model_id": "M1_Q20_visit_intent",
            "sample_scope": "模型1_Q20游览意愿",
            "target": TARGET_Q20,
            "include_q20": False,
            "formula": f"{TARGET_Q20} ~ {CORE_COL} + {control_formula}",
            "seed_offset": 11,
        },
        {
            "model_id": "M2_Q21_recommend_direct",
            "sample_scope": "模型2_Q21推荐意愿_不含Q20",
            "target": TARGET_Q21,
            "include_q20": False,
            "formula": f"{TARGET_Q21} ~ {CORE_COL} + {control_formula}",
            "seed_offset": 29,
        },
        {
            "model_id": "M3_Q21_recommend_with_q20",
            "sample_scope": "模型3_Q21推荐意愿_含Q20链路",
            "target": TARGET_Q21,
            "include_q20": True,
            "formula": f"{TARGET_Q21} ~ {CORE_COL} + {TARGET_Q20} + {control_formula}",
            "seed_offset": 47,
        },
    ]

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    pred_tables: list[pd.DataFrame] = []
    fitted: dict[str, dict] = {}

    for m in model_defs:
        model_df = build_model_df(base_df=base_df, target=m["target"], include_q20=m["include_q20"])
        fit_res, robust_res = fit_statsmodels_logit(model_df=model_df, formula=m["formula"], max_iter=args.max_iter)
        coef_rows.extend(extract_coef_rows(robust_res=robust_res, model_id=m["model_id"], sample_scope=m["sample_scope"]))

        p_oof, oof_method = build_oof_probs(
            model_df=model_df,
            target=m["target"],
            include_q20=m["include_q20"],
            cv_folds=args.cv_folds,
            random_state=args.random_state + int(m["seed_offset"]),
        )
        y_true = model_df[m["target"]].to_numpy(dtype=int)
        metric_rows.append(
            evaluate_model(
                y_true=y_true,
                p_pred=p_oof,
                fit_res=fit_res,
                model_id=m["model_id"],
                sample_scope=m["sample_scope"],
                target_name=m["target"],
                oof_method=oof_method,
            )
        )
        pred_tables.append(
            pd.DataFrame(
                {
                    "respondent_id": model_df["respondent_id"].to_numpy(),
                    "model_id": m["model_id"],
                    "sample_scope": m["sample_scope"],
                    "y_true": y_true,
                    "prob_oof": p_oof,
                    "pred_0_5": (p_oof >= 0.5).astype(int),
                }
            )
        )
        fitted[m["model_id"]] = {"fit_res": fit_res, "model_df": model_df}

    metric_df = pd.DataFrame(metric_rows).sort_values("model_id").reset_index(drop=True)
    coef_df = pd.DataFrame(coef_rows).sort_values(["model_id", "abs_coef"], ascending=[True, False]).reset_index(drop=True)
    pred_df = pd.concat(pred_tables, ignore_index=True).sort_values(["model_id", "respondent_id"]).reset_index(drop=True)

    cross_df = make_cross_table(base_df=base_df)
    gradient_df = make_gradient_table(
        m1_res=fitted["M1_Q20_visit_intent"]["fit_res"],
        m1_df=fitted["M1_Q20_visit_intent"]["model_df"],
        m2_res=fitted["M2_Q21_recommend_direct"]["fit_res"],
        m2_df=fitted["M2_Q21_recommend_direct"]["model_df"],
        m3_res=fitted["M3_Q21_recommend_with_q20"]["fit_res"],
        m3_df=fitted["M3_Q21_recommend_with_q20"]["model_df"],
    )

    header_map = get_header_map(Path(args.source_xlsx))
    fmap_rows = []
    for t in sorted(set(coef_df["term"].tolist())):
        fmap_rows.append(term_to_feature_map(t, header_map=header_map))
    fmap_df = pd.DataFrame(fmap_rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)

    metric_df.to_csv(out_dir / "Logit改进5_模型指标.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(out_dir / "Logit改进5_系数OR.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "Logit改进5_OOF预测.csv", index=False, encoding="utf-8-sig")
    cross_df.to_csv(out_dir / "Logit改进5_Q20Q21交叉分析.csv", index=False, encoding="utf-8-sig")
    gradient_df.to_csv(out_dir / "Logit改进5_认知增益梯度.csv", index=False, encoding="utf-8-sig")
    fmap_df.to_csv(out_dir / "Logit改进5_特征映射.csv", index=False, encoding="utf-8-sig")

    write_model_note(
        out_path=out_dir / "Logit改进5_模型说明.txt",
        metric_df=metric_df,
        coef_df=coef_df,
        cross_df=cross_df,
        gradient_df=gradient_df,
    )

    meta = {
        "generated_on": str(date.today()),
        "input_csv": str(args.input_csv),
        "source_xlsx": str(args.source_xlsx),
        "output_dir": str(out_dir),
        "hypothesis": "C088认知增益 -> Q20高游览意愿 -> Q21高推荐意愿（并检验C088对Q21直接效应）",
        "binary_rules": {
            "Q20(C090)": "1/2 => 1; 3/4/5 => 0",
            "Q21(C091)": "1/2 => 1; 3/4/5 => 0",
        },
        "controls": CONTROL_COLS,
        "models": [m["model_id"] for m in model_defs],
        "cv_folds": int(args.cv_folds),
        "random_state": int(args.random_state),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"logit_done: {out_dir}")


if __name__ == "__main__":
    main()
