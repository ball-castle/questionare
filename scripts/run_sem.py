#!/usr/bin/env python3
"""Run SEM with legacy/new model suite and export compatible + improved outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from semopy import Model, calc_stats
from semopy.inspector import inspect

from qp_io import read_xlsx_first_sheet

MODEL_DESC_BASELINE_CURRENT = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089

O_cognition ~ S_environment + S_service + S_activity
R_visit ~ O_cognition
R_recommend ~ O_cognition
"""

MODEL_DESC_LEGACY_COMPAT_V2 = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089

O_cognition ~ S_environment + S_service + S_activity
R_visit ~ O_cognition
R_recommend ~ O_cognition

S_environment ~~ S_service
S_environment ~~ S_activity
S_service ~~ S_activity
R_visit ~~ R_recommend

# same-dimension method-effect residuals (promotion/word-of-mouth/online seeding)
C062 ~~ C063
C063 ~~ C065
C062 ~~ C065
"""

MODEL_DESC_INTENT_PARTIAL_V1 = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089
R_intent =~ R_visit + R_recommend

O_cognition ~ S_environment + S_service + S_activity
R_intent ~ O_cognition + S_environment + S_service + S_activity

S_environment ~~ S_service
S_environment ~~ S_activity
S_service ~~ S_activity

# same-dimension method-effect residuals (promotion/word-of-mouth/online seeding)
C062 ~~ C063
C063 ~~ C065
C062 ~~ C065
"""

MODEL_DESC_CFA_INTENT_WITH_METHOD = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089
R_intent =~ R_visit + R_recommend

S_environment ~~ S_service
S_environment ~~ S_activity
S_service ~~ S_activity
S_environment ~~ O_cognition
S_service ~~ O_cognition
S_activity ~~ O_cognition
S_environment ~~ R_intent
S_service ~~ R_intent
S_activity ~~ R_intent
O_cognition ~~ R_intent

# same-dimension method-effect residuals (promotion/word-of-mouth/online seeding)
C062 ~~ C063
C063 ~~ C065
C062 ~~ C065
"""

MODEL_DESC_CFA_INTENT_NO_METHOD = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089
R_intent =~ R_visit + R_recommend

S_environment ~~ S_service
S_environment ~~ S_activity
S_service ~~ S_activity
S_environment ~~ O_cognition
S_service ~~ O_cognition
S_activity ~~ O_cognition
S_environment ~~ R_intent
S_service ~~ R_intent
S_activity ~~ R_intent
O_cognition ~~ R_intent
"""

MODEL_DESC_INTENT_OBS_RSUM_V1 = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089

O_cognition ~ S_environment + S_service + S_activity
R_sum ~ O_cognition + S_environment + S_service + S_activity

S_environment ~~ S_service
S_environment ~~ S_activity
S_service ~~ S_activity

# same-dimension method-effect residuals (promotion/word-of-mouth/online seeding)
C062 ~~ C063
C063 ~~ C065
C062 ~~ C065
"""

MEASUREMENT_ADJUSTMENTS = [
    "C062 ~~ C063",
    "C063 ~~ C065",
    "C062 ~~ C065",
]

S_SERVICE_COLS = ["C052", "C053", "C054", "C062", "C063", "C065"]
S_ENV_COLS = ["C058", "C059", "C060", "C061"]
S_ACTIVITY_COLS = ["C055", "C056", "C057"]
O_COLS = ["C086", "C087", "C088", "C089"]
INDICATOR_COLS = S_SERVICE_COLS + S_ENV_COLS + S_ACTIVITY_COLS + O_COLS
MODEL_COLS = INDICATOR_COLS + ["R_visit", "R_recommend"]
PRIMARY_COLUMNS = [f"C{i:03d}" for i in range(1, 109)]

CFA_LOADING_ORDER: dict[str, list[str]] = {
    "S_service": S_SERVICE_COLS,
    "S_environment": S_ENV_COLS,
    "S_activity": S_ACTIVITY_COLS,
    "O_cognition": O_COLS,
    "R_intent": ["R_visit", "R_recommend"],
}

LEGACY_DIRECT_DEFS: list[tuple[str, str, str, str]] = [
    ("H1", "S_env -> O_cognition", "O_cognition", "S_environment"),
    ("H2", "S_service -> O_cognition", "O_cognition", "S_service"),
    ("H3", "S_activity -> O_cognition", "O_cognition", "S_activity"),
    ("H4", "O_cognition -> R_visit", "R_visit", "O_cognition"),
    ("H5", "O_cognition -> R_recommend", "R_recommend", "O_cognition"),
]

LEGACY_INDIRECT_DEFS: list[
    tuple[str, str, tuple[str, str], tuple[str, str]]
] = [
    ("H6", "S_env -> O -> R", ("O_cognition", "S_environment"), ("R_visit", "O_cognition")),
    ("H7", "S_service -> O -> R", ("O_cognition", "S_service"), ("R_recommend", "O_cognition")),
    ("H8", "S_activity -> O -> R", ("O_cognition", "S_activity"), ("R_recommend", "O_cognition")),
]

NEW_DIRECT_DEFS: list[tuple[str, str, str, str]] = [
    ("NH1", "S_env -> O_cognition", "O_cognition", "S_environment"),
    ("NH2", "S_service -> O_cognition", "O_cognition", "S_service"),
    ("NH3", "S_activity -> O_cognition", "O_cognition", "S_activity"),
    ("NH4", "O_cognition -> R_intent", "R_intent", "O_cognition"),
    ("NH5", "S_env -> R_intent", "R_intent", "S_environment"),
    ("NH6", "S_service -> R_intent", "R_intent", "S_service"),
    ("NH7", "S_activity -> R_intent", "R_intent", "S_activity"),
]

NEW_INDIRECT_DEFS: list[
    tuple[str, str, tuple[str, str], tuple[str, str]]
] = [
    ("NH8", "S_env -> O -> R_intent", ("O_cognition", "S_environment"), ("R_intent", "O_cognition")),
    ("NH9", "S_service -> O -> R_intent", ("O_cognition", "S_service"), ("R_intent", "O_cognition")),
    ("NH10", "S_activity -> O -> R_intent", ("O_cognition", "S_activity"), ("R_intent", "O_cognition")),
]

REDUCED_DIRECT_DEFS: list[tuple[str, str, str, str]] = [
    ("RH1", "S_env -> O_cognition", "O_cognition", "S_environment"),
    ("RH2", "S_service -> O_cognition", "O_cognition", "S_service"),
    ("RH3", "S_activity -> O_cognition", "O_cognition", "S_activity"),
    ("RH4", "O_cognition -> R_sum", "R_sum", "O_cognition"),
    ("RH5", "S_env -> R_sum", "R_sum", "S_environment"),
    ("RH6", "S_service -> R_sum", "R_sum", "S_service"),
    ("RH7", "S_activity -> R_sum", "R_sum", "S_activity"),
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "baseline_current": {
        "model_desc": MODEL_DESC_BASELINE_CURRENT,
        "direct_defs": [],
        "indirect_defs": [],
        "display_name": "baseline_current",
    },
    "legacy_compat_v2": {
        "model_desc": MODEL_DESC_LEGACY_COMPAT_V2,
        "direct_defs": LEGACY_DIRECT_DEFS,
        "indirect_defs": LEGACY_INDIRECT_DEFS,
        "display_name": "legacy_compat_v2",
    },
    "intent_partial_v1": {
        "model_desc": MODEL_DESC_INTENT_PARTIAL_V1,
        "direct_defs": NEW_DIRECT_DEFS,
        "indirect_defs": NEW_INDIRECT_DEFS,
        "display_name": "intent_partial_v1",
    },
    "cfa_intent_with_method": {
        "model_desc": MODEL_DESC_CFA_INTENT_WITH_METHOD,
        "direct_defs": [],
        "indirect_defs": [],
        "display_name": "cfa_intent_with_method",
    },
    "cfa_intent_no_method": {
        "model_desc": MODEL_DESC_CFA_INTENT_NO_METHOD,
        "direct_defs": [],
        "indirect_defs": [],
        "display_name": "cfa_intent_no_method",
    },
    "intent_obs_rsum_v1": {
        "model_desc": MODEL_DESC_INTENT_OBS_RSUM_V1,
        "direct_defs": REDUCED_DIRECT_DEFS,
        "indirect_defs": [],
        "display_name": "intent_obs_rsum_v1",
    },
}

FIT_STANDARDS = {
    "national_common": {
        "CMIN/DF": ("lt", 3.0),
        "RMSEA": ("lt", 0.08),
        "CFI": ("gt", 0.90),
        "TLI": ("gt", 0.90),
        "SRMR": ("lt", 0.08),
    }
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEM and fill table 7-4 / 7-5.")
    parser.add_argument(
        "--model-suite",
        default="legacy_only",
        choices=["legacy_only", "dual"],
        help="Model suite: legacy_only(compat outputs) or dual(compat + new model outputs).",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Input SEM file path (.xlsx or .csv). Default: data/问卷数据_cleaned_for_SEM.xlsx",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Legacy alias for csv input. Ignored when --input-file is provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: data/data_SEM",
    )
    parser.add_argument(
        "--output-tables-dir",
        default=None,
        help="Legacy alias for --output-dir.",
    )
    parser.add_argument(
        "--run-robustness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run robustness SEM on attention_pass_eq1==1 subsample.",
    )
    parser.add_argument(
        "--bootstrap-n",
        default=2000,
        type=int,
        help="Bootstrap iterations for mediation effects.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--mapping",
        default="data_driven_v1",
        choices=["data_driven_v1"],
        help="Variable mapping profile.",
    )
    parser.add_argument(
        "--fit-standard",
        default="national_common",
        choices=["national_common"],
        help="Fit metric threshold profile.",
    )
    parser.add_argument(
        "--audit-json",
        default=None,
        help="Path to SEM audit json. Defaults to <output-dir>/SEM_建模审计.json.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite target outputs.",
    )
    return parser.parse_args()


def _resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input_file:
        return Path(args.input_file)
    if args.input_csv:
        return Path(args.input_csv)
    return Path("data/问卷数据_cleaned_for_SEM.xlsx")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.output_tables_dir:
        return Path(args.output_tables_dir)
    return Path("data/data_SEM")


def _load_xlsx(path: Path) -> tuple[pd.DataFrame, str]:
    headers, rows = read_xlsx_first_sheet(path)
    raw = pd.DataFrame(rows, columns=headers)

    if all(c in raw.columns for c in PRIMARY_COLUMNS):
        return raw.copy(), "xlsx_code_columns_direct"
    if any(c in raw.columns for c in PRIMARY_COLUMNS):
        return raw.copy(), "xlsx_partial_code_columns"

    if raw.shape[1] < 108:
        raise ValueError(f"XLSX has <108 columns and cannot map by position: {path}")
    df = raw.iloc[:, :108].copy()
    df.columns = PRIMARY_COLUMNS
    for extra in ["R_visit_rev", "R_recommend_rev", "attention_pass_eq1"]:
        if extra in raw.columns:
            df[extra] = raw[extra]
    return df, "xlsx_position_1_108_to_C001_C108"


def _load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if all(c in raw.columns for c in PRIMARY_COLUMNS):
        return raw.copy(), "csv_code_columns_direct"
    if any(c in raw.columns for c in PRIMARY_COLUMNS):
        return raw.copy(), "csv_partial_code_columns"

    if raw.shape[1] < 108:
        raise ValueError(f"CSV has <108 columns and cannot map by position: {path}")
    df = raw.iloc[:, :108].copy()
    df.columns = PRIMARY_COLUMNS
    for extra in ["R_visit_rev", "R_recommend_rev", "attention_pass_eq1"]:
        if extra in raw.columns:
            df[extra] = raw[extra]
    return df, "csv_position_1_108_to_C001_C108"


def _load_source_df(input_path: Path) -> tuple[pd.DataFrame, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(input_path)
    if suffix in {".xlsx", ".xlsm"}:
        return _load_xlsx(input_path)
    raise ValueError(f"Unsupported input file suffix: {suffix}. Only .xlsx/.xlsm/.csv are supported.")


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _validate_likert(series: pd.Series, name: str) -> None:
    arr = series.to_numpy(dtype=float)
    bad = (~np.isfinite(arr)) | (arr < 1) | (arr > 5) | (np.abs(arr - np.round(arr)) > 1e-6)
    if np.any(bad):
        bad_values = sorted({str(v) for v in series[bad].head(8).tolist()})
        raise ValueError(f"Likert range check failed for {name}. Examples: {bad_values}")


def _validate_binary(series: pd.Series, name: str) -> None:
    arr = series.to_numpy(dtype=float)
    bad = (~np.isfinite(arr)) | ((arr != 0) & (arr != 1))
    if np.any(bad):
        bad_values = sorted({str(v) for v in series[bad].head(8).tolist()})
        raise ValueError(f"Binary check failed for {name}. Examples: {bad_values}")


def load_sem_input(input_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    src, mapping_strategy = _load_source_df(input_path)

    missing_indicators = [c for c in INDICATOR_COLS if c not in src.columns]
    if missing_indicators:
        raise ValueError(f"Missing required SEM indicator columns: {missing_indicators}")

    out = pd.DataFrame(index=src.index)
    for c in INDICATOR_COLS:
        out[c] = _numeric(src, c)
        if out[c].isna().any():
            miss = int(out[c].isna().sum())
            raise ValueError(f"SEM indicator has missing/non-numeric values: {c} (n={miss})")
        _validate_likert(out[c], c)

    reverse_source = ""
    if "R_visit_rev" in src.columns and "R_recommend_rev" in src.columns:
        r_visit = _numeric(src, "R_visit_rev")
        r_recommend = _numeric(src, "R_recommend_rev")
        if r_visit.isna().any() or r_recommend.isna().any():
            raise ValueError("R_visit_rev/R_recommend_rev contain missing or non-numeric values.")
        reverse_source = "provided_R_visit_rev_R_recommend_rev"
    else:
        for c in ["C090", "C091"]:
            if c not in src.columns:
                raise ValueError("Missing C090/C091 and no R_visit_rev/R_recommend_rev provided.")
        c090 = _numeric(src, "C090")
        c091 = _numeric(src, "C091")
        if c090.isna().any() or c091.isna().any():
            raise ValueError("C090/C091 contain missing or non-numeric values.")
        _validate_likert(c090, "C090")
        _validate_likert(c091, "C091")
        r_visit = 6.0 - c090
        r_recommend = 6.0 - c091
        reverse_source = "derived_6_minus_C090_C091"

    out["R_visit"] = r_visit
    out["R_recommend"] = r_recommend
    _validate_likert(out["R_visit"], "R_visit")
    _validate_likert(out["R_recommend"], "R_recommend")

    c064: pd.Series | None = None
    if "C064" in src.columns:
        c064 = _numeric(src, "C064")
        if c064.isna().any():
            raise ValueError("C064 contains missing or non-numeric values.")
        _validate_likert(c064, "C064")

    if "attention_pass_eq1" in src.columns:
        attention = _numeric(src, "attention_pass_eq1")
        if attention.isna().any():
            raise ValueError("attention_pass_eq1 contains missing or non-numeric values.")
        _validate_binary(attention, "attention_pass_eq1")
        attention_source = "provided_attention_pass_eq1"
    else:
        if c064 is None:
            raise ValueError("Missing attention_pass_eq1 and C064; cannot derive robustness subset.")
        attention = (c064 == 1).astype(int)
        attention_source = "derived_C064_eq_1"
    out["attention_pass_eq1"] = attention.astype(int)
    _validate_binary(out["attention_pass_eq1"], "attention_pass_eq1")

    attention_eq5_source = None
    if c064 is not None:
        out["attention_pass_eq5"] = (c064 == 5).astype(int)
        _validate_binary(out["attention_pass_eq5"], "attention_pass_eq5")
        attention_eq5_source = "derived_C064_eq_5"

    if out[MODEL_COLS].isna().any().any():
        miss_n = int(out[MODEL_COLS].isna().sum().sum())
        raise ValueError(f"SEM input has missing values after processing: {miss_n}")

    meta = {
        "input_path": str(input_path),
        "input_suffix": input_path.suffix.lower(),
        "mapping_strategy": mapping_strategy,
        "reverse_scoring_source": reverse_source,
        "attention_source": attention_source,
        "attention_eq5_source": attention_eq5_source,
    }
    return out.reset_index(drop=True), meta


def fit_sem(df: pd.DataFrame, model_desc: str, retries: int = 1) -> tuple[Model, pd.DataFrame, str]:
    errors: list[str] = []
    solvers = ["SLSQP", "L-BFGS-B"]
    total_tries = max(1, retries + 1)
    for i in range(total_tries):
        solver = solvers[i % len(solvers)]
        try:
            model = Model(model_desc)
            model.fit(df, solver=solver)
            stats = calc_stats(model)
            if "Value" not in stats.index:
                raise RuntimeError("calc_stats output missing 'Value' row.")
            return model, stats, solver
        except Exception as e:
            errors.append(f"solver={solver}: {e}")
    raise RuntimeError("SEM fit failed after retries: " + " | ".join(errors))


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    if np.any(d <= 0) or np.any(~np.isfinite(d)):
        raise RuntimeError("Non-positive or invalid variances while converting covariance to correlation.")
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_srmr(model: Model, df: pd.DataFrame) -> float:
    obs = list(model.vars.get("observed", []))
    if not obs:
        raise RuntimeError("SEM observed-variable order unavailable (model.vars['observed']).")
    sample = df[obs].to_numpy(dtype=float)
    sample_cov = np.cov(sample.T, ddof=1)
    implied_cov = model.calc_sigma()[0]
    if implied_cov.shape != sample_cov.shape:
        raise RuntimeError(
            f"Covariance shape mismatch for SRMR: sample={sample_cov.shape}, implied={implied_cov.shape}"
        )
    s = cov_to_corr(sample_cov)
    s_hat = cov_to_corr(implied_cov)
    resid = s - s_hat
    mask = np.tril(np.ones_like(resid, dtype=bool), k=-1)
    vals = resid[mask]
    if vals.size == 0:
        raise RuntimeError("SRMR residual set is empty.")
    srmr = float(np.sqrt(np.mean(np.square(vals))))
    if not np.isfinite(srmr):
        raise RuntimeError("SRMR is non-finite.")
    return srmr


def _get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(c).lower().replace(" ", ""): str(c) for c in df.columns}
    for c in candidates:
        key = c.lower().replace(" ", "")
        if key in lookup:
            return lookup[key]
    for k, v in lookup.items():
        if any(c.lower().replace(" ", "") in k for c in candidates):
            return v
    raise KeyError(f"Missing expected column, candidates={candidates}, actual={list(df.columns)}")


def _to_float(v: Any, default: float = math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def extract_path(ins_df: pd.DataFrame, lval: str, rval: str) -> tuple[float, float]:
    std_col = _get_col(ins_df, ["Est. Std", "Std.Est", "Std. Est", "Est.Std"])
    p_col = _get_col(ins_df, ["p-value", "pvalue", "P-value", "PValue"])
    rows = ins_df[(ins_df["op"] == "~") & (ins_df["lval"] == lval) & (ins_df["rval"] == rval)]
    if rows.empty:
        raise KeyError(f"Path not found in inspect output: {lval} ~ {rval}")
    r = rows.iloc[0]
    beta = _to_float(r.get(std_col))
    pval = _to_float(r.get(p_col))
    if not np.isfinite(beta):
        raise RuntimeError(f"Invalid standardized beta for {lval}~{rval}: {r.get(std_col)}")
    if not np.isfinite(pval):
        pval = 1.0
    return beta, pval


def extract_path_detail(ins_df: pd.DataFrame, lval: str, rval: str) -> dict[str, float]:
    est_col = _get_col(ins_df, ["Estimate"])
    std_col = _get_col(ins_df, ["Est. Std", "Std.Est", "Std. Est", "Est.Std"])
    se_col = _get_col(ins_df, ["Std. Err", "Std.Err", "StdErr"])
    p_col = _get_col(ins_df, ["p-value", "pvalue", "P-value", "PValue"])
    rows = ins_df[(ins_df["op"] == "~") & (ins_df["lval"] == lval) & (ins_df["rval"] == rval)]
    if rows.empty:
        raise KeyError(f"Path not found in inspect output: {lval} ~ {rval}")
    r = rows.iloc[0]
    estimate = _to_float(r.get(est_col))
    est_std = _to_float(r.get(std_col))
    std_err = _to_float(r.get(se_col))
    pval = _to_float(r.get(p_col), default=1.0)
    if not np.isfinite(estimate):
        raise RuntimeError(f"Invalid estimate for {lval}~{rval}: {r.get(est_col)}")
    if not np.isfinite(est_std):
        raise RuntimeError(f"Invalid standardized estimate for {lval}~{rval}: {r.get(std_col)}")
    if not np.isfinite(std_err):
        std_err = math.nan
    if not np.isfinite(pval):
        pval = 1.0
    return {
        "estimate": float(estimate),
        "est_std": float(est_std),
        "std_err": float(std_err),
        "p": float(pval),
    }


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fit_metrics(stats: pd.DataFrame, srmr: float) -> dict[str, float]:
    row = stats.loc["Value"]
    dof = _to_float(row["DoF"])
    chi2 = _to_float(row["chi2"])
    rmsea = _to_float(row["RMSEA"])
    cfi = _to_float(row["CFI"])
    tli = _to_float(row["TLI"])
    if not np.isfinite(dof) or dof <= 0:
        raise RuntimeError(f"Invalid DoF for CMIN/DF: {dof}")
    cmin_df = chi2 / dof
    out = {
        "CMIN/DF": float(cmin_df),
        "RMSEA": float(rmsea),
        "CFI": float(cfi),
        "TLI": float(tli),
        "SRMR": float(srmr),
    }
    for k, v in out.items():
        if not np.isfinite(v):
            raise RuntimeError(f"Invalid fit metric {k}: {v}")
    return out


def _fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def metric_pass(metric: str, value: float, fit_standard: str) -> bool:
    op, threshold = FIT_STANDARDS[fit_standard][metric]
    if op == "lt":
        return value < threshold
    return value > threshold


def _threshold_conclusion(metric: str, value: float, fit_standard: str) -> tuple[str, str]:
    op, threshold = FIT_STANDARDS[fit_standard][metric]
    ok = metric_pass(metric, value, fit_standard)
    sign = "<" if op == "lt" else ">"
    return ("达标" if ok else "未达标", f"阈值：{sign}{threshold:.2f}（{fit_standard}）")


def _metric_checks(metrics: dict[str, float], fit_standard: str) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for metric in ["CMIN/DF", "RMSEA", "CFI", "TLI", "SRMR"]:
        op, threshold = FIT_STANDARDS[fit_standard][metric]
        checks[metric] = {
            "value": metrics[metric],
            "op": op,
            "threshold": threshold,
            "pass": metric_pass(metric, metrics[metric], fit_standard),
        }
    return checks


def write_table_74(path: Path, metrics: dict[str, float], solver: str, fit_standard: str) -> None:
    rows: list[dict[str, str]] = []
    for metric in ["CMIN/DF", "RMSEA", "CFI", "TLI", "SRMR"]:
        val = metrics[metric]
        conclusion, th = _threshold_conclusion(metric, val, fit_standard)
        rows.append(
            {
                "指标": metric,
                "值": _fmt(val, 4),
                "结论": conclusion,
                "状态": "ready",
                "备注": f"{th}; optimizer={solver}",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["指标", "值", "结论", "状态", "备注"])
        w.writeheader()
        w.writerows(rows)


def _weighted_metric(rows: list[tuple[int, dict[str, float]]], key: str) -> float:
    total_n = sum(max(0, int(n)) for n, _ in rows)
    if total_n <= 0:
        raise ValueError("Cannot compute weighted metric with empty groups.")
    num = sum(float(n) * float(m[key]) for n, m in rows)
    return float(num / total_n)


def _extract_cfa_loadings(ins_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    est_col = _get_col(ins_df, ["Estimate"])
    out: dict[tuple[str, str], float] = {}
    for factor, indicators in CFA_LOADING_ORDER.items():
        for indicator in indicators:
            rows = ins_df[(ins_df["op"] == "~") & (ins_df["lval"] == indicator) & (ins_df["rval"] == factor)]
            if rows.empty:
                raise KeyError(f"Loading not found: {factor} =~ {indicator}")
            est = _to_float(rows.iloc[0].get(est_col))
            if not np.isfinite(est):
                raise RuntimeError(f"Invalid loading estimate: {factor} =~ {indicator}")
            out[(factor, indicator)] = float(est)
    return out


def build_metric_fixed_cfa_desc(loadings: dict[tuple[str, str], float]) -> str:
    lines: list[str] = []
    for factor, indicators in CFA_LOADING_ORDER.items():
        terms = []
        for indicator in indicators:
            key = (factor, indicator)
            if key not in loadings:
                raise KeyError(f"Missing loading key: {key}")
            terms.append(f"{loadings[key]:.8f}*{indicator}")
        lines.append(f"{factor} =~ " + " + ".join(terms))
    lines.extend(
        [
            "",
            "S_environment ~~ S_service",
            "S_environment ~~ S_activity",
            "S_service ~~ S_activity",
            "S_environment ~~ O_cognition",
            "S_service ~~ O_cognition",
            "S_activity ~~ O_cognition",
            "S_environment ~~ R_intent",
            "S_service ~~ R_intent",
            "S_activity ~~ R_intent",
            "O_cognition ~~ R_intent",
            "",
            "C062 ~~ C063",
            "C063 ~~ C065",
            "C062 ~~ C065",
        ]
    )
    return "\n".join(lines)


def _path_key(lval: str, rval: str) -> str:
    return f"{lval}~{rval}"


def _bootstrap_indirect(
    df: pd.DataFrame,
    model_desc: str,
    indirect_defs: list[tuple[str, str, tuple[str, str], tuple[str, str]]],
    bootstrap_n: int,
    seed: int,
    min_success: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    rng = np.random.default_rng(seed)
    n = len(df)
    effects: dict[str, list[float]] = {hid: [] for hid, _, _, _ in indirect_defs}
    success = 0
    fail = 0

    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            model_b, _, _ = fit_sem(d, model_desc=model_desc, retries=1)
            ins_b = inspect(model_b, std_est=True)
            for hid, _, a_path, b_path in indirect_defs:
                a_beta, _ = extract_path(ins_b, a_path[0], a_path[1])
                b_beta, _ = extract_path(ins_b, b_path[0], b_path[1])
                effects[hid].append(a_beta * b_beta)
            success += 1
        except Exception:
            fail += 1

    if success < min_success:
        raise RuntimeError(
            f"Bootstrap success too low: success={success}, fail={fail}, required>={min_success}"
        )

    arr = {k: np.asarray(v, dtype=float) for k, v in effects.items()}
    return arr, success, fail


def bootstrap_summary(dist: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p_boot = 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return float(lo), float(hi), float(min(1.0, p_boot))


def run_one_model(
    model_id: str,
    model_desc: str,
    direct_defs: list[tuple[str, str, str, str]],
    indirect_defs: list[tuple[str, str, tuple[str, str], tuple[str, str]]],
    df: pd.DataFrame,
    fit_standard: str,
    bootstrap_n: int,
    seed: int,
    run_bootstrap: bool,
) -> dict[str, Any]:
    model, stats, solver = fit_sem(df, model_desc=model_desc, retries=1)
    ins = inspect(model, std_est=True)
    srmr = compute_srmr(model, df)
    metrics = fit_metrics(stats, srmr)
    metric_checks = _metric_checks(metrics, fit_standard)
    fit_all_pass = all(bool(v["pass"]) for v in metric_checks.values())

    required_paths: set[tuple[str, str]] = {(lval, rval) for _, _, lval, rval in direct_defs}
    for _, _, a_path, b_path in indirect_defs:
        required_paths.add(a_path)
        required_paths.add(b_path)

    all_paths: dict[str, tuple[float, float]] = {}
    for lval, rval in sorted(required_paths):
        all_paths[_path_key(lval, rval)] = extract_path(ins, lval, rval)

    direct: dict[str, tuple[float, float]] = {}
    for _, _, lval, rval in direct_defs:
        direct[_path_key(lval, rval)] = all_paths[_path_key(lval, rval)]

    indirect_point: dict[str, float] = {}
    for hid, _, a_path, b_path in indirect_defs:
        indirect_point[hid] = (
            all_paths[_path_key(a_path[0], a_path[1])][0]
            * all_paths[_path_key(b_path[0], b_path[1])][0]
        )

    bootstrap_payload: dict[str, Any] = {
        "n_requested": int(bootstrap_n) if run_bootstrap else 0,
        "min_success_required": int(math.ceil(bootstrap_n * 0.8)) if run_bootstrap else 0,
        "success": None,
        "fail": None,
        "executed": bool(run_bootstrap and bool(indirect_defs)),
    }
    indirect_dist: dict[str, np.ndarray] = {}
    if run_bootstrap and indirect_defs:
        min_success = int(math.ceil(bootstrap_n * 0.8))
        dist, success_n, fail_n = _bootstrap_indirect(
            df=df,
            model_desc=model_desc,
            indirect_defs=indirect_defs,
            bootstrap_n=bootstrap_n,
            seed=seed,
            min_success=min_success,
        )
        bootstrap_payload["success"] = int(success_n)
        bootstrap_payload["fail"] = int(fail_n)
        indirect_dist = dist

    return {
        "model_id": model_id,
        "solver": solver,
        "inspect": ins,
        "metrics": metrics,
        "metric_checks": metric_checks,
        "fit_all_pass": fit_all_pass,
        "direct": direct,
        "indirect_point": indirect_point,
        "indirect_dist": indirect_dist,
        "bootstrap": bootstrap_payload,
    }


def write_table_paths(
    path: Path,
    direct_defs: list[tuple[str, str, str, str]],
    direct_results: dict[str, tuple[float, float]],
    indirect_defs: list[tuple[str, str, tuple[str, str], tuple[str, str]]],
    indirect_point: dict[str, float],
    indirect_dist: dict[str, np.ndarray],
    bootstrap: dict[str, Any],
    direct_note: str,
) -> None:
    rows: list[dict[str, str]] = []

    for hid, label, lval, rval in direct_defs:
        beta, pval = direct_results[_path_key(lval, rval)]
        supported = (pval < 0.05) and (beta > 0)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "标准化系数β": _fmt(beta, 4),
                "p值": _fmt(pval, 4),
                "结论": "支持" if supported else "不支持",
                "状态": "ready",
                "备注": direct_note,
            }
        )

    for hid, label, _, _ in indirect_defs:
        if hid not in indirect_dist:
            rows.append(
                {
                    "假设": hid,
                    "路径": label,
                    "标准化系数β": f"indirect={_fmt(indirect_point.get(hid, math.nan), 4)}",
                    "p值": "p_boot=NA;95%CI=[NA,NA]",
                    "结论": "中介待补",
                    "状态": "pending",
                    "备注": "bootstrap未执行",
                }
            )
            continue
        lo, hi, p_boot = bootstrap_summary(indirect_dist[hid])
        sig = (lo > 0) or (hi < 0)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "标准化系数β": f"indirect={_fmt(indirect_point[hid], 4)}",
                "p值": f"p_boot={_fmt(p_boot,4)};95%CI=[{_fmt(lo,4)},{_fmt(hi,4)}]",
                "结论": "中介显著" if sig else "中介不显著",
                "状态": "ready",
                "备注": (
                    f"bootstrap={bootstrap.get('n_requested', 0)};"
                    f"success={bootstrap.get('success')};fail={bootstrap.get('fail')}"
                ),
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["假设", "路径", "标准化系数β", "p值", "结论", "状态", "备注"])
        w.writeheader()
        w.writerows(rows)


def write_table_direct_only(
    path: Path,
    direct_defs: list[tuple[str, str, str, str]],
    direct_results: dict[str, tuple[float, float]],
    note: str,
) -> None:
    rows: list[dict[str, str]] = []
    for hid, label, lval, rval in direct_defs:
        beta, pval = direct_results[_path_key(lval, rval)]
        supported = (pval < 0.05) and (beta > 0)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "标准化系数β": _fmt(beta, 4),
                "p值": _fmt(pval, 4),
                "结论": "支持" if supported else "不支持",
                "状态": "ready",
                "备注": note,
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["假设", "路径", "标准化系数β", "p值", "结论", "状态", "备注"])
        w.writeheader()
        w.writerows(rows)


def _sign_name(v: float) -> str:
    if v > 0:
        return "正向"
    if v < 0:
        return "负向"
    return "零"


def write_robustness_compare(
    path: Path,
    direct_defs: list[tuple[str, str, str, str]],
    direct_main: dict[str, tuple[float, float]],
    direct_attention: dict[str, tuple[float, float]],
) -> None:
    rows: list[dict[str, str]] = []
    for hid, label, lval, rval in direct_defs:
        key = _path_key(lval, rval)
        main_beta, main_p = direct_main[key]
        att_beta, att_p = direct_attention[key]
        main_sign = _sign_name(main_beta)
        att_sign = _sign_name(att_beta)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "主样本β": _fmt(main_beta, 4),
                "主样本p值": _fmt(main_p, 4),
                "稳健样本β": _fmt(att_beta, 4),
                "稳健样本p值": _fmt(att_p, 4),
                "主样本方向": main_sign,
                "稳健样本方向": att_sign,
                "方向一致性": str(main_sign == att_sign),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "假设",
                "路径",
                "主样本β",
                "主样本p值",
                "稳健样本β",
                "稳健样本p值",
                "主样本方向",
                "稳健样本方向",
                "方向一致性",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_model_compare(path: Path, rows: list[dict[str, Any]]) -> None:
    out_rows = []
    for r in rows:
        out_rows.append(
            {
                "模型ID": str(r["model_id"]),
                "模型说明": str(r["model_name"]),
                "CMIN/DF": _fmt(float(r["metrics"]["CMIN/DF"]), 4),
                "RMSEA": _fmt(float(r["metrics"]["RMSEA"]), 4),
                "CFI": _fmt(float(r["metrics"]["CFI"]), 4),
                "TLI": _fmt(float(r["metrics"]["TLI"]), 4),
                "SRMR": _fmt(float(r["metrics"]["SRMR"]), 4),
                "全部达标": str(bool(r["fit_all_pass"])),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["模型ID", "模型说明", "CMIN/DF", "RMSEA", "CFI", "TLI", "SRMR", "全部达标"],
        )
        w.writeheader()
        w.writerows(out_rows)


def write_cfa_method_compare(
    path: Path,
    with_method: dict[str, Any],
    without_method: dict[str, Any],
) -> None:
    delta_cfi = float(without_method["metrics"]["CFI"]) - float(with_method["metrics"]["CFI"])
    delta_rmsea = float(without_method["metrics"]["RMSEA"]) - float(with_method["metrics"]["RMSEA"])
    rows = [
        {
            "模型": "CFA_含残差相关",
            "CMIN/DF": _fmt(with_method["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(with_method["metrics"]["RMSEA"], 4),
            "CFI": _fmt(with_method["metrics"]["CFI"], 4),
            "TLI": _fmt(with_method["metrics"]["TLI"], 4),
            "SRMR": _fmt(with_method["metrics"]["SRMR"], 4),
            "全部达标": str(bool(with_method["fit_all_pass"])),
            "ΔCFI_相对含残差": _fmt(0.0, 4),
            "ΔRMSEA_相对含残差": _fmt(0.0, 4),
            "备注": "用于证明同方法效应已被测量层吸收",
        },
        {
            "模型": "CFA_去残差相关",
            "CMIN/DF": _fmt(without_method["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(without_method["metrics"]["RMSEA"], 4),
            "CFI": _fmt(without_method["metrics"]["CFI"], 4),
            "TLI": _fmt(without_method["metrics"]["TLI"], 4),
            "SRMR": _fmt(without_method["metrics"]["SRMR"], 4),
            "全部达标": str(bool(without_method["fit_all_pass"])),
            "ΔCFI_相对含残差": _fmt(delta_cfi, 4),
            "ΔRMSEA_相对含残差": _fmt(delta_rmsea, 4),
            "备注": "若CFI下降/RMSEA上升，支持残差相关为方法效应而非结构后验调参",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "模型",
                "CMIN/DF",
                "RMSEA",
                "CFI",
                "TLI",
                "SRMR",
                "全部达标",
                "ΔCFI_相对含残差",
                "ΔRMSEA_相对含残差",
                "备注",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_multigroup_cfa_compare(
    path: Path,
    n_pass: int,
    n_fail: int,
    config_pass: dict[str, Any],
    config_fail: dict[str, Any],
    metric_pass: dict[str, Any],
    metric_fail: dict[str, Any],
    cfi_configural_weighted: float,
    cfi_metric_weighted: float,
    delta_cfi: float,
) -> None:
    rows = [
        {
            "层级": "Configural",
            "分组": "attention_pass_eq1==1",
            "n": str(n_pass),
            "CMIN/DF": _fmt(config_pass["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(config_pass["metrics"]["RMSEA"], 4),
            "CFI": _fmt(config_pass["metrics"]["CFI"], 4),
            "TLI": _fmt(config_pass["metrics"]["TLI"], 4),
            "SRMR": _fmt(config_pass["metrics"]["SRMR"], 4),
            "全部达标": str(bool(config_pass["fit_all_pass"])),
            "备注": "同一因子结构，自由载荷",
        },
        {
            "层级": "Configural",
            "分组": "attention_pass_eq1==0",
            "n": str(n_fail),
            "CMIN/DF": _fmt(config_fail["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(config_fail["metrics"]["RMSEA"], 4),
            "CFI": _fmt(config_fail["metrics"]["CFI"], 4),
            "TLI": _fmt(config_fail["metrics"]["TLI"], 4),
            "SRMR": _fmt(config_fail["metrics"]["SRMR"], 4),
            "全部达标": str(bool(config_fail["fit_all_pass"])),
            "备注": "同一因子结构，自由载荷",
        },
        {
            "层级": "Metric(近似)",
            "分组": "attention_pass_eq1==1",
            "n": str(n_pass),
            "CMIN/DF": _fmt(metric_pass["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(metric_pass["metrics"]["RMSEA"], 4),
            "CFI": _fmt(metric_pass["metrics"]["CFI"], 4),
            "TLI": _fmt(metric_pass["metrics"]["TLI"], 4),
            "SRMR": _fmt(metric_pass["metrics"]["SRMR"], 4),
            "全部达标": str(bool(metric_pass["fit_all_pass"])),
            "备注": "载荷固定为共同参照值",
        },
        {
            "层级": "Metric(近似)",
            "分组": "attention_pass_eq1==0",
            "n": str(n_fail),
            "CMIN/DF": _fmt(metric_fail["metrics"]["CMIN/DF"], 4),
            "RMSEA": _fmt(metric_fail["metrics"]["RMSEA"], 4),
            "CFI": _fmt(metric_fail["metrics"]["CFI"], 4),
            "TLI": _fmt(metric_fail["metrics"]["TLI"], 4),
            "SRMR": _fmt(metric_fail["metrics"]["SRMR"], 4),
            "全部达标": str(bool(metric_fail["fit_all_pass"])),
            "备注": "载荷固定为共同参照值",
        },
        {
            "层级": "Summary",
            "分组": "weighted",
            "n": str(n_pass + n_fail),
            "CMIN/DF": "",
            "RMSEA": "",
            "CFI": _fmt(cfi_metric_weighted, 4),
            "TLI": "",
            "SRMR": "",
            "全部达标": str(bool(delta_cfi >= -0.01)),
            "备注": (
                f"weighted CFI: configural={_fmt(cfi_configural_weighted,4)}, "
                f"metric={_fmt(cfi_metric_weighted,4)}, ΔCFI(metric-configural)={_fmt(delta_cfi,4)}"
            ),
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["层级", "分组", "n", "CMIN/DF", "RMSEA", "CFI", "TLI", "SRMR", "全部达标", "备注"],
        )
        w.writeheader()
        w.writerows(rows)


def write_nh4_multigroup_test(
    path: Path,
    n_main: int,
    n_pass: int,
    n_fail: int,
    main_detail: dict[str, float],
    pass_detail: dict[str, float],
    fail_detail: dict[str, float],
    z_wald: float,
    p_wald: float,
) -> None:
    rows = [
        {
            "对比": "main",
            "n": str(n_main),
            "路径": "NH4: O_cognition -> R_intent",
            "标准化系数β": _fmt(main_detail["est_std"], 4),
            "非标准化系数": _fmt(main_detail["estimate"], 4),
            "Std.Err": _fmt(main_detail["std_err"], 4) if np.isfinite(main_detail["std_err"]) else "NA",
            "p值": _fmt(main_detail["p"], 4),
            "结论": "基准主样本",
            "备注": "用于正文主结果",
        },
        {
            "对比": "attention_pass_eq1==1",
            "n": str(n_pass),
            "路径": "NH4: O_cognition -> R_intent",
            "标准化系数β": _fmt(pass_detail["est_std"], 4),
            "非标准化系数": _fmt(pass_detail["estimate"], 4),
            "Std.Err": _fmt(pass_detail["std_err"], 4) if np.isfinite(pass_detail["std_err"]) else "NA",
            "p值": _fmt(pass_detail["p"], 4),
            "结论": "稳健性子样本",
            "备注": "子样本直接估计",
        },
        {
            "对比": "attention_pass_eq1==0",
            "n": str(n_fail),
            "路径": "NH4: O_cognition -> R_intent",
            "标准化系数β": _fmt(fail_detail["est_std"], 4),
            "非标准化系数": _fmt(fail_detail["estimate"], 4),
            "Std.Err": _fmt(fail_detail["std_err"], 4) if np.isfinite(fail_detail["std_err"]) else "NA",
            "p值": _fmt(fail_detail["p"], 4),
            "结论": "非通过组",
            "备注": "与通过组构成互斥两组",
        },
        {
            "对比": "Wald(pass_vs_fail)",
            "n": str(n_pass + n_fail),
            "路径": "NH4: O_cognition -> R_intent",
            "标准化系数β": "",
            "非标准化系数": "",
            "Std.Err": "",
            "p值": _fmt(p_wald, 4) if np.isfinite(p_wald) else "NA",
            "结论": "组间差异显著" if (np.isfinite(p_wald) and p_wald < 0.05) else "组间差异不显著",
            "备注": _fmt(z_wald, 4) if np.isfinite(z_wald) else "Wald统计量不可计算",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["对比", "n", "路径", "标准化系数β", "非标准化系数", "Std.Err", "p值", "结论", "备注"],
        )
        w.writeheader()
        w.writerows(rows)


def write_attention_eq5_sensitivity(
    path: Path,
    n_main: int,
    main_detail: dict[str, float] | None,
    n_eq1: int | None,
    eq1_detail: dict[str, float] | None,
    n_eq5: int | None,
    eq5_detail: dict[str, float] | None,
) -> None:
    def _row(rule: str, n: int | None, detail: dict[str, float] | None, note: str) -> dict[str, str]:
        n_val = int(n) if n is not None else 0
        ratio = (n_val / n_main) if n_main > 0 else 0.0
        if detail is None:
            return {
                "subset_rule": rule,
                "n": str(n_val),
                "占主样本比例": _fmt(ratio, 4),
                "NH4_标准化系数β": "NA",
                "NH4_p值": "NA",
                "方向": "NA",
                "备注": note,
            }
        beta = detail["est_std"]
        direction = _sign_name(beta)
        return {
            "subset_rule": rule,
            "n": str(n_val),
            "占主样本比例": _fmt(ratio, 4),
            "NH4_标准化系数β": _fmt(beta, 4),
            "NH4_p值": _fmt(detail["p"], 4),
            "方向": direction,
            "备注": note,
        }

    rows = [
        _row("main(all)", n_main, main_detail, "主样本基准"),
        _row("attention_pass_eq1==1", n_eq1, eq1_detail, "稳健性主口径"),
        _row("attention_pass_eq5==1", n_eq5, eq5_detail, "敏感性一行检查"),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["subset_rule", "n", "占主样本比例", "NH4_标准化系数β", "NH4_p值", "方向", "备注"],
        )
        w.writeheader()
        w.writerows(rows)


def _model_audit_block(
    model_name: str,
    res: dict[str, Any] | None,
    direct_defs: list[tuple[str, str, str, str]],
    indirect_defs: list[tuple[str, str, tuple[str, str], tuple[str, str]]],
) -> dict[str, Any] | None:
    if res is None:
        return None
    direct_rows = []
    for hid, label, lval, rval in direct_defs:
        beta, pval = res["direct"][_path_key(lval, rval)]
        direct_rows.append(
            {
                "id": hid,
                "label": label,
                "lval": lval,
                "rval": rval,
                "beta": beta,
                "p": pval,
            }
        )
    indirect_rows = []
    for hid, label, _, _ in indirect_defs:
        row: dict[str, Any] = {"id": hid, "label": label, "point": res["indirect_point"].get(hid)}
        dist = res["indirect_dist"].get(hid)
        if dist is not None and len(dist) > 0:
            lo, hi, p_boot = bootstrap_summary(dist)
            row["p_boot"] = p_boot
            row["ci95"] = [lo, hi]
        else:
            row["p_boot"] = None
            row["ci95"] = None
        indirect_rows.append(row)
    return {
        "model_name": model_name,
        "solver": res["solver"],
        "metrics": res["metrics"],
        "metric_checks": res["metric_checks"],
        "fit_all_pass": res["fit_all_pass"],
        "bootstrap": res["bootstrap"],
        "direct_paths": direct_rows,
        "indirect_paths": indirect_rows,
    }


def main() -> None:
    args = parse_args()
    if args.mapping != "data_driven_v1":
        raise ValueError(f"Unsupported mapping: {args.mapping}")
    if args.bootstrap_n <= 0:
        raise ValueError("--bootstrap-n must be > 0.")

    input_path = _resolve_input_path(args)
    out_dir = _resolve_output_dir(args)

    out_74 = out_dir / "表7-4_SEM模型拟合指标.csv"
    out_75 = out_dir / "表7-5_SEM路径系数与显著性.csv"
    out_input_main = out_dir / "SEM_输入数据_main.csv"
    out_75_attention = out_dir / "表7-5_SEM路径系数与显著性_attention_pass.csv"
    out_input_attention = out_dir / "SEM_输入数据_attention_pass.csv"
    out_robustness = out_dir / "SEM_稳健性对比.csv"

    out_74_new = out_dir / "表7-4_SEM模型拟合指标_新模型.csv"
    out_75_new = out_dir / "表7-5_SEM路径系数与显著性_新模型.csv"
    out_robustness_new = out_dir / "SEM_稳健性对比_新模型.csv"
    out_model_compare = out_dir / "SEM_模型对比.csv"
    out_cfa_compare = out_dir / "SEM_CFA对照_残差相关.csv"
    out_multigroup_cfa = out_dir / "SEM_多组检验_CFA.csv"
    out_multigroup_nh4 = out_dir / "SEM_多组检验_NH4.csv"
    out_attention_eq5 = out_dir / "SEM_注意力敏感性_eq5.csv"
    out_74_reduced = out_dir / "表7-4_SEM模型拟合指标_降复杂模型.csv"
    out_75_reduced = out_dir / "表7-5_SEM路径系数与显著性_降复杂模型.csv"
    out_75_reduced_attention = out_dir / "表7-5_SEM路径系数与显著性_attention_pass_降复杂模型.csv"
    out_robustness_reduced = out_dir / "SEM_稳健性对比_降复杂模型.csv"

    audit_json = Path(args.audit_json) if args.audit_json else (out_dir / "SEM_建模审计.json")

    targets = [out_74, out_75, out_input_main, audit_json]
    targets.extend([out_cfa_compare, out_74_reduced, out_75_reduced])
    if args.run_robustness:
        targets.extend(
            [
                out_75_attention,
                out_input_attention,
                out_robustness,
                out_75_reduced_attention,
                out_robustness_reduced,
            ]
        )
    if args.model_suite == "dual":
        targets.extend([out_74_new, out_75_new, out_model_compare])
        if args.run_robustness:
            targets.extend([out_robustness_new, out_multigroup_cfa, out_multigroup_nh4, out_attention_eq5])
    if (not args.overwrite) and any(p.exists() for p in targets):
        raise FileExistsError("Target outputs already exist and --no-overwrite is set.")

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    sem_data, input_meta = load_sem_input(input_path)
    df_main = sem_data[MODEL_COLS].copy()
    df_main.to_csv(out_input_main, encoding="utf-8-sig", index=False)

    legacy_main = run_one_model(
        model_id="legacy_compat_v2",
        model_desc=MODEL_SPECS["legacy_compat_v2"]["model_desc"],
        direct_defs=LEGACY_DIRECT_DEFS,
        indirect_defs=LEGACY_INDIRECT_DEFS,
        df=df_main,
        fit_standard=args.fit_standard,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        run_bootstrap=True,
    )
    write_table_74(out_74, legacy_main["metrics"], legacy_main["solver"], args.fit_standard)
    write_table_paths(
        out_75,
        LEGACY_DIRECT_DEFS,
        legacy_main["direct"],
        LEGACY_INDIRECT_DEFS,
        legacy_main["indirect_point"],
        legacy_main["indirect_dist"],
        legacy_main["bootstrap"],
        direct_note="直接路径（标准化系数）",
    )

    # Step 2: lightweight CFA comparison for method-effect residual justification.
    cfa_with_method = run_one_model(
        model_id="cfa_intent_with_method",
        model_desc=MODEL_SPECS["cfa_intent_with_method"]["model_desc"],
        direct_defs=[],
        indirect_defs=[],
        df=df_main,
        fit_standard=args.fit_standard,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        run_bootstrap=False,
    )
    cfa_without_method = run_one_model(
        model_id="cfa_intent_no_method",
        model_desc=MODEL_SPECS["cfa_intent_no_method"]["model_desc"],
        direct_defs=[],
        indirect_defs=[],
        df=df_main,
        fit_standard=args.fit_standard,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        run_bootstrap=False,
    )
    write_cfa_method_compare(out_cfa_compare, cfa_with_method, cfa_without_method)

    # Step 5: reduced-complexity robustness evidence (R_sum observed composite).
    df_main_reduced = df_main.copy()
    df_main_reduced["R_sum"] = (df_main_reduced["R_visit"] + df_main_reduced["R_recommend"]) / 2.0
    reduced_main = run_one_model(
        model_id="intent_obs_rsum_v1",
        model_desc=MODEL_SPECS["intent_obs_rsum_v1"]["model_desc"],
        direct_defs=REDUCED_DIRECT_DEFS,
        indirect_defs=[],
        df=df_main_reduced,
        fit_standard=args.fit_standard,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        run_bootstrap=False,
    )
    write_table_74(out_74_reduced, reduced_main["metrics"], reduced_main["solver"], args.fit_standard)
    write_table_direct_only(
        out_75_reduced,
        REDUCED_DIRECT_DEFS,
        reduced_main["direct"],
        note="降复杂度模型（R_sum观测合成）",
    )

    robustness_payload: dict[str, Any] = {
        "enabled": bool(args.run_robustness),
        "n_rows_attention_pass": None,
        "legacy_solver_attention_pass": None,
        "legacy_metrics_attention_pass": None,
        "legacy_metric_checks_attention_pass": None,
        "legacy_fit_all_pass_attention_pass": None,
        "table_7_5_attention_pass": None,
        "input_attention_pass_csv": None,
        "robustness_compare_csv": None,
        "new_solver_attention_pass": None,
        "new_metrics_attention_pass": None,
        "new_metric_checks_attention_pass": None,
        "new_fit_all_pass_attention_pass": None,
        "robustness_compare_new_model_csv": None,
        "cfa_compare_csv": str(out_cfa_compare),
        "multigroup_cfa_csv": None,
        "multigroup_nh4_csv": None,
        "attention_eq5_sensitivity_csv": None,
        "n_rows_attention_fail": None,
        "n_rows_attention_eq5": None,
        "nh4_wald_z_pass_vs_fail": None,
        "nh4_wald_p_pass_vs_fail": None,
        "cfa_delta_cfi_metric_minus_configural": None,
        "table_7_4_reduced_model": None,
        "table_7_5_reduced_model": None,
        "table_7_5_reduced_attention": None,
        "robustness_compare_reduced_model_csv": None,
    }
    robustness_payload.update(
        {
            "table_7_4_reduced_model": str(out_74_reduced),
            "table_7_5_reduced_model": str(out_75_reduced),
        }
    )

    df_attention: pd.DataFrame | None = None
    df_attention_fail: pd.DataFrame | None = None
    legacy_attention: dict[str, Any] | None = None
    reduced_attention: dict[str, Any] | None = None
    if args.run_robustness:
        att = sem_data[sem_data["attention_pass_eq1"] == 1].reset_index(drop=True)
        if att.empty:
            raise RuntimeError("Robustness sample is empty (attention_pass_eq1==1).")
        att_fail = sem_data[sem_data["attention_pass_eq1"] == 0].reset_index(drop=True)
        if att_fail.empty:
            raise RuntimeError("attention_pass_eq1==0 group is empty; cannot run group comparison.")
        df_attention = att[MODEL_COLS].copy()
        df_attention_fail = att_fail[MODEL_COLS].copy()
        df_attention.to_csv(out_input_attention, encoding="utf-8-sig", index=False)

        legacy_attention = run_one_model(
            model_id="legacy_compat_v2_attention",
            model_desc=MODEL_SPECS["legacy_compat_v2"]["model_desc"],
            direct_defs=LEGACY_DIRECT_DEFS,
            indirect_defs=LEGACY_INDIRECT_DEFS,
            df=df_attention,
            fit_standard=args.fit_standard,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            run_bootstrap=False,
        )
        write_table_direct_only(
            out_75_attention,
            LEGACY_DIRECT_DEFS,
            legacy_attention["direct"],
            note="attention_pass_eq1==1 子样本（直接路径）",
        )
        write_robustness_compare(
            out_robustness,
            LEGACY_DIRECT_DEFS,
            legacy_main["direct"],
            legacy_attention["direct"],
        )

        df_attention_reduced = df_attention.copy()
        df_attention_reduced["R_sum"] = (df_attention_reduced["R_visit"] + df_attention_reduced["R_recommend"]) / 2.0
        reduced_attention = run_one_model(
            model_id="intent_obs_rsum_v1_attention",
            model_desc=MODEL_SPECS["intent_obs_rsum_v1"]["model_desc"],
            direct_defs=REDUCED_DIRECT_DEFS,
            indirect_defs=[],
            df=df_attention_reduced,
            fit_standard=args.fit_standard,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            run_bootstrap=False,
        )
        write_table_direct_only(
            out_75_reduced_attention,
            REDUCED_DIRECT_DEFS,
            reduced_attention["direct"],
            note="attention_pass_eq1==1 子样本（降复杂度模型）",
        )
        write_robustness_compare(
            out_robustness_reduced,
            REDUCED_DIRECT_DEFS,
            reduced_main["direct"],
            reduced_attention["direct"],
        )
        robustness_payload.update(
            {
                "n_rows_attention_pass": int(len(df_attention)),
                "n_rows_attention_fail": int(len(df_attention_fail)),
                "legacy_solver_attention_pass": legacy_attention["solver"],
                "legacy_metrics_attention_pass": legacy_attention["metrics"],
                "legacy_metric_checks_attention_pass": legacy_attention["metric_checks"],
                "legacy_fit_all_pass_attention_pass": legacy_attention["fit_all_pass"],
                "table_7_5_attention_pass": str(out_75_attention),
                "input_attention_pass_csv": str(out_input_attention),
                "robustness_compare_csv": str(out_robustness),
                "table_7_5_reduced_attention": str(out_75_reduced_attention),
                "robustness_compare_reduced_model_csv": str(out_robustness_reduced),
            }
        )

    baseline_main: dict[str, Any] | None = None
    new_main: dict[str, Any] | None = None
    new_attention: dict[str, Any] | None = None
    new_fail: dict[str, Any] | None = None
    new_eq5: dict[str, Any] | None = None
    if args.model_suite == "dual":
        baseline_main = run_one_model(
            model_id="baseline_current",
            model_desc=MODEL_SPECS["baseline_current"]["model_desc"],
            direct_defs=[],
            indirect_defs=[],
            df=df_main,
            fit_standard=args.fit_standard,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            run_bootstrap=False,
        )

        new_main = run_one_model(
            model_id="intent_partial_v1",
            model_desc=MODEL_SPECS["intent_partial_v1"]["model_desc"],
            direct_defs=NEW_DIRECT_DEFS,
            indirect_defs=NEW_INDIRECT_DEFS,
            df=df_main,
            fit_standard=args.fit_standard,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            run_bootstrap=True,
        )
        write_table_74(out_74_new, new_main["metrics"], new_main["solver"], args.fit_standard)
        write_table_paths(
            out_75_new,
            NEW_DIRECT_DEFS,
            new_main["direct"],
            NEW_INDIRECT_DEFS,
            new_main["indirect_point"],
            new_main["indirect_dist"],
            new_main["bootstrap"],
            direct_note="直接路径（标准化系数）",
        )
        compare_rows = [
            {
                "model_id": "baseline_current",
                "model_name": "baseline_current",
                "metrics": baseline_main["metrics"],
                "fit_all_pass": baseline_main["fit_all_pass"],
            },
            {
                "model_id": "legacy_compat_v2",
                "model_name": "legacy_compat_v2",
                "metrics": legacy_main["metrics"],
                "fit_all_pass": legacy_main["fit_all_pass"],
            },
            {
                "model_id": "intent_partial_v1",
                "model_name": "intent_partial_v1",
                "metrics": new_main["metrics"],
                "fit_all_pass": new_main["fit_all_pass"],
            },
        ]
        write_model_compare(out_model_compare, compare_rows)

        if args.run_robustness and (df_attention is not None):
            new_attention = run_one_model(
                model_id="intent_partial_v1_attention",
                model_desc=MODEL_SPECS["intent_partial_v1"]["model_desc"],
                direct_defs=NEW_DIRECT_DEFS,
                indirect_defs=NEW_INDIRECT_DEFS,
                df=df_attention,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )
            write_robustness_compare(
                out_robustness_new,
                NEW_DIRECT_DEFS,
                new_main["direct"],
                new_attention["direct"],
            )

            if df_attention_fail is None:
                raise RuntimeError("attention_pass_eq1==0 group is unavailable for multi-group checks.")
            new_fail = run_one_model(
                model_id="intent_partial_v1_attention_fail",
                model_desc=MODEL_SPECS["intent_partial_v1"]["model_desc"],
                direct_defs=NEW_DIRECT_DEFS,
                indirect_defs=NEW_INDIRECT_DEFS,
                df=df_attention_fail,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )

            cfg_pass = run_one_model(
                model_id="cfa_intent_with_method_pass",
                model_desc=MODEL_SPECS["cfa_intent_with_method"]["model_desc"],
                direct_defs=[],
                indirect_defs=[],
                df=df_attention,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )
            cfg_fail = run_one_model(
                model_id="cfa_intent_with_method_fail",
                model_desc=MODEL_SPECS["cfa_intent_with_method"]["model_desc"],
                direct_defs=[],
                indirect_defs=[],
                df=df_attention_fail,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )
            fixed_loadings = _extract_cfa_loadings(cfa_with_method["inspect"])
            metric_desc = build_metric_fixed_cfa_desc(fixed_loadings)
            metric_pass = run_one_model(
                model_id="cfa_intent_metric_pass",
                model_desc=metric_desc,
                direct_defs=[],
                indirect_defs=[],
                df=df_attention,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )
            metric_fail = run_one_model(
                model_id="cfa_intent_metric_fail",
                model_desc=metric_desc,
                direct_defs=[],
                indirect_defs=[],
                df=df_attention_fail,
                fit_standard=args.fit_standard,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
                run_bootstrap=False,
            )
            n_pass = len(df_attention)
            n_fail = len(df_attention_fail)
            cfi_configural_weighted = _weighted_metric(
                [(n_pass, cfg_pass["metrics"]), (n_fail, cfg_fail["metrics"])], "CFI"
            )
            cfi_metric_weighted = _weighted_metric(
                [(n_pass, metric_pass["metrics"]), (n_fail, metric_fail["metrics"])], "CFI"
            )
            delta_cfi = cfi_metric_weighted - cfi_configural_weighted
            write_multigroup_cfa_compare(
                out_multigroup_cfa,
                n_pass=n_pass,
                n_fail=n_fail,
                config_pass=cfg_pass,
                config_fail=cfg_fail,
                metric_pass=metric_pass,
                metric_fail=metric_fail,
                cfi_configural_weighted=cfi_configural_weighted,
                cfi_metric_weighted=cfi_metric_weighted,
                delta_cfi=delta_cfi,
            )

            main_nh4 = extract_path_detail(new_main["inspect"], "R_intent", "O_cognition")
            pass_nh4 = extract_path_detail(new_attention["inspect"], "R_intent", "O_cognition")
            fail_nh4 = extract_path_detail(new_fail["inspect"], "R_intent", "O_cognition")
            z_wald = math.nan
            p_wald = math.nan
            if (
                np.isfinite(pass_nh4["std_err"])
                and np.isfinite(fail_nh4["std_err"])
                and pass_nh4["std_err"] > 0
                and fail_nh4["std_err"] > 0
            ):
                se_diff = math.sqrt(pass_nh4["std_err"] ** 2 + fail_nh4["std_err"] ** 2)
                if se_diff > 0:
                    z_wald = (pass_nh4["estimate"] - fail_nh4["estimate"]) / se_diff
                    p_wald = 2.0 * (1.0 - normal_cdf(abs(z_wald)))
            write_nh4_multigroup_test(
                out_multigroup_nh4,
                n_main=len(df_main),
                n_pass=n_pass,
                n_fail=n_fail,
                main_detail=main_nh4,
                pass_detail=pass_nh4,
                fail_detail=fail_nh4,
                z_wald=z_wald,
                p_wald=p_wald,
            )

            n_eq5: int | None = None
            eq5_nh4_detail: dict[str, float] | None = None
            if "attention_pass_eq5" in sem_data.columns:
                att_eq5 = sem_data[sem_data["attention_pass_eq5"] == 1].reset_index(drop=True)
                n_eq5 = int(len(att_eq5))
                if n_eq5 > 0:
                    df_eq5 = att_eq5[MODEL_COLS].copy()
                    new_eq5 = run_one_model(
                        model_id="intent_partial_v1_attention_eq5",
                        model_desc=MODEL_SPECS["intent_partial_v1"]["model_desc"],
                        direct_defs=NEW_DIRECT_DEFS,
                        indirect_defs=NEW_INDIRECT_DEFS,
                        df=df_eq5,
                        fit_standard=args.fit_standard,
                        bootstrap_n=args.bootstrap_n,
                        seed=args.seed,
                        run_bootstrap=False,
                    )
                    eq5_nh4_detail = extract_path_detail(new_eq5["inspect"], "R_intent", "O_cognition")
            write_attention_eq5_sensitivity(
                out_attention_eq5,
                n_main=len(df_main),
                main_detail=main_nh4,
                n_eq1=n_pass,
                eq1_detail=pass_nh4,
                n_eq5=n_eq5,
                eq5_detail=eq5_nh4_detail,
            )

            robustness_payload.update(
                {
                    "new_solver_attention_pass": new_attention["solver"],
                    "new_metrics_attention_pass": new_attention["metrics"],
                    "new_metric_checks_attention_pass": new_attention["metric_checks"],
                    "new_fit_all_pass_attention_pass": new_attention["fit_all_pass"],
                    "robustness_compare_new_model_csv": str(out_robustness_new),
                    "multigroup_cfa_csv": str(out_multigroup_cfa),
                    "multigroup_nh4_csv": str(out_multigroup_nh4),
                    "attention_eq5_sensitivity_csv": str(out_attention_eq5),
                    "n_rows_attention_eq5": n_eq5,
                    "nh4_wald_z_pass_vs_fail": z_wald if np.isfinite(z_wald) else None,
                    "nh4_wald_p_pass_vs_fail": p_wald if np.isfinite(p_wald) else None,
                    "cfa_delta_cfi_metric_minus_configural": delta_cfi,
                }
            )

    audit = {
        "generated_at_utc": now_iso(),
        "source_script": "run_sem.py",
        "sem_revision": "service_method_effect_v2",
        "measurement_adjustments": MEASUREMENT_ADJUSTMENTS,
        "model_suite": args.model_suite,
        "fit_standard": args.fit_standard,
        "input": input_meta,
        "output_dir": str(out_dir),
        "outputs": {
            "table_7_4": str(out_74),
            "table_7_5": str(out_75),
            "input_main_csv": str(out_input_main),
            "audit_json": str(audit_json),
            "cfa_compare_csv": robustness_payload["cfa_compare_csv"],
            "table_7_5_attention_pass": robustness_payload["table_7_5_attention_pass"],
            "input_attention_pass_csv": robustness_payload["input_attention_pass_csv"],
            "robustness_compare_csv": robustness_payload["robustness_compare_csv"],
            "multigroup_cfa_csv": robustness_payload["multigroup_cfa_csv"],
            "multigroup_nh4_csv": robustness_payload["multigroup_nh4_csv"],
            "attention_eq5_sensitivity_csv": robustness_payload["attention_eq5_sensitivity_csv"],
            "table_7_4_new_model": str(out_74_new) if args.model_suite == "dual" else None,
            "table_7_5_new_model": str(out_75_new) if args.model_suite == "dual" else None,
            "model_compare_csv": str(out_model_compare) if args.model_suite == "dual" else None,
            "robustness_compare_new_model_csv": robustness_payload["robustness_compare_new_model_csv"],
            "table_7_4_reduced_model": robustness_payload["table_7_4_reduced_model"],
            "table_7_5_reduced_model": robustness_payload["table_7_5_reduced_model"],
            "table_7_5_reduced_attention": robustness_payload["table_7_5_reduced_attention"],
            "robustness_compare_reduced_model_csv": robustness_payload["robustness_compare_reduced_model_csv"],
        },
        "n_rows_main": int(len(df_main)),
        "n_rows_attention_pass": robustness_payload["n_rows_attention_pass"],
        "n_rows_attention_fail": robustness_payload["n_rows_attention_fail"],
        "n_rows_attention_eq5": robustness_payload["n_rows_attention_eq5"],
        "solver_main": legacy_main["solver"],
        "solver_attention_pass": robustness_payload["legacy_solver_attention_pass"],
        "metrics_main": legacy_main["metrics"],
        "metrics_attention_pass": robustness_payload["legacy_metrics_attention_pass"],
        "metric_checks_main": legacy_main["metric_checks"],
        "metric_checks_attention_pass": robustness_payload["legacy_metric_checks_attention_pass"],
        "fit_all_pass_main": legacy_main["fit_all_pass"],
        "fit_all_pass_attention_pass": robustness_payload["legacy_fit_all_pass_attention_pass"],
        "bootstrap": legacy_main["bootstrap"],
        "robustness": {
            "enabled": bool(robustness_payload["enabled"]),
            "subset_rule": "attention_pass_eq1 == 1",
            "sensitivity_rule": "attention_pass_eq5 == 1",
            "nh4_wald_z_pass_vs_fail": robustness_payload["nh4_wald_z_pass_vs_fail"],
            "nh4_wald_p_pass_vs_fail": robustness_payload["nh4_wald_p_pass_vs_fail"],
            "cfa_delta_cfi_metric_minus_configural": robustness_payload["cfa_delta_cfi_metric_minus_configural"],
        },
        "models": {
            "legacy_compat_v2": _model_audit_block(
                "legacy_compat_v2",
                legacy_main,
                LEGACY_DIRECT_DEFS,
                LEGACY_INDIRECT_DEFS,
            ),
            "baseline_current": _model_audit_block(
                "baseline_current",
                baseline_main,
                [],
                [],
            ),
            "intent_partial_v1": _model_audit_block(
                "intent_partial_v1",
                new_main,
                NEW_DIRECT_DEFS,
                NEW_INDIRECT_DEFS,
            ),
            "cfa_intent_with_method": _model_audit_block(
                "cfa_intent_with_method",
                cfa_with_method,
                [],
                [],
            ),
            "cfa_intent_no_method": _model_audit_block(
                "cfa_intent_no_method",
                cfa_without_method,
                [],
                [],
            ),
            "intent_obs_rsum_v1": _model_audit_block(
                "intent_obs_rsum_v1",
                reduced_main,
                REDUCED_DIRECT_DEFS,
                [],
            ),
            "legacy_compat_v2_attention": _model_audit_block(
                "legacy_compat_v2_attention",
                legacy_attention,
                LEGACY_DIRECT_DEFS,
                [],
            ),
            "intent_partial_v1_attention": _model_audit_block(
                "intent_partial_v1_attention",
                new_attention,
                NEW_DIRECT_DEFS,
                [],
            ),
            "intent_partial_v1_attention_fail": _model_audit_block(
                "intent_partial_v1_attention_fail",
                new_fail,
                NEW_DIRECT_DEFS,
                [],
            ),
            "intent_partial_v1_attention_eq5": _model_audit_block(
                "intent_partial_v1_attention_eq5",
                new_eq5,
                NEW_DIRECT_DEFS,
                [],
            ),
            "intent_obs_rsum_v1_attention": _model_audit_block(
                "intent_obs_rsum_v1_attention",
                reduced_attention,
                REDUCED_DIRECT_DEFS,
                [],
            ),
        },
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "sem_done:",
        f"model_suite={args.model_suite}",
        f"n_main={len(df_main)}",
        f"n_attention_pass={robustness_payload['n_rows_attention_pass']}",
        f"solver_main={legacy_main['solver']}",
        f"fit_standard={args.fit_standard}",
        f"table74={out_74}",
        f"table75={out_75}",
        f"table74_new={out_74_new if args.model_suite == 'dual' else 'skipped'}",
        f"table75_new={out_75_new if args.model_suite == 'dual' else 'skipped'}",
        f"model_compare={out_model_compare if args.model_suite == 'dual' else 'skipped'}",
        f"cfa_compare={out_cfa_compare}",
        f"multigroup_cfa={out_multigroup_cfa if (args.model_suite == 'dual' and args.run_robustness) else 'skipped'}",
        f"multigroup_nh4={out_multigroup_nh4 if (args.model_suite == 'dual' and args.run_robustness) else 'skipped'}",
        f"attention_eq5={out_attention_eq5 if (args.model_suite == 'dual' and args.run_robustness) else 'skipped'}",
        f"table74_reduced={out_74_reduced}",
        f"table75_reduced={out_75_reduced}",
        f"robustness_reduced={out_robustness_reduced if args.run_robustness else 'skipped'}",
        f"audit={audit_json}",
    )


if __name__ == "__main__":
    main()
