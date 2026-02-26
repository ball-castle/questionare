#!/usr/bin/env python3
"""Run SEM on 880 sample and backfill table 7-4 / 7-5."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from semopy import Model, calc_stats
from semopy.inspector import inspect

MODEL_DESC_DATA_DRIVEN_V1 = """
S_service =~ C052 + C053 + C054 + C062 + C063 + C065
S_environment =~ C058 + C059 + C060 + C061
S_activity =~ C055 + C056 + C057
O_cognition =~ C086 + C087 + C088 + C089

O_cognition ~ S_environment + S_service + S_activity
C090 ~ O_cognition
C091 ~ O_cognition
"""

REQUIRED_COLS = [
    "C052",
    "C053",
    "C054",
    "C055",
    "C056",
    "C057",
    "C058",
    "C059",
    "C060",
    "C061",
    "C062",
    "C063",
    "C065",
    "C086",
    "C087",
    "C088",
    "C089",
    "C090",
    "C091",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEM and fill table 7-4 / 7-5.")
    parser.add_argument(
        "--input-csv",
        default="output_data_analysis/tables/survey_clean.csv",
        help="Input clean sample CSV.",
    )
    parser.add_argument(
        "--output-tables-dir",
        default="output/tables",
        help="Output tables directory.",
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
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite target 7-4/7-5 tables.",
    )
    return parser.parse_args()


def read_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required SEM columns: {missing}")
    x = df[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x.isna().any().any():
        miss_n = int(x.isna().sum().sum())
        raise ValueError(f"SEM input has missing/non-numeric values after conversion: {miss_n}")
    return x


def fit_sem(df: pd.DataFrame, retries: int = 1) -> tuple[Model, pd.DataFrame, str]:
    errors: list[str] = []
    solvers = ["SLSQP", "L-BFGS-B"]
    total_tries = max(1, retries + 1)
    for i in range(total_tries):
        solver = solvers[i % len(solvers)]
        try:
            model = Model(MODEL_DESC_DATA_DRIVEN_V1)
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
    # fuzzy contains
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
        # if p-value is unavailable, fall back to non-significant.
        pval = 1.0
    return beta, pval


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


def _threshold_conclusion(metric: str, value: float) -> tuple[str, str]:
    if metric == "CMIN/DF":
        ok = value < 3.0
        return ("达标" if ok else "未达标", "阈值：<3.0")
    if metric == "RMSEA":
        ok = value < 0.08
        return ("达标" if ok else "未达标", "阈值：<0.08")
    if metric == "CFI":
        ok = value > 0.90
        return ("达标" if ok else "未达标", "阈值：>0.90")
    if metric == "TLI":
        ok = value > 0.90
        return ("达标" if ok else "未达标", "阈值：>0.90")
    if metric == "SRMR":
        ok = value < 0.08
        return ("达标" if ok else "未达标", "阈值：<0.08")
    return ("待判定", "")


def write_table_74(path: Path, metrics: dict[str, float], solver: str) -> None:
    rows: list[dict[str, str]] = []
    for metric in ["CMIN/DF", "RMSEA", "CFI", "TLI", "SRMR"]:
        val = metrics[metric]
        conclusion, th = _threshold_conclusion(metric, val)
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


def bootstrap_indirect(
    df: pd.DataFrame,
    bootstrap_n: int,
    seed: int,
    min_success: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    rng = np.random.default_rng(seed)
    n = len(df)
    effects = {"H6": [], "H7": [], "H8": []}
    success = 0
    fail = 0

    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            model_b, _, _ = fit_sem(d, retries=1)
            ins_b = inspect(model_b, std_est=True)
            a_env, _ = extract_path(ins_b, "O_cognition", "S_environment")
            a_srv, _ = extract_path(ins_b, "O_cognition", "S_service")
            a_act, _ = extract_path(ins_b, "O_cognition", "S_activity")
            b_visit, _ = extract_path(ins_b, "C090", "O_cognition")
            b_rec, _ = extract_path(ins_b, "C091", "O_cognition")

            effects["H6"].append(a_env * b_visit)
            effects["H7"].append(a_srv * b_rec)
            effects["H8"].append(a_act * b_rec)
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


def write_table_75(
    path: Path,
    direct: dict[str, tuple[float, float]],
    indirect_point: dict[str, float],
    indirect_dist: dict[str, np.ndarray],
    bootstrap_n: int,
    success_n: int,
    fail_n: int,
) -> None:
    # H1-H5: direct paths
    rows: list[dict[str, str]] = []
    direct_specs = [
        ("H1", "S_env -> O_cognition", "O_cognition", "S_environment"),
        ("H2", "S_service -> O_cognition", "O_cognition", "S_service"),
        ("H3", "S_activity -> O_cognition", "O_cognition", "S_activity"),
        ("H4", "O_cognition -> R_visit", "C090", "O_cognition"),
        ("H5", "O_cognition -> R_recommend", "C091", "O_cognition"),
    ]
    for hid, label, lval, rval in direct_specs:
        beta, pval = direct[f"{lval}~{rval}"]
        supported = (pval < 0.05) and (beta > 0)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "标准化系数β": _fmt(beta, 4),
                "p值": _fmt(pval, 4),
                "结论": "支持" if supported else "不支持",
                "状态": "ready",
                "备注": "直接路径（标准化系数）",
            }
        )

    # H6-H8: mediation via bootstrap
    med_specs = [
        ("H6", "S_env -> O -> R", "H6"),
        ("H7", "S_service -> O -> R", "H7"),
        ("H8", "S_activity -> O -> R", "H8"),
    ]
    for hid, label, key in med_specs:
        lo, hi, p_boot = bootstrap_summary(indirect_dist[key])
        sig = (lo > 0) or (hi < 0)
        rows.append(
            {
                "假设": hid,
                "路径": label,
                "标准化系数β": f"indirect={_fmt(indirect_point[key], 4)}",
                "p值": f"p_boot={_fmt(p_boot,4)};95%CI=[{_fmt(lo,4)},{_fmt(hi,4)}]",
                "结论": "中介显著" if sig else "中介不显著",
                "状态": "ready",
                "备注": f"bootstrap={bootstrap_n};success={success_n};fail={fail_n}",
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["假设", "路径", "标准化系数β", "p值", "结论", "状态", "备注"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.mapping != "data_driven_v1":
        raise ValueError(f"Unsupported mapping: {args.mapping}")

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_tables_dir)
    out_74 = out_dir / "表7-4_SEM模型拟合指标.csv"
    out_75 = out_dir / "表7-5_SEM路径系数与显著性.csv"

    if (not args.overwrite) and (out_74.exists() or out_75.exists()):
        raise FileExistsError("Target tables already exist and --no-overwrite is set.")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(input_csv)

    # Main fit with retry.
    model, stats, solver = fit_sem(df, retries=1)
    ins = inspect(model, std_est=True)

    # Fit metrics and SRMR.
    srmr = compute_srmr(model, df)
    metrics = fit_metrics(stats, srmr)

    # Direct effects (H1-H5).
    direct = {}
    for lval, rval in [
        ("O_cognition", "S_environment"),
        ("O_cognition", "S_service"),
        ("O_cognition", "S_activity"),
        ("C090", "O_cognition"),
        ("C091", "O_cognition"),
    ]:
        direct[f"{lval}~{rval}"] = extract_path(ins, lval, rval)

    # Indirect point estimates from main fit.
    indirect_point = {
        "H6": direct["O_cognition~S_environment"][0] * direct["C090~O_cognition"][0],
        "H7": direct["O_cognition~S_service"][0] * direct["C091~O_cognition"][0],
        "H8": direct["O_cognition~S_activity"][0] * direct["C091~O_cognition"][0],
    }

    min_success = max(1600, int(math.ceil(args.bootstrap_n * 0.8)))
    dist, success_n, fail_n = bootstrap_indirect(
        df=df,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        min_success=min_success,
    )

    write_table_74(out_74, metrics, solver)
    write_table_75(out_75, direct, indirect_point, dist, args.bootstrap_n, success_n, fail_n)

    print(
        "sem_done:",
        f"n={len(df)}",
        f"solver={solver}",
        f"bootstrap_success={success_n}",
        f"bootstrap_fail={fail_n}",
        f"table74={out_74}",
        f"table75={out_75}",
    )


if __name__ == "__main__":
    main()
