#!/usr/bin/env python3
"""Data extraction helpers for chapter 6/7 report generation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float:
    s = str(value or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def to_int(value: str | None) -> int:
    return int(round(to_float(value)))


def to_float_nan(value: str | None) -> float:
    s = str(value or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def p_value_text(value: float) -> str:
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.4f}"


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _single_value(single_map: dict[tuple[int, int], dict[str, float]], col_idx: int, code: int) -> dict[str, float]:
    return single_map.get((col_idx, code), {"count": 0.0, "pct": 0.0})


def _top_single_by_pct(single_rows: list[dict[str, str]], col_idx: int) -> dict[str, float]:
    vals = [r for r in single_rows if to_int(r.get("col_idx")) == col_idx]
    if not vals:
        return {"code": 0.0, "count": 0.0, "pct": 0.0}
    best = max(vals, key=lambda x: to_float(x.get("pct")))
    return {
        "code": float(to_int(best.get("code"))),
        "count": to_float(best.get("count")),
        "pct": to_float(best.get("pct")),
    }


def _top_multi_by_pct(multi_rows: list[dict[str, str]], col_min: int, col_max: int, top_n: int) -> list[dict[str, float | str]]:
    vals = [r for r in multi_rows if col_min <= to_int(r.get("col_idx")) <= col_max]
    vals.sort(key=lambda x: to_float(x.get("selected_pct")), reverse=True)
    out: list[dict[str, float | str]] = []
    for row in vals[:top_n]:
        out.append(
            {
                "col_idx": to_float(row.get("col_idx")),
                "item_text": str(row.get("item_text", "")),
                "selected_pct": to_float(row.get("selected_pct")),
                "selected_count": to_float(row.get("selected_count")),
                "valid_n": to_float(row.get("valid_n")),
            }
        )
    return out


def _read_logit_row(rows: list[dict[str, str]], term: str) -> dict[str, float]:
    for r in rows:
        if str(r.get("term")) == term:
            return {
                "coef": to_float(r.get("coef")),
                "p_value": to_float(r.get("p_value")),
                "odds_ratio": to_float(r.get("odds_ratio")),
            }
    return {"coef": 0.0, "p_value": 1.0, "odds_ratio": 0.0}


def _extract_mca_top_cards(mca_text: str, marker: str, top_n: int = 5) -> list[str]:
    lines = [ln.strip() for ln in mca_text.splitlines()]
    out: list[str] = []
    active = False
    for line in lines:
        if marker in line:
            active = True
            continue
        if active:
            if not line:
                break
            if line.startswith("- "):
                out.append(line[2:])
    return out[:top_n]


def _figure_item(path: Path, caption: str) -> dict[str, str | bool]:
    return {"path": str(path), "caption": caption, "exists": path.exists()}


def _missing(path: Path, missing_policy: str, warnings: list[str]) -> None:
    msg = f"missing: {path}"
    if missing_policy == "fail":
        raise FileNotFoundError(msg)
    warnings.append(msg)


def _row_mean(row: dict[str, str], cols_1b: list[int]) -> float:
    vals = [to_float_nan(row.get(f"C{c:03d}")) for c in cols_1b]
    valid = [v for v in vals if not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def _corr(x: list[float], y: list[float]) -> dict[str, float]:
    pairs = [(a, b) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)]
    n = len(pairs)
    if n < 3:
        return {"r": float("nan"), "n": float(n)}
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / n
    my = sum(yv) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    var_x = sum((a - mx) ** 2 for a in xv)
    var_y = sum((b - my) ** 2 for b in yv)
    if var_x <= 0 or var_y <= 0:
        return {"r": float("nan"), "n": float(n)}
    r = cov / math.sqrt(var_x * var_y)
    return {"r": float(r), "n": float(n)}


def _group_means(metric: list[float], group_code: list[float], code: int) -> dict[str, float]:
    vals = [
        m
        for m, g in zip(metric, group_code)
        if (not math.isnan(m)) and (not math.isnan(g)) and int(round(g)) == int(code)
    ]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "n": 0.0}
    return {"mean": float(sum(vals) / n), "n": float(n)}


def extract_report_data(
    tables_dir: Path,
    figures_dir: Path,
    outline_md: Path,
    missing_policy: str = "keep_placeholder",
) -> dict:
    warnings: list[str] = []
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    outline_md = Path(outline_md)
    if not outline_md.exists():
        _missing(outline_md, missing_policy, warnings)

    must_have = [
        tables_dir / "单选题频数百分比表.csv",
        tables_dir / "多选题选择率表.csv",
        tables_dir / "交叉分析卡方汇总.csv",
        tables_dir / "MCA特征值.csv",
        tables_dir / "Logit模型指标.csv",
        tables_dir / "Logit回归结果_主样本.csv",
        tables_dir / "Logit回归结果_敏感性样本.csv",
        tables_dir / "二阶聚类画像卡.csv",
        tables_dir / "二阶聚类候选K评估.csv",
        tables_dir / "聚类稳定性对比表.csv",
        tables_dir / "IPA整改优先级表.csv",
        tables_dir / "IPA阈值敏感性表.csv",
        tables_dir / "问题-证据-建议对照表.csv",
        tables_dir / "建议落地行动矩阵.csv",
        tables_dir / "假设变量模型映射表.csv",
    ]
    for p in must_have:
        if not p.exists():
            _missing(p, missing_policy, warnings)

    single_rows = read_csv(tables_dir / "单选题频数百分比表.csv") if (tables_dir / "单选题频数百分比表.csv").exists() else []
    multi_rows = read_csv(tables_dir / "多选题选择率表.csv") if (tables_dir / "多选题选择率表.csv").exists() else []
    chi_rows = read_csv(tables_dir / "交叉分析卡方汇总.csv") if (tables_dir / "交叉分析卡方汇总.csv").exists() else []
    mca_eigen_rows = read_csv(tables_dir / "MCA特征值.csv") if (tables_dir / "MCA特征值.csv").exists() else []
    mca_loading_rows = read_csv(tables_dir / "MCA类别坐标与贡献.csv") if (tables_dir / "MCA类别坐标与贡献.csv").exists() else []
    mca_card_text = safe_read_text(tables_dir / "MCA群体解释卡.txt")

    logit_metric_rows = read_csv(tables_dir / "Logit模型指标.csv") if (tables_dir / "Logit模型指标.csv").exists() else []
    logit_main_rows = read_csv(tables_dir / "Logit回归结果_主样本.csv") if (tables_dir / "Logit回归结果_主样本.csv").exists() else []
    logit_sens_rows = read_csv(tables_dir / "Logit回归结果_敏感性样本.csv") if (tables_dir / "Logit回归结果_敏感性样本.csv").exists() else []
    logit_dir_rows = read_csv(tables_dir / "Logit稳健性方向对比.csv") if (tables_dir / "Logit稳健性方向对比.csv").exists() else []
    dual_rows = read_csv(tables_dir / "注意力题双口径对比.csv") if (tables_dir / "注意力题双口径对比.csv").exists() else []

    cluster_profile_rows = read_csv(tables_dir / "二阶聚类画像卡.csv") if (tables_dir / "二阶聚类画像卡.csv").exists() else []
    cluster_k_rows = read_csv(tables_dir / "二阶聚类候选K评估.csv") if (tables_dir / "二阶聚类候选K评估.csv").exists() else []
    cluster_stability_rows = read_csv(tables_dir / "聚类稳定性对比表.csv") if (tables_dir / "聚类稳定性对比表.csv").exists() else []

    ipa_priority_rows = read_csv(tables_dir / "IPA整改优先级表.csv") if (tables_dir / "IPA整改优先级表.csv").exists() else []
    ipa_sensitivity_rows = read_csv(tables_dir / "IPA阈值敏感性表.csv") if (tables_dir / "IPA阈值敏感性表.csv").exists() else []
    action_rows = read_csv(tables_dir / "建议落地行动矩阵.csv") if (tables_dir / "建议落地行动矩阵.csv").exists() else []
    problem_rows = read_csv(tables_dir / "问题-证据-建议对照表.csv") if (tables_dir / "问题-证据-建议对照表.csv").exists() else []
    sem_map_rows = read_csv(tables_dir / "假设变量模型映射表.csv") if (tables_dir / "假设变量模型映射表.csv").exists() else []
    clean_rows = read_csv(tables_dir / "survey_clean.csv") if (tables_dir / "survey_clean.csv").exists() else []

    run_meta_path = tables_dir.parent / "run_metadata.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}

    def _meta_int(key: str) -> int:
        try:
            return int(float(str(run_meta.get(key, "")).strip()))
        except Exception:
            return 0

    raw_n_meta = _meta_int("n_samples")
    main_n_meta = _meta_int("remain_n_revised")

    if raw_n_meta <= 0 or main_n_meta <= 0:
        flow_rows = []
        for flow_name in ["样本流转表_重筛.csv", "样本流转表.csv"]:
            p = tables_dir / flow_name
            if p.exists():
                flow_rows = read_csv(p)
                break
        if flow_rows:
            for r in flow_rows:
                step = str(r.get("step", "")).strip()
                n_val = to_int(r.get("n"))
                if step == "raw_input" and raw_n_meta <= 0:
                    raw_n_meta = n_val
                if step in {"remain_revised", "main_analysis"} and main_n_meta <= 0:
                    main_n_meta = n_val

    if main_n_meta <= 0:
        main_n_meta = len(clean_rows)
    if raw_n_meta <= 0:
        raw_n_meta = main_n_meta
    if not run_meta.get("quality_profile"):
        run_meta["quality_profile"] = "balanced_v20260221"
    run_meta["n_samples"] = raw_n_meta
    run_meta["remain_n_revised"] = main_n_meta

    perception_vec = [_row_mean(r, list(range(52, 64)) + [65]) for r in clean_rows]
    cognition_vec = [_row_mean(r, list(range(86, 90))) for r in clean_rows]
    intention_vec = [_row_mean(r, [90, 91]) for r in clean_rows]
    performance_vec = [_row_mean(r, list(range(76, 86))) for r in clean_rows]
    q8_vec = [to_float_nan(r.get("C008")) for r in clean_rows]

    corr_s_o = _corr(perception_vec, cognition_vec)
    corr_o_r = _corr(cognition_vec, intention_vec)
    corr_s_r = _corr(perception_vec, intention_vec)
    corr_perf_r = _corr(performance_vec, intention_vec)

    visit_yes = _group_means(intention_vec, q8_vec, 1)
    visit_no = _group_means(intention_vec, q8_vec, 2)
    visit_gap = (
        float(visit_yes["mean"] - visit_no["mean"])
        if (not math.isnan(visit_yes["mean"]) and not math.isnan(visit_no["mean"]))
        else float("nan")
    )

    single_map: dict[tuple[int, int], dict[str, float]] = {}
    for row in single_rows:
        key = (to_int(row.get("col_idx")), to_int(row.get("code")))
        single_map[key] = {"count": to_float(row.get("count")), "pct": to_float(row.get("pct"))}

    q1_male = _single_value(single_map, 1, 1)
    q1_female = _single_value(single_map, 1, 2)
    q2_age = {code: _single_value(single_map, 2, code) for code in [1, 3, 2, 4, 5]}
    q8_yes = _single_value(single_map, 8, 1)
    q8_no = _single_value(single_map, 8, 2)
    q24_top = _top_single_by_pct(single_rows, 24)
    q25_top = _top_single_by_pct(single_rows, 25)
    q90_pos = _single_value(single_map, 90, 4)["pct"] + _single_value(single_map, 90, 5)["pct"]
    q91_pos = _single_value(single_map, 91, 4)["pct"] + _single_value(single_map, 91, 5)["pct"]

    motive_top = _top_multi_by_pct(multi_rows, 16, 23, top_n=1)
    visited_pain_top = _top_multi_by_pct(multi_rows, 33, 42, top_n=3)
    unvisited_block_top = _top_multi_by_pct(multi_rows, 43, 51, top_n=2)
    new_project_top = _top_multi_by_pct(multi_rows, 92, 100, top_n=3)
    promo_top = _top_multi_by_pct(multi_rows, 101, 107, top_n=2)

    chi_focus = []
    for pair in [("C008", "C006"), ("C008", "C007"), ("C008", "C024"), ("C008", "C025")]:
        found = next((r for r in chi_rows if str(r.get("var1")) == pair[0] and str(r.get("var2")) == pair[1]), None)
        if found:
            pval = to_float(found.get("p_value"))
            chi_focus.append({"pair": f"{pair[0]}×{pair[1]}", "p_value": pval, "text": p_value_text(pval)})

    mca_dim1 = to_float(mca_eigen_rows[0].get("eigenvalue")) if len(mca_eigen_rows) >= 1 else 0.0
    mca_dim2 = to_float(mca_eigen_rows[1].get("eigenvalue")) if len(mca_eigen_rows) >= 2 else 0.0
    mca_cum = mca_dim1 + mca_dim2
    mca_dim1_top = sorted(mca_loading_rows, key=lambda x: abs(to_float(x.get("dim1_contrib"))), reverse=True)[:5]
    mca_dim2_top = sorted(mca_loading_rows, key=lambda x: abs(to_float(x.get("dim2_contrib"))), reverse=True)[:5]
    mca_cards_1 = _extract_mca_top_cards(mca_card_text, "维度1绝对载荷Top8：")
    mca_cards_2 = _extract_mca_top_cards(mca_card_text, "维度2绝对载荷Top8：")

    metric_map = {str(r.get("sample")): r for r in logit_metric_rows}
    main_metric = metric_map.get("main", {})
    sens_metric = metric_map.get("sensitivity", {})
    q8_sens = _read_logit_row(logit_sens_rows, "Q8_visit_status_code")
    reversed_n = sum(1 for r in logit_dir_rows if to_int(r.get("reversed")) == 1)
    dual_summary = [
        {
            "calibration": str(r.get("calibration", "")),
            "accuracy": to_float(r.get("accuracy")),
            "auc": to_float(r.get("auc")),
            "pseudo_r2": to_float(r.get("pseudo_r2")),
            "reversed_direction_n": to_int(r.get("reversed_direction_n")),
        }
        for r in dual_rows
    ]
    dual_summary.sort(key=lambda x: x["auc"], reverse=True)

    cluster_profiles = sorted(cluster_profile_rows, key=lambda x: to_int(x.get("cluster")))
    cluster_best = None
    for row in cluster_stability_rows:
        if "优先方案" in str(row.get("稳定性说明", "")):
            cluster_best = row
            break
    if cluster_best is None and cluster_k_rows:
        cluster_best = max(cluster_k_rows, key=lambda x: to_float(x.get("silhouette")))

    ipa_mean_rows = [r for r in ipa_sensitivity_rows if str(r.get("阈值方法")) == "均值阈值"]
    ipa_q1 = [r for r in ipa_mean_rows if str(r.get("quadrant")).startswith("Q1")]
    ipa_q2 = [r for r in ipa_mean_rows if str(r.get("quadrant")).startswith("Q2")]
    ipa_q3 = [r for r in ipa_mean_rows if str(r.get("quadrant")).startswith("Q3")]
    ipa_q4 = [r for r in ipa_mean_rows if str(r.get("quadrant")).startswith("Q4")]

    figures = [
        _figure_item(figures_dir / "MCA二维图.png", "图6-1 MCA二维图"),
        _figure_item(figures_dir / "二阶聚类画像图.png", "图7-1 二阶聚类画像图"),
        _figure_item(figures_dir / "IPA象限图.png", "图7-2 IPA象限图"),
        _figure_item(figures_dir / "核心画像图_core_profile.png", "图7-3 核心画像图"),
    ]
    for fig in figures:
        if not fig["exists"]:
            _missing(Path(str(fig["path"])), missing_policy, warnings)

    report_data = {
        "outline_path": str(outline_md),
        "outline_exists": outline_md.exists(),
        "run_meta": run_meta,
        "desc": {
            "q1_male": q1_male,
            "q1_female": q1_female,
            "q2_age": q2_age,
            "q8_yes": q8_yes,
            "q8_no": q8_no,
            "q24_top": q24_top,
            "q25_top": q25_top,
            "q90_pos": q90_pos,
            "q91_pos": q91_pos,
            "motive_top": motive_top,
            "visited_pain_top": visited_pain_top,
            "unvisited_block_top": unvisited_block_top,
            "chi_focus": chi_focus,
        },
        "mca": {
            "dim1": mca_dim1,
            "dim2": mca_dim2,
            "cum2": mca_cum,
            "dim1_top": mca_dim1_top,
            "dim2_top": mca_dim2_top,
            "cards1": mca_cards_1,
            "cards2": mca_cards_2,
        },
        "logit": {
            "main_metric": main_metric,
            "sens_metric": sens_metric,
            "q8_sens": q8_sens,
            "reversed_n": reversed_n,
            "dual_summary": dual_summary,
        },
        "mechanism": {
            "corr_s_o": corr_s_o,
            "corr_o_r": corr_o_r,
            "corr_s_r": corr_s_r,
            "corr_perf_r": corr_perf_r,
            "visit_yes": visit_yes,
            "visit_no": visit_no,
            "visit_gap": visit_gap,
        },
        "cluster": {
            "profiles": cluster_profiles,
            "best": cluster_best or {},
        },
        "ipa": {
            "priority_rows": ipa_priority_rows,
            "mean_rows": ipa_mean_rows,
            "q1": ipa_q1,
            "q2": ipa_q2,
            "q3": ipa_q3,
            "q4": ipa_q4,
        },
        "voice": {
            "new_project_top": new_project_top,
            "promo_top": promo_top,
            "problem_rows": problem_rows[:8],
            "action_rows": action_rows[:8],
            "sem_map_rows": sem_map_rows[:8],
        },
        "figures": figures,
        "warnings": warnings,
    }
    return report_data
