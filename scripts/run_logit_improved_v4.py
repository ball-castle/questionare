#!/usr/bin/env python3
"""Post-process and complete remaining items from 改进方案4.

This script runs v3 pipeline first (optional), then adds:
1) Cost-benefit tables in business language (touches per hit, FP, ROI placeholder).
2) Fold-level TopK stability and conservative strategy selection by lower bound.
3) Bootstrap confidence intervals.
4) External validation plan text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


TOP_KS = (5, 10, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete remaining improvements for Logit (方案4).")
    parser.add_argument("--output-dir", default="data/data_logit2", help="Output directory.")
    parser.add_argument("--base-script", default="scripts/run_logit_improved_v3.py", help="Base script path.")
    parser.add_argument("--run-base", action="store_true", help="Run base script before post-processing.")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds used in base run.")
    parser.add_argument("--base-random-state", type=int, default=42, help="Random state used by base run.")
    parser.add_argument("--target-recall", type=float, default=0.30, help="Target recall for threshold strategy.")
    parser.add_argument("--bootstrap-n", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260227, help="Bootstrap seed.")
    parser.add_argument("--cost-per-touch", type=float, default=1.0, help="Cost per outreach unit.")
    parser.add_argument("--value-per-hit", type=float, default=1.0, help="Value per true positive.")
    return parser.parse_args()


def topk_detail(y: np.ndarray, score: np.ndarray, k_pct: int) -> dict:
    n = len(y)
    base_rate = float(np.mean(y)) if n else np.nan
    top_n = max(1, int(np.ceil(n * (k_pct / 100.0))))
    order = np.argsort(-score)
    sel = order[:top_n]
    hit_n = int(np.sum(y[sel]))
    fp_n = int(top_n - hit_n)
    precision = float(hit_n / top_n) if top_n > 0 else np.nan
    recall = float(hit_n / max(int(np.sum(y)), 1))
    lift = float(precision / base_rate) if base_rate > 0 else np.nan
    touches_per_hit = float(top_n / hit_n) if hit_n > 0 else np.inf
    return {
        "top_k_pct": int(k_pct),
        "top_n": int(top_n),
        "hit_n": int(hit_n),
        "false_positive_n": int(fp_n),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "base_rate": base_rate,
        "touches_per_hit": touches_per_hit,
    }


def build_topk_table(scope: str, score_type: str, y: np.ndarray, score: np.ndarray) -> list[dict]:
    rows = []
    for k in TOP_KS:
        rec = {"sample_scope": scope, "score_type": score_type}
        rec.update(topk_detail(y=y, score=score, k_pct=k))
        rows.append(rec)
    return rows


def make_fold_ids(y: np.ndarray, n_splits: int, random_state: int) -> np.ndarray:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    use_splits = max(2, min(n_splits, pos, neg))
    cv = StratifiedKFold(n_splits=use_splits, shuffle=True, random_state=random_state)
    fold_ids = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        fold_ids[te] = i
    return fold_ids


def add_business_value(df: pd.DataFrame, cost_per_touch: float, value_per_hit: float) -> pd.DataFrame:
    out = df.copy()
    out["touch_cost"] = out["top_n"] * float(cost_per_touch)
    out["hit_value"] = out["hit_n"] * float(value_per_hit)
    out["net_value"] = out["hit_value"] - out["touch_cost"]
    out["roi"] = out["net_value"] / out["touch_cost"].replace(0, np.nan)
    return out


def agg_stability(fold_topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, score_type, k), g in fold_topk.groupby(["sample_scope", "score_type", "top_k_pct"], dropna=False):
        n = len(g)
        for metric in ["precision_at_k", "recall_at_k", "lift_at_k", "touches_per_hit"]:
            m = float(g[metric].mean())
            s = float(g[metric].std(ddof=1)) if n > 1 else 0.0
            se = s / np.sqrt(n) if n > 0 else np.nan
            lb = m - 1.96 * se if np.isfinite(se) else np.nan
            ub = m + 1.96 * se if np.isfinite(se) else np.nan
            rows.append(
                {
                    "sample_scope": scope,
                    "score_type": score_type,
                    "top_k_pct": int(k),
                    "metric": metric,
                    "n_folds": int(n),
                    "mean": m,
                    "std": s,
                    "ci95_low": lb,
                    "ci95_high": ub,
                }
            )
    return pd.DataFrame(rows)


def recommend_strategy(stability: pd.DataFrame) -> pd.DataFrame:
    tgt = stability[stability["metric"] == "precision_at_k"].copy()
    rows = []
    for (scope, score_type), g in tgt.groupby(["sample_scope", "score_type"], dropna=False):
        g = g.sort_values(["ci95_low", "mean"], ascending=[False, False]).reset_index(drop=True)
        best = g.iloc[0]
        rows.append(
            {
                "sample_scope": scope,
                "score_type": score_type,
                "recommended_top_k_pct": int(best["top_k_pct"]),
                "precision_mean": float(best["mean"]),
                "precision_ci95_low": float(best["ci95_low"]),
                "precision_ci95_high": float(best["ci95_high"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_scope", "score_type"]).reset_index(drop=True)


def bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    aucs = []
    prs = []
    p20s = []
    l20s = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        sb = score[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))
        prs.append(float(average_precision_score(yb, sb)))
        r20 = topk_detail(yb, sb, 20)
        p20s.append(float(r20["precision_at_k"]))
        l20s.append(float(r20["lift_at_k"]))

    def summarize(name: str, vals: list[float]) -> dict:
        arr = np.array(vals, dtype=float)
        return {
            "metric": name,
            "n_valid_bootstrap": int(len(arr)),
            "mean": float(np.mean(arr)) if len(arr) else np.nan,
            "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        }

    return [
        summarize("auc", aucs),
        summarize("pr_auc", prs),
        summarize("precision_at_20", p20s),
        summarize("lift_at_20", l20s),
    ]


def threshold_strategy_rows(
    scope: str,
    y: np.ndarray,
    score_cal: np.ndarray,
    thr_target: float,
    thr_f2: float,
    cost_per_touch: float,
    value_per_hit: float,
) -> list[dict]:
    rows = []
    for strat, thr in [("target_recall", thr_target), ("f2_optimal", thr_f2)]:
        pred = (score_cal >= float(thr)).astype(int)
        pred_n = int(pred.sum())
        hit_n = int(((pred == 1) & (y == 1)).sum())
        fp_n = int(((pred == 1) & (y == 0)).sum())
        fn_n = int(((pred == 0) & (y == 1)).sum())
        precision = float(hit_n / max(pred_n, 1))
        recall = float(hit_n / max(int(y.sum()), 1))
        touches_per_hit = float(pred_n / hit_n) if hit_n > 0 else np.inf
        touch_cost = pred_n * float(cost_per_touch)
        hit_value = hit_n * float(value_per_hit)
        net_value = hit_value - touch_cost
        roi = net_value / touch_cost if touch_cost > 0 else np.nan
        rows.append(
            {
                "sample_scope": scope,
                "strategy": strat,
                "threshold": float(thr),
                "predicted_positive_n": pred_n,
                "hit_n": hit_n,
                "false_positive_n": fp_n,
                "false_negative_n": fn_n,
                "precision": precision,
                "recall": recall,
                "touches_per_hit": touches_per_hit,
                "touch_cost": touch_cost,
                "hit_value": hit_value,
                "net_value": net_value,
                "roi": roi,
            }
        )
    return rows


def write_validation_plan(path: Path) -> None:
    lines = [
        "外部验证计划（最小可行 + 正式版）",
        "版本日期：2026-02-27",
        "",
        "A. 最小可行（当前数据）",
        "1) 固定当前模型与策略参数（TopK与阈值），不再调参。",
        "2) 报告AUC、PR-AUC、Precision@20、Lift@20的Bootstrap 95%置信区间。",
        "",
        "B. 正式外部验证（新样本）",
        "1) 在2026-03-31前收集一批新问卷/运营回流样本（建议N>=300）。",
        "2) 使用冻结模型直接打分，不做再训练。",
        "3) 复测TopK命中率、误报率、每命中触达人数，并与本期OOF对比。",
        "4) 若Top20 Precision下降超过20%，触发重训与特征更新流程。",
        "",
        "C. 上线策略",
        "1) 排序筛选用raw score；规模估算用calibrated probability。",
        "2) 策略选择采用“置信区间下界最大”原则，而不是仅看均值。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_note(
    path: Path,
    metric_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    upper_df: pd.DataFrame,
) -> None:
    lines = [
        "Logit改进4补齐说明",
        "1) 交付口径改为业务可用：TopK成本收益 + 每命中触达人数。",
        "2) 可信度采用连续权重(sample_weight)，不使用硬筛选子样本作为主模型。",
        "3) 输出双分数：raw用于排序筛选，calibrated用于规模估算。",
        "4) 模型/策略选择引入稳定性约束（折次CI下界）。",
        "",
    ]
    if not metric_df.empty:
        for _, r in metric_df.iterrows():
            lines.append(
                f"{r['sample_scope']}: auc_raw={r['auc_raw']:.3f}, auc_cal={r['auc_calibrated']:.3f}, "
                f"pr_auc_cal={r['pr_auc_calibrated']:.3f}, precision@target={r['precision_at_target_recall']:.3f}, "
                f"recall@target={r['recall_at_target_recall']:.3f}"
            )
    lines.append("")
    lines.append("TopK每命中触达（calibrated）:")
    for scope in sorted(cost_df["sample_scope"].unique().tolist()):
        g = cost_df[(cost_df["sample_scope"] == scope) & (cost_df["score_type"] == "calibrated") & (cost_df["top_k_pct"].isin([10, 20, 30]))]
        for _, r in g.iterrows():
            lines.append(
                f"{scope} Top{int(r['top_k_pct'])}%: precision={r['precision_at_k']:.3f}, "
                f"touches_per_hit={r['touches_per_hit']:.2f}, hit={int(r['hit_n'])}, fp={int(r['false_positive_n'])}"
            )
    lines.append("")
    lines.append("稳定性推荐策略（按precision CI下界）:")
    for _, r in rec_df.iterrows():
        lines.append(
            f"{r['sample_scope']} {r['score_type']}: Top{int(r['recommended_top_k_pct'])}% "
            f"(precision CI95下界={r['precision_ci95_low']:.3f})"
        )
    lines.append("")
    lines.append("上限测试:")
    for _, r in upper_df.iterrows():
        lines.append(
            f"{r['sample_scope']} {r['model_family']}[{r['engine']}]: auc={r['auc']:.3f}, pr_auc={r['pr_auc']:.3f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_base:
        cmd = [
            sys.executable,
            args.base_script,
            "--output-dir",
            str(out_dir),
            "--cv-folds",
            str(args.cv_folds),
            "--target-recall",
            str(args.target_recall),
            "--random-state",
            str(args.base_random_state),
        ]
        subprocess.run(cmd, check=True)

    metric_path = out_dir / "Logit改进3_模型指标.csv"
    topk_path = out_dir / "Logit改进3_TopK指标.csv"
    oof_path = out_dir / "Logit改进3_OOF预测.csv"
    upper_path = out_dir / "Logit改进3_上限测试.csv"
    meta_path = out_dir / "run_metadata.json"
    if not metric_path.exists() or not oof_path.exists() or not meta_path.exists():
        raise FileNotFoundError("未找到基础输出，请先运行v3或使用 --run-base。")

    metric_df = pd.read_csv(metric_path)
    _topk_old = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    oof_df = pd.read_csv(oof_path)
    upper_df = pd.read_csv(upper_path) if upper_path.exists() else pd.DataFrame()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    scope_order = list(meta.get("sample_defs", []))
    if not scope_order:
        scope_order = sorted(oof_df["sample_scope"].unique().tolist())

    topk_rows = []
    fold_rows = []
    boot_rows = []
    threshold_rows = []

    for idx, scope in enumerate(scope_order, start=1):
        sdf = oof_df[oof_df["sample_scope"] == scope].copy().reset_index(drop=True)
        if sdf.empty:
            continue
        y = sdf["y_true_high"].to_numpy(dtype=int)
        s_raw = sdf["prob_raw"].to_numpy(dtype=float)
        s_cal = sdf["prob_calibrated"].to_numpy(dtype=float)

        topk_rows.extend(build_topk_table(scope, "raw", y, s_raw))
        topk_rows.extend(build_topk_table(scope, "calibrated", y, s_cal))

        fold_seed = int(args.base_random_state + idx * 100 + 7)
        fold_ids = make_fold_ids(y=y, n_splits=args.cv_folds, random_state=fold_seed)
        sdf["fold_id"] = fold_ids
        for fold in sorted(np.unique(fold_ids).tolist()):
            g = sdf[sdf["fold_id"] == fold]
            y_f = g["y_true_high"].to_numpy(dtype=int)
            r_f = g["prob_raw"].to_numpy(dtype=float)
            c_f = g["prob_calibrated"].to_numpy(dtype=float)
            for rec in build_topk_table(scope, "raw", y_f, r_f):
                rec["fold_id"] = int(fold)
                fold_rows.append(rec)
            for rec in build_topk_table(scope, "calibrated", y_f, c_f):
                rec["fold_id"] = int(fold)
                fold_rows.append(rec)

        for score_type, score in [("raw", s_raw), ("calibrated", s_cal)]:
            b = bootstrap_ci(y=y, score=score, n_boot=args.bootstrap_n, seed=args.bootstrap_seed + idx * 1000 + (0 if score_type == "raw" else 500))
            for rec in b:
                rec["sample_scope"] = scope
                rec["score_type"] = score_type
                boot_rows.append(rec)

        mr = metric_df[metric_df["sample_scope"] == scope]
        if len(mr) == 1:
            thr_target = float(mr.iloc[0]["threshold_target_recall"])
            thr_f2 = float(mr.iloc[0]["threshold_f2"])
            threshold_rows.extend(
                threshold_strategy_rows(
                    scope=scope,
                    y=y,
                    score_cal=s_cal,
                    thr_target=thr_target,
                    thr_f2=thr_f2,
                    cost_per_touch=args.cost_per_touch,
                    value_per_hit=args.value_per_hit,
                )
            )

    topk_df = pd.DataFrame(topk_rows).sort_values(["sample_scope", "score_type", "top_k_pct"]).reset_index(drop=True)
    cost_df = add_business_value(topk_df, cost_per_touch=args.cost_per_touch, value_per_hit=args.value_per_hit)

    fold_topk_df = pd.DataFrame(fold_rows).sort_values(["sample_scope", "score_type", "top_k_pct", "fold_id"]).reset_index(drop=True)
    fold_stability_df = agg_stability(fold_topk_df)
    rec_df = recommend_strategy(fold_stability_df)

    boot_df = pd.DataFrame(boot_rows).sort_values(["sample_scope", "score_type", "metric"]).reset_index(drop=True)
    threshold_df = pd.DataFrame(threshold_rows).sort_values(["sample_scope", "strategy"]).reset_index(drop=True)

    # Scope-level business summary from threshold strategy.
    biz_rows = []
    for _, r in threshold_df.iterrows():
        biz_rows.append(
            {
                "sample_scope": r["sample_scope"],
                "strategy": r["strategy"],
                "threshold": r["threshold"],
                "precision": r["precision"],
                "recall": r["recall"],
                "hit_n": int(r["hit_n"]),
                "false_positive_n": int(r["false_positive_n"]),
                "touches_per_hit": r["touches_per_hit"],
            }
        )
    biz_df = pd.DataFrame(biz_rows).sort_values(["sample_scope", "strategy"]).reset_index(drop=True)

    topk_df.to_csv(out_dir / "Logit改进4_TopK指标.csv", index=False, encoding="utf-8-sig")
    cost_df.to_csv(out_dir / "Logit改进4_成本收益表.csv", index=False, encoding="utf-8-sig")
    fold_topk_df.to_csv(out_dir / "Logit改进4_TopK折次结果.csv", index=False, encoding="utf-8-sig")
    fold_stability_df.to_csv(out_dir / "Logit改进4_TopK稳定性汇总.csv", index=False, encoding="utf-8-sig")
    rec_df.to_csv(out_dir / "Logit改进4_推荐策略.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / "Logit改进4_阈值策略表.csv", index=False, encoding="utf-8-sig")
    biz_df.to_csv(out_dir / "Logit改进4_业务换算表.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(out_dir / "Logit改进4_BootstrapCI.csv", index=False, encoding="utf-8-sig")

    write_validation_plan(out_dir / "Logit改进4_外部验证计划.txt")
    write_summary_note(
        path=out_dir / "Logit改进4_模型说明.txt",
        metric_df=metric_df,
        cost_df=cost_df,
        rec_df=rec_df,
        upper_df=upper_df,
    )

    meta4 = {
        "generated_on": str(date.today()),
        "output_dir": str(out_dir),
        "base_script": args.base_script,
        "run_base": bool(args.run_base),
        "cv_folds": int(args.cv_folds),
        "base_random_state": int(args.base_random_state),
        "target_recall": float(args.target_recall),
        "top_ks": list(TOP_KS),
        "bootstrap_n": int(args.bootstrap_n),
        "bootstrap_seed": int(args.bootstrap_seed),
        "cost_per_touch": float(args.cost_per_touch),
        "value_per_hit": float(args.value_per_hit),
        "sample_defs": scope_order,
        "dual_output_policy": {
            "ranking": "raw_score",
            "capacity_estimation": "calibrated_probability",
        },
    }
    (out_dir / "run_metadata_改进4.json").write_text(json.dumps(meta4, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"logit_improved_v4_done: {out_dir}")


if __name__ == "__main__":
    main()
