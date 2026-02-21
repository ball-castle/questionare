#!/usr/bin/env python3
"""Compare two run directories, score them, and generate recommendation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _load_logit_metrics(run_dir: Path) -> Dict[str, float]:
    rows = read_csv(run_dir / "tables" / "Logit模型指标.csv")
    main = next((r for r in rows if r.get("sample") == "main"), {})
    sens = next((r for r in rows if r.get("sample") == "sensitivity"), {})
    return {
        "main_auc": clamp01(to_float(main.get("auc"), 0.0)),
        "main_pseudo_r2": clamp01(to_float(main.get("pseudo_r2"), 0.0)),
        "sensitivity_auc": clamp01(to_float(sens.get("auc"), 0.0)),
    }


def _load_direction_and_sensitivity(run_dir: Path, n_samples: int) -> Dict[str, float]:
    path = run_dir / "tables" / "注意力题双口径对比.csv"
    if not path.exists():
        return {"direction_stability": 0.0, "sensitivity_ratio": 0.0}
    rows = read_csv(path)
    row = next((r for r in rows if "口径A" in str(r.get("calibration", ""))), rows[0] if rows else {})
    cons = to_float(row.get("consistent_direction_n"), 0.0)
    rev = to_float(row.get("reversed_direction_n"), 0.0)
    sens_n = to_float(row.get("sensitivity_n"), 0.0)
    den = cons + rev
    direction_stability = clamp01(cons / den) if den > 0 else 0.0
    sensitivity_ratio = clamp01(sens_n / max(1.0, float(n_samples)))
    return {"direction_stability": direction_stability, "sensitivity_ratio": sensitivity_ratio}


def _load_reliability(run_dir: Path) -> float:
    rows = read_csv(run_dir / "tables" / "信度分析表.csv")
    vals = [clamp01(to_float(r.get("alpha"), 0.0)) for r in rows]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _load_validity(run_dir: Path) -> float:
    rows = read_csv(run_dir / "tables" / "效度分析表.csv")
    if not rows:
        return 0.0
    r = rows[0]
    kmo = clamp01(to_float(r.get("kmo"), 0.0))
    p = to_float(r.get("bartlett_p"), 1.0)
    sig_adj = 1.0 if p < 0.05 else 0.5
    return clamp01(kmo * sig_adj)


def _load_visit_balance(run_dir: Path) -> float:
    rows = read_csv(run_dir / "tables" / "单选题频数百分比表.csv")
    q8 = [r for r in rows if str(r.get("col_idx")) == "8"]
    c1 = sum(to_float(r.get("count"), 0.0) for r in q8 if str(r.get("code")) == "1")
    c2 = sum(to_float(r.get("count"), 0.0) for r in q8 if str(r.get("code")) == "2")
    total = c1 + c2
    if total <= 0:
        return 0.0
    p_visit = c1 / total
    # Target 60/40 per legacy survey allocation setting.
    return clamp01(1.0 - abs(p_visit - 0.60) / 0.60)


def load_run(run_dir: Path, label: str) -> Dict[str, float]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    audit_path = run_dir / "conversion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

    n = int(to_float(meta.get("n_samples"), 0.0))
    attention = int(to_float(meta.get("attention_flag_n"), 0.0))
    logic = int(to_float(meta.get("logic_flag_n"), 0.0))
    duplicate = int(to_float(meta.get("duplicate_flag_n"), 0.0))
    straight = int(to_float(meta.get("straightline_flag_n"), 0.0))
    cleanliness = clamp01(1.0 - (attention + logic + duplicate + straight) / max(1.0, 4.0 * n))

    reliability = _load_reliability(run_dir)
    validity = _load_validity(run_dir)
    conversion_integrity = clamp01(to_float(audit.get("conversion_integrity"), 1.0))

    model = _load_logit_metrics(run_dir)
    model2 = _load_direction_and_sensitivity(run_dir, n)
    visit_balance = _load_visit_balance(run_dir)

    return {
        "label": label,
        "run_dir": str(run_dir),
        "n_samples": n,
        "attention_flag_n": attention,
        "logic_flag_n": logic,
        "duplicate_flag_n": duplicate,
        "straightline_flag_n": straight,
        "cluster_best_silhouette": to_float(meta.get("cluster_best_silhouette"), 0.0),
        "cleanliness": cleanliness,
        "reliability": reliability,
        "validity": validity,
        "conversion_integrity": conversion_integrity,
        "main_auc": model["main_auc"],
        "main_pseudo_r2": model["main_pseudo_r2"],
        "sensitivity_auc": model["sensitivity_auc"],
        "direction_stability": model2["direction_stability"],
        "sensitivity_ratio": model2["sensitivity_ratio"],
        "visit_balance": visit_balance,
        "conversion_unknown_rate": to_float(audit.get("unknown_value_rate"), 0.0),
        "conversion_unknown_count": int(to_float(audit.get("unknown_value_count"), 0.0)),
        "conversion_branch_conflict_count": int(to_float(audit.get("branch_conflict_count"), 0.0)),
    }


def score_runs(items: List[Dict[str, float]]) -> None:
    max_n = max(1.0, max(float(x["n_samples"]) for x in items))
    for x in items:
        sample_size_score = clamp01(float(x["n_samples"]) / max_n)
        q = 100.0 * (
            0.40 * x["cleanliness"]
            + 0.30 * x["reliability"]
            + 0.20 * x["validity"]
            + 0.10 * x["conversion_integrity"]
        )
        m = 100.0 * (
            0.35 * x["main_auc"]
            + 0.20 * x["main_pseudo_r2"]
            + 0.15 * x["sensitivity_auc"]
            + 0.15 * x["direction_stability"]
            + 0.15 * x["sensitivity_ratio"]
        )
        c = 100.0 * (0.70 * sample_size_score + 0.30 * x["visit_balance"])
        t = 0.50 * q + 0.30 * m + 0.20 * c
        x["sample_size_score"] = sample_size_score
        x["Q_score"] = q
        x["M_score"] = m
        x["C_score"] = c
        x["T_score"] = t


def choose_recommendation(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, str]:
    dt = abs(a["T_score"] - b["T_score"])
    dq = abs(a["Q_score"] - b["Q_score"])
    if dt >= 5.0:
        winner = a if a["T_score"] >= b["T_score"] else b
        rule = "总分差>=5，直接选总分更高"
    else:
        if dq >= 2.0:
            winner = a if a["Q_score"] >= b["Q_score"] else b
            rule = "总分接近，按质量分Q优先"
        else:
            if abs(a["direction_stability"] - b["direction_stability"]) > 1e-12:
                winner = a if a["direction_stability"] >= b["direction_stability"] else b
                rule = "总分和质量分接近，按方向稳定性优先"
            else:
                if a["label"] == "amethyst":
                    winner = a
                elif b["label"] == "amethyst":
                    winner = b
                else:
                    winner = a
                rule = "总分/质量/稳定性均接近，按默认保守策略优先amethyst"
    return {"winner": winner["label"], "rule": rule}


def write_recommendation(out_dir: Path, a: Dict[str, float], b: Dict[str, float], rec: Dict[str, str]) -> None:
    by_label = {a["label"]: a, b["label"]: b}
    w = by_label[rec["winner"]]
    l = b if w is a else a
    lines = [
        "双数据并行复跑推荐结论",
        "=" * 50,
        f"推荐数据集：{w['label']}",
        f"判定规则：{rec['rule']}",
        "",
        "核心证据：",
        f"1) 综合总分 T：{w['label']}={w['T_score']:.3f}，{l['label']}={l['T_score']:.3f}",
        f"2) 质量分 Q：{w['label']}={w['Q_score']:.3f}，{l['label']}={l['Q_score']:.3f}",
        f"3) 模型分 M：{w['label']}={w['M_score']:.3f}，{l['label']}={l['M_score']:.3f}",
        f"4) 覆盖分 C：{w['label']}={w['C_score']:.3f}，{l['label']}={l['C_score']:.3f}",
        f"5) 主样本AUC：{w['label']}={w['main_auc']:.4f}，{l['label']}={l['main_auc']:.4f}",
        f"6) 方向稳定性：{w['label']}={w['direction_stability']:.4f}，{l['label']}={l['direction_stability']:.4f}",
        f"7) 样本量：{w['label']}={int(w['n_samples'])}，{l['label']}={int(l['n_samples'])}",
        "",
        "风险提示：",
        f"- {a['label']} 转换未知值占比={a['conversion_unknown_rate']:.6f}，分支冲突数={a['conversion_branch_conflict_count']}",
        f"- {b['label']} 转换未知值占比={b['conversion_unknown_rate']:.6f}，分支冲突数={b['conversion_branch_conflict_count']}",
        "",
        "答辩口径建议：",
        "- 本次采用质量/模型/覆盖=50/30/20的稳健加权体系；",
        "- 当总分差距不足5分时，优先质量分与方向稳定性，最后才考虑历史兼容默认项。",
    ]
    (out_dir / "推荐结论.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two run directories and recommend one dataset.")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-a", default="amethyst")
    parser.add_argument("--label-b", default="new961")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = load_run(Path(args.run_a), args.label_a)
    b = load_run(Path(args.run_b), args.label_b)
    items = [a, b]
    score_runs(items)
    items_sorted = sorted(items, key=lambda x: x["T_score"], reverse=True)

    detail_fields = [
        "label",
        "run_dir",
        "n_samples",
        "attention_flag_n",
        "logic_flag_n",
        "duplicate_flag_n",
        "straightline_flag_n",
        "cluster_best_silhouette",
        "cleanliness",
        "reliability",
        "validity",
        "conversion_integrity",
        "main_auc",
        "main_pseudo_r2",
        "sensitivity_auc",
        "direction_stability",
        "sensitivity_ratio",
        "sample_size_score",
        "visit_balance",
        "Q_score",
        "M_score",
        "C_score",
        "T_score",
        "conversion_unknown_count",
        "conversion_unknown_rate",
        "conversion_branch_conflict_count",
    ]
    write_csv(out_dir / "指标对比明细.csv", detail_fields, items)

    score_rows = []
    for rank, x in enumerate(items_sorted, start=1):
        score_rows.append(
            {
                "rank": rank,
                "label": x["label"],
                "Q_score": f"{x['Q_score']:.6f}",
                "M_score": f"{x['M_score']:.6f}",
                "C_score": f"{x['C_score']:.6f}",
                "T_score": f"{x['T_score']:.6f}",
            }
        )
    write_csv(out_dir / "综合评分表.csv", ["rank", "label", "Q_score", "M_score", "C_score", "T_score"], score_rows)

    key_rows = []
    key_metrics = [
        "n_samples",
        "cleanliness",
        "reliability",
        "validity",
        "main_auc",
        "main_pseudo_r2",
        "direction_stability",
        "cluster_best_silhouette",
        "conversion_unknown_rate",
    ]
    for metric in key_metrics:
        key_rows.append(
            {
                "metric": metric,
                args.label_a: a.get(metric),
                args.label_b: b.get(metric),
            }
        )
    write_csv(out_dir / "关键图表对比.csv", ["metric", args.label_a, args.label_b], key_rows)

    rec = choose_recommendation(a, b)
    write_recommendation(out_dir, a, b, rec)
    print(f"compare_done: winner={rec['winner']} out={out_dir}")


if __name__ == "__main__":
    main()

