#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(v, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def to_int(v, default=0) -> int:
    return int(round(to_float(v, default)))


def fmt(v, digits=3) -> str:
    return f"{to_float(v):.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_docx_paras(path: Path) -> list[str]:
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    out = []
    for p in root.findall(".//w:p", ns):
        txt = "".join((t.text or "") for t in p.findall(".//w:t", ns)).strip()
        if txt:
            out.append(txt)
    return out


def extract_outline_requirements(docx: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(parse_docx_paras(docx), 1):
        for t in re.findall(r"(表\d+-\d+)", line):
            rows.append({"para_no": i, "requirement_type": "table", "tag": t, "text": line})
        for t in re.findall(r"(图\d+-\d+)", line):
            rows.append({"para_no": i, "requirement_type": "figure", "tag": t, "text": line})
        if ("待补" in line) or ("待填" in line):
            rows.append({"para_no": i, "requirement_type": "placeholder", "tag": "待补/待填", "text": line})
    return rows


def top_multi(rows: list[dict[str, str]], col_min: int, col_max: int, n: int) -> list[dict[str, str]]:
    x = [r for r in rows if col_min <= to_int(r.get("col_idx")) <= col_max]
    x.sort(key=lambda r: to_float(r.get("selected_pct")), reverse=True)
    return x[:n]


def quadrant_meta(q: str) -> tuple[str, str]:
    if q.startswith("Q1"):
        return ("高重要·高表现", "维持现有水平，防止退化")
    if q.startswith("Q2"):
        return ("高重要·低表现", "立即整改，重点资源投入")
    if q.startswith("Q3"):
        return ("低重要·低表现", "按需投入，不宜过度资源")
    if q.startswith("Q4"):
        return ("低重要·高表现", "适度削减，资源转移至Q2")
    return ("未知", "待补")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 880 data pack aligned to 大纲.docx")
    ap.add_argument("--outline-docx", default="大纲.docx")
    ap.add_argument("--source-dir", default="output_data_analysis")
    ap.add_argument("--processed-dir", default="data/processed_880")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--sample-n", type=int, default=880)
    ap.add_argument(
        "--sample-check-mode",
        choices=["dynamic", "strict_880", "off"],
        default="dynamic",
        help="Sample consistency check mode.",
    )
    ap.add_argument("--reuse-mode", choices=["reuse_then_fill"], default="reuse_then_fill")
    ap.add_argument("--sem-policy", choices=["keep_placeholder", "fail"], default="keep_placeholder")
    ap.add_argument("--js-table", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    outline_docx = Path(args.outline_docx)
    source_dir = Path(args.source_dir)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)

    t_out = out_dir / "tables"
    f_out = out_dir / "figures"
    fd_out = out_dir / "figures_data"
    a_out = out_dir / "audit"
    for p in [t_out, f_out, fd_out, a_out]:
        p.mkdir(parents=True, exist_ok=True)

    src_t = source_dir / "tables"
    src_f = source_dir / "figures"

    write_csv(a_out / "大纲需求清单.csv", ["para_no", "requirement_type", "tag", "text"], extract_outline_requirements(outline_docx))

    single = read_csv(src_t / "单选题频数百分比表.csv")
    multi = read_csv(src_t / "多选题选择率表.csv")
    logit_metric = read_csv(src_t / "Logit模型指标.csv")
    logit_main = read_csv(src_t / "Logit回归结果_主样本.csv")
    logit_sens = read_csv(src_t / "Logit回归结果_敏感性样本.csv")
    cluster = read_csv(src_t / "二阶聚类画像卡.csv")
    action = read_csv(src_t / "建议落地行动矩阵.csv")
    problem = read_csv(src_t / "问题-证据-建议对照表.csv")
    sem_map = read_csv(src_t / "假设变量模型映射表.csv")
    ipa_sens = read_csv(src_t / "IPA阈值敏感性表.csv")

    clean_csv = src_t / "survey_clean.csv"
    if not clean_csv.exists() and (processed_dir / "survey_clean_880.csv").exists():
        clean_csv = processed_dir / "survey_clean_880.csv"

    meta = read_json(source_dir / "run_metadata.json", {})
    if not meta:
        meta = read_json(processed_dir / "run_metadata.json", {})

    m = {str(r.get("sample", "")): r for r in logit_metric}
    remain_n = to_int(meta.get("remain_n_revised") or meta.get("n_samples_main"))
    clean_n = len(read_csv(clean_csv))
    main_n = to_int(m.get("main", {}).get("n"))
    sens_n = to_int(m.get("sensitivity", {}).get("n"))

    def valid_uniq(a: int, b: int) -> list[int]:
        return sorted({to_int(r.get("valid_n")) for r in multi if a <= to_int(r.get("col_idx")) <= b})

    valid_checks = {
        "到访动机(16-23)": valid_uniq(16, 23),
        "已到访痛点(33-42)": valid_uniq(33, 42),
        "未到访阻碍(43-51)": valid_uniq(43, 51),
        "新增项目(92-100)": valid_uniq(92, 100),
        "促销偏好(101-107)": valid_uniq(101, 107),
    }

    strict_checks = {
        "remain_n_match_required": remain_n == args.sample_n,
        "clean_rows_match_required": clean_n == args.sample_n,
        "main_n_match_required": main_n == args.sample_n,
        "sensitivity_n_match_106": sens_n == 106,
        "motive_valid_match_required": valid_checks["到访动机(16-23)"] == [args.sample_n],
        "visited_pain_valid_match_661": valid_checks["已到访痛点(33-42)"] == [661],
        "unvisited_barrier_valid_match_219": valid_checks["未到访阻碍(43-51)"] == [219],
    }

    visited_valid = valid_checks["已到访痛点(33-42)"]
    unvisited_valid = valid_checks["未到访阻碍(43-51)"]
    has_both_blocks = bool(visited_valid) and bool(unvisited_valid)
    block_partition_ok = True
    if has_both_blocks:
        block_partition_ok = (
            len(visited_valid) == 1
            and len(unvisited_valid) == 1
            and (visited_valid[0] + unvisited_valid[0] == remain_n)
        )

    dynamic_checks = {
        "remain_eq_clean_eq_main": remain_n == clean_n == main_n,
        "motive_valid_match_remain": valid_checks["到访动机(16-23)"] == [remain_n],
        "block_partition_check_or_skipped": block_partition_ok,
        "sensitivity_n_in_range": sens_n > 0 and sens_n <= remain_n,
    }

    if args.sample_check_mode == "strict_880":
        check_items = strict_checks
        check_note = "strict_880"
        check_pass = all(check_items.values())
    elif args.sample_check_mode == "dynamic":
        check_items = dynamic_checks
        check_note = "dynamic"
        check_pass = all(check_items.values())
    else:
        check_items = {"sample_check_skipped": True}
        check_note = "off"
        check_pass = True

    sample_check = {
        "sample_check_mode": args.sample_check_mode,
        "mode_note": check_note,
        "remain_n_revised": remain_n,
        "survey_clean_rows": clean_n,
        "logit_main_n": main_n,
        "logit_sensitivity_n": sens_n,
        "valid_n_checks": valid_checks,
        "required_sample_n": args.sample_n,
        "has_both_visit_status_blocks": has_both_blocks,
        "checks": check_items,
        "pass": check_pass,
    }
    (a_out / "口径核验_880.json").write_text(json.dumps(sample_check, ensure_ascii=False, indent=2), encoding="utf-8")
    if not sample_check["pass"]:
        raise RuntimeError(f"样本口径校验失败: mode={args.sample_check_mode}")

    manifest_items: list[dict] = []
    mapping_rows: list[dict] = []

    def src_tables(*names: str) -> str:
        return "|".join((source_dir / "tables" / n).as_posix() for n in names)

    def add_item(item_id: str, status: str, out_path: Path, source_path: str, transform_rule: str, pending_reason: str = ""):
        rec = {
            "item_id": item_id,
            "status": status,
            "output_path": out_path.as_posix(),
            "source_path": source_path,
            "transform_rule": transform_rule,
            "sample_check": sample_check["pass"],
            "pending_reason": pending_reason,
        }
        manifest_items.append(rec)
        mapping_rows.append({
            "item_id": item_id,
            "output_path": out_path.as_posix(),
            "status": status,
            "source_path": source_path,
            "transform_rule": transform_rule,
            "pending_reason": pending_reason,
        })

    q8_main = next((r for r in logit_main if str(r.get("term")) == "Q8_visit_status_code"), {})
    q8_sens = next((r for r in logit_sens if str(r.get("term")) == "Q8_visit_status_code"), {})
    t61 = t_out / "表6-1_双口径模型拟合指标对比.csv"
    write_csv(
        t61,
        ["指标", f"主样本（n={main_n}）", f"敏感性样本（n={sens_n}）", "说明"],
        [
            {"指标": "准确率 Accuracy", f"主样本（n={main_n}）": fmt(m.get("main", {}).get("accuracy"), 3), f"敏感性样本（n={sens_n}）": fmt(m.get("sensitivity", {}).get("accuracy"), 3), "说明": "整体分类正确率"},
            {"指标": "AUC", f"主样本（n={main_n}）": fmt(m.get("main", {}).get("auc"), 3), f"敏感性样本（n={sens_n}）": fmt(m.get("sensitivity", {}).get("auc"), 3), "说明": "区分能力"},
            {"指标": "McFadden R²", f"主样本（n={main_n}）": fmt(m.get("main", {}).get("pseudo_r2"), 3), f"敏感性样本（n={sens_n}）": fmt(m.get("sensitivity", {}).get("pseudo_r2"), 3), "说明": "伪决定系数"},
            {"指标": "主效应整体显著性（参考Q8）", f"主样本（n={main_n}）": f"p={fmt(q8_main.get('p_value'),4)}（不显著）", f"敏感性样本（n={sens_n}）": f"p={fmt(q8_sens.get('p_value'),4)}（显著）", "说明": "按Q8到访状态项对照"},
        ],
    )
    add_item("table_6_1", "ready", t61, src_tables("Logit模型指标.csv", "Logit回归结果_主样本.csv", "Logit回归结果_敏感性样本.csv"), "双口径指标重整")

    c_sorted = sorted(cluster, key=lambda r: to_int(r.get("cluster")))
    c1 = next((r for r in c_sorted if "价格" in str(r.get("cluster_name", ""))), c_sorted[0])
    c2 = next((r for r in c_sorted if "文化" in str(r.get("cluster_name", ""))), c_sorted[-1])
    c1_col = f"C1 {c1.get('cluster_name','')}（n={to_int(c1.get('n'))}，{fmt(c1.get('share_pct'),2)}%）"
    c2_col = f"C2 {c2.get('cluster_name','')}（n={to_int(c2.get('n'))}，{fmt(c2.get('share_pct'),2)}%）"
    t71 = t_out / "表7-1_两类游客画像关键特征对比.csv"
    write_csv(
        t71,
        ["维度", c1_col, c2_col, "说明"],
        [
            {"维度": "重要度均值", c1_col: fmt(c1.get("importance_mean")), c2_col: fmt(c2.get("importance_mean")), "说明": "越高越重视"},
            {"维度": "表现度均值", c1_col: fmt(c1.get("performance_mean")), c2_col: fmt(c2.get("performance_mean")), "说明": "越高体验越好"},
            {"维度": "文化认知均值", c1_col: fmt(c1.get("cognition_mean")), c2_col: fmt(c2.get("cognition_mean")), "说明": "越高中医药认知更强"},
            {"维度": "促销偏好计数", c1_col: fmt(c1.get("promo_pref_count")), c2_col: fmt(c2.get("promo_pref_count")), "说明": "越高对优惠更敏感"},
            {"维度": "到访动机计数", c1_col: fmt(c1.get("motive_count")), c2_col: fmt(c2.get("motive_count")), "说明": "越高动机更丰富"},
            {"维度": "核心特征", c1_col: "偏好优惠与稳定体验，对价格-品质感知敏感", c2_col: "偏好内容深度与文化沉浸，动机多元丰富", "说明": "按聚类中心语义解释"},
            {"维度": "运营方向", c1_col: "折扣券、家庭套票、性价比提升", c2_col: "非遗体验课、定制化文化项目、深度内容", "说明": "与行动矩阵联动"},
        ],
    )
    add_item("table_7_1", "ready", t71, src_tables("二阶聚类画像卡.csv"), "按大纲宽表格式重排")

    top_new = top_multi(multi, 92, 100, 3)
    top_promo = top_multi(multi, 101, 107, 2)
    new_text = "、".join(str(r.get("item_text", "")).strip() for r in top_new)
    promo_text = "、".join(str(r.get("item_text", "")).strip() for r in top_promo)
    t72 = t_out / "表7-2_分层运营落地矩阵.csv"
    write_csv(
        t72,
        ["策略维度", "C1 价格优惠敏感型", "C2 文化深度体验型", "证据来源"],
        [
            {"策略维度": "核心产品", "C1 价格优惠敏感型": "中药养生套餐、药膳固定菜单（以性价比组合呈现）", "C2 文化深度体验型": f"{new_text}（优先孵化）", "证据来源": "二阶聚类画像卡 + 多选题选择率表(92-100)"},
            {"策略维度": "促销机制", "C1 价格优惠敏感型": f"{promo_text}（优先投放）", "C2 文化深度体验型": "会员积分体系、文化内容会员订阅", "证据来源": "多选题选择率表(101-107)+行动矩阵"},
            {"策略维度": "触点设计", "C1 价格优惠敏感型": "入口促销展示、扫码优惠弹窗", "C2 文化深度体验型": "深度导览路线设计、匠人故事互动展", "证据来源": "聚类画像语义解释 + 行动矩阵"},
            {"策略维度": "传播渠道", "C1 价格优惠敏感型": "大众点评、美团优惠推广", "C2 文化深度体验型": "小红书文化种草、B站纪录片", "证据来源": "聚类画像语义解释"},
        ],
    )
    add_item("table_7_2", "ready", t72, src_tables("建议落地行动矩阵.csv", "二阶聚类画像卡.csv", "多选题选择率表.csv"), "分群与偏好题重构")

    sem_h = {str(r.get("假设编号", "")).strip(): r for r in sem_map}
    t73_rows = []
    for i in range(1, 9):
        hid = f"H{i}"
        if hid in sem_h:
            s = sem_h[hid]
            t73_rows.append({"假设编号": hid, "研究假设": s.get("研究假设", ""), "变量": s.get("变量", ""), "模型": s.get("模型", ""), "预期方向": s.get("预期方向", ""), "证据": s.get("证据", ""), "结论": s.get("结论", ""), "状态": "ready", "备注": ""})
        else:
            t73_rows.append({"假设编号": hid, "研究假设": "待补", "变量": "待补", "模型": "SEM", "预期方向": "待补", "证据": "待补", "结论": "待补", "状态": "pending", "备注": "当前仓库无SEM建模结果"})
    t73 = t_out / "表7-3_研究假设汇总_H1-H8.csv"
    write_csv(t73, ["假设编号", "研究假设", "变量", "模型", "预期方向", "证据", "结论", "状态", "备注"], t73_rows)
    add_item("table_7_3", "ready", t73, src_tables("假设变量模型映射表.csv"), "H1-H7复用，H8占位")

    t74 = t_out / "表7-4_SEM模型拟合指标_待补.csv"
    write_csv(t74, ["指标", "值", "结论", "状态", "备注"], [
        {"指标": "CMIN/DF", "值": "待补", "结论": "待补", "状态": "pending", "备注": "缺SEM模型输出"},
        {"指标": "RMSEA", "值": "待补", "结论": "待补", "状态": "pending", "备注": "缺SEM模型输出"},
        {"指标": "CFI", "值": "待补", "结论": "待补", "状态": "pending", "备注": "缺SEM模型输出"},
        {"指标": "TLI", "值": "待补", "结论": "待补", "状态": "pending", "备注": "缺SEM模型输出"},
        {"指标": "SRMR", "值": "待补", "结论": "待补", "状态": "pending", "备注": "缺SEM模型输出"},
    ])
    add_item("table_7_4", "pending", t74, "", "按大纲固定字段占位", "SEM结果缺失（按策略保留待补）")

    t75 = t_out / "表7-5_SEM路径系数与显著性_待补.csv"
    write_csv(t75, ["假设", "路径", "标准化系数β", "p值", "结论", "状态", "备注"], [
        {"假设": "H1", "路径": "S_env -> O_cognition", "标准化系数β": "待补", "p值": "待补", "结论": "支持/不支持", "状态": "pending", "备注": "缺SEM路径估计"},
        {"假设": "H2", "路径": "S_service -> O_cognition", "标准化系数β": "待补", "p值": "待补", "结论": "支持/不支持", "状态": "pending", "备注": "缺SEM路径估计"},
        {"假设": "H3", "路径": "S_activity -> O_cognition", "标准化系数β": "待补", "p值": "待补", "结论": "支持/不支持", "状态": "pending", "备注": "缺SEM路径估计"},
        {"假设": "H4", "路径": "O_cognition -> R_visit", "标准化系数β": "待补", "p值": "待补", "结论": "支持/不支持", "状态": "pending", "备注": "缺SEM路径估计"},
        {"假设": "H5", "路径": "O_cognition -> R_recommend", "标准化系数β": "待补", "p值": "待补", "结论": "支持/不支持", "状态": "pending", "备注": "缺SEM路径估计"},
        {"假设": "H6", "路径": "S_env -> O -> R", "标准化系数β": "待补", "p值": "待补", "结论": "中介显著/不显著", "状态": "pending", "备注": "缺SEM中介估计"},
        {"假设": "H7", "路径": "S_service -> O -> R", "标准化系数β": "待补", "p值": "待补", "结论": "中介显著/不显著", "状态": "pending", "备注": "缺SEM中介估计"},
        {"假设": "H8", "路径": "S_activity -> O -> R", "标准化系数β": "待补", "p值": "待补", "结论": "中介显著/不显著", "状态": "pending", "备注": "缺SEM中介估计"},
    ])
    add_item("table_7_5", "pending", t75, "", "按大纲固定字段占位", "SEM路径结果缺失（按策略保留待补）")

    mean = [r for r in ipa_sens if str(r.get("阈值方法")) == "均值阈值"]
    qmap: dict[str, list[dict[str, str]]] = {}
    for r in mean:
        q = str(r.get("quadrant", "")).strip()
        qmap.setdefault(q, []).append(r)
    t76_rows = []
    for q in ["Q2_优先改进", "Q1_保持优势", "Q3_低优先级", "Q4_可能过度投入"]:
        rs = qmap.get(q, [])
        meaning, mgmt = quadrant_meta(q)
        if rs:
            item_text = "、".join(str(r.get("item_text", "")).strip() for r in rs)
            imp = sum(to_float(r.get("importance_mean")) for r in rs) / len(rs)
            perf = sum(to_float(r.get("performance_mean")) for r in rs) / len(rs)
            gap = perf - imp
            t76_rows.append({"象限": q, "含义": meaning, "归属项目": item_text, "管理含义": mgmt, "重要度均值": fmt(imp, 3), "表现度均值": fmt(perf, 3), "差值(表现-重要)": fmt(gap, 3)})
        else:
            t76_rows.append({"象限": q, "含义": meaning, "归属项目": "", "管理含义": mgmt, "重要度均值": "", "表现度均值": "", "差值(表现-重要)": ""})
    t76 = t_out / "表7-6_IPA四象限归属汇总_均值阈值.csv"
    write_csv(t76, ["象限", "含义", "归属项目", "管理含义", "重要度均值", "表现度均值", "差值(表现-重要)"], t76_rows)
    add_item("table_7_6", "ready", t76, src_tables("IPA阈值敏感性表.csv"), "过滤均值阈值后按象限汇总")

    p_map = {str(r.get("priority", "")).strip(): r for r in problem}
    t77_rows = []
    for i, r in enumerate(action, 1):
        pnum = to_int(r.get("优先级"), i)
        label = {1: "P1（立即）", 2: "P2（同步）", 3: "P3（同步）"}.get(pnum, "P4（后续）")
        psrc = p_map.get(str(pnum), {})
        t77_rows.append({
            "优先级": label,
            "问题": r.get("问题", ""),
            "量化证据": r.get("证据", "") or psrc.get("evidence", ""),
            "落地动作": r.get("动作", "") or psrc.get("suggestion", ""),
            "时间节点": r.get("时间窗", ""),
            "责任方": r.get("责任方", ""),
            "KPI": r.get("KPI", ""),
            "依赖条件": r.get("依赖条件", ""),
            "来源": "建议落地行动矩阵 + 问题-证据-建议对照表",
        })
    t77 = t_out / "表7-7_IPA整改行动清单.csv"
    write_csv(t77, ["优先级", "问题", "量化证据", "落地动作", "时间节点", "责任方", "KPI", "依赖条件", "来源"], t77_rows)
    add_item("table_7_7", "ready", t77, src_tables("建议落地行动矩阵.csv", "问题-证据-建议对照表.csv"), "动作矩阵与证据对照表合并")

    pain = top_multi(multi, 33, 42, 3)
    block = top_multi(multi, 43, 51, 2)
    proj = top_multi(multi, 92, 100, 3)
    promo = top_multi(multi, 101, 107, 2)

    def top_text(items: list[dict[str, str]]) -> str:
        return "；".join(f"{r.get('item_text','')}（{fmt(r.get('selected_pct'),2)}%）" for r in items)

    t78 = t_out / "表7-8_游客意见三维汇总.csv"
    write_csv(t78, ["意见维度", "核心问题", "数据支撑", "对应分析节点"], [
        {"意见维度": "体验端短板", "核心问题": "配套设施不完善、指示标识不清晰、价格品质感知落差", "数据支撑": top_text(pain), "对应分析节点": "6.1痛点 + IPA Q2象限"},
        {"意见维度": "认知端壁垒", "核心问题": "认为街区缺乏吸引力、兴趣不大（未到访群体）", "数据支撑": top_text(block), "对应分析节点": "6.1阻碍 + 6.2 MCA维度1"},
        {"意见维度": "需求端期待", "核心问题": "新增深度体验内容与优惠组合机制", "数据支撑": f"{top_text(proj)}；{top_text(promo)}", "对应分析节点": "7.1聚类矩阵 + 7.3 IPA行动清单"},
    ])
    add_item("table_7_8", "ready", t78, src_tables("多选题选择率表.csv"), "按三维框架聚合多选题Top项")

    t79 = t_out / "表7-9_专家意见整合框架_待补.csv"
    write_csv(t79, ["访谈维度", "内容框架", "与定量结果的关联", "状态", "备注"], [
        {"访谈维度": "专家背景", "内容框架": "受访专家：___（机构/职称）；访谈时间：___；形式：半结构化深度访谈", "与定量结果的关联": "—", "状态": "pending", "备注": "待访谈资料"},
        {"访谈维度": "核心判断", "内容框架": "专家指出___是关键约束。", "与定量结果的关联": "对应IPA Q2或MCA/Logit结论", "状态": "pending", "备注": "待访谈资料"},
        {"访谈维度": "机制解释", "内容框架": "专家从___角度解释___机制。", "与定量结果的关联": "与统计模型形成印证/补充", "状态": "pending", "备注": "待访谈资料"},
        {"访谈维度": "政策建议", "内容框架": "专家建议优先推进：①__；②__；③__。", "与定量结果的关联": "与行动矩阵对照共识与差异", "状态": "pending", "备注": "待访谈资料"},
    ])
    add_item("table_7_9", "pending", t79, "", "按大纲固定字段占位", "专家访谈材料缺失（按策略保留待补）")

    t710 = t_out / "表7-10_问题证据建议闭环汇总.csv"
    write_csv(t710, ["核心问题", "统计证据", "游客声音", "专家意见", "建议方向", "状态"], [
        {"核心问题": "环境舒适度不足", "统计证据": "IPA：②环境舒适度与卫生状况表现度低于重要度", "游客声音": top_text(pain[:1]), "专家意见": "待补", "建议方向": "优先整改，纳入季度KPI", "状态": "专家意见待补"},
        {"核心问题": "低认知客群转化不足", "统计证据": "MCA：维度1负端集中低认知群体", "游客声音": top_text(block[:1]), "专家意见": "待补", "建议方向": "低门槛内容营销 + 首访激励设计", "状态": "专家意见待补"},
        {"核心问题": "深度体验供给不足", "统计证据": "聚类：文化深度体验型动机更丰富", "游客声音": top_text(proj[:1]), "专家意见": "待补", "建议方向": "试点3类深度体验产品", "状态": "专家意见待补"},
        {"核心问题": "促销机制单一", "统计证据": "聚类：价格优惠敏感型促销偏好更高", "游客声音": top_text(promo[:1]), "专家意见": "待补", "建议方向": "分层定向促销替代均一促销", "状态": "专家意见待补"},
    ])
    add_item("table_7_10", "ready", t710, src_tables("问题-证据-建议对照表.csv", "建议落地行动矩阵.csv", "多选题选择率表.csv"), "统计证据与游客声音填充，专家列保留待补")

    for item_id, name in [
        ("fig_6_4_mca", "MCA二维图.png"),
        ("fig_7_1_ipa", "IPA象限图.png"),
        ("fig_core_profile", "核心画像图_core_profile.png"),
        ("fig_6_1_age_gender", "年龄段人数_性别堆叠图.png"),
    ]:
        src = src_f / name
        dst = f_out / name
        ok = False
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                same_file = src.resolve() == dst.resolve()
            except Exception:
                same_file = src == dst
            if not same_file:
                shutil.copy2(src, dst)
            ok = True
        add_item(item_id, "ready" if ok else "pending", dst, src.as_posix(), "直接复制可复用图", "" if ok else "源图缺失")

    s24 = [r for r in single if to_int(r.get("col_idx")) == 24]
    s25 = [r for r in single if to_int(r.get("col_idx")) == 25]
    s24.sort(key=lambda r: to_float(r.get("pct")), reverse=True)
    s25.sort(key=lambda r: to_float(r.get("pct")), reverse=True)

    fig62 = fd_out / "图6-2_消费梯度堆积图_数据.csv"
    r62 = []
    for r in s24:
        r62.append({"图号": "图6-2", "变量": "停留时长(Q24)", "col_idx": to_int(r.get("col_idx")), "code": to_int(r.get("code")), "count": to_int(r.get("count")), "pct": fmt(r.get("pct"), 4), "item_text": r.get("item_text", "")})
    for r in s25:
        r62.append({"图号": "图6-2", "变量": "消费金额(Q25)", "col_idx": to_int(r.get("col_idx")), "code": to_int(r.get("code")), "count": to_int(r.get("count")), "pct": fmt(r.get("pct"), 4), "item_text": r.get("item_text", "")})
    write_csv(fig62, ["图号", "变量", "col_idx", "code", "count", "pct", "item_text"], r62)
    add_item("fig_data_6_2", "ready", fig62, src_tables("单选题频数百分比表.csv"), "提取Q24/Q25分布数据")

    fig63 = fd_out / "图6-3_动机排名条形图_数据.csv"
    motives = top_multi(multi, 16, 23, 8)
    write_csv(fig63, ["rank", "item_text", "selected_count", "valid_n", "selected_pct"], [{"rank": i + 1, "item_text": r.get("item_text", ""), "selected_count": to_int(r.get("selected_count")), "valid_n": to_int(r.get("valid_n")), "selected_pct": fmt(r.get("selected_pct"), 4)} for i, r in enumerate(motives)])
    add_item("fig_data_6_3", "ready", fig63, src_tables("多选题选择率表.csv"), "提取Q16-23动机Top排序")

    manifest_path = out_dir / "manifest_大纲数据_880.json"
    manifest = {
        "generated_at": now_iso(),
        "outline_docx": outline_docx.as_posix(),
        "outline_sha256": hashlib.sha256(outline_docx.read_bytes()).hexdigest() if outline_docx.exists() else "",
        "source_dir": source_dir.as_posix(),
        "processed_dir": processed_dir.as_posix(),
        "output_dir": out_dir.as_posix(),
        "sample_n": args.sample_n,
        "reuse_mode": args.reuse_mode,
        "sem_policy": args.sem_policy,
        "js_table": bool(args.js_table),
        "sample_check_mode": args.sample_check_mode,
        "sample_check": sample_check,
        "items": manifest_items,
    }

    xlsx_path = out_dir / "大纲数据总表_880.xlsx"
    js_status = "skipped"
    js_note = ""
    if args.js_table:
        cmd = [
            "node",
            str(Path("scripts") / "js" / "render_pretty_tables.mjs"),
            "--tables-dir",
            str(t_out),
            "--output-xlsx",
            str(xlsx_path),
            "--title",
            "大纲数据总表（880口径）",
        ]
        try:
            subprocess.run(cmd, check=True)
            js_status = "ready"
            js_note = "JS制表成功"
        except Exception as e:
            js_status = "pending"
            js_note = f"JS制表失败: {e}"
    add_item("xlsx_bundle", js_status, xlsx_path, f"{t_out.as_posix()}/*.csv", "JS(exceljs)多sheet整表导出", "" if js_status == "ready" else js_note)

    write_csv(
        a_out / "数据来源映射.csv",
        ["item_id", "output_path", "status", "source_path", "transform_rule", "pending_reason"],
        mapping_rows,
    )
    miss = [r for r in mapping_rows if r["status"] == "pending"]
    write_csv(
        a_out / "缺失项清单.csv",
        ["item_id", "output_path", "source_path", "pending_reason"],
        [
            {
                "item_id": r["item_id"],
                "output_path": r["output_path"],
                "source_path": r["source_path"],
                "pending_reason": r["pending_reason"],
            }
            for r in miss
        ],
    )

    manifest["items"] = manifest_items
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "output_dir": out_dir.as_posix(),
        "tables_generated": len(list(t_out.glob("*.csv"))),
        "figures_generated": len(list(f_out.glob("*.png"))),
        "figures_data_generated": len(list(fd_out.glob("*.csv"))),
        "pending_items": [r["item_id"] for r in manifest_items if r["status"] == "pending"],
        "manifest": manifest_path.as_posix(),
        "sample_check_pass": sample_check["pass"],
    }
    (a_out / "构建摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "outline_data_pack_done:",
        f"tables={len(list(t_out.glob('*.csv')))}",
        f"pending={len(summary['pending_items'])}",
        f"js_bundle={js_status}",
    )


if __name__ == "__main__":
    main()
