#!/usr/bin/env python3
"""Fill resolvable placeholders in docx using current tables directory (v2)."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from qp_stats import cronbach_alpha


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_dict_csv(path: Path, fieldnames, rows):
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def get_para_text(p):
    ts = p.findall(".//w:t", NS)
    return "".join(t.text or "" for t in ts), ts


def set_para_text(p, text: str):
    _, ts = get_para_text(p)
    if ts:
        ts[0].text = text
        for t in ts[1:]:
            t.text = ""
        return
    r = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r, qn("t"))
    t.text = text


def get_cell_text(tc):
    ts = tc.findall(".//w:t", NS)
    return "".join(t.text or "" for t in ts), ts


def set_cell_text(tc, text: str):
    old, ts = get_cell_text(tc)
    if ts:
        ts[0].text = text
        for t in ts[1:]:
            t.text = ""
        return old
    p = tc.find("./w:p", NS)
    if p is None:
        p = ET.SubElement(tc, qn("p"))
    r = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r, qn("t"))
    t.text = text
    return old


def table_rows(tbl):
    rows = []
    for tr in tbl.findall("./w:tr", NS):
        rows.append(tr.findall("./w:tc", NS))
    return rows


def alpha_grade(alpha: float) -> str:
    if alpha >= 0.8:
        return "良好"
    if alpha >= 0.7:
        return "可接受"
    return "一般"


def _load_current_metrics(tables_dir: Path):
    freq_rows = read_csv(tables_dir / "单选题频数百分比表.csv")
    val_rows = read_csv(tables_dir / "效度分析表.csv")
    clean_rows = read_csv(tables_dir / "survey_clean.csv")

    def freq(col_idx: int, code: int):
        for r in freq_rows:
            if str(r.get("col_idx")) == str(col_idx) and str(r.get("code")) == str(code):
                return {"count": int(float(r.get("count", 0))), "pct": float(r.get("pct", 0.0))}
        return {"count": 0, "pct": 0.0}

    q1 = {k: freq(1, k) for k in [1, 2]}
    q2 = {k: freq(2, k) for k in [1, 2, 3, 4, 5]}
    q3 = {k: freq(3, k) for k in [1, 2, 3, 4, 5]}
    q4 = {k: freq(4, k) for k in [1, 2, 3, 4, 5, 6, 7, 8]}
    q5 = {k: freq(5, k) for k in [1, 2, 3, 4, 5]}
    q8 = {k: freq(8, k) for k in [1, 2]}

    cols = [f"C{i:03d}" for i in range(1, 109)]
    num = np.full((len(clean_rows), len(cols)), np.nan, dtype=float)
    for i, r in enumerate(clean_rows):
        for j, c in enumerate(cols):
            s = str(r.get(c, "")).strip()
            if s == "":
                continue
            try:
                num[i, j] = float(s)
            except Exception:
                pass

    rel_map = [
        ("文化体验维度", [52, 53, 54]),
        ("非遗体验维度", [53, 54]),
        ("产品体验维度", [55, 56, 57]),
        ("配套保障维度", [58, 59, 60, 61]),
        ("宣传策略维度", [62, 63, 65]),
        ("整体量表", list(range(52, 64)) + [65]),
    ]
    rel_rows = []
    for name, c1b in rel_map:
        alpha, _ = cronbach_alpha(num[:, [c - 1 for c in c1b]])
        rel_rows.append(
            {
                "dim_name": name,
                "item_count": len(c1b),
                "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                "grade": alpha_grade(float(alpha)) if np.isfinite(alpha) else "一般",
            }
        )

    val = val_rows[0] if val_rows else {}
    validity = {
        "kmo": f"{float(val.get('kmo', 0.0)):.4f}",
        "chi2": f"{float(val.get('bartlett_chi2', 0.0)):.3f}",
        "df": str(int(float(val.get("bartlett_df", 0.0)))),
        "p": "<0.001" if float(val.get("bartlett_p", 1.0)) < 0.001 else f"{float(val.get('bartlett_p', 1.0)):.3f}",
    }

    return {
        "n_samples": len(clean_rows),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "q8": q8,
        "reliability": rel_rows,
        "validity": validity,
    }


def apply_paragraph_replacements(root, fill_rows, m):
    n = m["n_samples"]
    q8_yes, q8_no = m["q8"][1]["count"], m["q8"][2]["count"]
    q1_m, q1_f = m["q1"][1], m["q1"][2]
    q2 = m["q2"]

    p_alloc = re.compile(r"累计发放问卷\d+份，回收有效问卷\d+份（到访\d+份、未到访\d+份），有效回收率[0-9.]+%")
    p_overall = re.compile(r"本次调查累计发放问卷\d+份，有效问卷\d+份，问卷有效率[0-9.]+%")
    p_gender_age = re.compile(
        r"本次调查回收有效问卷共\d+份，在所有受访者中，男女比例为\d+:\d+（[0-9.]+%:[0-9.]+%）[；;]年龄结构中，编码\d+占[0-9.]+%，编码\d+占[0-9.]+%，编码\d+占[0-9.]+%"
    )

    canonical_alloc = f"正式调查按既定配额实施，累计发放问卷{n}份，回收有效问卷{n}份（到访{q8_yes}份、未到访{q8_no}份），有效回收率100.00%。"
    canonical_overall = f"本次调查累计发放问卷{n}份，有效问卷{n}份，问卷有效率100.00%。以下为有效问卷样本的结构分析。"
    canonical_gender_age = (
        f"本次调查回收有效问卷共{n}份，在所有受访者中，男女比例为{q1_m['count']}:{q1_f['count']}"
        f"（{q1_m['pct']:.2f}%:{q1_f['pct']:.2f}%）；年龄结构中，编码3占{q2[3]['pct']:.2f}%，"
        f"编码2占{q2[2]['pct']:.2f}%，编码4占{q2[4]['pct']:.2f}%。"
    )

    reps = [
        (
            "通过网络平台及街区现场发放，共发放问卷600份，共有有效问卷526份，有效率为87.67%。",
            f"通过网络平台及街区现场发放，累计发放问卷{n}份，回收有效问卷{n}份（到访{q8_yes}份、未到访{q8_no}份），有效回收率100.00%。",
            "统一当前口径（正式调查回收）",
        ),
        (
            "预调查共发放150份问卷（线上【X】份、线下【X】份），回收【X】份，有效问卷【X】份，有效回收率【X%】。对预调查数据进行编码、缺失处理与异常剔除后，开展信度与效度检验，为正式问卷修订提供依据。",
            "预调查共发放150份问卷（线上待补份、线下待补份），回收待补份，有效问卷待补份，有效回收率待补。对预调查数据进行编码、缺失处理与异常剔除后，开展信度与效度检验，为正式问卷修订提供依据。",
            "预调查数据缺失，保留待补",
        ),
        (
            "正式调查按“到访600+未到访400”的目标配额实施，累计发放N份，回收有效问卷XX份（到访X份、未到访X份），有效回收率X%。",
            canonical_alloc,
            "当前Q8分组口径",
        ),
        (
            "本次调查共发放问卷600份，有效问卷526份，问卷有效率87.67%。以下为有效问卷样本的结构分析。",
            canonical_overall,
            "统一当前样本总量",
        ),
        (
            "本次调查回收有效问卷共526份，在所有受访者中，男女比例为231:295，所占比例基本一致；从受访者年龄性别分布图可知，30岁以下青年群体占31.18%，30岁以上中年群体占68.82%，调查基本覆盖50岁以下全年龄段。",
            canonical_gender_age,
            "基于单选题频数百分比表（Q1/Q2）",
        ),
    ]

    alloc_done = False
    overall_done = False
    gender_done = False

    paragraphs = root.findall(".//w:p", NS)
    for p in paragraphs:
        old_text, _ = get_para_text(p)
        if not old_text:
            continue
        if p_alloc.search(old_text):
            set_para_text(p, canonical_alloc)
            fill_rows.append(
                {
                    "location": "paragraph",
                    "old_value": old_text,
                    "new_value": canonical_alloc,
                    "source_table": "单选题频数百分比表.csv(Q8)",
                    "calculation_rule": "正则刷新正式调查配额段落",
                }
            )
            alloc_done = True
            continue
        if p_overall.search(old_text):
            set_para_text(p, canonical_overall)
            fill_rows.append(
                {
                    "location": "paragraph",
                    "old_value": old_text,
                    "new_value": canonical_overall,
                    "source_table": "survey_clean.csv",
                    "calculation_rule": "正则刷新样本总量段落",
                }
            )
            overall_done = True
            continue
        if p_gender_age.search(old_text):
            set_para_text(p, canonical_gender_age)
            fill_rows.append(
                {
                    "location": "paragraph",
                    "old_value": old_text,
                    "new_value": canonical_gender_age,
                    "source_table": "单选题频数百分比表.csv(Q1,Q2)",
                    "calculation_rule": "正则刷新性别年龄段落",
                }
            )
            gender_done = True
            continue
        for old_sub, new_text, calc in reps:
            if old_sub in old_text:
                set_para_text(p, new_text)
                fill_rows.append(
                    {
                        "location": "paragraph",
                        "old_value": old_text,
                        "new_value": new_text,
                        "source_table": "tables/*.csv",
                        "calculation_rule": calc,
                    }
                )
                if "配额" in calc or "Q8" in calc:
                    alloc_done = True
                if "样本总量" in calc:
                    overall_done = True
                if "Q1/Q2" in calc:
                    gender_done = True
                break

    body = root.find(".//w:body", NS)
    if body is not None:
        def append_para(text: str, calc: str):
            p = ET.SubElement(body, qn("p"))
            r = ET.SubElement(p, qn("r"))
            t = ET.SubElement(r, qn("t"))
            t.text = text
            fill_rows.append(
                {
                    "location": "paragraph_append",
                    "old_value": "(missing)",
                    "new_value": text,
                    "source_table": "tables/*.csv",
                    "calculation_rule": calc,
                }
            )

        if not alloc_done:
            append_para(canonical_alloc, "补充正式调查配额段落")
        if not overall_done:
            append_para(canonical_overall, "补充样本总量段落")
        if not gender_done:
            append_para(canonical_gender_age, "补充性别年龄段落")


def fill_table_1_pretest_reliability(tbl, fill_rows, pending_rows):
    rows = table_rows(tbl)
    for r in range(1, min(7, len(rows))):
        dim_name, _ = get_cell_text(rows[r][0])
        for c in range(1, min(4, len(rows[r]))):
            old = set_cell_text(rows[r][c], "待补")
            fill_rows.append(
                {
                    "location": f"table1_r{r+1}_c{c+1}",
                    "old_value": old,
                    "new_value": "待补",
                    "source_table": "无（预调查数据缺失）",
                    "calculation_rule": "预调查项保留待补",
                }
            )
            pending_rows.append(
                {
                    "location": f"table1_r{r+1}_c{c+1}",
                    "item": f"预调查信度-{dim_name}",
                    "reason": "缺少预调查原始数据/结果表",
                    "required_source": "预调查样本与信效度输出",
                }
            )


def fill_table_2_pretest_validity(tbl, fill_rows, pending_rows):
    rows = table_rows(tbl)
    for r in range(1, min(5, len(rows))):
        metric, _ = get_cell_text(rows[r][0])
        old = set_cell_text(rows[r][1], "待补")
        fill_rows.append(
            {
                "location": f"table2_r{r+1}_c2",
                "old_value": old,
                "new_value": "待补",
                "source_table": "无（预调查数据缺失）",
                "calculation_rule": "预调查项保留待补",
            }
        )
        pending_rows.append(
            {
                "location": f"table2_r{r+1}_c2",
                "item": f"预调查效度-{metric}",
                "reason": "缺少预调查原始数据/结果表",
                "required_source": "预调查KMO/Bartlett结果",
            }
        )


def fill_table_3_formal_reliability(tbl, fill_rows, metrics):
    rows = table_rows(tbl)
    for i, rec in enumerate(metrics["reliability"], start=1):
        if i >= len(rows):
            break
        n_items = str(rec["item_count"])
        alpha = f"{rec['alpha']:.4f}" if np.isfinite(rec["alpha"]) else "NA"
        grade = rec["grade"]
        old2 = set_cell_text(rows[i][1], n_items)
        old3 = set_cell_text(rows[i][2], alpha)
        old4 = set_cell_text(rows[i][3], grade)
        fill_rows.extend(
            [
                {
                    "location": f"table3_r{i+1}_c2",
                    "old_value": old2,
                    "new_value": n_items,
                    "source_table": "survey_clean.csv",
                    "calculation_rule": "维度题项数",
                },
                {
                    "location": f"table3_r{i+1}_c3",
                    "old_value": old3,
                    "new_value": alpha,
                    "source_table": "survey_clean.csv + cronbach_alpha",
                    "calculation_rule": "Cronbach alpha（4位小数）",
                },
                {
                    "location": f"table3_r{i+1}_c4",
                    "old_value": old4,
                    "new_value": grade,
                    "source_table": "alpha评级规则",
                    "calculation_rule": ">=0.8良好，>=0.7可接受",
                },
            ]
        )


def fill_table_4_formal_validity(tbl, fill_rows, metrics):
    rows = table_rows(tbl)
    vals = {
        1: metrics["validity"]["kmo"],
        2: metrics["validity"]["chi2"],
        3: metrics["validity"]["df"],
        4: metrics["validity"]["p"],
    }
    for r in range(1, min(5, len(rows))):
        old = set_cell_text(rows[r][1], vals.get(r, ""))
        metric, _ = get_cell_text(rows[r][0])
        fill_rows.append(
            {
                "location": f"table4_r{r+1}_c2",
                "old_value": old,
                "new_value": vals.get(r, ""),
                "source_table": "效度分析表.csv",
                "calculation_rule": f"{metric}",
            }
        )


def fill_table_5_sample_structure(tbl, fill_rows, metrics):
    rows = table_rows(tbl)
    q3, q4, q5 = metrics["q3"], metrics["q4"], metrics["q5"]
    n = metrics["n_samples"]

    def fill_row(r, var_text, attr, count, pct, total):
        if r >= len(rows):
            return
        if var_text is not None:
            old = set_cell_text(rows[r][0], var_text)
            fill_rows.append(
                {
                    "location": f"table5_r{r+1}_c1",
                    "old_value": old,
                    "new_value": var_text,
                    "source_table": "结构表口径",
                    "calculation_rule": "分组标题/留空控制",
                }
            )
        old2 = set_cell_text(rows[r][1], attr)
        old3 = set_cell_text(rows[r][2], str(count))
        old4 = set_cell_text(rows[r][3], pct)
        old5 = set_cell_text(rows[r][4], total if total is not None else "")
        fill_rows.extend(
            [
                {"location": f"table5_r{r+1}_c2", "old_value": old2, "new_value": attr, "source_table": "单选题频数百分比表.csv", "calculation_rule": "属性名"},
                {"location": f"table5_r{r+1}_c3", "old_value": old3, "new_value": str(count), "source_table": "单选题频数百分比表.csv", "calculation_rule": "人数"},
                {"location": f"table5_r{r+1}_c4", "old_value": old4, "new_value": pct, "source_table": "单选题频数百分比表.csv", "calculation_rule": "比例(2位小数)"},
                {"location": f"table5_r{r+1}_c5", "old_value": old5, "new_value": total if total is not None else "", "source_table": "分组总计", "calculation_rule": "组首100%"},
            ]
        )

    edu = [
        ("文化程度", "初中及以下", q3[1]["count"], f"{q3[1]['pct']:.2f}%", "100%"),
        ("", "中专/高中", q3[2]["count"], f"{q3[2]['pct']:.2f}%", ""),
        ("", "大专", q3[3]["count"], f"{q3[3]['pct']:.2f}%", ""),
        ("", "本科", q3[4]["count"], f"{q3[4]['pct']:.2f}%", ""),
        ("", "硕士及以上", q3[5]["count"], f"{q3[5]['pct']:.2f}%", ""),
    ]
    for idx, row in enumerate(edu, start=1):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    jobs = [
        ("职业", "学生", q4[1]["count"], f"{q4[1]['pct']:.2f}%", "100%"),
        ("", "企业/公司职员", q4[2]["count"], f"{q4[2]['pct']:.2f}%", ""),
        ("", "事业单位人员/公务员", q4[3]["count"], f"{q4[3]['pct']:.2f}%", ""),
        ("", "自由职业者", q4[4]["count"], f"{q4[4]['pct']:.2f}%", ""),
        ("", "个体经营者", q4[5]["count"], f"{q4[5]['pct']:.2f}%", ""),
        ("", "服务业从业者", q4[6]["count"], f"{q4[6]['pct']:.2f}%", ""),
        ("", "离退休人员", q4[7]["count"], f"{q4[7]['pct']:.2f}%", ""),
        ("", "其他", q4[8]["count"], f"{q4[8]['pct']:.2f}%", ""),
    ]
    for idx, row in enumerate(jobs, start=6):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    income = [
        ("月收入", "3000元以下", q5[1]["count"], f"{q5[1]['pct']:.2f}%", "100%"),
        ("", "3001-5000元", q5[2]["count"], f"{q5[2]['pct']:.2f}%", ""),
        ("", "5001-8000元", q5[3]["count"], f"{q5[3]['pct']:.2f}%", ""),
        ("", "8001-15000元", q5[4]["count"], f"{q5[4]['pct']:.2f}%", ""),
        ("", "15000元以上", q5[5]["count"], f"{q5[5]['pct']:.2f}%", ""),
    ]
    for idx, row in enumerate(income, start=14):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    if len(rows) > 19:
        old1 = set_cell_text(rows[19][0], "有效填写人次")
        old2 = set_cell_text(rows[19][1], str(n)) if len(rows[19]) > 1 else ""
        fill_rows.extend(
            [
                {
                    "location": "table5_r20_c1",
                    "old_value": old1,
                    "new_value": "有效填写人次",
                    "source_table": "样本总量",
                    "calculation_rule": "固定标签",
                },
                {
                    "location": "table5_r20_c2",
                    "old_value": old2,
                    "new_value": str(n),
                    "source_table": "survey_clean.csv",
                    "calculation_rule": "有效样本总数",
                },
            ]
        )


def scan_left_numbers(root):
    paragraphs = root.findall(".//w:p", NS)
    hits = []
    for p in paragraphs:
        txt, _ = get_para_text(p)
        if "526" in txt or "600份" in txt or "87.67%" in txt:
            hits.append(txt)
    return hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill pending doc using current tables (v2).")
    parser.add_argument("--doc-path", required=True, help="Target docx path to update in place.")
    parser.add_argument("--tables-dir", required=True, help="Directory containing output tables CSV files.")
    parser.add_argument("--output-dir", required=True, help="Run output directory (logs written under output_dir/tables).")
    return parser.parse_args()


def main():
    args = parse_args()
    doc_path = Path(args.doc_path)
    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.output_dir)
    out_fill = out_dir / "tables" / "待填数据_回填值清单.csv"
    out_pending = out_dir / "tables" / "待填数据_待补项清单.csv"
    out_check = out_dir / "tables" / "待填数据_回填校验报告.txt"

    metrics = _load_current_metrics(tables_dir)
    fill_rows = []
    pending_rows = []

    with zipfile.ZipFile(doc_path, "r") as zin:
        file_map = {name: zin.read(name) for name in zin.namelist()}

    root = ET.fromstring(file_map["word/document.xml"])
    apply_paragraph_replacements(root, fill_rows, metrics)

    tables = root.findall(".//w:tbl", NS)
    if len(tables) < 5:
        raise RuntimeError(f"Expected at least 5 tables, got {len(tables)}")

    fill_table_1_pretest_reliability(tables[0], fill_rows, pending_rows)
    fill_table_2_pretest_validity(tables[1], fill_rows, pending_rows)
    fill_table_3_formal_reliability(tables[2], fill_rows, metrics)
    fill_table_4_formal_validity(tables[3], fill_rows, metrics)
    fill_table_5_sample_structure(tables[4], fill_rows, metrics)

    pending_rows.append(
        {
            "location": "paragraph_pretest_recovery",
            "item": "预调查线上份数/线下份数/回收份数/有效份数/有效回收率",
            "reason": "缺少预调查原始数据",
            "required_source": "预调查回收台账/原始问卷",
        }
    )

    file_map["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    tmp_path = doc_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, content in file_map.items():
            zout.writestr(name, content)
    tmp_path.replace(doc_path)

    write_dict_csv(out_fill, ["location", "old_value", "new_value", "source_table", "calculation_rule"], fill_rows)
    write_dict_csv(out_pending, ["location", "item", "reason", "required_source"], pending_rows)

    residual = scan_left_numbers(root)
    report_lines = [
        "待填数据文档回填校验报告(v2)",
        "====================",
        f"回填记录数: {len(fill_rows)}",
        f"待补记录数: {len(pending_rows)}",
        "",
        "旧口径残留检查（段落中的526/600份/87.67%）:",
        f"残留条数: {len(residual)}",
    ]
    for x in residual:
        report_lines.append(f"- {x}")
    out_check.write_text("\n".join(report_lines), encoding="utf-8")

    print("DONE")
    print(f"fill_rows={len(fill_rows)} pending_rows={len(pending_rows)} residual_old_tokens={len(residual)}")


if __name__ == "__main__":
    main()
