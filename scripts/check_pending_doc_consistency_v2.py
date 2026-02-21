#!/usr/bin/env python3
"""Validate consistency between pending doc and current run outputs (v2)."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from convert_961_to_108 import convert_961_to_108
from qp_io import numeric_matrix, read_xlsx_first_sheet
from qp_stats import cronbach_alpha, kmo_bartlett, freq_table


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or ""))


def read_docx_xml_root(doc_path: Path):
    with zipfile.ZipFile(doc_path, "r") as zf:
        return ET.fromstring(zf.read("word/document.xml"))


def paragraph_texts(root):
    out = []
    for p in root.findall(".//w:p", NS):
        txt = "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()
        if txt:
            out.append(txt)
    return out


def table_rows(tbl):
    rows = []
    for tr in tbl.findall("./w:tr", NS):
        cells = []
        for tc in tr.findall("./w:tc", NS):
            txt = "".join(t.text or "" for t in tc.findall(".//w:t", NS)).strip()
            cells.append(txt)
        rows.append(cells)
    return rows


def safe_cell(rows, r, c):
    if r < 0 or c < 0:
        return ""
    if r >= len(rows):
        return ""
    if c >= len(rows[r]):
        return ""
    return (rows[r][c] or "").strip()


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def alpha_grade(alpha):
    if alpha >= 0.8:
        return "良好"
    if alpha >= 0.7:
        return "可接受"
    return "一般"


def freq_as_dict(vec):
    out = {}
    for r in freq_table(vec):
        out[int(r["code"])] = {"count": int(r["count"]), "pct": float(r["pct"])}
    return out


def format_pct(pct):
    return f"{pct:.2f}%"


def add_result(results, check_item, doc_value, recomputed_value, source, status, note):
    results.append(
        {
            "check_item": check_item,
            "doc_value": doc_value,
            "recomputed_value": recomputed_value,
            "source": source,
            "status": status,
            "note": note,
        }
    )


def add_compare(results, check_item, doc_value, recomputed_value, source, note):
    status = "PASS" if norm_text(doc_value) == norm_text(recomputed_value) else "FAIL"
    add_result(results, check_item, doc_value, recomputed_value, source, status, note)


def _load_raw_as_108(raw_xlsx_path: Path):
    headers, rows_dense = read_xlsx_first_sheet(raw_xlsx_path)
    if len(headers) == 64:
        conv = convert_961_to_108(headers, rows_dense)
        return conv.headers_108, conv.rows_108
    return headers, rows_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check consistency between pending doc and current outputs (v2).")
    parser.add_argument("--doc-path", required=True, help="Target docx path.")
    parser.add_argument("--raw-xlsx", required=True, help="Raw xlsx path (64 or 108 columns).")
    parser.add_argument("--tables-dir", required=True, help="Current output tables directory.")
    parser.add_argument("--output-dir", required=True, help="Current output root directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    doc_path = Path(args.doc_path)
    raw_xlsx_path = Path(args.raw_xlsx)
    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    out_path = output_dir / "tables" / "待填数据_一致性核查报告.csv"

    headers, rows_dense = _load_raw_as_108(raw_xlsx_path)
    num_raw, _ = numeric_matrix(rows_dense)
    num = num_raw.copy()
    num[:, 32:51] = np.where(num[:, 32:51] == -3, np.nan, num[:, 32:51])

    n_samples = len(rows_dense)
    q1 = freq_as_dict(num[:, 0])
    q2 = freq_as_dict(num[:, 1])
    q3 = freq_as_dict(num[:, 2])
    q4 = freq_as_dict(num[:, 3])
    q5 = freq_as_dict(num[:, 4])
    q8 = freq_as_dict(num[:, 7])

    val_cols = list(range(52, 64)) + [65] + list(range(66, 86)) + list(range(86, 90))
    validity = kmo_bartlett(num[:, [c - 1 for c in val_cols]])

    rel_map = [
        ("文化体验维度", [52, 53, 54], 1),
        ("非遗体验维度", [53, 54], 2),
        ("产品体验维度", [55, 56, 57], 3),
        ("配套保障维度", [58, 59, 60, 61], 4),
        ("宣传策略维度", [62, 63, 65], 5),
        ("整体量表", list(range(52, 64)) + [65], 6),
    ]

    root = read_docx_xml_root(doc_path)
    paras = paragraph_texts(root)
    tables = root.findall(".//w:tbl", NS)
    results = []

    old_hits = [p for p in paras if ("526" in p or "600份" in p or "87.67%" in p)]
    add_result(
        results,
        "paragraph.old_caliber_tokens",
        str(len(old_hits)),
        "0",
        str(doc_path),
        "PASS" if len(old_hits) == 0 else "FAIL",
        "检查段落是否残留526/600份/87.67%旧口径",
    )

    expected_alloc = (
        str(n_samples),
        str(n_samples),
        str(q8.get(1, {"count": 0})["count"]),
        str(q8.get(2, {"count": 0})["count"]),
        "100.00",
    )
    alloc_pattern = re.compile(r"累计发放问卷(\d+)份，回收有效问卷(\d+)份（到访(\d+)份、未到访(\d+)份），有效回收率([0-9.]+)%")
    alloc_matches = []
    for p in paras:
        alloc_matches.extend(alloc_pattern.findall(p))
    if not alloc_matches:
        add_result(
            results,
            "paragraph.formal_allocation",
            "(missing)",
            f"累计发放问卷{n_samples}份，回收有效问卷{n_samples}份（到访{expected_alloc[2]}份、未到访{expected_alloc[3]}份），有效回收率100.00%",
            f"{doc_path} + {raw_xlsx_path}(Q8)",
            "FAIL",
            "未找到正式调查配额段落",
        )
    else:
        for i, m in enumerate(alloc_matches, start=1):
            add_result(
                results,
                f"paragraph.formal_allocation.{i}",
                f"{m[0]}/{m[1]}/{m[2]}/{m[3]}/{m[4]}%",
                f"{expected_alloc[0]}/{expected_alloc[1]}/{expected_alloc[2]}/{expected_alloc[3]}/{expected_alloc[4]}%",
                f"{doc_path} + {raw_xlsx_path}(Q8)",
                "PASS" if tuple(m) == expected_alloc else "FAIL",
                "正式调查回收口径",
            )

    overall_pattern = re.compile(r"本次调查累计发放问卷(\d+)份，有效问卷(\d+)份，问卷有效率([0-9.]+)%")
    overall_matches = []
    for p in paras:
        overall_matches.extend(overall_pattern.findall(p))
    expected_overall = (str(n_samples), str(n_samples), "100.00")
    if not overall_matches:
        add_result(
            results,
            "paragraph.overall_sample",
            "(missing)",
            f"本次调查累计发放问卷{n_samples}份，有效问卷{n_samples}份，问卷有效率100.00%",
            f"{doc_path} + {raw_xlsx_path}",
            "FAIL",
            "未找到样本结构段落",
        )
    else:
        for i, m in enumerate(overall_matches, start=1):
            add_result(
                results,
                f"paragraph.overall_sample.{i}",
                f"{m[0]}/{m[1]}/{m[2]}%",
                f"{expected_overall[0]}/{expected_overall[1]}/{expected_overall[2]}%",
                f"{doc_path} + {raw_xlsx_path}",
                "PASS" if tuple(m) == expected_overall else "FAIL",
                "正式调查总样本口径",
            )

    gender_age_pattern = re.compile(
        r"本次调查回收有效问卷共(\d+)份，在所有受访者中，男女比例为(\d+):(\d+)（([0-9.]+)%:([0-9.]+)%）[；;]年龄结构中，编码(\d+)占([0-9.]+)%，编码(\d+)占([0-9.]+)%，编码(\d+)占([0-9.]+)%"
    )
    ga_matches = []
    for p in paras:
        ga_matches.extend(gender_age_pattern.findall(p))
    expected_ga = (
        str(n_samples),
        str(q1.get(1, {"count": 0})["count"]),
        str(q1.get(2, {"count": 0})["count"]),
        f"{q1.get(1, {'pct': 0.0})['pct']:.2f}",
        f"{q1.get(2, {'pct': 0.0})['pct']:.2f}",
        "3",
        f"{q2.get(3, {'pct': 0.0})['pct']:.2f}",
        "2",
        f"{q2.get(2, {'pct': 0.0})['pct']:.2f}",
        "4",
        f"{q2.get(4, {'pct': 0.0})['pct']:.2f}",
    )
    if not ga_matches:
        add_result(
            results,
            "paragraph.gender_age",
            "(missing)",
            "n;male:female;pct;age-top3",
            f"{doc_path} + {raw_xlsx_path}(Q1,Q2)",
            "FAIL",
            "未找到性别年龄段落",
        )
    else:
        for i, m in enumerate(ga_matches, start=1):
            add_result(
                results,
                f"paragraph.gender_age.{i}",
                ";".join(m),
                ";".join(expected_ga),
                f"{doc_path} + {raw_xlsx_path}(Q1,Q2)",
                "PASS" if tuple(m) == expected_ga else "FAIL",
                "样本结构（性别+年龄核心占比）",
            )

    if len(tables) < 5:
        add_result(
            results,
            "doc.table_count",
            str(len(tables)),
            ">=5",
            str(doc_path),
            "FAIL",
            "文档表格数量不足，无法继续逐表核查",
        )
        write_csv(out_path, ["check_item", "doc_value", "recomputed_value", "source", "status", "note"], results)
        print(f"DONE: report={out_path}")
        return

    t3 = table_rows(tables[2])
    for dim_name, cols, row_idx in rel_map:
        doc_dim = safe_cell(t3, row_idx, 0)
        doc_n = safe_cell(t3, row_idx, 1)
        doc_alpha = safe_cell(t3, row_idx, 2)
        doc_grade = safe_cell(t3, row_idx, 3)

        alpha, _ = cronbach_alpha(num[:, [c - 1 for c in cols]])
        calc_n = str(len(cols))
        calc_alpha = f"{alpha:.4f}"
        calc_grade = alpha_grade(alpha)
        col_note = ",".join([f"C{c:03d}" for c in cols])

        add_compare(results, f"table3.{dim_name}.dim_name", doc_dim, dim_name, f"{doc_path}(table3)", "维度名称一致性")
        add_compare(results, f"table3.{dim_name}.item_count", doc_n, calc_n, f"{raw_xlsx_path}({col_note})", "题项数")
        add_compare(results, f"table3.{dim_name}.alpha", doc_alpha, calc_alpha, f"{raw_xlsx_path}({col_note}) + cronbach_alpha", "Cronbach alpha（4位小数）")
        add_compare(results, f"table3.{dim_name}.grade", doc_grade, calc_grade, f"alpha={calc_alpha}", "评级规则：>=0.8良好，>=0.7可接受")

    t4 = table_rows(tables[3])
    doc_kmo = safe_cell(t4, 1, 1)
    doc_chi2 = safe_cell(t4, 2, 1)
    doc_df = safe_cell(t4, 3, 1)
    doc_p = safe_cell(t4, 4, 1)
    calc_kmo = f"{validity['kmo']:.4f}"
    calc_chi2 = f"{validity['bartlett_chi2']:.3f}"
    calc_df = str(int(validity["bartlett_df"]))
    calc_p = "<0.001" if float(validity["bartlett_p"]) < 0.001 else f"{float(validity['bartlett_p']):.3f}"
    add_compare(results, "table4.KMO", doc_kmo, calc_kmo, f"{raw_xlsx_path}(C052-C089) + kmo_bartlett", "KMO（4位小数）")
    add_compare(results, "table4.Bartlett_chi2", doc_chi2, calc_chi2, f"{raw_xlsx_path}(C052-C089) + kmo_bartlett", "Bartlett近似卡方（3位小数）")
    add_compare(results, "table4.Bartlett_df", doc_df, calc_df, f"{raw_xlsx_path}(C052-C089) + kmo_bartlett", "自由度df")
    add_compare(results, "table4.Bartlett_p", doc_p, calc_p, f"{raw_xlsx_path}(C052-C089) + kmo_bartlett", "显著性p")

    t5 = table_rows(tables[4])
    edu_rows = [(1, "文化程度", "初中及以下", 1, "100%"), (2, "", "中专/高中", 2, ""), (3, "", "大专", 3, ""), (4, "", "本科", 4, ""), (5, "", "硕士及以上", 5, "")]
    occ_rows = [
        (6, "职业", "学生", 1, "100%"),
        (7, "", "企业/公司职员", 2, ""),
        (8, "", "事业单位人员/公务员", 3, ""),
        (9, "", "自由职业者", 4, ""),
        (10, "", "个体经营者", 5, ""),
        (11, "", "服务业从业者", 6, ""),
        (12, "", "离退休人员", 7, ""),
        (13, "", "其他", 8, ""),
    ]
    inc_rows = [(14, "月收入", "3000元以下", 1, "100%"), (15, "", "3001-5000元", 2, ""), (16, "", "5001-8000元", 3, ""), (17, "", "8001-15000元", 4, ""), (18, "", "15000元以上", 5, "")]

    for row_idx, expected_var, expected_attr, code, expected_total in edu_rows:
        d = q3.get(code, {"count": 0, "pct": 0.0})
        add_compare(results, f"table5.row{row_idx+1}.var", safe_cell(t5, row_idx, 0), expected_var, f"{doc_path}(table5)", "分组标题")
        add_compare(results, f"table5.row{row_idx+1}.attr", safe_cell(t5, row_idx, 1), expected_attr, f"{doc_path}(table5)", "属性名称")
        add_compare(results, f"table5.row{row_idx+1}.count", safe_cell(t5, row_idx, 2), str(d["count"]), f"{raw_xlsx_path}(Q3)", "人数")
        add_compare(results, f"table5.row{row_idx+1}.pct", safe_cell(t5, row_idx, 3), format_pct(d["pct"]), f"{raw_xlsx_path}(Q3)", "比例（2位小数）")
        add_compare(results, f"table5.row{row_idx+1}.total", safe_cell(t5, row_idx, 4), expected_total, f"{doc_path}(table5)", "组首行为100%，其余留空")

    for row_idx, expected_var, expected_attr, code, expected_total in occ_rows:
        d = q4.get(code, {"count": 0, "pct": 0.0})
        add_compare(results, f"table5.row{row_idx+1}.var", safe_cell(t5, row_idx, 0), expected_var, f"{doc_path}(table5)", "分组标题")
        add_compare(results, f"table5.row{row_idx+1}.attr", safe_cell(t5, row_idx, 1), expected_attr, f"{doc_path}(table5)", "属性名称")
        add_compare(results, f"table5.row{row_idx+1}.count", safe_cell(t5, row_idx, 2), str(d["count"]), f"{raw_xlsx_path}(Q4)", "人数")
        add_compare(results, f"table5.row{row_idx+1}.pct", safe_cell(t5, row_idx, 3), format_pct(d["pct"]), f"{raw_xlsx_path}(Q4)", "比例（2位小数）")
        add_compare(results, f"table5.row{row_idx+1}.total", safe_cell(t5, row_idx, 4), expected_total, f"{doc_path}(table5)", "组首行为100%，其余留空")

    for row_idx, expected_var, expected_attr, code, expected_total in inc_rows:
        d = q5.get(code, {"count": 0, "pct": 0.0})
        add_compare(results, f"table5.row{row_idx+1}.var", safe_cell(t5, row_idx, 0), expected_var, f"{doc_path}(table5)", "分组标题")
        add_compare(results, f"table5.row{row_idx+1}.attr", safe_cell(t5, row_idx, 1), expected_attr, f"{doc_path}(table5)", "属性名称")
        add_compare(results, f"table5.row{row_idx+1}.count", safe_cell(t5, row_idx, 2), str(d["count"]), f"{raw_xlsx_path}(Q5)", "人数")
        add_compare(results, f"table5.row{row_idx+1}.pct", safe_cell(t5, row_idx, 3), format_pct(d["pct"]), f"{raw_xlsx_path}(Q5)", "比例（2位小数）")
        add_compare(results, f"table5.row{row_idx+1}.total", safe_cell(t5, row_idx, 4), expected_total, f"{doc_path}(table5)", "组首行为100%，其余留空")

    add_compare(results, "table5.valid_n.label", safe_cell(t5, 19, 0), "有效填写人次", f"{doc_path}(table5)", "汇总标签")
    add_compare(results, "table5.valid_n.value", safe_cell(t5, 19, 1), str(n_samples), str(raw_xlsx_path), "样本总数")

    pending_path = output_dir / "tables" / "待填数据_待补项清单.csv"
    if not pending_path.exists():
        alt = tables_dir / "待填数据_待补项清单.csv"
        pending_path = alt if alt.exists() else pending_path
    pending_rows = read_csv(pending_path) if pending_path.exists() else []
    for p in pending_rows:
        add_result(
            results,
            f"pending.{p.get('location', '')}",
            "待补",
            "待补",
            p.get("required_source", ""),
            "PENDING",
            p.get("reason", ""),
        )

    write_csv(out_path, ["check_item", "doc_value", "recomputed_value", "source", "status", "note"], results)
    pass_n = sum(1 for r in results if r["status"] == "PASS")
    fail_n = sum(1 for r in results if r["status"] == "FAIL")
    pending_n = sum(1 for r in results if r["status"] == "PENDING")
    print(f"DONE: report={out_path}")
    print(f"PASS={pass_n} FAIL={fail_n} PENDING={pending_n}")


if __name__ == "__main__":
    main()

