#!/usr/bin/env python3
"""Fill resolvable placeholders in 待填数据.docx and export fill/pending logs."""

import csv
import zipfile
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def apply_paragraph_replacements(root, fill_rows):
    replacements = [
        (
            "通过网络平台及街区现场发放，共发放问卷600份，共有有效问卷526份，有效率为87.67%。",
            "通过网络平台及街区现场发放，累计发放问卷813份，回收有效问卷813份（到访647份、未到访166份），有效回收率100.00%。",
            "统一813口径（正式调查回收）",
        ),
        (
            "预调查共发放150份问卷（线上【X】份、线下【X】份），回收【X】份，有效问卷【X】份，有效回收率【X%】。对预调查数据进行编码、缺失处理与异常剔除后，开展信度与效度检验，为正式问卷修订提供依据。",
            "预调查共发放150份问卷（线上待补份、线下待补份），回收待补份，有效问卷待补份，有效回收率待补。对预调查数据进行编码、缺失处理与异常剔除后，开展信度与效度检验，为正式问卷修订提供依据。",
            "预调查数据源缺失，先标记待补",
        ),
        (
            "正式调查按“到访600+未到访400”的目标配额实施，累计发放N份，回收有效问卷XX份（到访X份、未到访X份），有效回收率X%。",
            "正式调查按既定配额实施，累计发放813份，回收有效问卷813份（到访647份、未到访166份），有效回收率100.00%。",
            "统一813口径（Q8分组）",
        ),
        (
            "本次调查共发放问卷600份，有效问卷526份，问卷有效率87.67%。以下为有效问卷样本的结构分析。",
            "本次调查累计发放问卷813份，有效问卷813份，问卷有效率100.00%。以下为有效问卷样本的结构分析。",
            "统一813口径",
        ),
        (
            "本次调查回收有效问卷共526份，在所有受访者中，男女比例为231:295，所占比例基本一致；从受访者年龄性别分布图可知，30岁以下青年群体占31.18%，30岁以上中年群体占68.82%，调查基本覆盖50岁以下全年龄段。",
            "本次调查回收有效问卷共813份，在所有受访者中，男女比例为420:393（51.66%:48.34%）；年龄结构中，编码3占38.38%，编码2占28.54%，编码4占18.70%，样本覆盖较广。",
            "基于单选题频数百分比表（Q1/Q2）",
        ),
    ]

    paragraphs = root.findall(".//w:p", NS)
    for p in paragraphs:
        old_text, _ = get_para_text(p)
        if not old_text:
            continue
        for old_sub, new_text, calc in replacements:
            if old_sub in old_text:
                set_para_text(p, new_text)
                fill_rows.append(
                    {
                        "location": "paragraph",
                        "old_value": old_text,
                        "new_value": new_text,
                        "source_table": "output/tables/单选题频数百分比表.csv + 口径约定",
                        "calculation_rule": calc,
                    }
                )
                break


def fill_table_1_pretest_reliability(tbl, fill_rows, pending_rows):
    rows = table_rows(tbl)
    # rows 2..7, cols 2..4 -> 待补
    for r in range(1, 7):
        dim_name, _ = get_cell_text(rows[r][0])
        for c in range(1, 4):
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
    # rows 2..5, col 2 -> 待补
    for r in range(1, 5):
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


def fill_table_3_formal_reliability(tbl, fill_rows):
    rows = table_rows(tbl)
    # row index from 0
    values = {
        1: ("3", "0.8077", "良好"),
        2: ("2", "0.7070", "可接受"),
        3: ("3", "0.7441", "可接受"),
        4: ("4", "0.8273", "良好"),
        5: ("3", "0.7877", "可接受"),
        6: ("13", "0.8243", "良好"),
    }
    for r, (n_items, alpha, grade) in values.items():
        old2 = set_cell_text(rows[r][1], n_items)
        old3 = set_cell_text(rows[r][2], alpha)
        old4 = set_cell_text(rows[r][3], grade)
        dim_name, _ = get_cell_text(rows[r][0])
        fill_rows.extend(
            [
                {
                    "location": f"table3_r{r+1}_c2",
                    "old_value": old2,
                    "new_value": n_items,
                    "source_table": "output/tables/信度分析表.csv + 维度划分",
                    "calculation_rule": f"{dim_name}题项数",
                },
                {
                    "location": f"table3_r{r+1}_c3",
                    "old_value": old3,
                    "new_value": alpha,
                    "source_table": "output/tables/survey_clean.csv重算",
                    "calculation_rule": f"{dim_name} Cronbach alpha(4位小数)",
                },
                {
                    "location": f"table3_r{r+1}_c4",
                    "old_value": old4,
                    "new_value": grade,
                    "source_table": "alpha评级规则",
                    "calculation_rule": ">=0.8良好，>=0.7可接受",
                },
            ]
        )


def fill_table_4_formal_validity(tbl, fill_rows):
    rows = table_rows(tbl)
    vals = {1: "0.9382", 2: "14747.692", 3: "666", 4: "<0.001"}
    for r in range(1, 5):
        old = set_cell_text(rows[r][1], vals[r])
        metric, _ = get_cell_text(rows[r][0])
        fill_rows.append(
            {
                "location": f"table4_r{r+1}_c2",
                "old_value": old,
                "new_value": vals[r],
                "source_table": "output/tables/效度分析表.csv",
                "calculation_rule": f"{metric}（KMO4位/卡方3位）",
            }
        )


def fill_table_5_sample_structure(tbl, fill_rows):
    rows = table_rows(tbl)

    # helper: fill a normal 5-col row
    def fill_row(r, var_text, attr, count, pct, total):
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
                {
                    "location": f"table5_r{r+1}_c2",
                    "old_value": old2,
                    "new_value": attr,
                    "source_table": "output/tables/单选题频数百分比表.csv",
                    "calculation_rule": "属性名修订",
                },
                {
                    "location": f"table5_r{r+1}_c3",
                    "old_value": old3,
                    "new_value": str(count),
                    "source_table": "output/tables/单选题频数百分比表.csv",
                    "calculation_rule": "人数",
                },
                {
                    "location": f"table5_r{r+1}_c4",
                    "old_value": old4,
                    "new_value": pct,
                    "source_table": "output/tables/单选题频数百分比表.csv",
                    "calculation_rule": "比例(2位小数)",
                },
                {
                    "location": f"table5_r{r+1}_c5",
                    "old_value": old5,
                    "new_value": total if total is not None else "",
                    "source_table": "分组总计",
                    "calculation_rule": "组首行为100%，其余留空",
                },
            ]
        )

    # 文化程度 rows 2..6 (index 1..5)
    edu = [
        ("文化程度", "初中及以下", 69, "8.49%", "100%"),
        ("", "中专/高中", 95, "11.69%", ""),
        ("", "大专", 203, "24.97%", ""),
        ("", "本科", 285, "35.06%", ""),
        ("", "硕士及以上", 161, "19.80%", ""),
    ]
    for idx, row in enumerate(edu, start=1):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    # 职业 rows 7..14 (index 6..13)
    jobs = [
        ("职业", "学生", 0, "0.00%", "100%"),
        ("", "企业/公司职员", 335, "41.21%", ""),
        ("", "事业单位人员/公务员", 82, "10.09%", ""),
        ("", "自由职业者", 156, "19.19%", ""),
        ("", "个体经营者", 111, "13.65%", ""),
        ("", "服务业从业者", 89, "10.95%", ""),
        ("", "离退休人员", 0, "0.00%", ""),
        ("", "其他", 40, "4.92%", ""),
    ]
    for idx, row in enumerate(jobs, start=6):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    # 月收入 rows 15..19 (index 14..18)
    income = [
        ("月收入", "3000元以下", 86, "10.58%", "100%"),
        ("", "3001-5000元", 230, "28.29%", ""),
        ("", "5001-8000元", 309, "38.01%", ""),
        ("", "8001-15000元", 151, "18.57%", ""),
        ("", "15000元以上", 37, "4.55%", ""),
    ]
    for idx, row in enumerate(income, start=14):
        fill_row(idx, row[0], row[1], row[2], row[3], row[4])

    # Row 20: 有效填写人次 | 813
    # this row may have merged cells; set first two cells if exist.
    last = rows[19]
    old1 = set_cell_text(last[0], "有效填写人次")
    if len(last) > 1:
        old2 = set_cell_text(last[1], "813")
    else:
        old2 = ""
    fill_rows.extend(
        [
            {
                "location": "table5_r20_c1",
                "old_value": old1,
                "new_value": "有效填写人次",
                "source_table": "统一813口径",
                "calculation_rule": "固定标签",
            },
            {
                "location": "table5_r20_c2",
                "old_value": old2,
                "new_value": "813",
                "source_table": "output/tables/survey_clean.csv",
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


def main():
    doc_path = Path("待填数据.docx")
    out_fill = Path("output/tables/待填数据_回填值清单.csv")
    out_pending = Path("output/tables/待填数据_待补项清单.csv")
    out_check = Path("output/tables/待填数据_回填校验报告.txt")

    fill_rows = []
    pending_rows = []

    with zipfile.ZipFile(doc_path, "r") as zin:
        file_map = {name: zin.read(name) for name in zin.namelist()}

    root = ET.fromstring(file_map["word/document.xml"])

    # Paragraph replacements
    apply_paragraph_replacements(root, fill_rows)

    # Table replacements
    tables = root.findall(".//w:tbl", NS)
    if len(tables) < 5:
        raise RuntimeError(f"Expected at least 5 tables, got {len(tables)}")

    fill_table_1_pretest_reliability(tables[0], fill_rows, pending_rows)
    fill_table_2_pretest_validity(tables[1], fill_rows, pending_rows)
    fill_table_3_formal_reliability(tables[2], fill_rows)
    fill_table_4_formal_validity(tables[3], fill_rows)
    fill_table_5_sample_structure(tables[4], fill_rows)

    # Pretest pending paragraph items
    pending_rows.extend(
        [
            {
                "location": "paragraph_pretest_recovery",
                "item": "预调查线上份数/线下份数/回收份数/有效份数/有效回收率",
                "reason": "缺少预调查原始数据",
                "required_source": "预调查回收台账/原始问卷",
            }
        ]
    )

    # write updated xml back into docx
    file_map["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    tmp_path = doc_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, content in file_map.items():
            zout.writestr(name, content)
    tmp_path.replace(doc_path)

    # export logs
    write_dict_csv(
        out_fill,
        ["location", "old_value", "new_value", "source_table", "calculation_rule"],
        fill_rows,
    )
    write_dict_csv(
        out_pending,
        ["location", "item", "reason", "required_source"],
        pending_rows,
    )

    # validation report
    # check residual old口径 tokens in paragraphs
    residual = scan_left_numbers(root)
    report_lines = [
        "待填数据文档回填校验报告",
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

