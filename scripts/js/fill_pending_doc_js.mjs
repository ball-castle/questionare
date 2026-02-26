#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import JSZip from "jszip";
import { DOMParser, XMLSerializer } from "@xmldom/xmldom";
import xpath from "xpath";

const W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";

function parseArgs(argv) {
  const out = {
    template: "",
    metrics: "",
    output: "",
    fillLogCsv: "output_data_analysis/tables/js_待填数据_回填值清单.csv",
    pendingCsv: "output_data_analysis/tables/js_待填数据_待补项清单.csv",
    auditJson: "output_data_analysis/tables/js_待填数据_回填审计.json",
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--template") out.template = argv[++i] || "";
    else if (a === "--metrics") out.metrics = argv[++i] || "";
    else if (a === "--output") out.output = argv[++i] || "";
    else if (a === "--fill-log-csv") out.fillLogCsv = argv[++i] || out.fillLogCsv;
    else if (a === "--pending-csv") out.pendingCsv = argv[++i] || out.pendingCsv;
    else if (a === "--audit-json") out.auditJson = argv[++i] || out.auditJson;
  }
  return out;
}

function ensureParent(filePath) {
  const dir = path.dirname(filePath);
  return fs.mkdir(dir, { recursive: true });
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

async function writeCsv(filePath, header, rows) {
  await ensureParent(filePath);
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push(header.map((k) => csvEscape(row[k] ?? "")).join(","));
  }
  await fs.writeFile(filePath, `\uFEFF${lines.join("\n")}`, "utf-8");
}

function buildSelectors(doc) {
  const select = xpath.useNamespaces({ w: W_NS });
  const paragraphs = select("//w:p", doc);
  const tables = select("//w:tbl", doc);
  return { select, paragraphs, tables };
}

function getNodeText(node, select) {
  const textNodes = select(".//w:t", node);
  return textNodes.map((n) => n.textContent || "").join("");
}

function setNodeText(node, select, text) {
  const textNodes = select(".//w:t", node);
  if (!textNodes.length) return "";
  const old = textNodes.map((n) => n.textContent || "").join("");
  textNodes[0].textContent = text;
  for (let i = 1; i < textNodes.length; i += 1) textNodes[i].textContent = "";
  return old;
}

function getTableRows(tableNode, select) {
  return select("./w:tr", tableNode);
}

function getRowCells(rowNode, select) {
  return select("./w:tc", rowNode);
}

function setCellText(rows, rowIdx, colIdx, select, text) {
  if (rowIdx < 0 || rowIdx >= rows.length) return "";
  const cells = getRowCells(rows[rowIdx], select);
  if (colIdx < 0 || colIdx >= cells.length) return "";
  return setNodeText(cells[colIdx], select, text);
}

function fillTable3(tableNode, select, metrics, fillRows) {
  const rows = getTableRows(tableNode, select);
  const data = metrics.tables.table3_reliability || [];
  for (let i = 0; i < data.length; i += 1) {
    const r = data[i];
    const rowIdx = i + 1;
    const old2 = setCellText(rows, rowIdx, 1, select, String(r.item_count));
    const old3 = setCellText(rows, rowIdx, 2, select, r.alpha == null ? "NA" : r.alpha.toFixed(4));
    const old4 = setCellText(rows, rowIdx, 3, select, String(r.grade || ""));
    fillRows.push({ location: `table3_r${rowIdx + 1}_c2`, old_value: old2, new_value: String(r.item_count), source_table: "metrics.tables.table3_reliability", calculation_rule: "维度题项数" });
    fillRows.push({ location: `table3_r${rowIdx + 1}_c3`, old_value: old3, new_value: r.alpha == null ? "NA" : r.alpha.toFixed(4), source_table: "metrics.tables.table3_reliability", calculation_rule: "Cronbach alpha（4位小数）" });
    fillRows.push({ location: `table3_r${rowIdx + 1}_c4`, old_value: old4, new_value: String(r.grade || ""), source_table: "metrics.tables.table3_reliability", calculation_rule: "信度等级" });
  }
}

function fillTable4(tableNode, select, metrics, fillRows) {
  const rows = getTableRows(tableNode, select);
  const val = metrics.tables.table4_validity || {};
  const map = [
    String(val.kmo ?? ""),
    String(val.bartlett_chi2 ?? ""),
    String(val.bartlett_df ?? ""),
    String(val.bartlett_p ?? ""),
  ];
  for (let i = 0; i < map.length; i += 1) {
    const rowIdx = i + 1;
    const old = setCellText(rows, rowIdx, 1, select, map[i]);
    fillRows.push({
      location: `table4_r${rowIdx + 1}_c2`,
      old_value: old,
      new_value: map[i],
      source_table: "metrics.tables.table4_validity",
      calculation_rule: "KMO/Bartlett",
    });
  }
}

function fillTable5(tableNode, select, metrics, fillRows) {
  const rows = getTableRows(tableNode, select);
  const data = metrics.tables.table5_sample_structure || [];
  for (let i = 0; i < data.length; i += 1) {
    const rowIdx = i + 1;
    const r = data[i];
    const old1 = setCellText(rows, rowIdx, 0, select, String(r.var_name || ""));
    const old2 = setCellText(rows, rowIdx, 1, select, String(r.attr || ""));
    const old3 = setCellText(rows, rowIdx, 2, select, String(r.count));
    const old4 = setCellText(rows, rowIdx, 3, select, String(r.pct || ""));
    const old5 = setCellText(rows, rowIdx, 4, select, String(r.total || ""));
    fillRows.push({ location: `table5_r${rowIdx + 1}_c1`, old_value: old1, new_value: String(r.var_name || ""), source_table: "metrics.tables.table5_sample_structure", calculation_rule: "变量名" });
    fillRows.push({ location: `table5_r${rowIdx + 1}_c2`, old_value: old2, new_value: String(r.attr || ""), source_table: "metrics.tables.table5_sample_structure", calculation_rule: "属性" });
    fillRows.push({ location: `table5_r${rowIdx + 1}_c3`, old_value: old3, new_value: String(r.count), source_table: "metrics.tables.table5_sample_structure", calculation_rule: "人数" });
    fillRows.push({ location: `table5_r${rowIdx + 1}_c4`, old_value: old4, new_value: String(r.pct || ""), source_table: "metrics.tables.table5_sample_structure", calculation_rule: "比例" });
    fillRows.push({ location: `table5_r${rowIdx + 1}_c5`, old_value: old5, new_value: String(r.total || ""), source_table: "metrics.tables.table5_sample_structure", calculation_rule: "总计" });
  }
  if (rows.length > 19) {
    const old1 = setCellText(rows, 19, 0, select, "有效填写人次");
    const old2 = setCellText(rows, 19, 1, select, String(metrics.tables.table5_valid_count ?? ""));
    fillRows.push({ location: "table5_r20_c1", old_value: old1, new_value: "有效填写人次", source_table: "fixed", calculation_rule: "固定标签" });
    fillRows.push({ location: "table5_r20_c2", old_value: old2, new_value: String(metrics.tables.table5_valid_count ?? ""), source_table: "metrics.tables.table5_valid_count", calculation_rule: "有效样本数" });
  }
}

function replaceParagraphs(doc, select, metrics, fillRows) {
  const paragraphs = select("//w:p", doc);
  const allocRe = /累计发放问卷\d+份，回收问卷\d+份，质控后有效问卷\d+份（到访\d+份、未到访\d+份），有效率[0-9.]+%。?/;
  const overallRe = /本次调查累计发放问卷\d+份，回收问卷\d+份，质控后有效问卷\d+份（有效率[0-9.]+%）。?以下为有效问卷样本的结构分析。?/;
  const genderAgeRe = /本次调查回收有效问卷共\d+份，在所有受访者中，男女比例为\d+:\d+（[0-9.]+%:[0-9.]+%）。从年龄分布看，.*。?/;

  for (const p of paragraphs) {
    const txt = getNodeText(p, select);
    if (!txt) continue;
    if (allocRe.test(txt)) {
      const old = setNodeText(p, select, metrics.paragraphs.formal_allocation);
      fillRows.push({ location: "paragraph_formal_allocation", old_value: old, new_value: metrics.paragraphs.formal_allocation, source_table: "metrics.paragraphs", calculation_rule: "正式调查配额段" });
      continue;
    }
    if (overallRe.test(txt)) {
      const old = setNodeText(p, select, metrics.paragraphs.overall_summary);
      fillRows.push({ location: "paragraph_overall_summary", old_value: old, new_value: metrics.paragraphs.overall_summary, source_table: "metrics.paragraphs", calculation_rule: "样本总量段" });
      continue;
    }
    if (genderAgeRe.test(txt)) {
      const old = setNodeText(p, select, metrics.paragraphs.gender_age);
      fillRows.push({ location: "paragraph_gender_age", old_value: old, new_value: metrics.paragraphs.gender_age, source_table: "metrics.paragraphs", calculation_rule: "性别年龄段" });
    }
  }
}

function countResidualTokens(doc, select) {
  const paragraphs = select("//w:p", doc);
  const tokens = ["961份", "880份", "91.57%"];
  let count = 0;
  for (const p of paragraphs) {
    const txt = getNodeText(p, select);
    if (tokens.some((t) => txt.includes(t))) count += 1;
  }
  return count;
}

function buildPendingRows() {
  const dims = ["文化体验维度", "非遗体验维度", "产品体验维度", "配套保障维度", "宣传策略维度", "整体量表"];
  const rows = [];
  for (let i = 0; i < dims.length; i += 1) {
    const rowIdx = i + 2;
    rows.push({ location: `table1_r${rowIdx}_c2`, item: `预调查信度-${dims[i]}`, reason: "缺少预调查原始数据/结果表", required_source: "预调查样本与信效度输出" });
    rows.push({ location: `table1_r${rowIdx}_c3`, item: `预调查信度-${dims[i]}`, reason: "缺少预调查原始数据/结果表", required_source: "预调查样本与信效度输出" });
    rows.push({ location: `table1_r${rowIdx}_c4`, item: `预调查信度-${dims[i]}`, reason: "缺少预调查原始数据/结果表", required_source: "预调查样本与信效度输出" });
  }
  rows.push({ location: "table2_r2_c2", item: "预调查效度-KMO", reason: "缺少预调查原始数据/结果表", required_source: "预调查KMO/Bartlett结果" });
  rows.push({ location: "table2_r3_c2", item: "预调查效度-Bartlett球形检验近似卡方", reason: "缺少预调查原始数据/结果表", required_source: "预调查KMO/Bartlett结果" });
  rows.push({ location: "table2_r4_c2", item: "预调查效度-自由度df", reason: "缺少预调查原始数据/结果表", required_source: "预调查KMO/Bartlett结果" });
  rows.push({ location: "table2_r5_c2", item: "预调查效度-显著性p", reason: "缺少预调查原始数据/结果表", required_source: "预调查KMO/Bartlett结果" });
  rows.push({ location: "paragraph_pretest_recovery", item: "预调查线上份数/线下份数/回收份数/有效份数/有效回收率", reason: "缺少预调查原始数据", required_source: "预调查回收台账/原始问卷" });
  return rows;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.template || !args.metrics || !args.output) {
    console.error("Usage: node scripts/js/fill_pending_doc_js.mjs --template <docx> --metrics <json> --output <docx>");
    process.exit(1);
  }

  const templatePath = path.resolve(args.template);
  const metricsPath = path.resolve(args.metrics);
  const outputPath = path.resolve(args.output);
  const fillLogCsv = path.resolve(args.fillLogCsv);
  const pendingCsv = path.resolve(args.pendingCsv);
  const auditJson = path.resolve(args.auditJson);

  const metrics = JSON.parse(await fs.readFile(metricsPath, "utf-8"));
  const docxBuf = await fs.readFile(templatePath);
  const zip = await JSZip.loadAsync(docxBuf);
  const docXmlEntry = zip.file("word/document.xml");
  if (!docXmlEntry) throw new Error("word/document.xml not found in template");
  const xml = await docXmlEntry.async("text");
  const doc = new DOMParser().parseFromString(xml, "text/xml");
  const { select, tables } = buildSelectors(doc);

  if (tables.length < 5) {
    throw new Error(`Expected at least 5 tables in template, got ${tables.length}`);
  }

  const fillRows = [];
  replaceParagraphs(doc, select, metrics, fillRows);
  fillTable3(tables[2], select, metrics, fillRows);
  fillTable4(tables[3], select, metrics, fillRows);
  fillTable5(tables[4], select, metrics, fillRows);

  const newXml = new XMLSerializer().serializeToString(doc);
  zip.file("word/document.xml", newXml);
  await ensureParent(outputPath);
  const outBuf = await zip.generateAsync({ type: "nodebuffer" });
  await fs.writeFile(outputPath, outBuf);

  const pendingRows = buildPendingRows();
  await writeCsv(fillLogCsv, ["location", "old_value", "new_value", "source_table", "calculation_rule"], fillRows);
  await writeCsv(pendingCsv, ["location", "item", "reason", "required_source"], pendingRows);

  const residualOldTokens = countResidualTokens(doc, select);
  const audit = {
    output_docx: outputPath,
    fill_rows: fillRows.length,
    pending_rows: pendingRows.length,
    residual_old_tokens: residualOldTokens,
    required_zero_tokens: ["961份", "880份", "91.57%"],
  };
  await ensureParent(auditJson);
  await fs.writeFile(auditJson, `${JSON.stringify(audit, null, 2)}\n`, "utf-8");

  console.log(`fill_doc_done: output=${outputPath}`);
  console.log(`fill_rows=${fillRows.length} pending_rows=${pendingRows.length} residual_old_tokens=${residualOldTokens}`);
}

main().catch((err) => {
  console.error(String(err?.stack || err));
  process.exit(1);
});
