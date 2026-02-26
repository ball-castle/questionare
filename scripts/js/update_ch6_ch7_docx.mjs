#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import JSZip from "jszip";
import { DOMParser, XMLSerializer } from "@xmldom/xmldom";
import xpath from "xpath";

const W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";

const REQUIRED_TABLE_FILES = {
  "表6-1": "表6-1_双口径模型拟合指标对比.csv",
  "表7-1": "表7-1_两类游客画像关键特征对比.csv",
  "表7-2": "表7-2_分层运营落地矩阵.csv",
  "表7-3": "表7-3_研究假设汇总_H1-H8.csv",
  "表7-4": "表7-4_SEM模型拟合指标.csv",
  "表7-5": "表7-5_SEM路径系数与显著性.csv",
  "表7-6": "表7-6_IPA四象限归属汇总_均值阈值.csv",
  "表7-7": "表7-7_IPA整改行动清单.csv",
  "表7-8": "表7-8_游客意见三维汇总.csv",
  "表7-9": "表7-9_专家意见整合框架.csv",
  "表7-10": "表7-10_问题证据建议闭环汇总.csv",
};

const REQUIRED_TABLE_KEYS = Object.keys(REQUIRED_TABLE_FILES);

function parseArgs(argv) {
  const out = {
    template: "六七章.docx",
    dataDir: "data/data_analysis",
    sourceDir: "data/data_analysis/_source_analysis",
    output: "doc/六七章_基于data_analysis_更新.docx",
    changesCsv: "doc/六七章_更新明细.csv",
    auditJson: "doc/六七章_更新审计.json",
    strict: true,
    force: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--template") out.template = argv[++i] || out.template;
    else if (a === "--data-dir") out.dataDir = argv[++i] || out.dataDir;
    else if (a === "--source-dir") out.sourceDir = argv[++i] || out.sourceDir;
    else if (a === "--output") out.output = argv[++i] || out.output;
    else if (a === "--changes-csv") out.changesCsv = argv[++i] || out.changesCsv;
    else if (a === "--audit-json") out.auditJson = argv[++i] || out.auditJson;
    else if (a === "--strict") {
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        out.strict = parseBool(next, true);
        i += 1;
      } else {
        out.strict = true;
      }
    } else if (a === "--force") {
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        out.force = parseBool(next, true);
        i += 1;
      } else {
        out.force = true;
      }
    }
  }
  return out;
}

function parseBool(v, defaultValue) {
  if (v == null) return defaultValue;
  const s = String(v).trim().toLowerCase();
  if (["1", "true", "yes", "y", "on"].includes(s)) return true;
  if (["0", "false", "no", "n", "off"].includes(s)) return false;
  return defaultValue;
}

function ensureParent(filePath) {
  return fs.mkdir(path.dirname(filePath), { recursive: true });
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

async function writeCsv(filePath, headers, rows) {
  await ensureParent(filePath);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h] ?? "")).join(","));
  }
  await fs.writeFile(filePath, `\uFEFF${lines.join("\n")}`, "utf-8");
}

function parseCsvText(text) {
  const src = String(text || "").replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];

    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ",") {
      row.push(cell);
      cell = "";
      continue;
    }
    if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      continue;
    }
    if (ch === "\r") continue;

    cell += ch;
  }

  row.push(cell);
  if (row.length > 1 || row[0] !== "") rows.push(row);

  return rows;
}

async function readCsvAsObjects(filePath, strict = true) {
  const text = await fs.readFile(filePath, "utf-8");
  const matrix = parseCsvText(text);
  if (!matrix.length) {
    if (strict) throw new Error(`CSV empty: ${filePath}`);
    return { headers: [], rows: [] };
  }
  const headers = matrix[0].map((x) => String(x || "").trim());
  const rows = [];
  for (let i = 1; i < matrix.length; i += 1) {
    const rec = {};
    for (let j = 0; j < headers.length; j += 1) {
      rec[headers[j]] = matrix[i][j] == null ? "" : String(matrix[i][j]);
    }
    rows.push(rec);
  }
  return { headers, rows };
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function buildSelectors(doc) {
  return xpath.useNamespaces({ w: W_NS });
}

function getNodeText(node, select) {
  const ts = select(".//w:t", node);
  return ts.map((x) => x.textContent || "").join("");
}

function setNodeText(node, select, text) {
  const ts = select(".//w:t", node);
  if (!ts.length) {
    return { old: "", changed: false, noTextNode: true };
  }
  const old = ts.map((x) => x.textContent || "").join("");
  const next = String(text ?? "");
  const changed = old !== next;
  ts[0].textContent = next;
  for (let i = 1; i < ts.length; i += 1) ts[i].textContent = "";
  return { old, changed, noTextNode: false };
}

function getTableRows(tableNode, select) {
  return select("./w:tr", tableNode);
}

function getRowCells(rowNode, select) {
  return select("./w:tc", rowNode);
}

function countStructuralNodes(doc, select) {
  return {
    tbl: select("//w:tbl", doc).length,
    tr: select("//w:tr", doc).length,
    tc: select("//w:tc", doc).length,
    p: select("//w:p", doc).length,
    r: select("//w:r", doc).length,
  };
}

function buildDocIndex(doc, select) {
  const body = select("/w:document/w:body", doc)[0];
  if (!body) throw new Error("word/document.xml missing w:body");

  const blocks = [];
  const tableEntries = [];
  const tableMap = new Map();

  const children = [];
  for (let i = 0; i < body.childNodes.length; i += 1) {
    const node = body.childNodes[i];
    if (node.nodeType === 1) children.push(node);
  }

  let tableCounter = 0;
  for (const child of children) {
    const local = child.localName || "";
    if (local === "p") {
      const text = getNodeText(child, select).trim();
      blocks.push({ type: "p", node: child, text });
      continue;
    }
    if (local !== "tbl") continue;

    tableCounter += 1;
    let caption = "";
    for (let i = blocks.length - 1; i >= 0; i -= 1) {
      if (blocks[i].type === "p" && blocks[i].text) {
        caption = blocks[i].text;
        break;
      }
    }

    const m = caption.match(/(表[67]-\d+)/);
    const key = m ? m[1] : "";

    const entry = { index: tableCounter, node: child, caption, key };
    tableEntries.push(entry);
    if (key) {
      if (!tableMap.has(key)) tableMap.set(key, []);
      tableMap.get(key).push(entry);
    }
    blocks.push({ type: "tbl", node: child, caption, key });
  }

  return { blocks, tableEntries, tableMap };
}

function toMatrix(headers, rows) {
  const matrix = [headers.slice()];
  for (const r of rows) matrix.push(headers.map((h) => String(r[h] ?? "")));
  return matrix;
}

function uniqueJoin(values, sep = "；") {
  const seen = new Set();
  const out = [];
  for (const v of values) {
    const s = String(v ?? "").trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out.join(sep);
}

function normalizePathArrow(v) {
  return String(v ?? "").replace(/\s*->\s*/g, " → ");
}

function normalizePValue(v) {
  const s = String(v ?? "").trim();
  if (!s) return "";
  if (/^0(?:\.0+)?$/.test(s)) return "<0.001";
  return s;
}

function normalizeQuadrant(v) {
  const s = String(v ?? "").trim();
  const m = s.match(/^Q(\d)[_\s]*(.*)$/);
  if (!m) return s;
  const tail = (m[2] || "").trim();
  if (!tail) return `Q${m[1]}`;
  return `Q${m[1]}  ${tail}`;
}

function buildTableMatrices(tableCsv, strict) {
  const m = {};

  m["表6-1"] = toMatrix(tableCsv["表6-1"].headers, tableCsv["表6-1"].rows);

  {
    const h = tableCsv["表7-1"].headers.slice(0, 3);
    m["表7-1"] = toMatrix(h, tableCsv["表7-1"].rows);
  }

  {
    const h = tableCsv["表7-2"].headers.slice(0, 3);
    m["表7-2"] = toMatrix(h, tableCsv["表7-2"].rows);
  }

  {
    const src = tableCsv["表7-3"];
    const h = ["假设编号", "研究假设", "变量"];
    for (const x of h) {
      assert(src.headers.includes(x), `表7-3 缺少列: ${x}`);
    }
    m["表7-3"] = toMatrix(h, src.rows);
  }

  {
    const src = tableCsv["表7-4"];
    if (strict && src.rows.length < 5) {
      throw new Error(`表7-4 期望至少5行，实际${src.rows.length}`);
    }
    const rows = src.rows.slice(0, 5);
    const header = rows.map((r) => String(r["指标"] ?? ""));
    header.push("结论");

    let passN = 0;
    let failN = 0;
    let otherN = 0;
    const values = rows.map((r) => {
      const value = String(r["值"] ?? "").trim();
      const conclusion = String(r["结论"] ?? "").trim();
      if (conclusion.includes("未达标")) failN += 1;
      else if (conclusion.includes("达标")) passN += 1;
      else otherN += 1;
      return `${value}（${conclusion}）`;
    });

    const parts = [];
    if (failN > 0) parts.push(`${failN}项未达标`);
    if (passN > 0) parts.push(`${passN}项达标`);
    if (otherN > 0) parts.push(`${otherN}项待判定`);
    values.push(parts.length ? parts.join("，") : "待判定");

    m["表7-4"] = [header, values];
  }

  {
    const src = tableCsv["表7-5"];
    const h = ["假设", "路径", "标准化系数β", "p值", "结论", "备注"];
    for (const x of ["假设", "路径", "标准化系数β", "p值", "结论"]) {
      assert(src.headers.includes(x), `表7-5 缺少列: ${x}`);
    }
    const rows = src.rows.map((r) => ({
      假设: String(r["假设"] ?? ""),
      路径: normalizePathArrow(r["路径"] ?? ""),
      标准化系数β: String(r["标准化系数β"] ?? ""),
      p值: normalizePValue(r["p值"] ?? ""),
      结论: String(r["结论"] ?? ""),
      备注: String(r["备注"] ?? ""),
    }));
    m["表7-5"] = toMatrix(h, rows);
  }

  {
    const src = tableCsv["表7-6"];
    const h = src.headers.slice(0, 4);
    const rows = src.rows.map((r) => {
      const rec = {};
      for (let i = 0; i < h.length; i += 1) {
        rec[h[i]] = String(r[h[i]] ?? "");
      }
      rec[h[0]] = normalizeQuadrant(rec[h[0]]);
      return rec;
    });
    m["表7-6"] = toMatrix(h, rows);
  }

  {
    const src = tableCsv["表7-7"];
    const h = ["优先级", "问题", "量化证据", "落地动作", "时间节点"];
    for (const x of h) assert(src.headers.includes(x), `表7-7 缺少列: ${x}`);

    const groupOrder = ["P1", "P2", "P3", "P4"];
    const groups = new Map(groupOrder.map((k) => [k, []]));

    for (const r of src.rows) {
      const raw = String(r["优先级"] ?? "").trim();
      const mPri = raw.match(/^(P\d)/i);
      if (!mPri) {
        if (strict) throw new Error(`表7-7 存在无法识别的优先级: ${raw}`);
        continue;
      }
      const key = mPri[1].toUpperCase();
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }

    const outRows = [];
    for (const g of groupOrder) {
      const list = groups.get(g) || [];
      if (strict && list.length === 0) throw new Error(`表7-7 缺少分组: ${g}`);
      if (list.length === 0) continue;

      outRows.push({
        优先级: String(list[0]["优先级"] ?? g),
        问题: uniqueJoin(list.map((r) => r["问题"])),
        量化证据: uniqueJoin(list.map((r) => r["量化证据"])),
        落地动作: uniqueJoin(list.map((r) => r["落地动作"])),
        时间节点: uniqueJoin(list.map((r) => r["时间节点"])),
      });
    }

    if (strict && outRows.length !== 4) {
      throw new Error(`表7-7 聚合后行数应为4，实际${outRows.length}`);
    }

    m["表7-7"] = toMatrix(h, outRows);
  }

  m["表7-8"] = toMatrix(tableCsv["表7-8"].headers.slice(0, 4), tableCsv["表7-8"].rows);
  m["表7-9"] = toMatrix(tableCsv["表7-9"].headers.slice(0, 3), tableCsv["表7-9"].rows);
  m["表7-10"] = toMatrix(tableCsv["表7-10"].headers.slice(0, 5), tableCsv["表7-10"].rows);

  return m;
}

function fillTableByMatrix({ tableNode, tableKey, matrix, select, changes, strict, source }) {
  const rows = getTableRows(tableNode, select);

  if (strict && rows.length !== matrix.length) {
    throw new Error(`${tableKey} 行数不匹配: doc=${rows.length}, data=${matrix.length}`);
  }
  const rowCount = Math.min(rows.length, matrix.length);

  for (let r = 0; r < rowCount; r += 1) {
    const cells = getRowCells(rows[r], select);
    if (strict && cells.length < matrix[r].length) {
      throw new Error(`${tableKey} 第${r + 1}行列数不足: doc=${cells.length}, data=${matrix[r].length}`);
    }
    const colCount = Math.min(cells.length, matrix[r].length);
    for (let c = 0; c < colCount; c += 1) {
      const newValue = String(matrix[r][c] ?? "");
      const rec = setNodeText(cells[c], select, newValue);
      if (rec.noTextNode) {
        if (strict) throw new Error(`${tableKey}_r${r + 1}_c${c + 1} 缺少 w:t 节点`);
        continue;
      }
      if (rec.changed) {
        changes.push({
          location: `${tableKey}_r${r + 1}_c${c + 1}`,
          old_value: rec.old,
          new_value: newValue,
          source,
          rule: "table_text_replace",
        });
      }
    }
  }
}

function collectParagraphInfos(doc, select) {
  const ps = select("//w:p", doc);
  return ps.map((p) => ({ node: p, text: getNodeText(p, select) }));
}

function applyParagraphRule({ paragraphInfos, select, id, pattern, toText, expected = null, strict, changes, source, rule }) {
  let hit = 0;
  for (const p of paragraphInfos) {
    const old = p.text;
    if (!pattern.test(old)) continue;
    const next = typeof toText === "function" ? toText(old) : String(toText);
    if (next !== old) {
      const rec = setNodeText(p.node, select, next);
      if (rec.noTextNode) {
        if (strict) throw new Error(`${id} 命中段落但无 w:t`);
        continue;
      }
      p.text = next;
      changes.push({
        location: `paragraph:${id}`,
        old_value: old,
        new_value: next,
        source,
        rule,
      });
    }
    hit += 1;
  }

  if (expected != null) {
    if (typeof expected === "number") {
      if (hit !== expected && strict) {
        throw new Error(`${id} 命中数异常: expected=${expected}, actual=${hit}`);
      }
    } else {
      const min = expected.min ?? 0;
      const max = expected.max ?? Number.POSITIVE_INFINITY;
      if ((hit < min || hit > max) && strict) {
        throw new Error(`${id} 命中数异常: expected=[${min},${max}], actual=${hit}`);
      }
    }
  }

  return hit;
}

function countResiduals(doc, select) {
  const ps = select("//w:p", doc);
  const counters = {
    pending_token: 0,
    pending_suffix: 0,
    sample_n_880: 0,
  };

  for (const p of ps) {
    const t = getNodeText(p, select);
    if (!t) continue;
    if (t.includes("【待补】")) counters.pending_token += 1;
    if (t.includes("（待补）")) counters.pending_suffix += 1;
    if (t.includes("n=880")) counters.sample_n_880 += 1;
  }

  return counters;
}

function sameCounts(a, b) {
  return a.tbl === b.tbl && a.tr === b.tr && a.tc === b.tc && a.p === b.p && a.r === b.r;
}

async function readRequiredData(dataDir, sourceDir, strict) {
  const tableDir = path.join(dataDir, "tables");
  const tableCsv = {};

  for (const [key, file] of Object.entries(REQUIRED_TABLE_FILES)) {
    const p = path.join(tableDir, file);
    tableCsv[key] = await readCsvAsObjects(p, true);
  }

  const runMetaPath = path.join(sourceDir, "run_metadata.json");
  const runMeta = JSON.parse(await fs.readFile(runMetaPath, "utf-8"));
  const mainN = Number(runMeta.n_samples_main || runMeta.remain_n_revised || 0);
  if (strict && !(mainN > 0)) throw new Error(`run_metadata.json 缺少有效 n_samples_main: ${runMetaPath}`);

  return { tableCsv, runMeta, mainN };
}

async function main() {
  const args = parseArgs(process.argv);

  const templatePath = path.resolve(args.template);
  const dataDir = path.resolve(args.dataDir);
  const sourceDir = path.resolve(args.sourceDir);
  const outputPath = path.resolve(args.output);
  const changesCsvPath = path.resolve(args.changesCsv);
  const auditJsonPath = path.resolve(args.auditJson);

  if (!args.force) {
    try {
      await fs.access(outputPath);
      throw new Error(`输出文件已存在（使用 --force 覆盖）: ${outputPath}`);
    } catch (err) {
      if (String(err.message || err).includes("输出文件已存在")) throw err;
    }
  }

  const { tableCsv, mainN } = await readRequiredData(dataDir, sourceDir, args.strict);

  const tableMatrices = buildTableMatrices(tableCsv, args.strict);

  const docxBuf = await fs.readFile(templatePath);
  const zip = await JSZip.loadAsync(docxBuf);
  const docXmlEntry = zip.file("word/document.xml");
  if (!docXmlEntry) throw new Error("模板缺少 word/document.xml");
  const xml = await docXmlEntry.async("text");

  const doc = new DOMParser().parseFromString(xml, "text/xml");
  const select = buildSelectors(doc);

  const beforeCounts = countStructuralNodes(doc, select);

  const { tableMap } = buildDocIndex(doc, select);
  for (const key of REQUIRED_TABLE_KEYS) {
    const hits = tableMap.get(key) || [];
    if (args.strict && hits.length !== 1) {
      throw new Error(`${key} 定位失败或不唯一: hits=${hits.length}`);
    }
  }

  const changes = [];

  for (const key of REQUIRED_TABLE_KEYS) {
    const entry = (tableMap.get(key) || [])[0];
    if (!entry) continue;
    const matrix = tableMatrices[key];
    fillTableByMatrix({
      tableNode: entry.node,
      tableKey: key,
      matrix,
      select,
      changes,
      strict: args.strict,
      source: path.join(dataDir, "tables", REQUIRED_TABLE_FILES[key]).replace(/\\/g, "/"),
    });
  }

  const paragraphs = collectParagraphInfos(doc, select);

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "cover_sample_n",
    pattern: /样本量\s*n=\d+（重筛主样本）/,
    toText: `样本量 n=${mainN}（重筛主样本）`,
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(sourceDir, "run_metadata.json").replace(/\\/g, "/"),
    rule: "anchor_full_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "source_paragraph_sample_n",
    pattern: /调查问卷（n=\d+，经数据清洗后保留有效样本）/,
    toText: (old) => old.replace(/调查问卷（n=\d+，经数据清洗后保留有效样本）/, `调查问卷（n=${mainN}，经数据清洗后保留有效样本）`),
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(sourceDir, "run_metadata.json").replace(/\\/g, "/"),
    rule: "anchor_regex_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "dual_sample_main_n",
    pattern: /双口径设计：主样本（n=\d+）报告整体效应；敏感性样本（n=\d+，严格筛选口径）/,
    toText: (old) => old.replace(/双口径设计：主样本（n=\d+）/, `双口径设计：主样本（n=${mainN}）`),
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(sourceDir, "run_metadata.json").replace(/\\/g, "/"),
    rule: "anchor_regex_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "title_7_4_3_proxy",
    pattern: /^7\.4\.3\s+专家访谈意见（待补）$/,
    toText: "7.4.3  专家意见整合分析（代理口径）",
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(dataDir, "tables", REQUIRED_TABLE_FILES["表7-9"]).replace(/\\/g, "/"),
    rule: "anchor_full_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "title_table7_9_remove_pending",
    pattern: /^表7-9\s+专家意见整合框架（待补）$/,
    toText: "表7-9  专家意见整合框架",
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(dataDir, "tables", REQUIRED_TABLE_FILES["表7-9"]).replace(/\\/g, "/"),
    rule: "anchor_full_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "proxy_intro_replace",
    pattern: /以下为占位框架，待访谈材料到位后替换。.*整合：明确标注专家观点与哪项统计发现形成印证、补充或质疑，以提升论证层次。/,
    toText: "以下内容基于标杆案例结构与本地定量证据形成代理专家口径，用于与统计分析结论进行对照、互证与行动闭环设计。",
    expected: 1,
    strict: args.strict,
    changes,
    source: path.join(dataDir, "tables", REQUIRED_TABLE_FILES["表7-9"]).replace(/\\/g, "/"),
    rule: "anchor_full_replace",
  });

  applyParagraphRule({
    paragraphInfos: paragraphs,
    select,
    id: "final_proxy_conclusion",
    pattern: /待专家访谈材料补齐后，本章将形成「量化\+质性」双轨闭环的完整研究报告。/,
    toText: (old) => old.replace(/待专家访谈材料补齐后，本章将形成「量化\+质性」双轨闭环的完整研究报告。/, "本章已形成「量化+质性（代理口径）」双轨闭环研究报告。"),
    expected: { min: 0, max: 1 },
    strict: args.strict,
    changes,
    source: path.join(dataDir, "tables", REQUIRED_TABLE_FILES["表7-9"]).replace(/\\/g, "/"),
    rule: "anchor_regex_replace",
  });

  const afterCounts = countStructuralNodes(doc, select);
  if (!sameCounts(beforeCounts, afterCounts)) {
    throw new Error(
      `结构校验失败: before=${JSON.stringify(beforeCounts)} after=${JSON.stringify(afterCounts)}`,
    );
  }

  const residuals = countResiduals(doc, select);
  if (args.strict && (residuals.pending_token > 0 || residuals.pending_suffix > 0)) {
    throw new Error(`仍存在待补占位符: ${JSON.stringify(residuals)}`);
  }

  const newXml = new XMLSerializer().serializeToString(doc);
  zip.file("word/document.xml", newXml);

  await ensureParent(outputPath);
  const outBuf = await zip.generateAsync({ type: "nodebuffer" });
  await fs.writeFile(outputPath, outBuf);

  await writeCsv(changesCsvPath, ["location", "old_value", "new_value", "source", "rule"], changes);

  const audit = {
    generated_at: new Date().toISOString(),
    template: templatePath,
    output: outputPath,
    data_dir: dataDir,
    source_dir: sourceDir,
    strict: Boolean(args.strict),
    force: Boolean(args.force),
    main_sample_n: mainN,
    required_tables: REQUIRED_TABLE_KEYS,
    changed_rows: changes.length,
    structure_before: beforeCounts,
    structure_after: afterCounts,
    structure_unchanged: true,
    residuals,
    changes_csv: changesCsvPath,
    checks: {
      table_count_expected: 11,
      table_count_actual: beforeCounts.tbl,
      pending_placeholder_zero: residuals.pending_token === 0 && residuals.pending_suffix === 0,
    },
  };

  await ensureParent(auditJsonPath);
  await fs.writeFile(auditJsonPath, `${JSON.stringify(audit, null, 2)}\n`, "utf-8");

  console.log(`update_done: output=${outputPath}`);
  console.log(`changed_rows=${changes.length}`);
  console.log(`residuals=${JSON.stringify(residuals)}`);
}

main().catch((err) => {
  console.error(String(err?.stack || err));
  process.exit(1);
});
