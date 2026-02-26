#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

function parseArgs(argv) {
  const out = {
    tablesDir: 'data/data_analysis/tables',
    outputCsv: 'data/data_analysis/tables/全部表格_合并.csv',
    includePending: false,
    bom: true,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--tables-dir') out.tablesDir = argv[++i];
    else if (a === '--output-csv') out.outputCsv = argv[++i];
    else if (a === '--include-pending') out.includePending = true;
    else if (a === '--no-bom') out.bom = false;
  }
  return out;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let i = 0;
  let inQuotes = false;
  const src = text.replace(/^\uFEFF/, '');
  while (i < src.length) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          cell += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      cell += ch;
      i += 1;
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (ch === ',') {
      row.push(cell);
      cell = '';
      i += 1;
      continue;
    }
    if (ch === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
      i += 1;
      continue;
    }
    if (ch === '\r') {
      i += 1;
      continue;
    }
    cell += ch;
    i += 1;
  }
  row.push(cell);
  if (row.length > 1 || row[0] !== '') rows.push(row);
  return rows;
}

function makeUniqueHeaders(headers) {
  const seen = new Map();
  return headers.map((raw, idx) => {
    const base = String(raw ?? '').trim() || `列${idx + 1}`;
    const n = seen.get(base) || 0;
    seen.set(base, n + 1);
    if (n === 0) return base;
    return `${base}_${n + 1}`;
  });
}

function csvCell(v) {
  const s = String(v ?? '');
  if (/[",\r\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function run() {
  const args = parseArgs(process.argv);
  const tablesDir = path.resolve(args.tablesDir);
  const outputCsv = path.resolve(args.outputCsv);
  const outputName = path.basename(outputCsv).toLowerCase();

  if (!fs.existsSync(tablesDir)) {
    throw new Error(`tables dir not found: ${tablesDir}`);
  }

  const csvFiles = fs.readdirSync(tablesDir)
    .filter((x) => x.toLowerCase().endsWith('.csv'))
    .filter((x) => x.toLowerCase() !== outputName)
    .filter((x) => args.includePending || !x.includes('_待补'))
    .sort((a, b) => a.localeCompare(b, 'zh-CN'));

  const allHeaders = [];
  const headerSet = new Set();
  const records = [];

  csvFiles.forEach((file, tableIdx) => {
    const fullPath = path.join(tablesDir, file);
    const rows = parseCsv(fs.readFileSync(fullPath, 'utf8'));
    if (!rows.length) return;

    const headers = makeUniqueHeaders(rows[0]);
    headers.forEach((h) => {
      if (!headerSet.has(h)) {
        headerSet.add(h);
        allHeaders.push(h);
      }
    });

    for (let i = 1; i < rows.length; i += 1) {
      const row = rows[i];
      const rec = {
        __table_order: tableIdx + 1,
        __table_file: file,
        __table_name: file.replace(/\.csv$/i, ''),
        __table_row_no: i,
      };
      headers.forEach((h, j) => {
        rec[h] = row[j] ?? '';
      });
      records.push(rec);
    }
  });

  const outputHeaders = [
    '__table_order',
    '__table_file',
    '__table_name',
    '__table_row_no',
    ...allHeaders,
  ];

  const lines = [];
  lines.push(outputHeaders.map(csvCell).join(','));
  records.forEach((r) => {
    lines.push(outputHeaders.map((h) => csvCell(r[h] ?? '')).join(','));
  });

  fs.mkdirSync(path.dirname(outputCsv), { recursive: true });
  const text = `${lines.join('\n')}\n`;
  fs.writeFileSync(outputCsv, args.bom ? `\uFEFF${text}` : text, 'utf8');
  console.log(
    `merge_tables_done: files=${csvFiles.length} rows=${records.length} output=${outputCsv}`,
  );
}

try {
  run();
} catch (err) {
  console.error(String(err?.stack || err));
  process.exit(1);
}
