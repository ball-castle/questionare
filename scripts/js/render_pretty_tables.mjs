#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import ExcelJS from 'exceljs';

function parseArgs(argv) {
  const out = {
    tablesDir: 'output/tables',
    outputXlsx: 'output/大纲数据总表_880.xlsx',
    title: '大纲数据总表（880口径）',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--tables-dir') out.tablesDir = argv[++i];
    else if (a === '--output-xlsx') out.outputXlsx = argv[++i];
    else if (a === '--title') out.title = argv[++i];
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

function safeSheetName(name, used) {
  let n = name.replace(/[\\/*?:\[\]]/g, '_');
  if (n.length > 31) n = n.slice(0, 31);
  let base = n;
  let idx = 1;
  while (used.has(n)) {
    const suffix = `_${idx}`;
    n = (base.slice(0, 31 - suffix.length) + suffix);
    idx += 1;
  }
  used.add(n);
  return n;
}

function styleSheet(ws) {
  ws.views = [{ state: 'frozen', ySplit: 1 }];
  const header = ws.getRow(1);
  header.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Microsoft YaHei UI', size: 11 };
  header.alignment = { vertical: 'middle', horizontal: 'center', wrapText: true };
  header.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1F4E78' } };

  ws.eachRow((row, rowNumber) => {
    row.eachCell((cell) => {
      cell.border = {
        top: { style: 'thin', color: { argb: 'FFD9D9D9' } },
        left: { style: 'thin', color: { argb: 'FFD9D9D9' } },
        bottom: { style: 'thin', color: { argb: 'FFD9D9D9' } },
        right: { style: 'thin', color: { argb: 'FFD9D9D9' } },
      };
      if (rowNumber > 1) {
        cell.font = { name: 'Microsoft YaHei UI', size: 10 };
        cell.alignment = { vertical: 'top', horizontal: 'left', wrapText: true };
      }
    });
  });

  ws.columns.forEach((col) => {
    let max = 10;
    col.eachCell({ includeEmpty: true }, (cell) => {
      const v = cell.value == null ? '' : String(cell.value);
      max = Math.max(max, Math.min(60, v.length + 2));
    });
    col.width = max;
  });
}

async function run() {
  const args = parseArgs(process.argv);
  const tablesDir = path.resolve(args.tablesDir);
  const outputXlsx = path.resolve(args.outputXlsx);
  const outDir = path.dirname(outputXlsx);

  if (!fs.existsSync(tablesDir)) {
    throw new Error(`tables dir not found: ${tablesDir}`);
  }

  const csvFiles = fs.readdirSync(tablesDir)
    .filter((x) => x.toLowerCase().endsWith('.csv'))
    .sort((a, b) => a.localeCompare(b, 'zh-CN'));

  const wb = new ExcelJS.Workbook();
  wb.creator = 'Codex';
  wb.created = new Date();

  const readme = wb.addWorksheet('README');
  readme.addRow([args.title]);
  readme.addRow(['generated_at', new Date().toISOString()]);
  readme.addRow(['tables_dir', tablesDir]);
  readme.addRow(['csv_count', csvFiles.length]);
  readme.addRow([]);
  readme.addRow(['sheet_name', 'source_file']);

  const used = new Set(['README']);

  for (const file of csvFiles) {
    const p = path.join(tablesDir, file);
    const text = fs.readFileSync(p, 'utf8');
    const rows = parseCsv(text);
    if (!rows.length) continue;

    const sheet = safeSheetName(file.replace(/\.csv$/i, ''), used);
    readme.addRow([sheet, file]);

    const ws = wb.addWorksheet(sheet);
    for (const r of rows) ws.addRow(r);
    styleSheet(ws);
  }

  styleSheet(readme);
  fs.mkdirSync(outDir, { recursive: true });
  await wb.xlsx.writeFile(outputXlsx);
  console.log(`pretty_tables_done: ${outputXlsx}`);
}

run().catch((err) => {
  console.error(String(err?.stack || err));
  process.exit(1);
});
