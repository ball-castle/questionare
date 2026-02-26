#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";

function parseArgs(argv) {
  const out = {
    inputXlsx: "data/叶开泰问卷数据.xlsx",
    template: "docs/待填数据.docx",
    outputDoc: "docs/待填数据_已回填.docx",
    outputJson: "output_data_analysis/js_report_metrics.json",
    ageFigure: "output_data_analysis/figures/年龄段人数_性别堆叠图.png",
    distributedN: "1000",
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--input-xlsx") out.inputXlsx = argv[++i] || out.inputXlsx;
    else if (a === "--template") out.template = argv[++i] || out.template;
    else if (a === "--output-doc") out.outputDoc = argv[++i] || out.outputDoc;
    else if (a === "--output-json") out.outputJson = argv[++i] || out.outputJson;
    else if (a === "--age-figure") out.ageFigure = argv[++i] || out.ageFigure;
    else if (a === "--distributed-n") out.distributedN = argv[++i] || out.distributedN;
  }
  return out;
}

function run(cmd, args, cwd) {
  const rec = `${cmd} ${args.join(" ")}`;
  console.log(`RUN ${rec}`);
  const r = spawnSync(cmd, args, {
    cwd,
    stdio: "inherit",
    shell: false,
    encoding: "utf-8",
  });
  if (r.status !== 0) {
    throw new Error(`Command failed (${r.status}): ${rec}`);
  }
}

function main() {
  const args = parseArgs(process.argv);
  const cwd = process.cwd();
  const pyScript = path.resolve("scripts/build_report_metrics.py");
  const jsScript = path.resolve("scripts/js/fill_pending_doc_js.mjs");

  run("uv", [
    "run",
    "python",
    pyScript,
    "--input-xlsx",
    args.inputXlsx,
    "--distributed-n",
    args.distributedN,
    "--output-json",
    args.outputJson,
    "--age-figure",
    args.ageFigure,
  ], cwd);

  run("node", [
    jsScript,
    "--template",
    args.template,
    "--metrics",
    args.outputJson,
    "--output",
    args.outputDoc,
  ], cwd);

  console.log(`report_fill_done: doc=${path.resolve(args.outputDoc)}`);
}

main();
