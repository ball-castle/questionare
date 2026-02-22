## Current Pipeline (Single Track)

This project runs on a single official data source:

- `data/raw/数据.xlsx`
- `docs/待填数据.docx`

Main pipeline command (default output directory: `output`):

```bash
uv run python scripts/run_current_pipeline.py --input-xlsx data/raw/数据.xlsx --doc-path docs/待填数据.docx
```

Rescreen-only command (persist `961->880` core artifacts to `data/processed_880`):

```bash
uv run python scripts/run_rescreen_to_data.py
```

Reliability/validity-only command (on `data/processed_880/survey_clean_880.csv`, output to `output_data_analysis`):

```bash
uv run python scripts/run_reliability_validity_880.py
```

With chapter 6/7 report generation enabled:

```bash
uv run python scripts/run_current_pipeline.py --input-xlsx data/raw/数据.xlsx --doc-path docs/待填数据.docx --build-report
```

Pipeline sequence:

1. questionnaire analysis
2. award booster tables
3. pending doc fill
4. pending doc consistency check
5. chapter 6/7 report generation (optional, enabled by `--build-report`)

Main outputs:

- `output/问卷数据处理与分析_结果摘要.txt`
- `output/tables/待填数据_回填值清单.csv`
- `output/tables/待填数据_待补项清单.csv`
- `output/tables/待填数据_一致性核查报告.csv`
- `output/pipeline_manifest.json`

Report outputs:

- `output/reports/六七部分_完整报告.md`
- `output/reports/六七部分_完整报告.docx`
- `output/reports/六七章_证据索引.csv`
- `output/reports/六七章_生成日志.json`
- `output/reports/六七部分_输入核查报告.json`

Standalone report commands:

- Prepare only (input audit):

```bash
uv run python scripts/generate_ch6_ch7_report.py --prepare-only
```

- Generate report:

```bash
uv run python scripts/generate_ch6_ch7_report.py
```

Notes:

- Default outline path: project root `六七大纲.md`
- Missing input behavior default: `--missing-policy keep_placeholder`
- Auto migration is enabled by default: when `output_report/` does not exist but `output_current/` exists, report command copies `output_current -> output_report` before audit/generation.
- `run_rescreen_to_data.py` and `run_reliability_validity_880.py` default to idempotent mode: if key outputs already exist, they print `skipped` and do nothing.
- Add `--force` to either script to overwrite existing artifacts.

