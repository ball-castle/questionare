## Current Pipeline (Single Track)

This project now runs on a single official data source:

- `data/raw/数据.xlsx`
- `docs/待填数据.docx`

Main command:

```bash
uv run python scripts/run_current_pipeline.py --input-xlsx data/raw/数据.xlsx --doc-path docs/待填数据.docx --output-dir output_current
```

Pipeline sequence:

1. questionnaire analysis
2. award booster tables
3. pending doc fill
4. pending doc consistency check

Main outputs:

- `output_current/问卷数据处理与分析_结果摘要.txt`
- `output_current/tables/待填数据_回填值清单.csv`
- `output_current/tables/待填数据_待补项清单.csv`
- `output_current/tables/待填数据_一致性核查报告.csv`
- `output_current/pipeline_manifest.json`

Legacy scripts and outputs are archived under `archive_legacy/`.
