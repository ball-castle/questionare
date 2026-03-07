# Questionnaire Analysis Pipeline

公开的 analysis-only Python 项目，保留问卷数据清洗、统计分析、建模、聚类与 SEM 流程，不提交原始问卷数据和批量研究结果。

## Requirements

- Python `3.11+`
- `uv`

安装依赖：

```bash
uv sync --locked
```

统一 CLI：

```bash
uv run questionnaire-analysis --help
```

兼容入口仍可用：

```bash
uv run python main.py --help
```

## Quick Start

不依赖私有问卷文件的 demo：

```bash
uv run questionnaire-analysis demo --output-dir demo-output
```

这会生成一套合成输出，便于检查目录结构、CLI 和 CI 是否正常。

使用本地真实数据运行主流程：

```bash
uv run questionnaire-analysis pipeline --input-xlsx data/叶开泰问卷数据.xlsx
```

常用命令：

```bash
uv run questionnaire-analysis rescreen-880 --input-xlsx data/叶开泰问卷数据.xlsx --force
uv run questionnaire-analysis reliability-880 --force
uv run questionnaire-analysis logit --input-csv data/data_analysis/_source_analysis/tables/survey_clean.csv
uv run questionnaire-analysis clustering --input-csv data/data_analysis/_source_analysis/tables/survey_clean.csv
uv run questionnaire-analysis sem --input-file data/问卷数据_cleaned_for_SEM.xlsx
```

## Repository Layout

- `src/questionnaire_analysis/`: 核心包与正式 CLI 模块
- `scripts/`: 兼容旧路径的轻量 wrapper
- `samples/`: 非敏感样例输出
- `data/`: 本地数据目录，占位文件会保留，真实数据默认不跟踪
- `tests/`: 最小 smoke tests
- `.github/workflows/ci.yml`: GitHub Actions CI

## Development

运行测试：

```bash
uv run python -m unittest discover -s tests -v
```

说明：

- 请优先使用 `uv run ...`，避免落到系统 Python 或 Conda 环境。
- `scripts/` 下的文件保留给旧命令和手动脚本入口；新开发建议直接从 `questionnaire_analysis` 包导入。
- `samples/` 只展示产物结构，不代表正式研究结论。
