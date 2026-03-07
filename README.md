# Questionnaire Analysis Pipeline

本仓库是一个 analysis-only 的 Python 项目，保留问卷数据清洗、统计分析、建模、聚类与 SEM 脚本，不包含文档生成链路，也不提交原始问卷数据和批量生成结果。

## Environment

- Python `3.13`
- dependency manager: `uv`

安装依赖：

```bash
uv sync
```

说明：

- 请优先使用 `uv run ...` 执行命令。
- 系统自带 `python` 不保证已经安装本项目依赖。
- 本地输入数据请放在 `data/` 目录，仓库默认不跟踪该目录下的真实数据文件。

## Unified CLI

查看统一入口帮助：

```bash
uv run python main.py --help
```

主流程：

```bash
uv run python main.py pipeline --input-xlsx data/叶开泰问卷数据.xlsx
```

880 样本重筛：

```bash
uv run python main.py rescreen-880 --input-xlsx data/叶开泰问卷数据.xlsx --force
```

880 样本信效度：

```bash
uv run python main.py reliability-880 --force
```

Logit：

```bash
uv run python main.py logit --input-csv data/data_analysis/_source_analysis/tables/survey_clean.csv
```

聚类：

```bash
uv run python main.py clustering --input-csv data/data_analysis/_source_analysis/tables/survey_clean.csv
```

SEM：

```bash
uv run python main.py sem --input-file data/问卷数据_cleaned_for_SEM.xlsx
```

## Main Outputs

主流程默认输出到 `output/`，核心产物包括：

- `output/问卷数据处理与分析_结果摘要.txt`
- `output/tables/survey_clean.csv`
- `output/tables/信度分析表.csv`
- `output/tables/效度分析表.csv`
- `output/tables/建议落地行动矩阵.csv`
- `output/pipeline_manifest.json`

`run_rescreen_to_data.py` 和 `run_reliability_validity_880.py` 默认幂等；关键产物已存在时会输出 `skipped`，需要覆盖时加 `--force`。

## Repository Layout

- `main.py`: 统一 CLI 分发入口
- `scripts/`: 核心分析脚本与内部模块
- `docs/scripts-core.md`: 当前保留脚本说明
- `samples/`: 非行级、非原始数据的样例输出
- `data/README.md`: 本地数据放置说明

