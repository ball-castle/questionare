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
- 本仓库依赖安装在项目本地虚拟环境 `.venv/` 中。
- 系统自带 `python` 不保证已经安装本项目依赖。
- 如果直接执行 `python ...`，很可能会落到 Conda 或系统 Python，而不是 `.venv`。
- 可用 `uv run python -c "import sys; print(sys.executable)"` 检查当前实际解释器。
- 如需手动激活环境，Windows PowerShell 下使用 `.\.venv\Scripts\Activate.ps1`。
- 本地输入数据请放在 `data/` 目录，仓库默认不跟踪该目录下的真实数据文件。

常见现象：

- `uv run python ...` 可以导入 `pandas/scipy/matplotlib`
- 但直接 `python ...` 报 `ModuleNotFoundError`

这通常不是依赖没装，而是当前终端没有进入本项目的 `.venv`。

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

