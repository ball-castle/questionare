# Scripts Core

## 公开入口脚本

- `scripts/run_current_pipeline.py`: 运行主分析流程，串联问卷分析与补强表生成。
- `scripts/run_rescreen_to_data.py`: 执行 `961 -> 880` 样本重筛并固化核心产物。
- `scripts/run_reliability_validity_880.py`: 对 880 主样本执行信度与效度分析。
- `scripts/run_logit_improved_v5.py`: 运行改进版 Logit 建模。
- `scripts/run_clustering_optimized_v3.py`: 运行聚类分析并输出画像、稳定性与外部效度结果。
- `scripts/run_sem.py`: 运行结构方程模型分析并输出拟合与路径结果。

## 内部共享模块

- `scripts/qp_io.py`: XLSX/CSV 读写与通用格式化工具。
- `scripts/qp_stats.py`: 信效度、Logit、聚类等基础统计函数。
- `scripts/convert_961_to_108.py`: 64 列问卷到 108 列标准结构的转换与审计。
- `scripts/run_questionnaire_analysis.py`: 主分析核心实现。
- `scripts/run_questionnaire_analysis_v2.py`: 兼容 64/108 列输入的隔离执行包装器。
- `scripts/generate_award_boosters.py`: 国奖补强表生成核心实现。
- `scripts/generate_award_boosters_v2.py`: 可配置路径的国奖补强包装器。

## 不再维护脚本

- 文档生成与 Node 依赖链脚本已移除。
- 一次性绘图脚本、备选重绘脚本和历史版本 v2/v3/v4 实验脚本已移除。
- `run_user_rescreen.py`、`generate_current_status_outputs.py`、`build_table17_payment_independence.py` 已移除。
