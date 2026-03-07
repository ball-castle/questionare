# Local Data Directory

将本地输入数据放在这个目录中运行分析，例如：

- `data/叶开泰问卷数据.xlsx`
- `data/问卷数据_cleaned_for_SEM.xlsx`

说明：

- 该目录默认不跟踪真实原始数据、清洗结果和批量输出。
- 仓库只保留本说明文件，避免将敏感或体量较大的研究资料上传到 GitHub。
- 若只想验证仓库结构与 CLI，可直接运行 `uv run questionnaire-analysis demo`，不需要在此目录放置真实数据。
