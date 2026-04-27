# Analysis Sandbox

当用户要求解释已有结果、统计结果、绘制统计图、筛选事件、对比震相、分析 catalog/picks 文件时，优先使用 `run_analysis_sandbox`。

不要重新运行完整 workflow，除非用户明确要求重新计算。

优先使用当前会话 `runtime_results` 中的：
- `last_picks_csv`
- `last_catalog_csv`
- `last_catalog_json`
- `last_artifacts`
- `last_continuous_monitoring`

可执行任务：
- 统计 P/S 波数量（`picks_summary`）
- 统计各台站拾取数量（`picks_by_station`）
- 绘制震级分布（`catalog_magnitude_hist`）
- 绘制深度分布（`catalog_depth_hist`）
- 绘制时间序列（`catalog_time_series`）
- 绘制震级-深度散点图（`catalog_mag_depth_scatter`）
- 按索引查看目录事件（`catalog_event_index`）

调用建议：
1. 优先传 `session_id` + `input_artifact_key`（如 `last_catalog_csv`）。
2. 如果用户明确指定文件，再传 `input_path`。
3. 只做轻量分析和绘图，不扫描全局目录。

输出要求：
- 图像保存到 `data/analysis/{session_id}/`
- 结果表格保存为 CSV
- 返回 `artifacts`
- `message` 保持简洁
- 不删除文件
- 不访问互联网
