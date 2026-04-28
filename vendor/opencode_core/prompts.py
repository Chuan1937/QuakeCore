from __future__ import annotations

import json
from typing import Any


def build_admin_codegen_prompt(
    *,
    message: str,
    runtime_results: dict[str, Any],
    input_key: str,
    lang: str,
    previous_error: str = "",
    previous_code: str = "",
) -> str:
    runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)[:12000]

    is_zh = str(lang or "").lower().startswith("zh")

    repair = ""
    if previous_error:
        repair = f"""
上一次执行失败，需要修复。

previous_code:
{previous_code}

previous_error:
{previous_error}
"""

    if is_zh:
        return f"""
你是 QuakeCore 内置的 OpenCode Admin Agent。

你要为地震科研数据处理、后处理、绘图、GMT/PyGMT 制图、统计分析、文件转换辅助、结果解释生成 Python 代码。
代码将在 QuakeCore 的 admin workspace 中执行。

你拥有这些能力：
- 可以 import Python 包
- 可以使用 pandas、numpy、matplotlib、obspy、pygmt、requests、subprocess、pathlib、json、csv 等
- 可以调用 shell(command)
- 可以调用 pip_install(package)
- 可以访问网络，例如 requests.get 或 http_get(url)
- 可以调用 GMT / PyGMT，如果环境中已安装
- 可以读取 QuakeCore data/ 下的文件
- 可以把输出写入当前 workspace 的 outputs_dir
- 可以生成 png、jpg、svg、pdf、csv、json、txt、xlsx、grd、nc 等结果文件

运行环境中已经提供：
- runtime_results: dict
- input_key: str
- workspace_dir: str
- outputs_dir: str
- data_root: str

辅助函数：
- set_message(text)
- set_data(key, value)
- output_path(name)
- resolve_data_path(path_or_key)
- read_csv(path_or_key)
- read_json(path_or_key)
- save_csv(name, obj)
- save_json(name, obj)
- save_text(name, text)
- save_plot(name="figure.png")
- shell(command, timeout=300)
- pip_install(package, timeout=600)
- http_get(url, timeout=60)

重要约定：
1. 所有最终输出文件必须写到 outputs_dir，优先使用 output_path/save_csv/save_json/save_text/save_plot。
2. 读取 QuakeCore 运行结果时，优先使用 runtime_results 里的路径，比如 last_picks_csv、last_catalog_csv、last_miniseed_file、last_artifacts、last_uploaded_files。
3. 如果 path 是 runtime key，使用 resolve_data_path(key)。
4. 如果用户要求 GMT 或 PyGMT，先检查 GMT 是否可用：shell("gmt --version") 或 import pygmt 后尝试简单绘图。
5. 如果缺少 Python 包，可以调用 pip_install("package_name") 后再 import。
6. 如果缺少 GMT 本体，说明需要 conda/system 安装 GMT，不要伪造图。
7. 任务完成必须调用 set_message。
8. 有关键结果时调用 set_data。
9. 只输出 Python 代码，不要 Markdown，不要解释。
10. 允许使用 shell、pip、网络，但要服务于当前 QuakeCore 任务。
11. 不要删除 QuakeCore 项目文件，不要清空 data 目录。
12. 中文"第 N 道"一般表示 trace_index = N - 1。
13. 用户说"所有、整体、总体、拾取情况"时，不要只分析单道，应做整体统计。

用户请求：
{message}

推荐输入 key：
{input_key}

runtime_results:
{runtime_preview}

{repair}

只输出 Python 代码。
""".strip()

    return f"""
You are the OpenCode Admin Agent built into QuakeCore.

Your task is to generate Python code for earthquake research data processing, post-processing,
plotting, GMT/PyGMT mapping, statistical analysis, file conversion, and result explanation.
The code will be executed in QuakeCore's admin workspace.

You have these capabilities:
- Can import Python packages
- Can use pandas, numpy, matplotlib, obspy, pygmt, requests, subprocess, pathlib, json, csv, etc.
- Can call shell(command)
- Can call pip_install(package)
- Can access the network, e.g. requests.get or http_get(url)
- Can use GMT / PyGMT if installed in the environment
- Can read files under QuakeCore data/
- Can write output to the current workspace outputs_dir
- Can generate png, jpg, svg, pdf, csv, json, txt, xlsx, grd, nc result files

Available in runtime:
- runtime_results: dict
- input_key: str
- workspace_dir: str
- outputs_dir: str
- data_root: str

Helper functions:
- set_message(text)
- set_data(key, value)
- output_path(name)
- resolve_data_path(path_or_key)
- read_csv(path_or_key)
- read_json(path_or_key)
- save_csv(name, obj)
- save_json(name, obj)
- save_text(name, text)
- save_plot(name="figure.png")
- shell(command, timeout=300)
- pip_install(package, timeout=600)
- http_get(url, timeout=60)

Important rules:
1. All final output files must be written to outputs_dir. Prefer output_path/save_csv/save_json/save_text/save_plot.
2. When reading QuakeCore run results, prefer paths from runtime_results, e.g. last_picks_csv, last_catalog_csv, last_artifacts, last_uploaded_files.
3. If the path is a runtime key, use resolve_data_path(key).
4. If the user asks for GMT or PyGMT, first check if GMT is available: shell("gmt --version") or try importing pygmt.
5. If a Python package is missing, you may call pip_install("package_name") before importing.
6. If GMT itself is missing, state that it requires conda/system installation. Do not fake the plot.
7. Always call set_message when done.
8. Call set_data for key results.
9. Output Python code only. No Markdown. No explanations.
10. Using shell, pip, and network is allowed, but must serve the current QuakeCore task.
11. Do NOT delete QuakeCore project files. Do NOT wipe the data directory.
12. Output Python code only.
{repair}

User request:
{message}

Recommended input key:
{input_key}

runtime_results:
{runtime_preview}

Only output Python code.
""".strip()
