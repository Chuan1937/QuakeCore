# QuakeCore 部署与开发指南

面向新成员的完整指南：从零搭建环境、理解架构、到添加你自己的地震学工具。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [项目架构](#2-项目架构)
3. [核心概念](#3-核心概念)
4. [添加新工具（完整流程）](#4-添加新工具完整流程)
5. [工具开发详解](#5-工具开发详解)
6. [集成外部 Python 包](#6-集成外部-python-包)
7. [集成命令行工具](#7-集成命令行工具)
8. [生成图片和文件产物](#8-生成图片和文件产物)
9. [测试与调试](#9-测试与调试)
10. [API 参考](#10-api-参考)
11. [常见问题](#11-常见问题)

---

## 1. 环境搭建

### 1.1 克隆仓库

```bash
git clone https://github.com/Chuan1937/QuakeCore.git
cd QuakeCore
```

### 1.2 创建 Python 环境

```bash
conda create -n quakecore python=3.12 -y
conda activate quakecore
```

### 1.3 安装依赖

```bash
pip install -r requirements.txt
pip install -r requirements-backend.txt
```

### 1.4 配置 LLM

```bash
# DeepSeek API（推荐，速度快）
export DEEPSEEK_API_KEY=your_key_here

# 或者使用本地 Ollama
ollama pull qwen2.5:3b
```

### 1.5 启动服务

```bash
# 终端 1：启动后端
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：启动前端
cd frontend
npm install
npm run dev
```

打开 http://localhost:3000 ，上传一个地震数据文件试试。

### 1.6 验证环境

```bash
conda activate quakecore
python -c "import quakecore_tools; from quakecore_tools.registry import _REGISTRY; print(f'已注册 {len(_REGISTRY)} 个工具')"
```

应输出 `已注册 50 个工具`。

---

## 2. 项目架构

```
QuakeCore/
├── quakecore_tools/          ← 你的工具放这里（核心目录）
│   ├── registry.py           # 工具注册表（@register_tool 装饰器）
│   ├── helpers.py            # 共享辅助函数
│   ├── context.py            # 会话上下文管理
│   ├── __init__.py           # 自动发现 + 遗留工具桥接
│   ├── stats_tools.py        # 示例：新工具
│   ├── dsa_tools.py          # DSA 深度扫描
│   ├── seispolarity_tools.py # 极性预测
│   └── ...                   # 其他工具模块
│
├── agent/                    # AI Agent 核心
│   ├── core.py               # Agent 创建（从注册表获取工具）
│   └── tools.py              # 遗留工具定义（逐步迁移中）
│
├── backend/                  # FastAPI 后端
│   ├── routes/               # API 路由
│   └── services/             # 业务逻辑
│
├── frontend/                 # Next.js 前端
├── skills/                   # 技能描述文件
├── data/                     # 数据输出目录
└── tests/                    # 测试
```

### 请求处理流程

```
用户消息
  → 前端发送 /api/chat/stream
  → RouterService 路由意图（tool_request / result_analysis / settings / general_chat）
  → Agent（DeepSeek tool_calls）选择工具
  → 工具执行（quakecore_tools/）
  → 返回结果 + 产物（图片/文件）
  → 前端展示
```

---

## 3. 核心概念

### 3.1 工具注册表

所有工具通过 `@register_tool` 装饰器注册到全局注册表。Agent 启动时自动发现所有已注册工具。

### 3.2 会话上下文

文件上传后，路径存储在 `context.py` 的 `FileContext` 中。工具通过 `get_context()` 获取当前加载的文件。

### 3.3 工具返回值

所有工具返回 JSON 字符串，使用 `tool_success()` 或 `tool_error()` 辅助函数。

### 3.4 产物（Artifacts）

工具生成的图片、CSV、HDF5 等文件，通过 `artifacts` 数组传递给前端。

---

## 4. 添加新工具（完整流程）

**只需创建一个文件，不需要修改任何其他文件。**

### 4.1 创建工具文件

在 `quakecore_tools/` 下创建 `my_tool.py`：

```python
"""我的地震学工具 — 简要描述功能。"""

import json
import os

from langchain.tools import tool

from quakecore_tools.registry import register_tool
from quakecore_tools.helpers import (
    parse_param_dict,
    tool_success,
    tool_error,
    build_artifact_entry,
    DEFAULT_STRUCTURE_DIR,
)


@register_tool(
    name="run_my_analysis",
    category="analysis",
    description="运行自定义地震分析，计算波形的频率特征",
    triggers=["频率分析", "frequency analysis", "频谱", "自定义分析"],
    needs_file=True,
    file_types=[".mseed", ".miniseed", ".sac", ".segy", ".sgy"],
)
@tool
def run_my_analysis(params: str | dict | None = None) -> str:
    """
    Run custom seismic analysis on loaded waveform data.
    Use when the user asks for 'frequency analysis', '频谱', or '自定义分析'.
    """
    from quakecore_tools.context import get_context

    parsed = parse_param_dict(params)
    ctx = get_context()

    # 1. 获取文件路径
    path = parsed.get("path") or ctx.active_path
    if not path or not os.path.exists(path):
        return tool_error("未找到波形文件。请先上传数据。")

    try:
        import numpy as np
        from obspy import read as obspy_read

        # 2. 读取波形
        st = obspy_read(path)
        tr = st[0]
        data = tr.data.astype(float)
        sr = float(tr.stats.sampling_rate)

        # 3. FFT 分析
        fft_vals = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=1.0 / sr)
        amplitudes = np.abs(fft_vals)
        dominant_freq = float(freqs[np.argmax(amplitudes[1:]) + 1])

        # 4. 画图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, amplitudes, color="steelblue", linewidth=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Dominant Frequency: {dominant_freq:.2f} Hz")
        fig.tight_layout()

        plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "my_analysis.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 5. 构建产物
        artifacts = []
        entry = build_artifact_entry(plot_path, "image")
        if entry:
            artifacts.append(entry)

        return tool_success(
            f"分析完成。主频: {dominant_freq:.2f} Hz",
            data={
                "dominant_frequency_hz": dominant_freq,
                "sampling_rate": sr,
                "num_samples": len(data),
            },
            artifacts=artifacts,
        )

    except ImportError:
        return tool_error("需要安装 ObsPy 和 NumPy。运行: pip install obspy numpy")
    except Exception as e:
        return tool_error(f"分析失败: {e}")
```

### 4.2 重启后端

```bash
# 停止后端 (Ctrl+C)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4.3 验证

```bash
python -c "import quakecore_tools; from quakecore_tools.registry import _REGISTRY; assert 'run_my_analysis' in _REGISTRY; print('✓ 工具已注册')"
```

### 4.4 使用

在 QuakeCore 前端对话框中输入 "帮我做频率分析" 或 "frequency analysis"，AI 会自动调用你的工具。

**就这么简单。不需要修改 `agent/core.py`、`tools_facade.py` 或任何其他文件。**

---

## 5. 工具开发详解

### 5.1 @register_tool 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 工具名称（默认为函数名） |
| `category` | str | 分类：`file` / `conversion` / `picking` / `location` / `monitoring` / `analysis` / `professional` |
| `description` | str | 功能描述（AI 根据这个选择工具） |
| `triggers` | list[str] | 触发关键词 |
| `needs_file` | bool | 是否需要已加载的文件 |
| `file_types` | list[str] | 支持的文件类型 |

### 5.2 参数解析

```python
from quakecore_tools.helpers import parse_param_dict

parsed = parse_param_dict(params)  # 兼容 str / dict / None
method = parsed.get("method", "default")
threshold = float(parsed.get("threshold", 0.5))
```

### 5.3 获取文件上下文

```python
from quakecore_tools.context import get_context

ctx = get_context()
path = ctx.active_path          # 当前活跃文件
path = ctx.miniseed_path        # MiniSEED 文件
path = ctx.segy_path            # SEGY 文件
files = ctx.uploaded_files      # 所有上传文件
```

### 5.4 返回值

```python
from quakecore_tools.helpers import tool_success, tool_error

# 成功
return tool_success("操作完成", data={"key": "value"}, artifacts=[...])

# 失败
return tool_error("错误描述")
```

### 5.5 生成产物

```python
from quakecore_tools.helpers import build_artifact_entry, DEFAULT_STRUCTURE_DIR

# 保存图片
plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "my_plot.png")
fig.savefig(plot_path, dpi=150)

# 构建产物条目
entry = build_artifact_entry(plot_path, "image")
# → {"type": "image", "name": "my_plot.png", "path": "structure/my_plot.png", "url": "/api/artifacts/structure/my_plot.png"}
```

---

## 6. 集成外部 Python 包

如果你的工具有 pip 包（如 `seispolarity`、`obspy`），直接导入：

```python
@register_tool(name="run_polarity", category="professional", ...)
@tool
def run_polarity(params: str | dict | None = None) -> str:
    try:
        from seispolarity.inference import Predictor
    except ImportError:
        return tool_error("需要安装 seispolarity。运行: pip install seispolarity")

    try:
        predictor = Predictor(model_name="ROSS_SCSN")
        predictions = predictor.predict(waveforms)
        return tool_success("预测完成", data={"predictions": predictions})
    except Exception as e:
        return tool_error(f"预测失败: {e}")
```

参考：`quakecore_tools/seispolarity_tools.py`

---

## 7. 集成命令行工具

用 `subprocess` 调用外部脚本：

```python
@register_tool(name="run_external", category="professional", ...)
@tool
def run_external(params: str | dict | None = None) -> str:
    import subprocess, sys
    script_path = "/path/to/your/script.py"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        if result.returncode != 0:
            return tool_error(f"执行失败: {result.stderr[-500:]}")
        return tool_success("执行完成", data={"stdout": result.stdout[-2000:]})
    except subprocess.TimeoutExpired:
        return tool_error("执行超时（5 分钟）")
```

参考：`quakecore_tools/dsa_tools.py`

---

## 8. 生成图片和文件产物

### 图片

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from quakecore_tools.helpers import DEFAULT_STRUCTURE_DIR, build_artifact_entry

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data)
fig.tight_layout()

os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "my_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)

artifacts = [build_artifact_entry(plot_path, "image")]
```

### 文件

```python
from quakecore_tools.helpers import DEFAULT_CONVERT_DIR, build_artifact_entry

csv_path = os.path.join(DEFAULT_CONVERT_DIR, "result.csv")
# ... 写入 CSV ...
artifacts = [build_artifact_entry(csv_path, "file")]
```

### 多个产物

```python
artifacts = []
for path in [plot_path, csv_path]:
    entry = build_artifact_entry(path, "image" if path.endswith(".png") else "file")
    if entry:
        artifacts.append(entry)

return tool_success("完成", data={...}, artifacts=artifacts)
```

---

## 9. 测试与调试

### 测试工具函数

```bash
conda activate quakecore
python -c "
from quakecore_tools.my_tool import run_my_analysis
result = run_my_analysis('{\"path\": \"example_data/test.mseed\"}')
print(result)
"
```

### 验证注册

```bash
python -c "
import quakecore_tools
from quakecore_tools.registry import _REGISTRY
assert 'run_my_analysis' in _REGISTRY, '工具未注册!'
print(f'已注册: {_REGISTRY[\"run_my_analysis\"].name}')
print(f'分类: {_REGISTRY[\"run_my_analysis\"].category}')
print(f'触发词: {_REGISTRY[\"run_my_analysis\"].triggers}')
"
```

### 运行项目测试

```bash
python -m pytest tests/ -q
```

### 调试技巧

- 返回 `tool_error("描述")` 而不是抛异常
- `print()` 输出会出现在后端日志中
- 图片路径用相对路径（相对于 `data/`）

---

## 10. API 参考

### 辅助函数

```python
from quakecore_tools.helpers import (
    tool_success,         # tool_success(message, data=None, artifacts=None) -> str
    tool_error,           # tool_error(message) -> str
    parse_param_dict,     # parse_param_dict(raw_params) -> dict
    coerce_int,           # coerce_int(value, allow_none=False, default=None) -> int
    coerce_float,         # coerce_float(value, allow_none=False, default=None) -> float
    build_artifact_entry, # build_artifact_entry(path, type) -> dict
    resolve_output_path,  # resolve_output_path(path, default_filename, base_dir) -> str
)
```

### 会话上下文

```python
from quakecore_tools.context import get_context

ctx = get_context()
ctx.active_path      # 当前活跃文件路径
ctx.current_type     # 文件类型 ("segy"/"miniseed"/"hdf5"/"sac"/None)
ctx.miniseed_path    # MiniSEED 文件路径
ctx.segy_path        # SEGY 文件路径
ctx.hdf5_path        # HDF5 文件路径
ctx.sac_path         # SAC 文件路径
ctx.uploaded_files   # 所有上传文件路径列表
```

### 输出目录

```python
from quakecore_tools.helpers import (
    DEFAULT_CONVERT_DIR,    # "data/convert"
    DEFAULT_STRUCTURE_DIR,  # "data/structure"
    DEFAULT_PICKS_DIR,      # "data/picks"
    DEFAULT_LOCATION_DIR,   # "data/location"
)
```
