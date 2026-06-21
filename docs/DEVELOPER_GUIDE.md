# QuakeCore 开发者指南

本指南面向希望将自己的地震学工具集成到 QuakeCore 的开发者。从零开始，一步一步教你完成。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [项目结构速览](#2-项目结构速览)
3. [添加你的第一个工具（5 分钟）](#3-添加你的第一个工具5-分钟)
4. [工具开发详解](#4-工具开发详解)
5. [进阶：集成外部 Python 包](#5-进阶集成外部-python-包)
6. [进阶：集成命令行工具](#6-进阶集成命令行工具)
7. [进阶：生成图片和文件产物](#7-进阶生成图片和文件产物)
8. [测试与调试](#8-测试与调试)
9. [API 参考](#9-api-参考)
10. [常见问题](#10-常见问题)

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
# DeepSeek API（推荐）
export DEEPSEEK_API_KEY=your_key_here

# 或者使用本地 Ollama
ollama pull qwen2.5:3b
```

### 1.5 验证环境

```bash
# 启动后端
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 另一个终端，启动前端
cd frontend
npm install
npm run dev
```

打开 http://localhost:3000 ，上传一个地震数据文件，试试对话。

---

## 2. 项目结构速览

你只需要关心两个目录：

```
QuakeCore/
├── quakecore_tools/          ← 你的工具放这里
│   ├── registry.py           # 工具注册表（不需要改）
│   ├── helpers.py            # 工具辅助函数（不需要改）
│   ├── context.py            # 会话上下文（不需要改）
│   ├── __init__.py           # 自动发现（不需要改）
│   ├── stats_tools.py        # 示例工具
│   ├── dsa_tools.py          # DSA 深度扫描
│   ├── seispolarity_tools.py # 极性预测
│   └── ...
│
├── skills/                   ← 技能描述文件（可选）
│   ├── phase_picking.md
│   ├── earthquake_location.md
│   └── ...
│
├── agent/                    # AI Agent 核心（不需要改）
├── backend/                  # 后端 API（不需要改）
├── frontend/                 # 前端 UI（不需要改）
└── tests/                    # 测试文件
```

**添加新工具只需要在 `quakecore_tools/` 下创建一个 Python 文件。不需要修改任何其他文件。**

---

## 3. 添加你的第一个工具（5 分钟）

假设你要添加一个工具，用来计算波形数据的频谱分析。

### 第一步：创建文件

在 `quakecore_tools/` 下创建 `spectral_tools.py`：

```python
"""频谱分析工具 — 计算波形的频率谱。"""

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
    name="compute_spectrum",
    category="analysis",
    description="计算当前波形数据的频率谱（FFT），返回主频和频谱图",
    triggers=["频谱", "spectrum", "fft", "频率谱", "主频"],
    needs_file=True,
    file_types=[".mseed", ".miniseed", ".sac", ".segy", ".sgy"],
)
@tool
def compute_spectrum(params: str | dict | None = None) -> str:
    """
    Compute the frequency spectrum (FFT) of the loaded waveform.
    Returns dominant frequency and a spectrum plot.
    Use when the user asks for 'spectrum', '频谱', 'FFT', or '主频'.
    """
    from quakecore_tools.context import get_context

    parsed = parse_param_dict(params)
    ctx = get_context()

    # 获取文件路径：优先用参数指定的，否则用当前加载的
    path = parsed.get("path") or ctx.active_path
    if not path or not os.path.exists(path):
        return tool_error("未找到波形文件。请先上传数据。")

    try:
        import numpy as np
        from obspy import read as obspy_read

        # 读取波形
        st = obspy_read(path)
        if not st or len(st) == 0:
            return tool_error("文件中未找到波形数据。")

        # 取第一条道做频谱分析
        tr = st[0]
        data = tr.data.astype(float)
        sr = float(tr.stats.sampling_rate)

        # FFT
        fft_vals = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=1.0 / sr)
        amplitudes = np.abs(fft_vals)

        # 找主频
        dominant_freq = float(freqs[np.argmax(amplitudes[1:]) + 1])

        # 画频谱图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, amplitudes, color="steelblue", linewidth=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Frequency Spectrum — Dominant: {dominant_freq:.2f} Hz")
        ax.set_xlim(0, sr / 2)
        fig.tight_layout()

        plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "spectrum.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 构建返回结果
        artifacts = []
        entry = build_artifact_entry(plot_path, "image")
        if entry:
            artifacts.append(entry)

        return tool_success(
            f"频谱分析完成。主频: {dominant_freq:.2f} Hz",
            data={
                "dominant_frequency_hz": dominant_freq,
                "sampling_rate": sr,
                "num_samples": len(data),
                "trace": f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}",
            },
            artifacts=artifacts,
        )

    except ImportError:
        return tool_error("需要安装 ObsPy 和 NumPy。运行: pip install obspy numpy")
    except Exception as e:
        return tool_error(f"频谱分析失败: {e}")
```

### 第二步：重启后端

```bash
# 停止后端 (Ctrl+C)，然后重新启动
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 第三步：使用

在 QuakeCore 前端对话框中输入：

- "帮我做频谱分析"
- "what is the dominant frequency?"
- "FFT"

AI 会自动调用你的 `compute_spectrum` 工具，返回主频和频谱图。

**就这么简单。不需要改任何其他文件。**

---

## 4. 工具开发详解

### 4.1 工具文件的基本结构

每个工具文件由三部分组成：

```python
# 1. 导入
from langchain.tools import tool
from quakecore_tools.registry import register_tool
from quakecore_tools.helpers import parse_param_dict, tool_success, tool_error

# 2. 注册 + 定义
@register_tool(
    name="工具名称",           # 必填，AI 和路由使用这个名字
    category="分类",           # 必填，见下方分类列表
    description="功能描述",    # 必填，给 AI 看的
    triggers=["关键词1", "关键词2"],  # 可选，路由匹配用
    needs_file=True,           # 可选，是否需要已加载的文件
    file_types=[".mseed"],     # 可选，支持的文件类型
)
@tool
def my_tool(params: str | dict | None = None) -> str:
    """给 LangChain 看的 docstring，AI 也会读这个。"""
    # 3. 实现逻辑
    ...
```

### 4.2 工具分类（category）

| 分类 | 说明 | 示例 |
|------|------|------|
| `file` | 文件读取、元数据 | get_file_structure |
| `conversion` | 格式转换 | convert_segy_to_numpy |
| `waveform` | 波形读取 | read_file_trace |
| `picking` | 震相拾取 | pick_first_arrivals |
| `location` | 地震定位 | locate_earthquake |
| `monitoring` | 连续监测 | run_continuous_monitoring |
| `analysis` | 数据分析、统计 | compute_spectrum |
| `professional` | 专业算法 | run_dsa_depth_scanning |

### 4.3 参数解析

AI 传入的参数可能是字符串、字典或 None。使用 `parse_param_dict` 统一处理：

```python
parsed = parse_param_dict(params)

# 获取参数，提供默认值
method = parsed.get("method", "default")
threshold = float(parsed.get("threshold", 0.5))
trace_index = int(parsed.get("trace_index", 0))
```

### 4.4 获取当前加载的文件

```python
from quakecore_tools.context import get_context

ctx = get_context()

# 方式 1：获取当前活跃文件（最简单）
path = ctx.active_path

# 方式 2：按类型获取
path = ctx.miniseed_path   # MiniSEED 文件
path = ctx.segy_path       # SEGY 文件
path = ctx.hdf5_path       # HDF5 文件
path = ctx.sac_path        # SAC 文件

# 方式 3：从参数获取（用户可能指定了路径）
path = parsed.get("path") or ctx.active_path
```

### 4.5 返回值规范

**必须返回 JSON 字符串**，使用辅助函数：

```python
from quakecore_tools.helpers import tool_success, tool_error

# 成功：返回 tool_success(消息, 数据, 产物)
return tool_success(
    "分析完成",
    data={"dominant_freq": 5.2, "traces": 3},
    artifacts=[image_entry, file_entry],
)

# 失败：返回 tool_error(错误消息)
return tool_error("文件未找到，请先上传数据。")
```

### 4.6 触发词（triggers）

`triggers` 是一个字符串列表，当用户消息中包含这些关键词时，路由器会优先考虑你的工具。

```python
triggers=["频谱", "spectrum", "fft", "频率谱", "主频"]
```

**建议**：中英文都写上，覆盖面更广。

---

## 5. 进阶：集成外部 Python 包

如果你的工具有现成的 pip 包（如 `seispolarity`、`obspy`），直接导入使用：

```python
@register_tool(name="run_polarity", category="professional", ...)
@tool
def run_polarity(params: str | dict | None = None) -> str:
    """Predict P-wave first-motion polarity."""
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

**关键**：用 `try/except ImportError` 处理缺失依赖，给用户清晰的安装提示。

参考实现：`quakecore_tools/seispolarity_tools.py`

---

## 6. 进阶：集成命令行工具

如果你的工具是一个独立的 Python 脚本或命令行程序，用 `subprocess` 调用：

```python
import subprocess
import sys
import os
from pathlib import Path

@register_tool(name="run_focal_mechanism", category="professional", ...)
@tool
def run_focal_mechanism(params: str | dict | None = None) -> str:
    """Run focal mechanism inversion using external script."""
    parsed = parse_param_dict(params)

    # 找到脚本位置
    script_path = Path(__file__).parent.parent / "external_tools" / "focal_mechanism.py"
    if not script_path.exists():
        return tool_error(f"脚本未找到: {script_path}")

    # 运行
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
            env={**os.environ, "MPLBACKEND": "Agg"},  # 无头模式画图
        )
        if result.returncode != 0:
            return tool_error(f"执行失败: {result.stderr[-500:]}")

        return tool_success(
            "震源机制反演完成",
            data={"stdout": result.stdout[-2000:]},
            artifacts=collect_plots(output_dir),  # 收集生成的图片
        )
    except subprocess.TimeoutExpired:
        return tool_error("执行超时（5 分钟）")
    except Exception as e:
        return tool_error(f"执行出错: {e}")
```

参考实现：`quakecore_tools/dsa_tools.py`

---

## 7. 进阶：生成图片和文件产物

### 7.1 生成图片

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from quakecore_tools.helpers import DEFAULT_STRUCTURE_DIR, build_artifact_entry

# 画图
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data)
ax.set_title("My Plot")
fig.tight_layout()

# 保存
os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "my_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# 构建产物条目
entry = build_artifact_entry(plot_path, "image")
# entry = {"type": "image", "name": "my_plot.png", "path": "structure/my_plot.png", "url": "/api/artifacts/structure/my_plot.png"}
```

### 7.2 生成文件（CSV、HDF5 等）

```python
from quakecore_tools.helpers import DEFAULT_CONVERT_DIR, build_artifact_entry

# 保存文件
os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
csv_path = os.path.join(DEFAULT_CONVERT_DIR, "my_result.csv")
# ... 写入 CSV ...

# 构建产物条目
entry = build_artifact_entry(csv_path, "file")
```

### 7.3 多个产物

```python
artifacts = []
for path in [plot_path, csv_path]:
    entry = build_artifact_entry(path, "image" if path.endswith(".png") else "file")
    if entry:
        artifacts.append(entry)

return tool_success("完成", data={...}, artifacts=artifacts)
```

---

## 8. 测试与调试

### 9.1 测试工具函数

```bash
conda activate quakecore

# 直接调用测试
python -c "
from quakecore_tools.spectral_tools import compute_spectrum
result = compute_spectrum('{\"path\": \"example_data/test.mseed\"}')
print(result)
"
```

### 9.2 验证工具注册

```bash
python -c "
import quakecore_tools
from quakecore_tools.registry import _REGISTRY
assert 'compute_spectrum' in _REGISTRY, '工具未注册!'
print(f'已注册: {_REGISTRY[\"compute_spectrum\"].name}')
print(f'分类: {_REGISTRY[\"compute_spectrum\"].category}')
print(f'触发词: {_REGISTRY[\"compute_spectrum\"].triggers}')
"
```

### 9.3 运行项目测试

```bash
python -m pytest tests/ -q
```

### 9.4 调试技巧

- 工具函数出错时，返回 `tool_error("描述")` 而不是抛异常
- 使用 `print()` 调试，输出会出现在后端日志中
- 图片路径用相对路径（相对于 `data/`），不要用绝对路径

---

## 9. API 参考

### 10.1 `@register_tool` 装饰器

```python
from quakecore_tools.registry import register_tool

@register_tool(
    name: str,                    # 工具名称（默认为函数名）
    category: str,                # 分类
    description: str = "",        # 功能描述
    triggers: list[str] = None,   # 触发关键词
    examples: list[str] = None,   # 使用示例
    needs_file: bool = False,     # 是否需要已加载文件
    file_types: list[str] = None, # 支持的文件类型
)
```

### 10.2 辅助函数

```python
from quakecore_tools.helpers import (
    tool_success,       # tool_success(message, data=None, artifacts=None) -> str
    tool_error,         # tool_error(message) -> str
    parse_param_dict,   # parse_param_dict(raw_params) -> dict
    coerce_int,         # coerce_int(value, allow_none=False, default=None) -> int
    coerce_float,       # coerce_float(value, allow_none=False, default=None) -> float
    build_artifact_entry,       # build_artifact_entry(path, type) -> dict
    build_artifact_response,    # build_artifact_response(result, ...) -> str
    resolve_output_path,        # resolve_output_path(path, default_filename, base_dir) -> str
)
```

### 10.3 会话上下文

```python
from quakecore_tools.context import get_context

ctx = get_context()
ctx.active_path      # 当前活跃文件路径
ctx.current_type     # 文件类型 ("segy"/"miniseed"/"hdf5"/"sac"/None)
ctx.miniseed_path    # MiniSEED 文件路径
ctx.miniseed_paths   # 所有 MiniSEED 文件路径列表
ctx.segy_path        # SEGY 文件路径
ctx.hdf5_path        # HDF5 文件路径
ctx.sac_path         # SAC 文件路径
ctx.uploaded_files   # 所有上传的文件路径列表
ctx.lang             # 当前语言 ("en"/"zh")
```

### 10.4 常用输出目录

```python
from quakecore_tools.helpers import (
    DEFAULT_CONVERT_DIR,    # "data/convert" — 格式转换输出
    DEFAULT_STRUCTURE_DIR,  # "data/structure" — 结构分析输出
    DEFAULT_PICKS_DIR,      # "data/picks" — 拾取结果输出
    DEFAULT_LOCATION_DIR,   # "data/location" — 定位结果输出
)
```

---

## 10. 常见问题

### Q: 我的工具需要 GPU 怎么办？

在工具内部检测 CUDA 可用性，自动 fallback：

```python
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
```

### Q: 工具运行时间很长怎么办？

对于超过 30 秒的任务，使用后台任务模式。参考 `quakecore_tools/monitoring_tools.py` 中的 `run_continuous_monitoring`，它返回 `job_id` 让前端轮询进度。

### Q: 我的工具有多个步骤怎么办？

在一个 `@tool` 函数中完成所有步骤。如果步骤之间需要用户确认，可以在中间返回提示信息，让用户继续。

### Q: 如何让用户选择不同的算法？

通过参数传递：

```python
parsed = parse_param_dict(params)
method = parsed.get("method", "default")  # 用户可选: "default", "fast", "accurate"

if method == "fast":
    result = fast_algorithm(data)
elif method == "accurate":
    result = accurate_algorithm(data)
else:
    result = default_algorithm(data)
```

### Q: 我的工具依赖特定的数据格式怎么办？

在工具内部做格式检查和转换：

```python
from quakecore_tools.context import get_context

ctx = get_context()
if ctx.current_type not in ("miniseed", "sac"):
    return tool_error("此工具仅支持 MiniSEED 和 SAC 格式。请先转换数据。")
```

### Q: 我想修改现有工具怎么办？

直接编辑 `quakecore_tools/` 下对应的文件。重启后端即可生效。

### Q: 工具注册表里的工具太多，AI 会不会选错？

不会。AI 会根据用户的自然语言描述和工具的 `description` + `triggers` 来选择最合适的工具。如果用户描述模糊，AI 会先询问确认。
