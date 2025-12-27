# QuakeCore AI Agent

这是一个基于 AI 的地震数据处理智能体框架。它允许用户上传 MiniSEED、SAC、SEG-Y、HDF5 等多种地震数据格式，并通过自然语言与 AI 对话来分析文件结构、获取统计信息或执行相位拾取。

## 功能特点

*   **多格式支持**: 一次性接入 SEGY、MiniSEED、SAC、HDF5、NumPy 数组等主流格式，自动识别采样率与起始时间。
*   **智能拾取**: 内置 STA/LTA、AIC、频率比、AR 模型等多种传统拾取算法，统一归一化评分并输出摘要。
*   **HDF5 自适应读取**: 自动遍历数据集并解码自定义键名/起始时间字段，减少手动配置。
*   **Web 界面**: 类似 GPT 的聊天界面，基于 Streamlit 构建。
*   **本地/云端 AI**: 集成 LangChain + Ollama（本地）以及 DeepSeek API（云端），按需切换推理路径。
*   **智能工具**: AI 可自动调用读取头信息、数据导出、相位拾取等工具，完成常见地震处理任务。

## 快速开始

### 1. 环境准备

确保你已经安装了 Python 3.8+。

```bash
# 克隆仓库 (如果适用)
git clone git@github.com:Chuan1937/QuakeCore.git

# 安装依赖
pip install -r requirements.txt
```

### 2. 选择推理方式

#### 方案 A：本地 Ollama

1.  前往 [Ollama 官网](https://ollama.com/) 下载并安装 Ollama。
2.  拉取一个模型 (推荐 **Qwen2.5:3b**，体积小且中文能力强):
    ```bash
    ollama pull qwen2.5:3b
    ```
3.  保持 `ollama serve` 或 `ollama run <model>` 处于运行状态。

#### 方案 B：DeepSeek API

1.  确认 `requirements.txt` 中的 `openai` / `langchain-openai` 已安装。
2.  在桌面环境变量或 `.env` 中配置 API Key：
    ```bash
    export DEEPSEEK_API_KEY="你的密钥"
    ```
3.  也可以直接在应用侧边栏输入 API Key、模型名称（默认 `deepseek-chat`）与 Base URL (`https://api.deepseek.com`)。

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开 (通常是 `http://localhost:8501`)。

## 使用指南

1.  在左侧边栏选择推理方式：
    *   **本地 Ollama**：填入模型名称（默认 `qwen2.5:3b`），并确保服务正在运行。
    *   **DeepSeek API**：填入模型名称、Base URL 和 API Key（也可通过 `DEEPSEEK_API_KEY` 环境变量注入）。
2.  在“数据源”区域上传 `.segy`/`.sgy`/`.mseed`/`.sac`/`.h5` 文件，或直接使用仓库中的示例文件（如 `data/example.mseed`、`data/example.h5`、`data/viking_small.segy`）。
3.  在聊天框中输入指令，例如：
    *   "读取segy文件，给我说明其内部的结构"
    *   "显示这个文件的文本头信息"
    *   "这个文件的采样率是多少？"
    *   "读取第0道的统计数据"
    *   "对当前加载的波形做初至拾取"
    *   "将第100到200道导出为Excel文件，保存在data/convert/目录下"

## 相位拾取工具

聊天对话触发“run_phase_picking”后，Agent 会自动判断数据类型并运行多种传统拾取算法。可选参数：

*   `source_type`: 数据来源（`mseed`、`sac`、`segy`、`hdf5`、`npy` 等），缺省时按文件扩展名或当前上下文推断。
*   `dataset`: 针对 HDF5 指定数据集名称；留空时会自动遍历并选取首个可用数据集。
*   `sampling_rate`: 当文件缺失采样率元数据时手动提供（单位 Hz）。

结果会列出每条 Trace 的多种拾取方法（STA/LTA、AIC、频率比、AR 模型、特征阈值、自相关等），并附带统一归一化分数与综合摘要，便于快速确认最佳拾取时间。

## 项目结构

*   `app.py`: Streamlit 前端主程序。
*   `agent/`: AI 智能体相关代码。
    *   `core.py`: Agent 初始化和配置，注册所有工具。
    *   `tools.py`: 定义文件读取、转换、相位拾取等 LangChain 工具。
*   `utils/`: 底层工具库。
    *   `segy_handler.py` / `miniseed_handler.py`: 针对不同格式的读取与转换封装。
    *   `phase_picker.py`: 统一的波形预处理与多算法拾取实现。
    *   `hdf5_handler.py`: 自适应数据集解析和导出逻辑。

## 扩展

*   **模型支持**: 已内置 Ollama 与 DeepSeek，如需扩展更多 OpenAI 兼容接口，可在 `agent/core.py` 的 `_build_llm` 中新增 provider。
*   **功能增强**: 在 `utils/segy_handler.py`、`utils/phase_picker.py` 与 `agent/tools.py` 中扩展更多地震处理算法（如频谱分析、自动剪切、机器学习拾取等）。

