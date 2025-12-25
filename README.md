# QuakeCore AI Agent

这是一个基于 AI 的地震数据处理智能体框架。它允许用户上传 SEGY 文件，并通过自然语言与 AI 对话来分析文件结构和内容。

## 功能特点

*   **Web 界面**: 类似 GPT 的聊天界面，基于 Streamlit 构建。
*   **SEGY 支持**: 支持上传和解析标准 SEGY 地震数据文件。
*   **本地 AI**: 集成 LangChain 和 Ollama，支持在本地运行 LLM (如 Llama 3) 进行推理，保护数据隐私。
*   **双模推理**: 也可切换至 DeepSeek API（OpenAI SDK 兼容），满足云端推理需求。
*   **智能工具**: AI 可以自动调用工具读取 SEGY 头信息（文本头、二进制头）和道集数据。

## 快速开始

### 1. 环境准备

确保你已经安装了 Python 3.8+。

```bash
# 克隆仓库 (如果适用)
# git clone ...

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
2.  在“数据源”区域上传 `.segy`/`.sgy` 文件，或直接指向仓库中的示例文件 `data/viking_small.segy`。
3.  在聊天框中输入指令，例如：
    *   "读取segy文件，给我说明其内部的结构"
    *   "显示这个文件的文本头信息"
    *   "这个文件的采样率是多少？"
    *   "读取第0道的统计数据"

## 项目结构

*   `app.py`: Streamlit 前端主程序。
*   `agent/`: AI 智能体相关代码。
    *   `core.py`: Agent 初始化和配置。
    *   `tools.py`: 定义 AI 可调用的 SEGY 处理工具。
*   `utils/`: 底层工具库。
    *   `segy_handler.py`: 基于 `segyio` 的文件读取封装。

## 扩展

*   **模型支持**: 已内置 Ollama 与 DeepSeek，如需扩展更多 OpenAI 兼容接口，可在 `agent/core.py` 的 `_build_llm` 中新增 provider。
*   **功能增强**: 在 `utils/segy_handler.py` 和 `agent/tools.py` 中添加更多地震处理算法（如频谱分析、增益控制等）。

