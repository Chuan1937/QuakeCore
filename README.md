# QuakeCore AI Agent

QuakeCore is an AI-based seismic data processing agent framework. It allows users to upload various seismic data formats (MiniSEED, SAC, SEG-Y, HDF5) and interact with an AI via natural language to analyze file structures, get statistics, and perform phase picking.

## Features
- **Multi-Format Support**: Reads SEGY, MiniSEED, SAC, HDF5, NumPy arrays.
- **Smart Phase Picking**: Built-in STA/LTA, AIC, and other traditional picking algorithms.
- **API Backend**: FastAPI routes for chat, uploads, config, skills, and artifacts.
- **Frontend**: A ChatGPT-like Next.js chat UI with integrated click upload, drag-and-drop upload, and paste upload.
- **Local/Cloud AI Support**: Integrates with local Ollama or cloud-based DeepSeek APIs.

## Quick Start

### 1. Installation
```bash
git clone https://github.com/Chuan1937/QuakeCore.git
cd QuakeCore

# Create and activate a conda environment
conda create -n quakecore python=3.12 -y
conda activate quakecore

# Install dependencies
pip install -r requirements.txt

# Install backend API dependencies
pip install -r requirements-backend.txt
```

### 2. Configure LLM

QuakeCore supports two providers:

- **DeepSeek API**
- **Ollama**

Recommended DeepSeek setup:

```bash
export DEEPSEEK_API_KEY=your_key
```

Recommended defaults:

- Provider: `deepseek`
- Model: `deepseek-v4-flash`
- Base URL: `https://api.deepseek.com`

Ollama example:

```bash
ollama pull qwen2.5:3b
```

Default Ollama base URL:

- `http://localhost:11434`

### 3. Start the Backend API

```bash
conda activate quakecore
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

- API: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

- Frontend: `http://localhost:3000`

If the frontend should talk to a non-default backend, set:

```bash
export NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

By default, the frontend uses `http://127.0.0.1:8000`.

## Configuration

### Frontend Settings Page

After starting backend and frontend, open the settings page in the web UI and save model settings there.

- DeepSeek:
  - Provider: `deepseek`
  - Model: `deepseek-v4-flash`
  - API Key: can be left empty if `DEEPSEEK_API_KEY` is already exported
  - Base URL: `https://api.deepseek.com`
- Ollama:
  - Provider: `ollama`
  - Base URL: usually `http://localhost:11434`
  - Model: detected from the local Ollama server

### Backend Config Persistence

Backend LLM settings are persisted to:

- `data/config/llm_config.json`

If that file does not exist, backend defaults are used. For DeepSeek, when `api_key` is empty in config, the backend falls back to:

- `DEEPSEEK_API_KEY`

Example config file:

```json
{
  "provider": "deepseek",
  "model_name": "deepseek-v4-flash",
  "api_key": null,
  "base_url": "https://api.deepseek.com"
}
```

You can also read or update config through the backend API:

- `GET /api/config/defaults`
- `GET /api/config/llm`
- `POST /api/config/llm`

## Startup Notes

Recommended startup order:

1. Start the backend on `127.0.0.1:8000`.
2. Start the frontend on `localhost:3000`.
3. Open the frontend settings page and confirm the LLM provider.
4. Upload data and use chat or workflow routes.

Current local CORS defaults allow:

- `http://localhost:3000`
- `http://127.0.0.1:3000`

## Usage
1. Configure your LLM settings in the sidebar.
2. Upload a seismic data file (or use the examples in `example_data/`).
3. Chat with the AI! Try prompts like:
   - *"Analyze this SEGY file's structure."*
   - *"What is the sampling rate of this file?"*
   - *"Perform phase picking on the loaded waveform."*
   - *"Convert this data to HDF5 format."*
4. In the new frontend chat page, you can upload files with:
   - Click upload (`+` button in composer)
   - Drag-and-drop to the chat page
   - Paste file(s) from clipboard
5. After upload, trigger capabilities with natural language directly:
   - *"请分析当前文件结构"*
   - *"对当前波形做初至拾取"*
   - *"使用当前数据进行地震定位"*
   - *"帮我做连续地震监测"*
   - *"对加州2019年7月4日的17到18点进行地震监测"*

## Natural Language Parameter Resolution

QuakeCore now applies a shared parameter-understanding layer before deterministic tools are executed.

- `RouterService` identifies the intent route, such as `phase_picking`, `earthquake_location`, or `continuous_monitoring`.
- `ToolPlanner` converts natural language into structured tool parameters before tool execution.
- Deterministic routes no longer rely only on raw tool kwargs; they can use LLM planning plus rule-based fallback.
- Continuous monitoring requests support natural-language time windows and region names such as:
  - *"对加州2019年7月4日的17到18点进行地震监测"*
  - *"2019年7月4日的17到18点进行地震监测"*
  - *"对南加州最近 10 小时做连续监测"*

Typical `continuous_monitoring` planning output looks like:

```json
{
  "route": "continuous_monitoring",
  "tool": "run_continuous_monitoring",
  "params": {
    "region": "加州",
    "start": "2019-07-04T17:00:00",
    "end": "2019-07-04T18:00:00"
  },
  "need_rerun": true,
  "confidence": 0.9
}
```

This same parameter resolution is now used by both:

- `/api/chat`
- `/api/workflows/continuous/start`

## Validation

Backend:
```bash
conda run -n quakecore python -m pytest tests/test_backend_health.py tests/test_backend_files.py tests/test_backend_chat_schema.py tests/test_backend_artifacts_route.py tests/test_backend_config.py tests/test_backend_skills.py -q
conda run -n quakecore python -m pytest tests/test_router_service.py -q
conda run -n quakecore python -m pytest tests/test_tool_planner.py tests/test_session_file_context.py tests/test_location_workflow_route.py -q
conda run -n quakecore python -m pytest tests -q
```

Frontend:
```bash
cd frontend
npm run build
```

Smoke scripts (in-process mode, no local socket dependency):
```bash
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_backend.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_upload.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_chat.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_upload_then_chat.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_location_workflow.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_full_web_agent.py
```

Live DeepSeek v4 Flash smoke:
```bash
export DEEPSEEK_API_KEY=your_key
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
python scripts/smoke_deepseek_v4_flash.py
```

Pre-merge consolidated checks:
```bash
pytest tests -q
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_location_workflow.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
cd frontend && npm run build
```

Optional live pytest:
```bash
pytest tests/test_deepseek_live.py -q
```

## Security Notes

- Artifact download route enforces path containment under `data/` and blocks path traversal.
- Upload endpoint accepts unknown extensions as `unknown`; unknown files are not bound to agent current-file state.
- Chat artifact metadata includes `type`, `name`, `path`, and `url` for explicit frontend rendering.
- Do not commit real API keys or `.env` files to the repository.

## Current Limitations

- The project still keeps compatibility with legacy `agent.tools` behaviors.
- Deterministic fast paths now cover `earthquake_location` and `continuous_monitoring`, but many other routes still depend on Agent + tool orchestration.
- Natural-language parameter resolution for deterministic tools is improving, but highly ambiguous prompts may still need clarification or explicit parameters.
- LangGraph is disabled by default (`QUAKECORE_USE_LANGGRAPH=0`).
- RAG and Python Runner are not enabled by default.

No Docker is used in this repository.
