# QuakeCore AI Agent

QuakeCore is an AI-based seismic data processing agent framework. It allows users to upload various seismic data formats (MiniSEED, SAC, SEG-Y, HDF5) and interact with an AI via natural language to analyze file structures, get statistics, and perform phase picking.

## Features
- **Multi-Format Support**: Reads SEGY, MiniSEED, SAC, HDF5, NumPy arrays.
- **Smart Phase Picking**: Built-in STA/LTA, AIC, and other traditional picking algorithms.
- **Web UI**: A GPT-like chat interface built with Streamlit.
- **API Backend**: FastAPI routes for chat, uploads, config, skills, and artifacts.
- **Frontend**: A Next.js + TypeScript chat/settings/skills UI.
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

### 2. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Run the Backend API
```bash
conda activate quakecore
uvicorn backend.main:app --reload
```
The API runs on `http://localhost:8000`.

### 4. Run the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open your browser to `http://localhost:3000`.

### 5. Setup LLM
- **Local (Ollama)**: Install [Ollama](https://ollama.com/) and pull a model (e.g., `ollama pull qwen2.5:3b`). Configure the model name in the app's sidebar.
- **Cloud (DeepSeek API)**: Enter your DeepSeek API key and base URL in the app's sidebar settings.
- Recommended DeepSeek model: `deepseek-v4-flash`.
- Streamlit and FastAPI both read `DEEPSEEK_API_KEY` from environment when config file key is empty.

## Usage
1. Configure your LLM settings in the sidebar.
2. Upload a seismic data file (or use the examples in `example_data/`).
3. Chat with the AI! Try prompts like:
   - *"Analyze this SEGY file's structure."*
   - *"What is the sampling rate of this file?"*
   - *"Perform phase picking on the loaded waveform."*
   - *"Convert this data to HDF5 format."*

## Validation

Backend:
```bash
conda run -n quakecore python -m pytest tests/test_backend_health.py tests/test_backend_files.py tests/test_backend_chat_schema.py tests/test_backend_artifacts_route.py tests/test_backend_config.py tests/test_backend_skills.py -q
conda run -n quakecore python -m pytest tests/test_router_service.py -q
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
```

Live DeepSeek v4 Flash smoke:
```bash
export DEEPSEEK_API_KEY=your_key
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
python scripts/smoke_deepseek_v4_flash.py
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

No Docker is used in this repository.
