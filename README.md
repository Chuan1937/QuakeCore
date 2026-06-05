# QuakeCore AI Agent

QuakeCore is an AI-based seismic data processing agent framework. It allows users to upload various seismic data formats (MiniSEED, SAC, SEG-Y, HDF5) and interact with an AI via natural language to analyze file structures, get statistics, and perform phase picking.

![Main Page](deploy/main_page.png)

## Features
- **Multi-Format Support**: Reads SEGY, MiniSEED, SAC, HDF5, NumPy arrays.
- **Smart Phase Picking**: Built-in STA/LTA, AIC, and other traditional picking algorithms.
- **API Backend**: FastAPI routes for chat, uploads, config, skills, and artifacts.
- **Frontend**: A ChatGPT-like Next.js chat UI with integrated click upload, drag-and-drop upload, and paste upload.
- **Local/Cloud AI Support**: Integrates with local Ollama or cloud-based DeepSeek APIs.

## Quick Start

### 1. Prerequisites
For All Platforms, you must have the following installed:
- **Conda** — for managing the Python environment ([Miniforge](https://github.com/conda-forge/miniforge) recommended, or [Miniconda](https://docs.anaconda.com/miniconda/) / [Anaconda](https://www.anaconda.com/download))
- **Node.js & npm** — for running the frontend ([Download](https://nodejs.org/))

### 2. Platform-Specific Installation

#### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/Chuan1937/QuakeCore.git
cd QuakeCore

# 2. Create and activate conda environment
conda create -n quakecore python=3.12 -y
conda activate quakecore

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-backend.txt

# 4. Install frontend dependencies
cd frontend
npm install
cd ..
```

#### Linux (Ubuntu/Debian)

```bash
# 1. Install system dependencies (if needed)
sudo apt update
sudo apt install -y git curl

# 2. Clone the repository
git clone https://github.com/Chuan1937/QuakeCore.git
cd QuakeCore

# 3. Create and activate conda environment
conda create -n quakecore python=3.12 -y
conda activate quakecore

# 4. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-backend.txt

# 5. Install frontend dependencies
cd frontend
npm install
cd ..
```

#### macOS

```bash
# 1. Install Homebrew (if not installed)
# Not China
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# China
/bin/bash -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"

# 2. Install system dependencies
brew install git curl

# 3. Clone the repository
git clone https://github.com/Chuan1937/QuakeCore.git
cd QuakeCore

# 4. Create and activate conda environment
conda create -n quakecore python=3.12 -y
conda activate quakecore

# 5. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-backend.txt

# 6. Install frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Configure LLM

QuakeCore supports two providers:

- **DeepSeek API** (Recommended)
- **Ollama**

#### Windows (PowerShell)

```powershell
# DeepSeek API Key
$env:DEEPSEEK_API_KEY="your_key"

# Or set permanently
[System.Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your_key", "User")
```

#### Linux / macOS (Bash/Zsh)

```bash
# DeepSeek API Key
export DEEPSEEK_API_KEY=your_key

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export DEEPSEEK_API_KEY=your_key' >> ~/.bashrc
```

**Recommended defaults:**

- Provider: `deepseek`
- Model: `deepseek-v4-flash`
- Base URL: `https://api.deepseek.com`

**Ollama example:**

```bash
ollama pull qwen2.5:3b
```

**Default Ollama base URL:**

- `http://localhost:11434`

### 4. Start the Backend API

#### Windows (PowerShell)

```powershell
conda activate quakecore
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Linux / macOS (Bash/Zsh)

```bash
conda activate quakecore
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Backend URLs:**

- API: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

### 5. Start the Frontend

#### Windows (PowerShell)

```powershell
cd frontend
npm run dev
```

#### Linux / macOS (Bash/Zsh)

```bash
cd frontend
npm run dev
```

**Open:**

- Frontend: `http://localhost:3000`

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
   - *"please do phase picking"*
   - *"请分析当前文件结构"*
   - *"对当前波形做初至拾取"*
   - *"使用当前数据进行地震定位"*
   - *"帮我做连续地震监测"*
   - *"对加州2019年7月4日的17到18点进行地震监测"*

![Phase picking result](deploy/phase_pick.png)
*Example: Phase picking results displayed in the chat interface.*

