# QuakeCore AI Agent

QuakeCore is an AI-based seismic data processing agent framework. It allows users to upload various seismic data formats (MiniSEED, SAC, SEG-Y, HDF5) and interact with an AI via natural language to analyze file structures, get statistics, and perform phase picking.

## Features
- **Multi-Format Support**: Reads SEGY, MiniSEED, SAC, HDF5, NumPy arrays.
- **Smart Phase Picking**: Built-in STA/LTA, AIC, and other traditional picking algorithms.
- **Web UI**: A GPT-like chat interface built with Streamlit.
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
```

### 2. Run the App
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Setup LLM
- **Local (Ollama)**: Install [Ollama](https://ollama.com/) and pull a model (e.g., `ollama pull qwen2.5:3b`). Configure the model name in the app's sidebar.
- **Cloud (DeepSeek API)**: Enter your DeepSeek API key and base URL in the app's sidebar settings.

## Usage
1. Configure your LLM settings in the sidebar.
2. Upload a seismic data file (or use the examples in `example_data/`).
3. Chat with the AI! Try prompts like:
   - *"Analyze this SEGY file's structure."*
   - *"What is the sampling rate of this file?"*
   - *"Perform phase picking on the loaded waveform."*
   - *"Convert this data to HDF5 format."*

