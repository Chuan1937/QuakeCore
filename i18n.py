"""
Internationalization (i18n) support for QuakeCore UI.
Supports Chinese (zh) and English (en).
"""

TRANSLATIONS = {
    # Header & language toggle
    "lang_toggle": {"zh": "EN", "en": "中"},
    "lang_tooltip": {"zh": "切换为英文", "en": "Switch to Chinese"},
    "settings_tooltip": {"zh": "模型配置", "en": "Model Settings"},

    # Welcome screen
    "app_title": {"zh": "震元引擎", "en": "QuakeCore Engine"},
    "app_subtitle": {"zh": "智能地震数据分析助手", "en": "Intelligent Seismic Data Analysis Assistant"},
    "app_formats": {
        "zh": "支持 SEGY / MiniSEED / HDF5 / SAC 格式 · 拖拽上传",
        "en": "SEGY / MiniSEED / HDF5 / SAC supported · Drag & drop",
    },

    # Settings dialog
    "settings_title": {"zh": "模型配置", "en": "Model Configuration"},
    "select_engine": {"zh": "选择推理引擎", "en": "Select Inference Engine"},
    "engine_label": {"zh": "推理引擎", "en": "Inference Engine"},
    "deepseek_option": {"zh": "DeepSeek API", "en": "DeepSeek API"},
    "ollama_option": {"zh": "本地 Ollama", "en": "Local Ollama"},
    "model_name": {"zh": "模型名称", "en": "Model Name"},
    "model_label": {"zh": "模型", "en": "Model"},
    "api_key_label": {"zh": "API Key", "en": "API Key"},
    "ollama_hint": {
        "zh": "确保本地已安装 Ollama 并运行对应模型",
        "en": "Make sure Ollama is installed and the model is running",
    },
    "save": {"zh": "保存", "en": "Save"},
    "cancel": {"zh": "取消", "en": "Cancel"},

    # Chat
    "chat_placeholder": {
        "zh": "输入问题或拖拽文件...",
        "en": "Type a question or drag files...",
    },
    "thinking": {"zh": "思考中...", "en": "Thinking..."},
    "done": {"zh": "完成", "en": "Done"},
    "thinking_process": {"zh": "思考过程", "en": "Thinking Process"},
    "error": {"zh": "错误", "en": "Error"},
    "uploaded": {"zh": "上传", "en": "Uploaded"},
    "read_file_default": {
        "zh": "请读取这个文件的基本信息。",
        "en": "Please read the basic info of this file.",
    },
}


def t(key: str, lang: str = "zh") -> str:
    """Get translated text for a given key."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("zh", key))
