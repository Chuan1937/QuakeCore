"""LLM config persistence and defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLM_CONFIG = {
    "provider": "deepseek",
    "model_name": "deepseek-v4-flash",
    "api_key": "",
    "base_url": "https://api.deepseek.com",
}

DEFAULT_CONFIGS = {
    "deepseek": {
        "provider": "deepseek",
        "model_name": "deepseek-v4-flash",
        "api_key": "",
        "base_url": "https://api.deepseek.com",
    },
    "ollama": {
        "provider": "ollama",
        "model_name": "qwen2.5:3b",
        "api_key": None,
        "base_url": None,
    },
}


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model_name: str
    api_key: str | None = None
    base_url: str | None = None


class ConfigService:
    def __init__(self, config_dir: str | Path = "data/config"):
        self.config_dir = Path(config_dir)
        self.llm_config_path = self.config_dir / "llm_config.json"

    def get_defaults(self) -> dict:
        return {
            "providers": ["deepseek", "ollama"],
            "default_llm_config": dict(DEFAULT_LLM_CONFIG),
            "provider_defaults": DEFAULT_CONFIGS,
        }

    def get_llm_config(self) -> dict:
        if not self.llm_config_path.exists():
            config = dict(DEFAULT_LLM_CONFIG)
            config["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
            return config
        try:
            with self.llm_config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            config = dict(DEFAULT_LLM_CONFIG)
            config["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
            return config

        config = dict(DEFAULT_LLM_CONFIG)
        if isinstance(data, dict):
            for key in ("provider", "model_name", "api_key", "base_url"):
                if key in data:
                    config[key] = data[key]
        if config.get("provider") == "deepseek" and not config.get("api_key"):
            config["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
        if config.get("provider") == "deepseek" and not config.get("base_url"):
            config["base_url"] = "https://api.deepseek.com"
        return config

    def save_llm_config(self, config: LlmConfig | dict) -> dict:
        payload = config.__dict__ if isinstance(config, LlmConfig) else dict(config)
        normalized = {
            "provider": payload.get("provider", DEFAULT_LLM_CONFIG["provider"]),
            "model_name": payload.get("model_name", DEFAULT_LLM_CONFIG["model_name"]),
            "api_key": payload.get("api_key"),
            "base_url": payload.get("base_url"),
        }
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with self.llm_config_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=False, indent=2)
        return normalized
