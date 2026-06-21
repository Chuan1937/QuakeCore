"""Intent routing and artifact extraction for chat responses.

The router does lightweight classification only:
- Is this a tool request or general chat?
- For tool requests, return a broad category; the Agent picks the specific tool.

Tool selection is handled by the Agent's ReAct loop, not by the router.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from backend.services.artifact_utils import to_data_relative_path


@dataclass(frozen=True)
class ArtifactItem:
    type: str
    url: str
    name: str
    path: str


# Broad intent categories. The router picks one; the Agent picks the specific tool.
_INTENT_CATEGORIES = {
    "tool_request": "用户想要执行一个地震学操作（分析、转换、拾取、定位、监测、绘图等）",
    "result_analysis": "用户想查看、分析、统计、解释之前操作的结果",
    "settings": "用户想修改配置、模型、API Key 等设置",
    "general_chat": "用户在问概念性问题、打招呼、或请求不属于工具操作的帮助",
}


class RouterService:
    _IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
    _PATH_PATTERN = re.compile(
        r"(?P<path>(?:/api/artifacts/|\.?/)?data/[^\s`'\"<>()，。；！？：:,]+|/api/artifacts/[^\s`'\"<>()，。；！？：:,]+)",
        re.IGNORECASE,
    )

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazy-init the LLM for routing."""
        if self._llm is not None:
            return self._llm
        try:
            from agent.core import _build_llm
            from backend.services.config_service import ConfigService
            cfg = ConfigService().get_llm_config()
            self._llm = _build_llm(
                provider=cfg.get("provider", "deepseek"),
                model_name=cfg.get("model_name", "deepseek-v4-flash"),
                api_key=cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
                base_url=cfg.get("base_url"),
                streaming=False,
            )
        except Exception:
            self._llm = None
        return self._llm

    def route_intent(self, message: str) -> str:
        """Classify user intent into a broad category.

        For tool requests, we return "tool_request" — the Agent's ReAct loop
        will pick the correct tool based on the full context.
        """
        text = str(message or "").strip()
        if not text:
            return "general_chat"

        text_lower = text.lower()

        # Short messages are likely follow-ups — let the Agent handle them
        if len(text) < 10:
            return "tool_request"

        # Fast path: obvious settings requests
        if any(kw in text_lower for kw in ("设置模型", "settings", "配置模型", "api key", "切换模型", "change model")):
            return "settings"

        # Fast path: obvious result analysis (follow-up on previous results)
        if any(kw in text_lower for kw in ("解释结果", "explain result", "统计一下", "分析结果", "看看第")):
            return "result_analysis"

        # Use LLM for everything else
        llm = self._get_llm()
        if llm is None:
            return "tool_request"

        prompt = f"""分类用户意图。只输出 JSON：{{"intent": "类别"}}

类别：
- tool_request: 用户想执行地震学操作（拾取、定位、监测、转换、分析数据、下载数据、读取文件、绘图、频谱分析等任何工具操作）
- result_analysis: 用户想查看/分析/统计之前操作的结果
- settings: 用户想修改模型配置、API Key、切换模型
- general_chat: 概念性问题、打招呼

注意：简短的回复（如数字、"是"、"确认"、"6个通道"）通常是对话的延续，应归为 tool_request。

用户消息：{text}

JSON："""

        try:
            out = llm.invoke(prompt)
            content = str(getattr(out, "content", "") or "").strip()
            match = re.search(r'\{[^}]+\}', content)
            if match:
                parsed = json.loads(match.group())
                intent = str(parsed.get("intent", "")).strip()
                if intent in _INTENT_CATEGORIES:
                    return intent
        except Exception:
            pass

        # Fallback
        if any(kw in text_lower for kw in ("hi", "hello", "你好", "help", "帮助")):
            return "general_chat"
        return "tool_request"

    def extract_artifacts(self, answer: str) -> list[ArtifactItem]:
        artifacts: list[ArtifactItem] = []
        data_root = Path("data")
        sources: list[tuple[str, str]] = []
        text = str(answer or "")
        for match in self._IMAGE_PATTERN.findall(text):
            sources.append(("image", match))
        for match in self._PATH_PATTERN.finditer(text):
            sources.append(("auto", match.group("path")))

        seen_paths: set[str] = set()
        for hint_type, source_raw in sources:
            source = str(source_raw).strip().strip("'\"`")
            if not source:
                continue
            if source.startswith(("http://", "https://")):
                continue
            normalized = to_data_relative_path(source)
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            resolved = self.resolve_artifact_path(data_root, normalized)
            if resolved is None or not resolved.is_file():
                continue
            suffix = resolved.suffix.lower()
            artifact_type = hint_type
            if artifact_type == "auto":
                artifact_type = "image" if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"} else "file"
            name = Path(normalized).name
            artifacts.append(
                ArtifactItem(
                    type=artifact_type,
                    url=f"/api/artifacts/{normalized}",
                    name=name,
                    path=normalized,
                )
            )
        return artifacts

    @staticmethod
    def resolve_artifact_path(data_root: str | Path, artifact_path: str) -> Path | None:
        root = Path(data_root).resolve()
        target = (root / artifact_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return target
