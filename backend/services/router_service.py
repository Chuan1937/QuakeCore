"""Intent routing and artifact extraction for chat responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactItem:
    type: str
    url: str
    name: str
    path: str


class RouterService:
    _IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
    _ROUTE_KEYWORDS = {
        "earthquake_location": (
            "定位",
            "locate",
            "location",
            "hypocenter",
            "epicenter",
            "震源",
            "震中",
            "event location",
        ),
        "phase_picking": (
            "拾取",
            "phase picking",
            "pick phases",
            "pick",
            "phase",
            "picking",
            "震相",
        ),
        "file_structure": (
            "结构",
            "file structure",
            "structure",
            "inspect file",
            "show structure",
            "read this file",
            "当前文件",
        ),
        "waveform_reading": (
            "waveform",
            "trace",
            "channel",
            "read waveform",
            "read trace",
            "第0道",
            "第 0 道",
            "读取波形",
        ),
        "format_conversion": (
            "convert",
            "conversion",
            "format conversion",
            "convert to",
            "convert into",
            "转换",
            "转成",
            "转为",
        ),
        "continuous_monitoring": (
            "continuous",
            "monitoring",
            "monitor",
            "recent",
            "latest",
            "连续",
            "最近",
            "监测",
        ),
        "map_plotting": (
            "map",
            "plot map",
            "plotting",
            "plot",
            "地图",
        ),
        "seismo_qa": (
            "qa",
            "question answering",
            "faq",
            "ask",
            "explain",
            "what is",
            "seismo qa",
            "地震问答",
        ),
        "settings": (
            "setting",
            "settings",
            "config",
            "configuration",
            "llm",
            "api key",
            "provider",
            "模型",
            "配置",
        ),
    }

    def route_intent(self, message: str) -> str:
        text = str(message or "").lower()
        for route, keywords in self._ROUTE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return route
        if any(token in text for token in ("chat", "hello", "hi", "help")):
            return "general_chat"
        return "general_chat"

    def extract_artifacts(self, answer: str) -> list[ArtifactItem]:
        artifacts: list[ArtifactItem] = []
        for match in self._IMAGE_PATTERN.findall(str(answer or "")):
            source = match.strip().strip("'\"")
            if not source:
                continue
            if source.startswith(("http://", "https://", "/api/artifacts/")):
                continue
            normalized = source.replace("\\", "/")
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if normalized.startswith("data/"):
                normalized = normalized[5:]
            normalized = normalized.lstrip("/")
            if not normalized:
                continue
            name = Path(normalized).name
            artifacts.append(
                ArtifactItem(
                    type="image",
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
