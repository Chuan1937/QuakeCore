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
    _PATH_PATTERN = re.compile(
        r"(?P<path>(?:/api/artifacts/|\.?/)?data/[^\s`'\"<>()，。；！？：:,]+|/api/artifacts/[^\s`'\"<>()，。；！？：:,]+)",
        re.IGNORECASE,
    )
    _ROUTE_KEYWORDS = {
        "result_explanation": (
            "解释结果",
            "解读结果",
            "解释一下结果",
            "结果说明",
            "result explanation",
            "explain the result",
            "interpret result",
            "interpret the result",
        ),
        "result_analysis": (
            "看看第",
            "第3个事件",
            "第 3 个事件",
            "统计",
            "分布",
            "直方图",
            "scatter",
            "histogram",
            "magnitude distribution",
            "depth distribution",
            "事件筛选",
            "筛选事件",
            "分析结果",
        ),
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
            "初至",
            "初至拾取",
            "到时拾取",
            "first arrival",
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
            "continuous seismic monitoring",
            "连续地震监测",
            "地震监测",
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
        if any(keyword in text for keyword in ("定位结果", "location result", "result of location", "解释结果", "解读结果")):
            return "result_explanation"
        if any(keyword in text for keyword in ("第3个事件", "第 3 个事件", "第三个事件", "magnitude distribution", "depth distribution")):
            return "result_analysis"
        for route, keywords in self._ROUTE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return route
        if any(token in text for token in ("chat", "hello", "hi", "help")):
            return "general_chat"
        return "general_chat"

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
            normalized = source.replace("\\", "/").strip()
            if normalized.startswith("/api/artifacts/"):
                normalized = normalized[len("/api/artifacts/"):]
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if "/data/" in normalized:
                normalized = normalized.split("/data/", 1)[1]
            if normalized.startswith("data/"):
                normalized = normalized[5:]
            normalized = normalized.lstrip("/")
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
