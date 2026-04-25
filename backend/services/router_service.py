"""Intent routing and artifact extraction for chat responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactItem:
    type: str
    url: str


class RouterService:
    _IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

    def route_intent(self, message: str) -> str:
        text = str(message or "").lower()
        if "定位" in text or "location" in text or "locate" in text:
            return "earthquake_location"
        if "拾取" in text or "pick" in text or "phase" in text:
            return "phase_picking"
        if "结构" in text or "structure" in text:
            return "file_structure"
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
            artifacts.append(ArtifactItem(type="image", url=f"/api/artifacts/{normalized}"))
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

