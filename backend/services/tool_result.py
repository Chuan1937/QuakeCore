"""Common result container for tool/agent execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    raw: Any = None

    @classmethod
    def from_response(cls, response: Any) -> "ToolResult":
        if isinstance(response, dict):
            return cls(ok=True, output=str(response.get("output", "")), raw=response)
        return cls(ok=True, output=str(response), raw=response)

    @classmethod
    def from_error(cls, error: Exception, raw: Any = None) -> "ToolResult":
        return cls(ok=False, output="", error=str(error), raw=raw)


@dataclass
class NormalizedToolResult:
    success: bool = True
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None


def _extract_markdown_artifacts(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    from backend.services.router_service import RouterService

    router = RouterService()
    return [
        {"type": item.type, "name": item.name, "path": item.path, "url": item.url}
        for item in router.extract_artifacts(text)
    ]


_IMAGE_LINE_RE = re.compile(
    r"^\s*.*?(波形图如下所示|结果图如下所示|图如下所示).*?$|^\s*!\[[^\]]*]\([^)]+\)\s*$",
    re.M,
)


def _strip_artifact_markdown(message: str) -> str:
    if not message:
        return message
    cleaned = _IMAGE_LINE_RE.sub("", message)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _to_data_relative_path(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("data/"):
        normalized = normalized[5:]
    return normalized.lstrip("/")


def _normalize_artifact_payload(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        stripped = item.strip()
        if not stripped:
            return None
        item = {"type": "image", "path": stripped}

    if not isinstance(item, dict):
        return None

    artifact_type = str(item.get("type", "image"))
    path = str(item.get("path", "")).strip() or None
    name = str(item.get("name", "")).strip() or None
    url = str(item.get("url", "")).strip() or None

    if path:
        path = _to_data_relative_path(path)

    if url:
        if url.startswith("/api/artifacts/"):
            path = path or url[len("/api/artifacts/"):]
        elif not url.startswith(("http://", "https://")):
            path = path or _to_data_relative_path(url)
            url = f"/api/artifacts/{path}" if path else url

    if not url and path:
        url = f"/api/artifacts/{path}"

    if not url:
        return None

    if not name and path:
        name = Path(path).name
    if not name and url:
        name = Path(url).name

    return {
        "type": artifact_type,
        "name": name or "",
        "path": path or name or "",
        "url": url,
    }


def _normalize_artifacts(raw_artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_artifacts, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_artifacts:
        parsed = _normalize_artifact_payload(item)
        if parsed is not None:
            normalized.append(parsed)
    return normalized


def _normalize_from_dict(payload: dict[str, Any], raw: str | None = None) -> NormalizedToolResult:
    reserved_keys = {"success", "message", "data", "artifacts", "raw", "error", "output"}

    message_value = payload.get("message")
    if message_value is None:
        message_value = payload.get("output", "")
    message = str(message_value or "")

    data_value = payload.get("data")
    if isinstance(data_value, dict):
        data = dict(data_value)
    else:
        data = {}

    if not data:
        data = {key: value for key, value in payload.items() if key not in reserved_keys}

    artifacts = _normalize_artifacts(payload.get("artifacts"))
    if not artifacts:
        artifacts = _extract_markdown_artifacts(message)

    if artifacts:
        message = _strip_artifact_markdown(message)

    error_value = payload.get("error")
    error = str(error_value) if error_value not in (None, "") else None
    success_value = payload.get("success")
    if success_value is None:
        success = error is None
    else:
        success = bool(success_value)

    normalized_raw = payload.get("raw")
    if isinstance(normalized_raw, str):
        raw = normalized_raw

    return NormalizedToolResult(
        success=success,
        message=message,
        data=data,
        artifacts=artifacts,
        raw=raw,
        error=error,
    )


def normalize_tool_output(output: Any) -> NormalizedToolResult:
    if isinstance(output, NormalizedToolResult):
        return output

    if isinstance(output, ToolResult):
        seed = output.raw if output.raw is not None else output.output
        normalized = normalize_tool_output(seed)
        if not normalized.message and output.output:
            normalized.message = str(output.output)
        if output.error:
            normalized.success = False
            normalized.error = output.error
        if normalized.raw is None and isinstance(output.output, str) and output.output:
            normalized.raw = output.output
        return normalized

    if isinstance(output, Exception):
        return NormalizedToolResult(
            success=False,
            message="",
            data={},
            artifacts=[],
            raw=None,
            error=str(output),
        )

    if output is None:
        return NormalizedToolResult(success=True, message="", data={}, artifacts=[], raw=None, error=None)

    if isinstance(output, dict):
        return _normalize_from_dict(output)

    if isinstance(output, str):
        stripped = output.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                normalized = _normalize_from_dict(parsed, raw=output)
                if normalized.raw is None:
                    normalized.raw = output
                return normalized

        artifacts = _extract_markdown_artifacts(output)
        message = _strip_artifact_markdown(output) if artifacts else output
        return NormalizedToolResult(
            success=True,
            message=message,
            data={},
            artifacts=artifacts,
            raw=output,
            error=None,
        )

    return NormalizedToolResult(
        success=True,
        message=str(output),
        data={},
        artifacts=[],
        raw=str(output),
        error=None,
    )
