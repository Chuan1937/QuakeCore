"""Common result container for tool/agent execution."""

from __future__ import annotations

import json
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
        normalized_path = path.replace("\\", "/")
        if normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]
        if normalized_path.startswith("data/"):
            normalized_path = normalized_path[5:]
        path = normalized_path.lstrip("/")

    if url:
        if url.startswith("./"):
            url = url[2:]
        if url.startswith("data/"):
            path = path or url[5:].lstrip("/")
            url = f"/api/artifacts/{url[5:].lstrip('/')}"
        elif not url.startswith(("http://", "https://", "/api/artifacts/")):
            normalized_path = url.replace("\\", "/")
            if normalized_path.startswith("./"):
                normalized_path = normalized_path[2:]
            if normalized_path.startswith("data/"):
                normalized_path = normalized_path[5:]
            normalized_path = normalized_path.lstrip("/")
            path = path or normalized_path
            url = f"/api/artifacts/{normalized_path}"

    if not path and url and url.startswith("/api/artifacts/"):
        path = url[len("/api/artifacts/"):]

    if not name and path:
        name = Path(path).name
    if not name and url:
        name = Path(url).name

    if not url and path:
        url = f"/api/artifacts/{path}"

    if not url:
        return None

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
        return NormalizedToolResult(
            success=True,
            message=output,
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
