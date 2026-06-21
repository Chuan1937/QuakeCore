"""Shared helpers for QuakeCore tools — parameter parsing, type coercion, artifact building."""

from __future__ import annotations

import json
import os
from typing import Any, Union

import numpy as np


# ──────────────────────────────────────────────
# Standardized tool return values
# ──────────────────────────────────────────────


def tool_success(message: str, data: dict | None = None, artifacts: list | None = None) -> str:
    """Return a standardized success JSON string from a tool."""
    return json.dumps(
        {
            "success": True,
            "message": message,
            "data": data or {},
            "artifacts": artifacts or [],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def tool_error(message: str) -> str:
    """Return a standardized error JSON string from a tool."""
    return json.dumps(
        {"success": False, "message": message, "artifacts": []},
        ensure_ascii=False,
        indent=2,
    )


# ──────────────────────────────────────────────
# Parameter parsing
# ──────────────────────────────────────────────


def _resolve_file_path(path: str) -> str:
    """Resolve a file path, trying ``data/`` prefix as fallback."""
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    data_path = os.path.join("data", path)
    if os.path.exists(data_path):
        return data_path
    cwd_data_path = os.path.join(os.getcwd(), "data", path)
    if os.path.exists(cwd_data_path):
        return cwd_data_path
    return path


def parse_param_dict(raw_params: Any) -> dict:
    """Normalize incoming tool arguments to a dict.

    Accepts ``None``, ``dict``, JSON string, or comma-separated key=value pairs.
    Automatically resolves the ``path`` key via :func:`_resolve_file_path`.
    """
    if raw_params is None:
        return {}
    if isinstance(raw_params, dict):
        result = dict(raw_params)
    elif isinstance(raw_params, str):
        candidate = raw_params.strip()
        if not candidate:
            return {}
        try:
            result = json.loads(candidate)
            if not isinstance(result, dict):
                result = {}
        except json.JSONDecodeError:
            params: dict[str, str] = {}
            for chunk in candidate.split(","):
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                    params[key.strip()] = value.strip()
            result = params
    else:
        return {}

    if "path" in result and result["path"]:
        result["path"] = _resolve_file_path(result["path"])
    return result


# ──────────────────────────────────────────────
# Type coercion
# ──────────────────────────────────────────────


def coerce_int(value: Any, *, allow_none: bool = False, default: int | None = None, field_name: str = "value") -> int | None:
    """Coerce *value* to ``int``.

    Handles ``int``, ``float`` (e.g. ``3.0``), and string representations
    (``"3"``, ``"3.0"``).  Returns *default* when *value* is ``None`` and
    *allow_none* is ``True``.
    """
    if value is None:
        if allow_none:
            return None
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if allow_none and lowered in {"none", "null", ""}:
            return None
        try:
            return int(lowered)
        except ValueError:
            return int(float(lowered))
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def coerce_float(value: Any, *, allow_none: bool = False, default: float | None = None, field_name: str = "value") -> float | None:
    """Coerce *value* to ``float``."""
    if value is None:
        if allow_none:
            return default
        if default is not None:
            return float(default)
        raise ValueError(f"{field_name} must be provided")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip()
        if allow_none and lowered.lower() in {"none", "null", ""}:
            return default
        return float(lowered)
    raise ValueError(f"{field_name} must be a float, got {value!r}")


# ──────────────────────────────────────────────
# Artifact helpers
# ──────────────────────────────────────────────

DEFAULT_CONVERT_DIR = "data/convert"
DEFAULT_STRUCTURE_DIR = "data/structure"
DEFAULT_PICKS_DIR = "data/picks"
DEFAULT_LOCATION_DIR = "data/location"


def resolve_output_path(output_path: str | None, *, default_filename: str, base_dir: str | None = None) -> str:
    """Resolve output file path and ensure parent directory exists."""
    if base_dir is None:
        base_dir = DEFAULT_CONVERT_DIR

    if not output_path or not str(output_path).strip():
        final_path = os.path.join(base_dir, default_filename)
    else:
        output_path = str(output_path).strip()
        if os.path.dirname(output_path) == "":
            final_path = os.path.join(base_dir, output_path)
        else:
            final_path = output_path

    parent = os.path.dirname(final_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return final_path


def to_relative_artifact_path(path: str) -> str:
    """Convert an absolute or data-prefixed path to a ``data/``-relative URL path."""
    rel = str(path).replace("\\", "/")
    if "/data/" in rel:
        rel = rel.split("/data/", 1)[1]
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("data/"):
        rel = rel[5:]
    return rel.lstrip("/")


def build_artifact_entry(path: str, artifact_type: str = "file") -> dict[str, str] | None:
    """Build a single artifact dict suitable for ``artifacts`` arrays."""
    rel = to_relative_artifact_path(path)
    if not rel:
        return None
    return {
        "type": artifact_type,
        "name": os.path.basename(path),
        "path": rel,
        "url": f"/api/artifacts/{rel}",
    }


def build_artifact_response(
    result: dict,
    message: str | None = None,
    output_path: str | None = None,
    input_path: str | None = None,
) -> str:
    """Wrap a result dict into the standard ``{success, message, artifacts, data}`` JSON format."""
    result = dict(result)
    plot_path = result.pop("plot_path", None)
    saved_to = result.pop("saved_to", None) or output_path

    if not message:
        input_basename = os.path.basename(input_path) if input_path else ""
        output_basename = os.path.basename(saved_to) if saved_to else ""
        message = f'文件 "{input_basename}" 已成功转换为 {output_basename}'
        if "trace_count" in result:
            message += f'，共包含 {result["trace_count"]} 条迹'

    response: dict[str, Any] = {"success": True, "message": message}
    artifacts: list[dict[str, str]] = []

    if saved_to:
        entry = build_artifact_entry(saved_to, "file")
        if entry:
            artifacts.append(entry)

    if plot_path:
        entry = build_artifact_entry(plot_path, "image")
        if entry:
            artifacts.append(entry)

    response["artifacts"] = artifacts
    if result:
        response["data"] = result

    return json.dumps(response, indent=2, ensure_ascii=False, default=str)
