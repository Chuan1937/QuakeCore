"""Deterministic location workflow orchestrated outside prompt logic."""

from __future__ import annotations

import time
from typing import Any, Callable

from agent.tools_facade import (
    add_station_coordinates,
    get_loaded_context,
    load_local_data,
    locate_uploaded_data_nearseismic,
    pick_all_miniseed_files,
    plot_location_map,
    prepare_nearseismic_taup_cache,
)
from backend.services.session_store import get_session_store
from backend.services.tool_result import NormalizedToolResult, normalize_tool_output

STATUS_SUCCESS = "success"
STATUS_PARTIAL_SUCCESS = "partial_success"
STATUS_FAILED = "failed"

STEP_REQUIRED = {
    "get_loaded_context",
    "pick_all_miniseed_files",
    "add_station_coordinates",
    "locate_uploaded_data_nearseismic",
}


def _invoke_tool(tool_obj: Any, payload: Any = None) -> Any:
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(payload if payload is not None else {})
    if callable(tool_obj):
        if payload is None:
            return tool_obj()
        return tool_obj(payload)
    raise TypeError(f"Unsupported tool object: {tool_obj!r}")


def _record_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    fn: Callable[[], Any] | None,
    required: bool,
    skip_reason: str | None = None,
) -> NormalizedToolResult | None:
    start = time.perf_counter()
    if fn is None:
        duration_ms = int((time.perf_counter() - start) * 1000)
        steps.append(
            {
                "name": name,
                "status": "skipped",
                "required": required,
                "error": None,
                "message": skip_reason or "",
                "data": {},
                "artifacts": [],
                "duration_ms": duration_ms,
            }
        )
        return None

    try:
        normalized = normalize_tool_output(fn())
    except Exception as exc:  # pragma: no cover - defensive layer
        normalized = normalize_tool_output(exc)

    duration_ms = int((time.perf_counter() - start) * 1000)
    step_status = "ok" if normalized.success else ("error" if required else "warning")
    steps.append(
        {
            "name": name,
            "status": step_status,
            "required": required,
            "error": normalized.error,
            "message": normalized.message,
            "data": normalized.data,
            "artifacts": normalized.artifacts,
            "duration_ms": duration_ms,
        }
    )
    return normalized


def _dedupe_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        url = str(artifact.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(artifact)
    return deduped


def _find_step(steps: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for step in steps:
        if step.get("name") == name:
            return step
    return None


def run_location_workflow(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    current_file = session_store.get_current_file(session_id)
    steps: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    location: dict[str, Any] = {}

    _record_step(
        steps,
        name="get_loaded_context",
        fn=lambda: _invoke_tool(get_loaded_context, {}),
        required=True,
    )

    _record_step(
        steps,
        name="load_local_data",
        fn=(lambda: _invoke_tool(load_local_data, {"path": "example_data"})) if not current_file else None,
        required=False,
        skip_reason="Current file exists in session context.",
    )

    _record_step(
        steps,
        name="pick_all_miniseed_files",
        fn=lambda: _invoke_tool(pick_all_miniseed_files, {}),
        required=True,
    )

    _record_step(
        steps,
        name="prepare_nearseismic_taup_cache",
        fn=lambda: _invoke_tool(prepare_nearseismic_taup_cache, {}),
        required=False,
    )

    _record_step(
        steps,
        name="add_station_coordinates",
        fn=lambda: _invoke_tool(add_station_coordinates, {}),
        required=True,
    )

    locate_result = _record_step(
        steps,
        name="locate_uploaded_data_nearseismic",
        fn=lambda: _invoke_tool(locate_uploaded_data_nearseismic, {}),
        required=True,
    )
    if locate_result:
        if isinstance(locate_result.data, dict):
            location = locate_result.data
        artifacts.extend(locate_result.artifacts)

    plot_result = _record_step(
        steps,
        name="plot_location_map",
        fn=lambda: _invoke_tool(plot_location_map, {}),
        required=False,
    )
    if plot_result:
        artifacts.extend(plot_result.artifacts)

    artifacts = _dedupe_artifacts(artifacts)

    required_errors = [
        step for step in steps if step.get("required") and step.get("status") == "error"
    ]
    optional_warnings = [
        step for step in steps if (not step.get("required")) and step.get("status") in {"warning", "error"}
    ]
    locate_step = _find_step(steps, "locate_uploaded_data_nearseismic")
    locate_success = bool(
        locate_step and locate_step.get("status") in {"ok", "warning"} and (location or artifacts)
    )

    status = STATUS_FAILED
    error: str | None = None
    summary_parts: list[str] = []
    if locate_success and not required_errors:
        status = STATUS_SUCCESS if not optional_warnings else STATUS_PARTIAL_SUCCESS
    elif locate_success:
        status = STATUS_PARTIAL_SUCCESS
    else:
        status = STATUS_FAILED
        if locate_step and locate_step.get("error"):
            error = str(locate_step.get("error"))
        elif required_errors:
            error = str(required_errors[0].get("error") or required_errors[0].get("name"))
        else:
            error = "Location workflow failed before producing location data."

    if status == STATUS_SUCCESS:
        summary_parts.append("Location workflow completed successfully.")
    elif status == STATUS_PARTIAL_SUCCESS:
        summary_parts.append("Location workflow completed with warnings.")
    else:
        summary_parts.append("Location workflow failed.")

    if required_errors:
        names = ", ".join(step["name"] for step in required_errors)
        summary_parts.append(f"Required step issues: {names}.")
    if optional_warnings:
        names = ", ".join(step["name"] for step in optional_warnings)
        summary_parts.append(f"Optional step issues: {names}.")

    summary = " ".join(summary_parts).strip()
    message = summary

    return {
        "success": status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS},
        "status": status,
        "message": message,
        "summary": summary,
        "steps": steps,
        "location": location,
        "artifacts": artifacts,
        "error": error,
    }
