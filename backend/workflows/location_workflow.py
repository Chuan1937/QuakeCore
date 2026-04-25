"""Deterministic location workflow orchestrated outside prompt logic."""

from __future__ import annotations

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
    skip_reason: str | None = None,
) -> NormalizedToolResult | None:
    if fn is None:
        steps.append(
            {
                "name": name,
                "status": "skipped",
                "error": None,
                "message": skip_reason or "",
                "data": {},
                "artifacts": [],
            }
        )
        return None

    try:
        normalized = normalize_tool_output(fn())
    except Exception as exc:  # pragma: no cover - defensive layer
        normalized = normalize_tool_output(exc)

    steps.append(
        {
            "name": name,
            "status": "ok" if normalized.success else "error",
            "error": normalized.error,
            "message": normalized.message,
            "data": normalized.data,
            "artifacts": normalized.artifacts,
        }
    )
    return normalized


def run_location_workflow(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    current_file = session_store.get_current_file(session_id)
    steps: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    location: dict[str, Any] = {}
    summary_parts: list[str] = []

    _record_step(
        steps,
        name="get_loaded_context",
        fn=lambda: _invoke_tool(get_loaded_context, {}),
    )

    load_result = _record_step(
        steps,
        name="load_local_data",
        fn=(lambda: _invoke_tool(load_local_data, {"path": "example_data"})) if not current_file else None,
        skip_reason="Current file exists in session context.",
    )
    if load_result and load_result.success:
        summary_parts.append("Loaded local data fallback from example_data.")

    _record_step(
        steps,
        name="pick_all_miniseed_files",
        fn=lambda: _invoke_tool(pick_all_miniseed_files, {}),
    )

    _record_step(
        steps,
        name="prepare_nearseismic_taup_cache",
        fn=lambda: _invoke_tool(prepare_nearseismic_taup_cache, {}),
    )

    _record_step(
        steps,
        name="add_station_coordinates",
        fn=lambda: _invoke_tool(add_station_coordinates, {}),
    )

    locate_result = _record_step(
        steps,
        name="locate_uploaded_data_nearseismic",
        fn=lambda: _invoke_tool(locate_uploaded_data_nearseismic, {}),
    )
    if locate_result:
        location = locate_result.data if isinstance(locate_result.data, dict) else {}
        artifacts.extend(locate_result.artifacts)

    plot_result = _record_step(
        steps,
        name="plot_location_map",
        fn=lambda: _invoke_tool(plot_location_map, {}),
    )
    if plot_result:
        artifacts.extend(plot_result.artifacts)

    deduped_artifacts: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        url = str(artifact.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_artifacts.append(artifact)

    critical_steps = {"pick_all_miniseed_files", "add_station_coordinates", "locate_uploaded_data_nearseismic"}
    critical_ok = all(
        step.get("status") == "ok"
        for step in steps
        if step.get("name") in critical_steps
    )

    if critical_ok:
        summary_parts.append("Location workflow completed with deterministic step orchestration.")
    else:
        summary_parts.append("Location workflow finished with partial failures; inspect step statuses.")

    return {
        "success": critical_ok,
        "steps": steps,
        "location": location,
        "artifacts": deduped_artifacts,
        "message": " ".join(summary_parts).strip(),
    }
