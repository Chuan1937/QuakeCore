"""Workflow API routes."""

from __future__ import annotations

import json
import re
from threading import Thread
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from backend.schemas import LocationWorkflowRunRequest
from backend.services.session_store import get_session_store
from backend.services.tool_planner import ToolPlanner
from backend.workflows.continuous_jobs import (
    create_continuous_job,
    fail_continuous_job,
    finish_continuous_job,
    get_continuous_job,
)
from backend.workflows.location_workflow import run_location_workflow
from quakecore_tools.monitoring_tools import run_continuous_monitoring

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.post("/location/run")
def run_location(payload: LocationWorkflowRunRequest):
    session_id = payload.session_id or uuid4().hex
    result = run_location_workflow(session_id)
    return {"session_id": session_id, **result}


def _extract_continuous_params_from_message(message: str) -> dict:
    text = str(message or "").strip()
    if not text:
        return {}

    # 1) JSON object payload
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    params: dict[str, str | int | float] = {}

    # 2) key=value pairs split by comma/space
    for token in re.split(r"[,\n]+", text):
        chunk = token.strip()
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            params[key] = value

    # 3) common Chinese shortcuts
    if "南加州" in text and "region" not in params:
        params["region"] = "南加州"
    elif "北加州" in text and "region" not in params:
        params["region"] = "北加州"
    elif "加州" in text and "region" not in params:
        params["region"] = "加州"

    # date / hours helpers
    if "date" not in params:
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if m:
            params["date"] = m.group(1)
    if "hours" not in params and "duration_hours" not in params:
        m = re.search(r"(\d+)\s*(?:小时|h|hours?)", text, flags=re.IGNORECASE)
        if m:
            params["hours"] = int(m.group(1))

    # Chinese datetime range, e.g.:
    # "2019年7月4日17点到18点" / "2019年07月04日 17时-18时"
    # -> start/end ISO strings for continuous workflow.
    zh_range = re.search(
        r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日?"
        r"(?:的)?\s*(?P<h1>\d{1,2})\s*(?:点|时)?"
        r"(?:\s*(?:到|至|\-|~|—|–)\s*(?P<h2>\d{1,2})\s*(?:点|时)?)?",
        text,
    )
    if zh_range:
        year = int(zh_range.group("y"))
        month = int(zh_range.group("m"))
        day = int(zh_range.group("d"))
        start_hour = int(zh_range.group("h1"))
        end_hour_raw = zh_range.group("h2")
        end_hour = int(end_hour_raw) if end_hour_raw is not None else (start_hour + 1)
        # Clamp into valid hour range with simple rollover for next-day edge case.
        start_hour = max(0, min(23, start_hour))
        if end_hour <= start_hour:
            end_hour = start_hour + 1
        if end_hour <= 24:
            start_str = f"{year:04d}-{month:02d}-{day:02d}T{start_hour:02d}:00:00"
            if end_hour == 24:
                # next day 00:00
                end_str = f"{year:04d}-{month:02d}-{day:02d}T23:59:59"
            else:
                end_str = f"{year:04d}-{month:02d}-{day:02d}T{end_hour:02d}:00:00"
            params["start"] = start_str
            params["end"] = end_str
            params.setdefault("date", f"{year:04d}-{month:02d}-{day:02d}")

    return params


def _plan_continuous_params(session_id: str, message: str, lang: str) -> dict:
    planner = ToolPlanner()
    store = get_session_store()
    plan = planner.plan(
        message=message,
        route="continuous_monitoring",
        runtime_results=store.get_runtime_results(session_id),
        uploaded_files=store.get_uploaded_files(session_id),
        current_file=store.get_current_file(session_id),
        lang=lang,
    )
    return dict(plan.params or {})


def _normalize_continuous_result(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"success": False, "message": raw, "error": raw}
    return {"success": False, "message": str(raw), "error": str(raw)}


def _to_runtime_path(value: str | None) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/api/artifacts/"):
        normalized = normalized[len("/api/artifacts/"):]
    if normalized.startswith("data/"):
        normalized = normalized[5:]
    return normalized.lstrip("/")


def _persist_continuous_runtime_results(session_id: str, result: dict) -> None:
    if not session_id or not isinstance(result, dict):
        return
    payload = result.get("data")
    if not isinstance(payload, dict):
        payload = {}
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []

    updates: dict[str, object] = {
        "last_route": "continuous_monitoring",
        "last_continuous_monitoring": payload,
    }

    for key, target in (
        ("catalog_csv", "last_catalog_csv"),
        ("catalog_json", "last_catalog_json"),
        ("location_map", "last_location_image"),
        ("location_3view", "last_location_image"),
        ("catalog_3view", "last_location_image"),
    ):
        path = _to_runtime_path(payload.get(key))
        if path:
            updates[target] = path

    runtime_artifacts = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = _to_runtime_path(item.get("path") or item.get("url"))
        url = str(item.get("url", "") or "").strip()
        if not url:
            continue
        runtime_artifacts.append(
            {
                "type": str(item.get("type", "file") or "file"),
                "name": str(item.get("name", "") or ""),
                "path": path,
                "url": url,
            }
        )
    if runtime_artifacts:
        updates["last_artifacts"] = runtime_artifacts

    if not updates:
        return
    get_session_store().update_runtime_results(session_id, updates)


def _run_continuous_job(job_id: str, session_id: str, params: dict, lang: str | None) -> None:
    try:
        from agent.tools import set_current_lang

        set_current_lang(lang or "zh")
        payload = dict(params or {})
        payload["job_id"] = job_id
        raw_result = run_continuous_monitoring.invoke({"params": payload})
        result = _normalize_continuous_result(raw_result)
        if (
            isinstance(result, dict)
            and "start/end or date+hours are required" in str(result.get("error") or result.get("message") or "")
            and payload.get("start")
            and payload.get("end")
        ):
            # Defensive retry: call tool with stringified params if wrapper parsing mismatches.
            retry_raw = run_continuous_monitoring.invoke({"params": json.dumps(payload, ensure_ascii=False)})
            result = _normalize_continuous_result(retry_raw)
        _persist_continuous_runtime_results(session_id, result)
        failed = bool(result.get("error")) or result.get("success") is False
        finish_continuous_job(job_id, result, failed=failed)
    except Exception as exc:
        fail_continuous_job(job_id, str(exc))


@router.post("/continuous/start")
def start_continuous_workflow(payload: dict):
    session_id = payload.get("session_id") or uuid4().hex
    message = str(payload.get("message") or "")
    lang = payload.get("lang") or "zh"
    request_params = payload.get("params")
    if isinstance(request_params, dict):
        params = dict(request_params)
    else:
        extracted = _extract_continuous_params_from_message(message)
        planned = _plan_continuous_params(session_id, message, lang)
        params = {**extracted, **planned}
        if planned.get("start") and planned.get("end"):
            params.pop("date", None)
            params.pop("hours", None)
            params.pop("duration_hours", None)
    job_id = uuid4().hex

    create_continuous_job(job_id, session_id=session_id, message=message)
    worker = Thread(target=_run_continuous_job, args=(job_id, session_id, params, lang), daemon=True)
    worker.start()
    return {"job_id": job_id, "session_id": session_id, "status": "running"}


@router.get("/continuous/{job_id}/progress")
def get_continuous_progress(job_id: str):
    job = get_continuous_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "session_id": job.get("session_id"),
        "status": job.get("status"),
        "step": job.get("step"),
        "percent": job.get("percent", 0),
        "logs": job.get("logs", []),
        "error": job.get("error"),
    }


@router.get("/continuous/{job_id}/result")
def get_continuous_result(job_id: str):
    job = get_continuous_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") == "running":
        return {
            "job_id": job_id,
            "status": "running",
            "step": job.get("step"),
            "percent": job.get("percent", 0),
        }
    result = job.get("result") or {}
    return {
        "job_id": job_id,
        "session_id": job.get("session_id"),
        "status": job.get("status"),
        "result": result,
        "error": job.get("error"),
    }
