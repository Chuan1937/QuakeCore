"""In-memory job store for continuous monitoring workflows."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = Lock()

_STAGE_BASE_PERCENT = {
    "queued": 2,
    "download": 8,
    "picking": 45,
    "association": 65,
    "location": 78,
    "plot": 90,
    "complete": 100,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_continuous_job(job_id: str, *, session_id: str | None = None, message: str = "") -> None:
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "message": message,
            "status": "running",
            "step": "Preparing task",
            "percent": _STAGE_BASE_PERCENT["queued"],
            "logs": [],
            "result": None,
            "error": None,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }


def update_continuous_job_progress(job_id: str, item: dict[str, Any]) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return

        stage = str(item.get("stage") or "download")
        message = str(item.get("message") or "").strip() or job.get("step", "Processing")
        downloaded = item.get("downloaded")
        total = item.get("total")
        base = _STAGE_BASE_PERCENT.get(stage, 10)
        percent = float(job.get("percent", base))

        if stage == "download" and isinstance(downloaded, (int, float)) and isinstance(total, (int, float)) and total:
            ratio = max(0.0, min(1.0, float(downloaded) / float(total)))
            percent = max(percent, base + ratio * 30.0)
        elif stage in {"picking", "association", "location", "plot"}:
            percent = max(percent, float(base))

        if job.get("status") == "running":
            job["step"] = message
            job["percent"] = max(0.0, min(99.0, percent))
            job["updated_at"] = _utc_now_iso()

        logs = job.setdefault("logs", [])
        logs.append(
            {
                "stage": stage,
                "message": message,
                "downloaded": downloaded,
                "failed": item.get("failed"),
                "total": total,
                "timestamp": _utc_now_iso(),
            }
        )
        if len(logs) > 300:
            del logs[:-300]


def finish_continuous_job(job_id: str, result: dict[str, Any], *, failed: bool = False) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return

        job["result"] = result
        job["updated_at"] = _utc_now_iso()
        if failed:
            job["status"] = "failed"
            job["error"] = str(result.get("error") or result.get("message") or "Workflow failed")
            job["step"] = "Failed"
            job["percent"] = float(job.get("percent", 0.0))
        else:
            job["status"] = "completed"
            job["error"] = None
            job["step"] = "Completed"
            job["percent"] = 100.0


def fail_continuous_job(job_id: str, error: str) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "failed"
        job["error"] = error
        job["step"] = "Failed"
        job["updated_at"] = _utc_now_iso()


def get_continuous_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return deepcopy(job) if job is not None else None

