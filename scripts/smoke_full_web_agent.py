"""Full smoke test for health + upload + chat + location workflow visibility."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"
STRUCTURE_MESSAGE = "请分析当前文件结构"
LOCATION_MESSAGE = "使用当前数据进行地震定位"


def _find_sample_file() -> Path | None:
    candidates: list[Path] = []
    for root in ("example_data", "data"):
        root_path = Path(root)
        if not root_path.exists():
            continue
        for suffix in (".mseed", ".miniseed", ".sac", ".segy", ".sgy", ".h5", ".hdf5"):
            candidates.extend(root_path.rglob(f"*{suffix}"))
    if not candidates:
        return None
    return min(candidates, key=lambda path: path.stat().st_size)


def _is_json(response: Any) -> bool:
    try:
        response.json()
    except Exception:
        return False
    return True


def _validate_location_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload.get("workflow"), dict):
        return False, "location response missing workflow"
    workflow = payload["workflow"]
    status = workflow.get("status")
    if status not in {"success", "partial_success", "failed"}:
        return False, f"invalid workflow status={status!r}"
    steps = workflow.get("steps")
    if not isinstance(steps, list):
        return False, "workflow steps must be list"
    return True, ""


def _run_with_client(client: Any) -> int:
    def _get(path: str):
        if hasattr(client, "base_url"):
            return client.get(path)
        return client.get(f"{BASE_URL}{path}", timeout=30)

    def _post(path: str, **kwargs: Any):
        if hasattr(client, "base_url"):
            return client.post(path, **kwargs)
        return client.post(f"{BASE_URL}{path}", timeout=60, **kwargs)

    health_response = _get("/health")
    if health_response.status_code != 200:
        print("[FAIL] health")
        return 1
    print("[OK] health")

    session_id: str | None = None
    sample_file = _find_sample_file()
    if sample_file is not None:
        with sample_file.open("rb") as handle:
            upload_response = _post(
                "/api/files/upload",
                files={"file": (sample_file.name, handle, "application/octet-stream")},
            )
        if upload_response.status_code != 200:
            print("[FAIL] upload file")
            return 1
        upload_payload = upload_response.json()
        session_id = upload_payload.get("session_id")
        print(f"[OK] upload file: {sample_file.name}")
    else:
        print("[SKIP] upload file: no sample file found under example_data/ or data/")

    structure_response = _post(
        "/api/chat",
        json={"message": STRUCTURE_MESSAGE, "session_id": session_id, "lang": "zh"},
    )
    if structure_response.status_code != 200:
        print("[FAIL] file_structure request status")
        return 1
    if not _is_json(structure_response):
        print("[FAIL] file_structure response not json")
        return 1
    structure_payload = structure_response.json()
    if not structure_payload.get("session_id") or not structure_payload.get("route"):
        print("[FAIL] file_structure response missing session_id/route")
        return 1
    if not isinstance(structure_payload.get("artifacts"), list):
        print("[FAIL] file_structure artifacts is not list")
        return 1
    if os.getenv("DEEPSEEK_API_KEY"):
        answer = str(structure_payload.get("answer", "") or "").strip()
        if not answer:
            print("[FAIL] file_structure empty answer with DEEPSEEK_API_KEY")
            return 1
    print(f"[OK] file_structure route={structure_payload.get('route')}")

    location_response = _post(
        "/api/chat",
        json={"message": LOCATION_MESSAGE, "session_id": structure_payload.get("session_id"), "lang": "zh"},
    )
    if location_response.status_code != 200:
        print("[FAIL] location request status")
        return 1
    if not _is_json(location_response):
        print("[FAIL] location response not json")
        return 1
    location_payload = location_response.json()
    if not location_payload.get("session_id") or not location_payload.get("route"):
        print("[FAIL] location response missing session_id/route")
        return 1
    if not isinstance(location_payload.get("artifacts"), list):
        print("[FAIL] location artifacts is not list")
        return 1
    ok, reason = _validate_location_payload(location_payload)
    if not ok:
        print(f"[FAIL] {reason}")
        return 1
    workflow = location_payload["workflow"]
    print(
        f"[OK] location route={location_payload.get('route')} "
        f"workflow_status={workflow.get('status')} steps={len(workflow.get('steps', []))}"
    )
    print(f"[OK] artifacts={len(location_payload.get('artifacts', []))}")
    print("[DONE] full web agent smoke passed")
    return 0


def main() -> int:
    if os.getenv("QUAKECORE_SMOKE_INPROCESS") == "1":
        try:
            from fastapi.testclient import TestClient

            from backend.main import app
        except Exception as exc:  # pragma: no cover - smoke fallback
            print(f"In-process smoke setup failed: {exc}", file=sys.stderr)
            return 1
        return _run_with_client(TestClient(app))

    try:
        return _run_with_client(requests.Session())
    except requests.RequestException:
        print(
            "Backend not reachable. Start it first with:\n"
            "uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
