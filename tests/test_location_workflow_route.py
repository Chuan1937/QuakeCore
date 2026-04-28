from fastapi.testclient import TestClient

from backend.main import app
from backend.routes.workflows import _extract_continuous_params_from_message


def test_location_workflow_route_returns_structured_payload(monkeypatch):
    monkeypatch.setattr(
        "backend.routes.workflows.run_location_workflow",
        lambda _sid: {
            "success": True,
            "status": "partial_success",
            "message": "workflow done",
            "summary": "workflow done",
            "steps": [{"name": "x", "status": "ok", "duration_ms": 1}],
            "location": {},
            "artifacts": [],
            "error": None,
        },
    )
    client = TestClient(app)

    response = client.post("/api/workflows/location/run", json={"session_id": "sid-route"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sid-route"
    assert payload["status"] == "partial_success"
    assert isinstance(payload["steps"], list)
    assert isinstance(payload["artifacts"], list)


def test_extract_continuous_params_from_message_handles_zh_time_range():
    payload = _extract_continuous_params_from_message("对加州2019年7月4日的17到18点进行地震监测")
    assert payload["region"] == "加州"
    assert payload["start"] == "2019-07-04T17:00:00"
    assert payload["end"] == "2019-07-04T18:00:00"


def test_continuous_workflow_route_prefers_planner_params(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.routes.workflows._plan_continuous_params",
        lambda session_id, message, lang: {
            "region": "加州",
            "start": "2019-07-04T17:00:00",
            "end": "2019-07-04T18:00:00",
        },
    )
    monkeypatch.setattr("backend.routes.workflows.create_continuous_job", lambda job_id, session_id, message: captured.update({"job_id": job_id, "session_id": session_id, "message": message}))

    class _NoopThread:
        def __init__(self, target=None, args=(), daemon=None):
            captured["thread_args"] = args

        def start(self):
            captured["started"] = True

    monkeypatch.setattr("backend.routes.workflows.Thread", _NoopThread)

    client = TestClient(app)
    response = client.post(
        "/api/workflows/continuous/start",
        json={"session_id": "sid-cont", "message": "对加州2019年7月4日的17到18点进行地震监测", "lang": "zh"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sid-cont"
    assert payload["status"] == "running"
    assert captured["started"] is True
    assert captured["thread_args"][2] == {
        "region": "加州",
        "start": "2019-07-04T17:00:00",
        "end": "2019-07-04T18:00:00",
    }
