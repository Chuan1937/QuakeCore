from fastapi.testclient import TestClient

from backend.main import app


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
