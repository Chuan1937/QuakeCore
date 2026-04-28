from fastapi.testclient import TestClient

from backend.main import app
from backend.services.agent_service import ChatResult
from backend.services.router_service import ArtifactItem


class FakeAgentService:
    def chat(self, message: str, session_id: str | None = None, lang: str | None = "en") -> ChatResult:
        if message == "boom":
            return ChatResult(
                session_id=session_id or "sid-err",
                answer="",
                error="mock failure",
                route="phase_picking",
                artifacts=[],
            )
        if message == "locate":
            return ChatResult(
                session_id=session_id or "sid-loc",
                answer="Location workflow completed with warnings.",
                error=None,
                route="earthquake_location",
                artifacts=[],
                workflow={
                    "status": "partial_success",
                    "summary": "Location workflow completed with warnings.",
                    "message": "Location workflow completed with warnings.",
                    "steps": [
                        {
                            "name": "locate_uploaded_data_nearseismic",
                            "status": "ok",
                            "required": True,
                            "message": "located",
                            "error": None,
                            "data": {"latitude": 10.0, "longitude": 20.0},
                            "artifacts": [],
                            "duration_ms": 2,
                        }
                    ],
                    "location": {"latitude": 10.0, "longitude": 20.0},
                    "artifacts": [],
                    "error": None,
                },
            )
        if message == "locate_failed":
            return ChatResult(
                session_id=session_id or "sid-loc-fail",
                answer="地震定位工作流执行失败，但已完成部分步骤。请查看步骤详情。",
                error="location failed",
                route="earthquake_location",
                artifacts=[],
                workflow={
                    "status": "failed",
                    "summary": "workflow failed",
                    "message": "workflow failed",
                    "steps": [
                        {
                            "name": "locate_uploaded_data_nearseismic",
                            "status": "error",
                            "required": True,
                            "message": "failed",
                            "error": "location failed",
                            "data": {},
                            "artifacts": [],
                            "duration_ms": 3,
                        }
                    ],
                    "location": {},
                    "artifacts": [],
                    "error": "location failed",
                },
            )
        return ChatResult(
            session_id=session_id or "sid-ok",
            answer=f"echo:{message}\n![img](data/plot.png)",
            error=None,
            route="file_structure",
            artifacts=[
                ArtifactItem(
                    type="image",
                    name="plot.png",
                    path="plot.png",
                    url="/api/artifacts/plot.png",
                )
            ],
        )


def test_chat_endpoint_returns_schema(monkeypatch):
    monkeypatch.setattr("backend.routes.chat._agent_service", FakeAgentService())
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "session_id": "s1", "lang": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "s1"
    assert payload["error"] is None
    assert payload["route"] == "file_structure"
    assert payload["workflow"] is None
    assert payload["artifacts"] == [
        {
            "type": "image",
            "name": "plot.png",
            "path": "plot.png",
            "url": "/api/artifacts/plot.png",
        }
    ]


def test_chat_endpoint_error_field_when_agent_fails(monkeypatch):
    monkeypatch.setattr("backend.routes.chat._agent_service", FakeAgentService())
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "boom", "lang": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sid-err"
    assert payload["answer"] == ""
    assert payload["error"] == "mock failure"
    assert payload["route"] == "phase_picking"
    assert payload["workflow"] is None
    assert payload["artifacts"] == []


def test_chat_endpoint_returns_workflow_payload_for_location_route(monkeypatch):
    monkeypatch.setattr("backend.routes.chat._agent_service", FakeAgentService())
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "locate", "lang": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sid-loc"
    assert payload["route"] == "earthquake_location"
    assert payload["workflow"]["status"] == "partial_success"
    assert isinstance(payload["workflow"]["steps"], list)


def test_chat_endpoint_keeps_http_200_when_workflow_failed(monkeypatch):
    monkeypatch.setattr("backend.routes.chat._agent_service", FakeAgentService())
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "locate_failed", "lang": "zh"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sid-loc-fail"
    assert payload["route"] == "earthquake_location"
    assert payload["error"] == "location failed"
    assert payload["workflow"]["status"] == "failed"
    assert isinstance(payload["workflow"]["steps"], list)
