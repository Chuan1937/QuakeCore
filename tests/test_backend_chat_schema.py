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
        return ChatResult(
            session_id=session_id or "sid-ok",
            answer=f"echo:{message}\n![img](data/plot.png)",
            error=None,
            route="file_structure",
            artifacts=[ArtifactItem(type="image", url="/api/artifacts/plot.png")],
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
    assert payload["artifacts"] == [{"type": "image", "url": "/api/artifacts/plot.png"}]


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
    assert payload["artifacts"] == []
