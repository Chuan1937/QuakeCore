from backend.services.agent_service import AgentService
from backend.services.session_store import AgentSession, SessionStore


class _DummyAgent:
    def invoke(self, _payload):
        return {"output": "ok"}


def test_session_file_context_does_not_leak_between_sessions(monkeypatch):
    store = SessionStore()
    store.add_file("session-a", "/tmp/a.mseed")
    store.set_current_file("session-a", "/tmp/a.mseed")
    store.add_file("session-b", "/tmp/b.segy")
    store.set_current_file("session-b", "/tmp/b.segy")

    service = AgentService(session_store=store)

    def _fake_build_agent_session(session_id: str, lang: str) -> AgentSession:
        return AgentSession(session_id=session_id, lang=lang, agent=_DummyAgent())

    monkeypatch.setattr(service, "_build_agent_session", _fake_build_agent_session)

    injected: list[tuple[str, str]] = []

    def _fake_bind(path: str, file_type: str) -> bool:
        injected.append((path, file_type))
        return True

    monkeypatch.setattr("backend.services.agent_service.bind_uploaded_file_to_agent", _fake_bind)

    result_a = service.chat("hello", session_id="session-a", lang="en")
    result_b = service.chat("hello", session_id="session-b", lang="en")

    assert result_a.error is None
    assert result_b.error is None
    assert injected == [
        ("/tmp/a.mseed", "miniseed"),
        ("/tmp/b.segy", "segy"),
    ]
