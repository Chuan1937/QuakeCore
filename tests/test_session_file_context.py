import json

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


def test_agent_service_returns_structured_fallback_on_react_parse_error(monkeypatch):
    store = SessionStore()
    service = AgentService(session_store=store)

    class _BrokenAgent:
        def invoke(self, _payload):
            raise RuntimeError("Invalid Format: Missing 'Action:' after 'Thought:'")

    def _fake_build_agent_session(session_id: str, lang: str) -> AgentSession:
        return AgentSession(session_id=session_id, lang=lang, agent=_BrokenAgent())

    monkeypatch.setattr(service, "_build_agent_session", _fake_build_agent_session)

    result = service.chat("对当前波形做初至拾取", session_id="sid-react", lang="zh")

    assert result.route == "phase_picking"
    assert result.answer == "工具执行已完成，但 Agent 输出格式异常。请查看已生成结果或重试。"
    assert "Invalid Format" in (result.error or "")


def test_session_store_runtime_results_roundtrip():
    store = SessionStore()
    store.set_runtime_result("sid", "last_catalog_csv", "location/a.csv")
    store.update_runtime_results("sid", {"last_catalog_json": "location/a.json"})

    payload = store.get_runtime_results("sid")
    assert payload["last_catalog_csv"] == "location/a.csv"
    assert payload["last_catalog_json"] == "location/a.json"


def test_agent_service_injects_runtime_context_and_updates_runtime(monkeypatch):
    store = SessionStore()
    store.update_runtime_results("sid", {"last_catalog_csv": "location/old.csv"})
    captured: dict[str, str] = {}

    class _CaptureAgent:
        def invoke(self, payload):
            captured["input"] = str(payload.get("input", ""))
            return {
                "output": json.dumps(
                    {
                        "success": True,
                        "message": "ok",
                        "data": {"catalog_csv": "data/location/new.csv"},
                        "artifacts": [
                            {
                                "type": "file",
                                "name": "new.csv",
                                "path": "location/new.csv",
                                "url": "/api/artifacts/location/new.csv",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }

    service = AgentService(session_store=store)

    def _fake_build_agent_session(session_id: str, lang: str) -> AgentSession:
        return AgentSession(session_id=session_id, lang=lang, agent=_CaptureAgent())

    monkeypatch.setattr(service, "_build_agent_session", _fake_build_agent_session)
    result = service.chat("hello", session_id="sid", lang="zh")

    assert result.error is None
    assert "当前会话已有结果上下文" in captured["input"]
    runtime = store.get_runtime_results("sid")
    assert runtime["last_catalog_csv"] == "location/new.csv"


def test_agent_service_trace_pick_fast_path_skips_react(monkeypatch):
    store = SessionStore()
    service = AgentService(session_store=store)

    class _ShouldNotRunAgent:
        def invoke(self, _payload):
            raise AssertionError("ReAct agent should not be invoked for trace-pick fast path")

    def _fake_build_agent_session(session_id: str, lang: str) -> AgentSession:
        return AgentSession(session_id=session_id, lang=lang, agent=_ShouldNotRunAgent())

    monkeypatch.setattr(service, "_build_agent_session", _fake_build_agent_session)
    monkeypatch.setattr(
        "backend.services.agent_service.pick_first_arrivals",
        type(
            "_DummyTool",
            (),
            {
                "invoke": staticmethod(
                    lambda payload: {
                        "success": True,
                        "message": "初至拾取已完成。\\n图表已保存至：`data/picks/demo.png`",
                        "artifacts": [
                            {
                                "type": "image",
                                "name": "demo.png",
                                "path": "picks/demo.png",
                                "url": "/api/artifacts/picks/demo.png",
                            }
                        ],
                        "data": {"plot_path": "picks/demo.png"},
                    }
                )
            },
        ),
    )

    result = service.chat("查看 trace 1 的拾取结果", session_id="sid-fast", lang="zh")
    assert result.error is None
    assert result.route == "phase_picking"
    assert result.artifacts and result.artifacts[0].url == "/api/artifacts/picks/demo.png"


def test_agent_service_file_structure_fast_path(monkeypatch):
    store = SessionStore()
    service = AgentService(session_store=store)

    class _ShouldNotRunAgent:
        def invoke(self, _payload):
            raise AssertionError("ReAct agent should not be invoked for file_structure fast path")

    def _fake_build_agent_session(session_id: str, lang: str) -> AgentSession:
        return AgentSession(session_id=session_id, lang=lang, agent=_ShouldNotRunAgent())

    monkeypatch.setattr(service, "_build_agent_session", _fake_build_agent_session)
    monkeypatch.setattr(
        "backend.services.agent_service.get_file_structure",
        type("_DummyTool", (), {"invoke": staticmethod(lambda _payload: "{\"type\":\"miniseed\"}")}),
    )

    result = service.chat("读取当前文件结构", session_id="sid-struct", lang="zh")
    assert result.error is None
    assert result.route == "file_structure"
    assert "miniseed" in result.answer
