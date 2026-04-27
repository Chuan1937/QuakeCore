import json

from backend.services.langgraph_runtime import LangGraphRuntime
from backend.services.tool_result import normalize_tool_output


class _DummyAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"output": "fallback-ok"}


def test_langgraph_runtime_fallback_for_non_analysis_route():
    runtime = LangGraphRuntime(enabled=True)
    agent = _DummyAgent()

    result = runtime.invoke(
        session_id="sid",
        message="hello",
        lang="en",
        fallback_agent=agent,
    )
    normalized = normalize_tool_output(result)

    assert normalized.message == "fallback-ok"
    assert agent.calls and agent.calls[0]["input"] == "hello"


def test_langgraph_runtime_result_analysis_runs_sandbox(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {
                    "success": True,
                    "message": "analysis-ok",
                    "data": {"template": "catalog_event_index"},
                    "artifacts": [],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())

    message = (
        "看看第3个事件\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-1", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "analysis-ok"
    params = captured["payload"]["params"]
    assert params["template"] == "catalog_event_index"
    assert params["event_index"] == 3
    assert params["input_artifact_key"] == "last_catalog_csv"


def test_langgraph_runtime_result_explanation_returns_runtime_summary():
    runtime = LangGraphRuntime(enabled=True)
    message = (
        "解释一下结果\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps(
            {
                "last_continuous_monitoring": {
                    "n_events_detected": 4,
                    "n_picks": 120,
                    "n_stations": 15,
                }
            },
            ensure_ascii=False,
        )
    )

    result = runtime.invoke(session_id="sid-2", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert "识别事件 4 个" in normalized.message


def test_langgraph_runtime_result_analysis_falls_back_to_code(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {
                    "success": True,
                    "message": "code-fallback-ok",
                    "data": {},
                    "artifacts": [],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(
        runtime,
        "_generate_analysis_code",
        lambda **_: "set_message('code-fallback-ok')\nset_data('x', 1)",
    )

    message = (
        "帮我做一个自定义分析：按小时统计再输出表格\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-code", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "code-fallback-ok"
    params = captured["payload"]["params"]
    assert params["allow_code"] is True
    assert "code" in params
