import json

from backend.services.langgraph_runtime import LangGraphRuntime
from backend.services.session_store import get_session_store
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
                    "data": {"mode": "code"},
                    "artifacts": [],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "set_message('analysis-ok')")

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
    assert params["allow_code"] is True
    assert "code" in params
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


def test_langgraph_runtime_trace_pick_analysis_uses_picks_template(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {"success": True, "message": "trace-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "")
    message = (
        "看看 trace 3 的拾取结果\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-trace", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    params = captured["payload"]["params"]
    assert params["template"] == "picks_trace_detail"
    assert params["trace_index"] == 3
    assert params["input_artifact_key"] == "last_picks_csv"


def test_langgraph_runtime_trace_pick_plot_analysis_uses_plot_template(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {"success": True, "message": "trace-plot-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "")
    message = (
        "看看 trace 3 的拾取结果图\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-trace-plot", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    params = captured["payload"]["params"]
    assert params["template"] == "picks_trace_plot"
    assert params["trace_index"] == 3


def test_langgraph_runtime_result_analysis_fallbacks_to_template_when_code_fails(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    calls = []

    class _DummySandbox:
        def invoke(self, payload):
            calls.append(payload)
            params = payload.get("params", {})
            if params.get("allow_code"):
                return json.dumps(
                    {"success": False, "message": "code failed", "error": "code failed", "artifacts": []},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"success": True, "message": "template ok", "data": {"template": "catalog_event_index"}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "set_message('x')")

    message = (
        "看看第3个事件\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-fallback", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "template ok"
    assert len(calls) >= 2
    assert calls[0]["params"]["allow_code"] is True
    assert calls[-1]["params"]["template"] == "catalog_event_index"


def test_langgraph_runtime_result_analysis_reads_runtime_from_session_store(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {"success": True, "message": "store-runtime-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "set_message('store-runtime-ok')")

    sid = "sid-store-runtime"
    get_session_store().update_runtime_results(sid, {"last_picks_csv": "picks/a.csv"})
    result = runtime.invoke(
        session_id=sid,
        message="看看第一道的拾取图像",
        lang="zh",
        fallback_agent=_DummyAgent(),
    )
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "store-runtime-ok"
    params = captured["payload"]["params"]
    assert params["input_artifact_key"] == "last_picks_csv"
