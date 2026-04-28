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


class _DummyOpenCode:
    """Mock the real opencode CLI invocation."""
    def __init__(self, result=None):
        self._result = result or {"success": True, "message": "opencode-handled", "data": {}, "artifacts": []}
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self._result)


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


def test_langgraph_runtime_result_analysis_uses_opencode(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({"success": True, "message": "opencode-analysis-done", "data": {}, "artifacts": []})
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )

    message = (
        "看看第3个事件\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-1", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "opencode-analysis-done"
    assert len(dummy.calls) == 1
    assert dummy.calls[0]["message"]


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


def test_langgraph_runtime_result_analysis_code_mode(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({"success": True, "message": "custom-analysis-ok", "data": {"x": 1}, "artifacts": []})
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )

    message = (
        "帮我做一个自定义分析：按小时统计再输出表格\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-code", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "custom-analysis-ok"


def test_langgraph_runtime_trace_pick_analysis_uses_opencode(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({"success": True, "message": "trace-done", "data": {}, "artifacts": []})
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )
    message = (
        "看看 trace 3 的拾取结果\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-trace", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert len(dummy.calls) == 1
    assert "picks" in dummy.calls[0]["message"].lower()


def test_langgraph_runtime_extract_trace_index_supports_chinese_numerals():
    assert LangGraphRuntime._extract_trace_index("看看第二道数据的拾取图像") == 1


def test_langgraph_runtime_result_analysis_returns_failure(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({"success": False, "message": "opencode failed", "data": {}, "artifacts": []})
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )

    message = (
        "看看第3个事件\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-fallback", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is False


def test_langgraph_runtime_result_analysis_reads_runtime(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({"success": True, "message": "store-runtime-ok", "data": {}, "artifacts": []})
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )

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
    assert len(dummy.calls) == 1
    assert isinstance(dummy.calls[0].get("runtime_results"), dict)
    assert dummy.calls[0]["runtime_results"]["last_picks_csv"] == "picks/a.csv"


def test_langgraph_runtime_opencode_receives_artifacts(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    dummy = _DummyOpenCode({
        "success": True,
        "message": "plot generated",
        "data": {},
        "artifacts": [
            {"type": "image", "name": "figure.png", "path": "analysis/figure.png", "url": "/api/artifacts/analysis/figure.png"}
        ],
    })
    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: dummy,
    )

    message = (
        "画一个震级分布图\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_catalog_csv": "location/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-art", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert len(normalized.artifacts) == 1
    assert normalized.artifacts[0]["name"] == "figure.png"


def test_langgraph_runtime_opencode_invocation_error(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)

    class _FailingOpenCode:
        def execute(self, **kwargs):
            raise RuntimeError("opencode binary not found")

    monkeypatch.setattr(
        "backend.services.langgraph_runtime.OpenCodeAdminRuntime",
        lambda **_: _FailingOpenCode(),
    )

    message = (
        "分析一下\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-err", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is False
