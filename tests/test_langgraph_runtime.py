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


def test_langgraph_runtime_trace_pick_analysis_uses_code_first(monkeypatch):
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
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "set_message('trace-ok')")
    message = (
        "看看 trace 3 的拾取结果\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-trace", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    params = captured["payload"]["params"]
    assert params["allow_code"] is True
    assert "code" in params
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
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "set_message('trace-plot-ok')")
    message = (
        "看看 trace 3 的拾取结果图\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-trace-plot", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    params = captured["payload"]["params"]
    assert params["allow_code"] is True
    assert "code" in params
    assert params["input_artifact_key"] == "last_picks_csv"


def test_langgraph_runtime_extract_trace_index_supports_chinese_numerals():
    assert LangGraphRuntime._extract_trace_index("看看第二道数据的拾取图像") == 1


def test_langgraph_runtime_result_analysis_returns_code_failure_when_code_fails(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    calls = []

    class _DummySandbox:
        def invoke(self, payload):
            calls.append(payload)
            return json.dumps(
                {"success": False, "message": "code failed", "error": "code failed", "artifacts": []},
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

    assert normalized.success is False
    assert "code failed" in normalized.message
    assert len(calls) >= 1
    assert calls[0]["params"]["allow_code"] is True


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
    assert isinstance(params.get("runtime_results"), dict)


def test_langgraph_runtime_sanitizes_import_before_sandbox(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["code"] = payload.get("params", {}).get("code", "")
            if "import " in captured["code"] or "from " in captured["code"]:
                return json.dumps(
                    {"success": False, "message": "Blocked syntax in code: Import", "error": "Blocked syntax in code: Import"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"success": True, "message": "sanitized-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(
        runtime,
        "_generate_analysis_code",
        lambda **_: "import os\nfrom math import sqrt\nset_message('sanitized-ok')",
    )

    message = (
        "看看第二道数据的拾取图像\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-sanitize", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "sanitized-ok"
    assert "import " not in captured["code"]
    assert "from " not in captured["code"]


def test_langgraph_runtime_trace_pick_image_uses_builtin_codegen_fallback(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    captured = {}

    class _DummySandbox:
        def invoke(self, payload):
            captured["payload"] = payload
            return json.dumps(
                {"success": True, "message": "builtin-fallback-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: (_ for _ in ()).throw(RuntimeError("Connection error.")))

    message = (
        "看看第二道数据的拾取图像\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv", "last_miniseed_file": "uploads/x.mseed"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-builtin", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "builtin-fallback-ok"
    code = str(captured["payload"]["params"].get("code", ""))
    assert "read_waveform" in code
    assert "last_picks_csv" in code


def test_langgraph_runtime_trace_pick_image_blocked_syntax_uses_builtin_fallback(monkeypatch):
    runtime = LangGraphRuntime(enabled=True)
    calls = []

    class _DummySandbox:
        def invoke(self, payload):
            calls.append(payload)
            if len(calls) == 1:
                return json.dumps(
                    {"success": False, "message": "Blocked syntax in code: Try", "error": "Blocked syntax in code: Try"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"success": True, "message": "fallback-after-block-ok", "data": {}, "artifacts": []},
                ensure_ascii=False,
            )

    monkeypatch.setattr("backend.services.langgraph_runtime.run_analysis_sandbox", _DummySandbox())
    monkeypatch.setattr(runtime, "_generate_analysis_code", lambda **_: "try:\n    set_message('x')\nexcept:\n    pass")

    message = (
        "看看第二道数据的拾取图像\n\n"
        "【当前会话已有结果上下文】\n"
        + json.dumps({"last_picks_csv": "picks/a.csv", "last_miniseed_file": "uploads/x.mseed"}, ensure_ascii=False)
    )
    result = runtime.invoke(session_id="sid-block", message=message, lang="zh", fallback_agent=_DummyAgent())
    normalized = normalize_tool_output(result)

    assert normalized.success is True
    assert normalized.message == "fallback-after-block-ok"
    assert len(calls) == 2
    second_code = str(calls[1]["params"].get("code", ""))
    assert "read_waveform" in second_code
    assert "try:" not in second_code
