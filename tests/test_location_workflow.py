from backend.workflows.location_workflow import run_location_workflow
from backend.services.agent_service import AgentService
from backend.services.session_store import SessionStore


class _ToolStub:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _params):
        return self._payload


def test_location_workflow_returns_structured_result(monkeypatch):
    monkeypatch.setattr(
        "backend.workflows.location_workflow.get_loaded_context",
        _ToolStub({"success": True, "message": "context", "data": {"files": []}}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.load_local_data",
        _ToolStub({"success": True, "message": "loaded"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.pick_all_miniseed_files",
        _ToolStub({"success": True, "message": "picked"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.prepare_nearseismic_taup_cache",
        _ToolStub({"success": True, "message": "taup ready"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.add_station_coordinates",
        _ToolStub({"success": True, "message": "stations ready"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.locate_uploaded_data_nearseismic",
        _ToolStub(
            {
                "success": True,
                "message": "located",
                "data": {"latitude": 29.67, "longitude": 102.28},
                "artifacts": [{"type": "image", "path": "data/location/workflow_map.png"}],
            }
        ),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.plot_location_map",
        _ToolStub(
            {
                "success": True,
                "message": "map plotted",
                "artifacts": [{"type": "image", "path": "data/location/final_map.png"}],
            }
        ),
    )

    result = run_location_workflow("session-workflow")

    assert isinstance(result, dict)
    assert "success" in result
    assert result["status"] == "success"
    assert "summary" in result
    assert "message" in result
    assert "error" in result
    assert "steps" in result
    assert "artifacts" in result
    assert len(result["steps"]) == 7
    assert isinstance(result["artifacts"], list)
    for step in result["steps"]:
        assert "duration_ms" in step
        assert isinstance(step["duration_ms"], int)
        assert "required" in step


def test_location_workflow_tolerates_tool_failures(monkeypatch):
    class _FailTool:
        def invoke(self, _params):
            raise RuntimeError("tool failed")

    monkeypatch.setattr("backend.workflows.location_workflow.get_loaded_context", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.load_local_data", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.pick_all_miniseed_files", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.prepare_nearseismic_taup_cache", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.add_station_coordinates", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.locate_uploaded_data_nearseismic", _FailTool())
    monkeypatch.setattr("backend.workflows.location_workflow.plot_location_map", _FailTool())

    result = run_location_workflow("session-fail")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["status"] == "failed"
    assert len(result["steps"]) == 7
    assert all(step["status"] in {"error", "warning", "skipped"} for step in result["steps"])


def test_location_workflow_plot_failure_is_not_fatal(monkeypatch):
    class _FailTool:
        def invoke(self, _params):
            raise RuntimeError("plot failed")

    monkeypatch.setattr(
        "backend.workflows.location_workflow.get_loaded_context",
        _ToolStub({"success": True, "message": "context"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.load_local_data",
        _ToolStub({"success": True, "message": "loaded"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.pick_all_miniseed_files",
        _ToolStub({"success": True, "message": "picked"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.prepare_nearseismic_taup_cache",
        _ToolStub({"success": True, "message": "taup ready"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.add_station_coordinates",
        _ToolStub({"success": True, "message": "stations ready"}),
    )
    monkeypatch.setattr(
        "backend.workflows.location_workflow.locate_uploaded_data_nearseismic",
        _ToolStub(
            {
                "success": True,
                "message": "located",
                "data": {"latitude": 30.0, "longitude": 100.0},
                "artifacts": [{"type": "image", "path": "data/location/location.png"}],
            }
        ),
    )
    monkeypatch.setattr("backend.workflows.location_workflow.plot_location_map", _FailTool())

    result = run_location_workflow("session-plot-fail")

    assert result["success"] is True
    assert result["status"] == "partial_success"
    assert any(step["name"] == "plot_location_map" and step["status"] == "warning" for step in result["steps"])


def test_agent_service_prefers_location_workflow_when_route_matches(monkeypatch):
    monkeypatch.setattr(
        "backend.services.agent_service.run_location_workflow",
        lambda _session_id: {
            "success": True,
            "status": "success",
            "steps": [
                {
                    "name": "locate_uploaded_data_nearseismic",
                    "status": "ok",
                    "required": True,
                    "message": "located",
                    "error": None,
                    "data": {"latitude": 1.0},
                    "artifacts": [],
                    "duration_ms": 1,
                }
            ],
            "location": {"latitude": 1.0},
            "artifacts": [
                {
                    "type": "image",
                    "name": "workflow_map.png",
                    "path": "location/workflow_map.png",
                    "url": "/api/artifacts/location/workflow_map.png",
                }
            ],
            "message": "Location workflow completed.",
            "summary": "Location workflow completed.",
            "error": None,
        },
    )
    service = AgentService(session_store=SessionStore())

    result = service.chat("Please locate the earthquake", session_id="sid-loc", lang="en")

    assert result.route == "earthquake_location"
    assert result.error is None
    assert result.answer == "Location workflow completed."
    assert len(result.artifacts) == 1
    assert result.artifacts[0].url == "/api/artifacts/location/workflow_map.png"
    assert result.workflow is not None
    assert result.workflow["status"] == "success"
    assert isinstance(result.workflow["steps"], list)


def test_agent_service_returns_failed_workflow_summary_without_llm_fallback(monkeypatch):
    monkeypatch.setattr(
        "backend.services.agent_service.run_location_workflow",
        lambda _session_id: {
            "success": False,
            "status": "failed",
            "steps": [{"name": "locate_uploaded_data_nearseismic", "status": "error"}],
            "location": {},
            "artifacts": [],
            "message": "Workflow failed in location step.",
            "summary": "Workflow failed in location step.",
            "error": "location failed",
        },
    )
    service = AgentService(session_store=SessionStore())

    result = service.chat("locate this event", session_id="sid-fail", lang="en")

    assert result.route == "earthquake_location"
    assert result.answer == "Workflow failed in location step."
    assert result.error == "location failed"
    assert result.workflow is not None
    assert result.workflow["status"] == "failed"
    assert isinstance(result.workflow["steps"], list)
