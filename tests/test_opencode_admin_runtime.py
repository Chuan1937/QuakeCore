"""Unit tests for OpenCodeAdminRuntime helper methods."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend.services.opencode_admin_runtime import OpenCodeAdminRuntime


# ---------------------------------------------------------------------------
# _compact_runtime_results
# ---------------------------------------------------------------------------

def test_compact_runtime_results_keeps_key_fields():
    runtime = {
        "active_file": "test.mseed",
        "last_picks_csv": "picks/test.csv",
        "last_picks_image": "picks/test.png",
        "last_miniseed_file": "waveforms/test.mseed",
        "last_catalog_csv": "catalog/events.csv",
        "last_catalog_json": "catalog/events.json",
        "last_location_image": "location/map.png",
        "last_converted_file": "output/test.h5",
        "last_uploaded_files": ["f1.mseed", "f2.sac"],
        "last_artifacts": [{"type": "image", "name": "a.png"}],
        "last_visible_artifacts": [{"type": "image", "name": "b.png"}],
        "files": {"f1": {"source_file": "x.mseed"}},
        # Should be stripped
        "last_continuous_monitoring": {"n_events": 10},
        "last_route": "phase_picking",
        "internal_state": "secret",
        "debug_log": "verbose...",
    }
    compact = OpenCodeAdminRuntime._compact_runtime_results(runtime)

    expected_keys = {
        "active_file", "last_picks_csv", "last_picks_image",
        "last_miniseed_file", "last_catalog_csv", "last_catalog_json",
        "last_location_image", "last_converted_file", "last_uploaded_files",
        "last_artifacts", "last_visible_artifacts", "files",
    }
    assert set(compact.keys()) == expected_keys
    assert compact["last_picks_csv"] == "picks/test.csv"


def test_compact_runtime_results_truncates_artifact_lists():
    runtime = {
        "last_artifacts": [{"n": i} for i in range(20)],
        "last_visible_artifacts": [{"n": i} for i in range(15)],
    }
    compact = OpenCodeAdminRuntime._compact_runtime_results(runtime)

    assert len(compact["last_artifacts"]) == 8
    assert len(compact["last_visible_artifacts"]) == 8
    # Should keep the most recent (last 8)
    assert compact["last_artifacts"][-1] == {"n": 19}


def test_compact_runtime_results_empty():
    assert OpenCodeAdminRuntime._compact_runtime_results({}) == {}


# ---------------------------------------------------------------------------
# _clean_image_contradiction
# ---------------------------------------------------------------------------

def test_clean_image_contradiction_no_image_artifact():
    result = OpenCodeAdminRuntime._clean_image_contradiction(
        "该模型无法显示图像。但结果正确。",
        [{"type": "file", "name": "data.csv"}],
    )
    assert "无法显示图像" in result  # unchanged


def test_clean_image_contradiction_removes_contradiction():
    result = OpenCodeAdminRuntime._clean_image_contradiction(
        "该模型无法显示图像，但我已生成并保存了图片。分析完成。",
        [{"type": "image", "name": "plot.png"}],
    )
    assert "无法显示图像" not in result
    assert "分析完成" in result


def test_clean_image_contradiction_no_false_positive():
    result = OpenCodeAdminRuntime._clean_image_contradiction(
        "图像已生成，见下方结果。分析完成。",
        [{"type": "image", "name": "plot.png"}],
    )
    assert "图像已生成" in result


def test_clean_image_contradiction_empty_message():
    result = OpenCodeAdminRuntime._clean_image_contradiction(
        "",
        [{"type": "image", "name": "plot.png"}],
    )
    assert result == ""


def test_clean_image_contradiction_removes_path_line():
    result = OpenCodeAdminRuntime._clean_image_contradiction(
        "图片已保存至 `data/plots/result.png`，任务完成。",
        [{"type": "image", "name": "result.png"}],
    )
    assert "data/plots/result.png" not in result


# ---------------------------------------------------------------------------
# _scan_artifacts (workspace-scoped)
# ---------------------------------------------------------------------------

def test_scan_artifacts_respects_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "workspace"
        ws.mkdir(parents=True)
        outputs = ws / "outputs"
        outputs.mkdir()

        # Create a file in workspace/outputs
        (outputs / "result.png").write_text("fake png")
        # Create a file outside workspace (should be ignored without scan_since)
        (Path(tmp) / "other.png").write_text("outside")

        artifacts = OpenCodeAdminRuntime._scan_artifacts(
            written_files=set(),
            scan_since=None,
            workspace=ws,
        )
        # Without scan_since, only written_files are processed
        assert artifacts == []


def test_scan_artifacts_with_scan_since_scopes_outputs():
    import time
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "workspace"
        ws.mkdir(parents=True)
        outputs = ws / "outputs"
        outputs.mkdir()

        # Create file in outputs
        img = outputs / "trace_plot.png"
        img.write_text("fake png")

        scan_start = time.time() - 1  # slightly in the past

        # Need to mock Path.cwd() — we scan real cwd data/ which may have files.
        # Just verify that written_files from the workspace are picked up.
        artifacts = OpenCodeAdminRuntime._scan_artifacts(
            written_files={str(img)},
            scan_since=scan_start,
            workspace=ws,
        )
        names = [a["name"] for a in artifacts]
        assert "trace_plot.png" in names


def test_scan_artifacts_written_file_relative_to_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        outputs = ws / "outputs"
        outputs.mkdir()

        csv_path = outputs / "data.csv"
        csv_path.write_text("a,b,c")

        artifacts = OpenCodeAdminRuntime._scan_artifacts(
            written_files={"outputs/data.csv"},
            scan_since=None,
            workspace=ws,
        )
        names = [a["name"] for a in artifacts]
        assert "data.csv" in names


# ---------------------------------------------------------------------------
# _parse_single_event
# ---------------------------------------------------------------------------

def test_parse_single_event_step_start():
    raw = {"type": "step_start", "timestamp": 1000}
    parsed = OpenCodeAdminRuntime._parse_single_event(raw)
    assert parsed is not None
    assert parsed["type"] == "step"
    assert parsed["status"] == "running"
    assert parsed["summary"] == "Agent is thinking..."


def test_parse_single_event_step_finish():
    raw = {
        "type": "step_finish",
        "timestamp": 2000,
        "part": {
            "reason": "stop",
            "tokens": {"total": 500, "cost": 0.001},
        },
    }
    parsed = OpenCodeAdminRuntime._parse_single_event(raw)
    assert parsed["status"] == "completed"
    assert parsed["_tokens"]["total"] == 500


def test_parse_single_event_text():
    raw = {
        "type": "text",
        "timestamp": 3000,
        "part": {"text": "分析已完成"},
    }
    parsed = OpenCodeAdminRuntime._parse_single_event(raw)
    assert parsed["_text"] == "分析已完成"
    assert parsed["summary"] == "分析已完成"


def test_parse_single_event_tool_use_write():
    raw = {
        "type": "tool_use",
        "timestamp": 4000,
        "part": {
            "tool": "write",
            "state": {
                "input": {"filePath": "data/plots/result.png"},
                "output": "ok",
                "title": "",
            },
        },
    }
    written: set[str] = set()
    parsed = OpenCodeAdminRuntime._parse_single_event(raw, written_files=written)
    assert parsed["tool"] == "write"
    assert "data/plots/result.png" in written


def test_parse_single_event_unknown_type():
    raw = {"type": "unknown_type", "timestamp": 5000}
    parsed = OpenCodeAdminRuntime._parse_single_event(raw)
    assert parsed is None
