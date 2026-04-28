import json
from pathlib import Path

from quakecore_tools.analysis_tools import run_analysis_sandbox
from backend.services.session_store import get_session_store


def _write_csv(path: Path):
    path.write_text("time,magnitude_pred,depth_km\n2020-01-01T00:00:00,2.1,10\n", encoding="utf-8")


def _write_picks_csv(path: Path):
    path.write_text(
        "trace_index,phase_type,method,sample_index,normalized_score\n"
        "3,P,phasenet,100,0.8\n"
        "3,S,eqtransformer,250,0.4\n"
        "1,P,phasenet,50,0.9\n",
        encoding="utf-8",
    )


def test_analysis_sandbox_code_mode_disabled(tmp_path, monkeypatch):
    src = tmp_path / "catalog.csv"
    _write_csv(src)
    monkeypatch.delenv("QUAKECORE_ANALYSIS_ALLOW_CODE", raising=False)

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "input_path": str(src),
                "code": "set_message('ok')",
                "allow_code": False,
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is False
    assert "Code mode is disabled" in payload["message"]


def test_analysis_sandbox_code_mode_enabled(tmp_path, monkeypatch):
    src = tmp_path / "catalog.csv"
    _write_csv(src)
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "session_id": "sid-code-test",
                "input_path": str(src),
                "allow_code": True,
                "code": """
counts = {'rows': len(rows)}
set_data('rows', counts['rows'])
save_csv('code_rows.csv', [counts])
set_message('code ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "code ok"
    assert payload["data"]["rows"] == 1
    assert any(item.get("name") == "code_rows.csv" for item in payload.get("artifacts", []))


def test_analysis_sandbox_picks_trace_detail(tmp_path):
    src = tmp_path / "picks.csv"
    _write_picks_csv(src)

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "input_path": str(src),
                "template": "picks_trace_detail",
                "trace_index": 3,
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["data"]["trace_index"] == 3
    assert payload["data"]["pick_count"] == 2
    assert any(item.get("name", "").startswith("trace_3_picks_") for item in payload["artifacts"])


def test_analysis_sandbox_picks_trace_plot(tmp_path):
    src = tmp_path / "picks.csv"
    _write_picks_csv(src)

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "input_path": str(src),
                "template": "picks_trace_plot",
                "trace_index": 3,
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is False
    assert "fixed template is disabled" in payload["message"]


def test_analysis_sandbox_code_mode_runtime_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "analysis"
    data_dir.mkdir(parents=True, exist_ok=True)
    src = data_dir / "catalog.csv"
    _write_csv(src)

    sid = "sid-runtime-helper"
    store = get_session_store()
    store.update_runtime_results(sid, {"last_catalog_csv": "analysis/catalog.csv"})
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "session_id": sid,
                "input_path": str(src),
                "allow_code": True,
                "code": """
rows2 = read_csv('last_catalog_csv')
set_data('rows2', len(rows2))
set_data('resolved', resolve_data_path('last_catalog_csv'))
set_message('helpers ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "helpers ok"
    assert payload["data"]["rows2"] == 1
    assert str(payload["data"]["resolved"]).endswith("data/analysis/catalog.csv")


def test_analysis_sandbox_code_mode_without_default_input_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    src = picks_dir / "demo_picks.csv"
    src.write_text(
        "trace_index,phase_type,sample_index,normalized_score\n1,P,120,0.9\n",
        encoding="utf-8",
    )

    sid = "sid-no-default-input"
    store = get_session_store()
    store.update_runtime_results(
        sid,
        {
            "last_picks_csv": "picks/demo_picks.csv",
            "last_artifacts": [
                {
                    "type": "file",
                    "name": "demo_picks.csv",
                    "path": "picks/demo_picks.csv",
                    "url": "/api/artifacts/picks/demo_picks.csv",
                }
            ],
        },
    )
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "session_id": sid,
                "allow_code": True,
                "code": """
picks_csv = runtime_results.get('last_picks_csv') or get_runtime_artifact_path('picks_csv')
rows2 = read_csv(picks_csv)
set_data('rows2', len(rows2))
set_message('code no default input ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "code no default input ok"
    assert payload["data"]["rows2"] == 1


def test_analysis_sandbox_code_mode_uses_inline_runtime_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    src = picks_dir / "inline_picks.csv"
    src.write_text(
        "trace_index,phase_type,sample_index,normalized_score\n0,P,10,0.8\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "allow_code": True,
                "runtime_results": {"last_picks_csv": "picks/inline_picks.csv"},
                "code": """
rows2 = read_csv('last_picks_csv')
set_data('rows2', len(rows2))
set_message('inline runtime ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "inline runtime ok"
    assert payload["data"]["rows2"] == 1


def test_analysis_sandbox_code_mode_runtime_file_path_helper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wave = tmp_path / "data" / "uploads" / "demo.mseed"
    wave.parent.mkdir(parents=True, exist_ok=True)
    wave.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "allow_code": True,
                "runtime_results": {"last_uploaded_files": ["uploads/demo.mseed"]},
                "code": """
path = get_runtime_file_path('miniseed')
set_data('path', path)
set_message('runtime file helper ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "runtime file helper ok"
    assert payload["data"]["path"] == "uploads/demo.mseed"


def test_analysis_sandbox_code_mode_visible_artifact_and_file_record_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "1")

    raw = run_analysis_sandbox.invoke(
        {
            "params": {
                "allow_code": True,
                "runtime_results": {
                    "active_file": "CI.IDO.mseed",
                    "files": {
                        "CI.IDO.mseed": {"picks_csv": "picks/ido.csv"},
                    },
                    "last_visible_artifacts": [
                        {"type": "image", "path": "analysis/a.png"},
                        {"type": "image", "path": "analysis/b.png"},
                    ],
                },
                "code": """
set_data('img2', get_visible_artifact(2, 'image'))
set_data('active', get_active_file_record().get('picks_csv', ''))
set_data('ido', get_file_record('CI.IDO.mseed').get('picks_csv', ''))
set_message('helper set ok')
""",
            }
        }
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "helper set ok"
    assert payload["data"]["img2"] == "analysis/b.png"
    assert payload["data"]["active"] == "picks/ido.csv"
    assert payload["data"]["ido"] == "picks/ido.csv"
