import json
from pathlib import Path

from quakecore_tools.analysis_tools import run_analysis_sandbox


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
