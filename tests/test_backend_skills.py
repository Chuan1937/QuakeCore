from fastapi.testclient import TestClient

from backend.main import app
from backend.services.skills_service import SkillsService


def test_skills_list_and_detail(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "phase_picking.md").write_text("# Phase Picking\nPick phases.", encoding="utf-8")
    (skills_dir / "earthquake_location.md").write_text("# Location\nLocate events.", encoding="utf-8")
    service = SkillsService(skills_dir)
    monkeypatch.setattr("backend.routes.skills._skills_service", service)
    client = TestClient(app)

    list_response = client.get("/api/skills")
    assert list_response.status_code == 200
    names = [item["name"] for item in list_response.json()["skills"]]
    assert names == ["earthquake_location", "phase_picking"]

    detail_response = client.get("/api/skills/phase_picking")
    assert detail_response.status_code == 200
    payload = detail_response.json()
    assert payload["name"] == "phase_picking"
    assert "Pick phases." in payload["content"]


def test_skills_detail_missing_returns_404(tmp_path, monkeypatch):
    service = SkillsService(tmp_path / "skills")
    monkeypatch.setattr("backend.routes.skills._skills_service", service)
    client = TestClient(app)

    response = client.get("/api/skills/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"
