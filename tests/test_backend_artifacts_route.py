from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def test_artifact_route_serves_file(tmp_path, monkeypatch):
    sample = tmp_path / "plot.png"
    sample.write_bytes(b"fake-image")
    monkeypatch.setattr("backend.routes.artifacts._data_root", tmp_path)
    client = TestClient(app)

    response = client.get("/api/artifacts/plot.png")

    assert response.status_code == 200
    assert response.content == b"fake-image"


def test_artifact_route_blocks_path_traversal(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"x")
    monkeypatch.setattr("backend.routes.artifacts._data_root", tmp_path)
    client = TestClient(app)

    response = client.get("/api/artifacts/../outside.png")

    assert response.status_code == 404
