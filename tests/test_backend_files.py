from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.file_service import FileService


def test_infer_file_type_maps_expected_suffixes():
    assert FileService.infer_file_type("wave.mseed") == "miniseed"
    assert FileService.infer_file_type("wave.sgy") == "segy"
    assert FileService.infer_file_type("wave.segy") == "segy"
    assert FileService.infer_file_type("wave.h5") == "hdf5"
    assert FileService.infer_file_type("wave.sac") == "sac"


def test_upload_file_persists_and_returns_type(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routes.files.FileService", lambda: FileService(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/files/upload",
        files={"file": ("sample.mseed", BytesIO(b"mini-seed-data"), "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.mseed"
    assert payload["file_type"] == "miniseed"
    saved_path = Path(payload["path"])
    assert saved_path.exists()
    assert saved_path.parent == tmp_path


def test_upload_file_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routes.files.FileService", lambda: FileService(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/files/upload",
        files={"file": ("sample.txt", BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type: .txt"
