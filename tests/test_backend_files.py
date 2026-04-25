from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.file_service import FileService


def test_infer_file_type_maps_expected_suffixes():
    assert FileService.infer_file_type("a.miniseed") == "miniseed"
    assert FileService.infer_file_type("a.segy") == "segy"
    assert FileService.infer_file_type("a.sgy") == "segy"
    assert FileService.infer_file_type("a.h5") == "hdf5"
    assert FileService.infer_file_type("a.hdf5") == "hdf5"
    assert FileService.infer_file_type("a.sac") == "sac"
    assert FileService.infer_file_type("a.npy") == "npy"
    assert FileService.infer_file_type("a.npz") == "npz"
    assert FileService.infer_file_type("a.txt") == "unknown"


def test_upload_file_mseed_binds_agent_state(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routes.files.FileService", lambda: FileService(tmp_path))
    monkeypatch.setattr(
        "backend.routes.files.bind_uploaded_file_to_agent",
        lambda _path, file_type: file_type == "miniseed",
    )
    client = TestClient(app)

    response = client.post(
        "/api/files/upload",
        files={"file": ("demo.mseed", BytesIO(b"mini-seed-data"), "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "demo.mseed"
    assert payload["file_type"] == "miniseed"
    assert payload["bound_to_agent"] is True
    saved_path = Path(payload["path"])
    assert saved_path.exists()
    assert saved_path.parent == tmp_path


def test_upload_file_txt_is_unknown_without_binding(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routes.files.FileService", lambda: FileService(tmp_path))
    monkeypatch.setattr(
        "backend.routes.files.bind_uploaded_file_to_agent",
        lambda _path, file_type: file_type in {"segy", "miniseed", "hdf5", "sac"},
    )
    client = TestClient(app)

    response = client.post(
        "/api/files/upload",
        files={"file": ("demo.txt", BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "demo.txt"
    assert payload["file_type"] == "unknown"
    assert payload["bound_to_agent"] is False
    saved_path = Path(payload["path"])
    assert saved_path.exists()
    assert saved_path.parent == tmp_path
