from backend.services.config_service import ConfigService
from fastapi.testclient import TestClient

from backend.main import app


def test_config_defaults_endpoint_returns_expected_defaults():
    client = TestClient(app)

    response = client.get("/api/config/defaults")

    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"] == ["deepseek", "ollama"]
    assert payload["default_llm_config"]["provider"] == "deepseek"
    assert payload["default_llm_config"]["model_name"] == "deepseek-v4-flash"
    assert payload["provider_defaults"]["ollama"]["model_name"] == "qwen2.5:3b"


def test_config_llm_round_trip_persists_to_disk(tmp_path, monkeypatch):
    service = ConfigService(tmp_path)
    monkeypatch.setattr("backend.routes.config._config_service", service)
    client = TestClient(app)

    response = client.post(
        "/api/config/llm",
        json={
            "provider": "ollama",
            "model_name": "qwen2.5:7b",
            "api_key": None,
            "base_url": None,
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    assert service.llm_config_path.exists()

    get_response = client.get("/api/config/llm")
    assert get_response.status_code == 200
    assert get_response.json()["model_name"] == "qwen2.5:7b"

