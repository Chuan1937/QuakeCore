import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app

pytestmark = pytest.mark.live_api


def test_deepseek_live_chat_path():
    if os.getenv("RUN_LIVE_API_TESTS") != "1":
        pytest.skip("RUN_LIVE_API_TESTS is not enabled")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "请用一句话说明 QuakeCore 可以做什么。", "lang": "zh"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("route")
    assert isinstance(payload.get("answer"), str)
    assert payload.get("answer", "").strip() != ""
    assert payload.get("error") in (None, "")
