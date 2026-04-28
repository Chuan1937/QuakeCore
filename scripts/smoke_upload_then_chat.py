"""Smoke test for upload followed by chat."""

import os
import sys
from io import BytesIO

import requests


BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    if os.getenv("QUAKECORE_SMOKE_INPROCESS") == "1":
        try:
            from fastapi.testclient import TestClient

            from backend.main import app
        except Exception as exc:  # pragma: no cover - smoke fallback
            print(f"In-process smoke setup failed: {exc}", file=sys.stderr)
            return 1

        client = TestClient(app)
        health_response = client.get("/health")
        if health_response.status_code != 200 or health_response.json() != {"status": "ok"}:
            print(f"Unexpected health payload: {health_response.text}", file=sys.stderr)
            return 1

        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("smoke.mseed", BytesIO(b"smoke-data"), "application/octet-stream")},
        )
        if upload_response.status_code != 200:
            print(f"Upload smoke failed: {upload_response.text}", file=sys.stderr)
            return 1
        upload_payload = upload_response.json()
        if upload_payload.get("file_type") != "miniseed" or not upload_payload.get("bound_to_agent"):
            print(f"Unexpected upload payload: {upload_payload}", file=sys.stderr)
            return 1

        chat_response = client.post(
            "/api/chat",
            json={"message": "Analyze the current file structure.", "lang": "en"},
        )
        if chat_response.status_code != 200:
            print(f"Chat smoke failed: {chat_response.text}", file=sys.stderr)
            return 1
        chat_payload = chat_response.json()
        required_keys = {"session_id", "answer", "error", "route", "artifacts"}
        if not required_keys.issubset(chat_payload):
            print(f"Unexpected chat payload: {chat_payload}", file=sys.stderr)
            return 1
        if chat_payload.get("route") != "file_structure":
            print(f"Unexpected chat route: {chat_payload}", file=sys.stderr)
            return 1
        if not isinstance(chat_payload.get("artifacts"), list):
            print(f"Invalid chat artifacts: {chat_payload}", file=sys.stderr)
            return 1

        print("Upload then chat smoke check passed (in-process).")
        return 0

    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
        if health_response.json() != {"status": "ok"}:
            print(f"Unexpected health payload: {health_response.text}", file=sys.stderr)
            return 1

        upload_response = requests.post(
            f"{BASE_URL}/api/files/upload",
            files={"file": ("smoke.mseed", BytesIO(b"smoke-data"), "application/octet-stream")},
            timeout=10,
        )
        upload_response.raise_for_status()
        upload_payload = upload_response.json()
        if upload_payload.get("file_type") != "miniseed" or not upload_payload.get("bound_to_agent"):
            print(f"Unexpected upload payload: {upload_payload}", file=sys.stderr)
            return 1

        chat_response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": "Analyze the current file structure.", "lang": "en"},
            timeout=30,
        )
        chat_response.raise_for_status()
        chat_payload = chat_response.json()
        required_keys = {"session_id", "answer", "error", "route", "artifacts"}
        if not required_keys.issubset(chat_payload):
            print(f"Unexpected chat payload: {chat_payload}", file=sys.stderr)
            return 1
        if chat_payload.get("route") != "file_structure":
            print(f"Unexpected chat route: {chat_payload}", file=sys.stderr)
            return 1
        if not isinstance(chat_payload.get("artifacts"), list):
            print(f"Invalid chat artifacts: {chat_payload}", file=sys.stderr)
            return 1
    except requests.RequestException as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        return 1

    print("Upload then chat smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
