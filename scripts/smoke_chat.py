"""Minimal smoke test for the chat endpoint schema."""

import os
import sys

import requests


def main() -> int:
    if os.getenv("QUAKECORE_SMOKE_INPROCESS") == "1":
        try:
            from fastapi.testclient import TestClient

            from backend.main import app
        except Exception as exc:  # pragma: no cover - smoke fallback
            print(f"In-process smoke setup failed: {exc}", file=sys.stderr)
            return 1

        response = TestClient(app).post(
            "/api/chat",
            json={"message": "hello", "lang": "en"},
        )
        if response.status_code != 200:
            print(f"Smoke check failed for /api/chat: {response.text}", file=sys.stderr)
            return 1
        payload = response.json()
        required_keys = {"session_id", "answer", "error", "route", "artifacts"}
        if not required_keys.issubset(payload):
            print(f"Unexpected payload from /api/chat: {payload}", file=sys.stderr)
            return 1
        if not payload.get("session_id"):
            print(f"Missing session_id in payload: {payload}", file=sys.stderr)
            return 1
        if not isinstance(payload.get("artifacts"), list):
            print(f"Invalid artifacts in payload: {payload}", file=sys.stderr)
            return 1
        print("Chat smoke check passed (in-process).")
        return 0

    url = "http://127.0.0.1:8000/api/chat"
    try:
        response = requests.post(
            url,
            json={"message": "hello", "lang": "en"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Smoke check failed for {url}: {exc}", file=sys.stderr)
        return 1

    payload = response.json()
    required_keys = {"session_id", "answer", "error", "route", "artifacts"}
    if not required_keys.issubset(payload):
        print(f"Unexpected payload from {url}: {payload}", file=sys.stderr)
        return 1
    if not payload.get("session_id"):
        print(f"Missing session_id in payload: {payload}", file=sys.stderr)
        return 1
    if not isinstance(payload.get("artifacts"), list):
        print(f"Invalid artifacts in payload: {payload}", file=sys.stderr)
        return 1

    print("Chat smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
