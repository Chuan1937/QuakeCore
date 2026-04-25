"""Smoke test for location workflow API."""

import os
import sys

import requests


BASE_URL = "http://127.0.0.1:8000"


def _validate_payload(payload: dict) -> bool:
    required = {"status", "steps", "artifacts"}
    if not required.issubset(payload):
        print(f"Missing workflow fields: {payload}", file=sys.stderr)
        return False
    if payload.get("status") not in {"success", "partial_success", "failed"}:
        print(f"Invalid status: {payload}", file=sys.stderr)
        return False
    if not isinstance(payload.get("steps"), list):
        print(f"Invalid steps type: {payload}", file=sys.stderr)
        return False
    if not isinstance(payload.get("artifacts"), list):
        print(f"Invalid artifacts type: {payload}", file=sys.stderr)
        return False
    return True


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
        if health_response.status_code != 200:
            print(f"Health check failed: {health_response.text}", file=sys.stderr)
            return 1
        response = client.post("/api/workflows/location/run", json={})
        if response.status_code != 200:
            print(f"Workflow route failed: {response.text}", file=sys.stderr)
            return 1
        if not _validate_payload(response.json()):
            return 1
        print("Location workflow smoke check passed (in-process).")
        return 0

    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
    except requests.RequestException:
        print(
            "Backend not reachable. Start it first with:\n"
            "uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload",
            file=sys.stderr,
        )
        return 1

    try:
        response = requests.post(
            f"{BASE_URL}/api/workflows/location/run",
            json={},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Workflow smoke failed: {exc}", file=sys.stderr)
        return 1

    if not _validate_payload(response.json()):
        return 1

    print("Location workflow smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
