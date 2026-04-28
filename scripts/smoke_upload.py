"""Minimal smoke test for the file upload endpoint."""

import os
import sys
from io import BytesIO

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
            "/api/files/upload",
            files={"file": ("smoke.mseed", BytesIO(b"smoke-data"), "application/octet-stream")},
        )
        if response.status_code != 200:
            print(f"Smoke check failed for /api/files/upload: {response.text}", file=sys.stderr)
            return 1
        payload = response.json()
        if (
            payload.get("filename") != "smoke.mseed"
            or payload.get("file_type") != "miniseed"
            or payload.get("bound_to_agent") is not True
        ):
            print(f"Unexpected payload from /api/files/upload: {payload}", file=sys.stderr)
            return 1
        print("File upload smoke check passed (in-process).")
        return 0

    url = "http://127.0.0.1:8000/api/files/upload"
    try:
        response = requests.post(
            url,
            files={"file": ("smoke.mseed", BytesIO(b"smoke-data"), "application/octet-stream")},
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Smoke check failed for {url}: {exc}", file=sys.stderr)
        return 1

    payload = response.json()
    if (
        payload.get("filename") != "smoke.mseed"
        or payload.get("file_type") != "miniseed"
        or payload.get("bound_to_agent") is not True
    ):
        print(f"Unexpected payload from {url}: {payload}", file=sys.stderr)
        return 1

    print("File upload smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
