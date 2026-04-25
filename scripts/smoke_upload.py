"""Minimal smoke test for the file upload endpoint."""

import sys
from io import BytesIO

import requests


def main() -> int:
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
    if payload.get("filename") != "smoke.mseed" or payload.get("file_type") != "miniseed":
        print(f"Unexpected payload from {url}: {payload}", file=sys.stderr)
        return 1

    print("File upload smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
