"""Minimal smoke test for the FastAPI backend."""

import sys

import requests


def main() -> int:
    url = "http://127.0.0.1:8000/health"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Smoke check failed for {url}: {exc}", file=sys.stderr)
        return 1

    payload = response.json()
    if payload != {"status": "ok"}:
        print(f"Unexpected payload from {url}: {payload}", file=sys.stderr)
        return 1

    print("Backend health smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
