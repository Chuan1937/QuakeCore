"""Live smoke check for DeepSeek v4 Flash via backend /api/chat."""

import os
import sys

import requests


API_URL = "http://127.0.0.1:8000/api/chat"


def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[SKIP] DEEPSEEK_API_KEY not set")
        return 0

    payload = {
        "message": "请用一句话说明 QuakeCore 可以做什么。",
        "language": "zh",
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
    except requests.RequestException:
        print(
            "Backend not reachable. Start it first with:\n"
            "uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload",
            file=sys.stderr,
        )
        return 1

    if response.status_code != 200:
        print(f"Unexpected status {response.status_code}: {response.text}", file=sys.stderr)
        return 1

    try:
        body = response.json()
    except Exception:
        print(f"Response is not JSON: {response.text}", file=sys.stderr)
        return 1

    answer = str(body.get("answer", "") or "").strip()
    error = body.get("error")
    route = body.get("route")

    if not answer:
        print(f"Empty answer payload: {body}", file=sys.stderr)
        return 1
    if error not in (None, ""):
        print(f"Unexpected error payload: {body}", file=sys.stderr)
        return 1
    if not route:
        print(f"Missing route payload: {body}", file=sys.stderr)
        return 1

    print("model_name=deepseek-v4-flash")
    print(f"route={route}")
    print(f"answer_preview={answer[:200]}")
    print("DeepSeek v4 Flash smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
