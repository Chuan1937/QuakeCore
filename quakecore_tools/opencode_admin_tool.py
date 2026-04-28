from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.tools import tool

from backend.services.artifact_utils import make_artifact, to_data_relative_path


OPENCODE_ADMIN_ROOT = Path("data/opencode_admin")


def _parse_params(params: str | dict | None) -> dict[str, Any]:
    if isinstance(params, dict):
        return dict(params)
    if params is None:
        return {}
    try:
        parsed = json.loads(str(params))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _result(
    success: bool,
    message: str,
    data: dict[str, Any] | None = None,
    artifacts: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "success": bool(success),
            "message": message,
            "data": data or {},
            "artifacts": artifacts or [],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _make_workspace(session_id: str) -> tuple[Path, Path, Path]:
    sid = str(session_id or "default").strip() or "default"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = (OPENCODE_ADMIN_ROOT / sid / stamp).resolve()
    workspace = root / "workspace"
    outputs = root / "outputs"
    logs = root / "logs"
    for p in (workspace, outputs, logs):
        p.mkdir(parents=True, exist_ok=True)
    return workspace, outputs, logs


def _collect_artifacts(outputs: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
    file_suffixes = {
        ".csv", ".json", ".txt", ".xlsx", ".xls", ".pdf", ".ps", ".eps",
        ".nc", ".grd", ".h5", ".hdf5", ".npy", ".npz", ".mseed",
        ".miniseed", ".sac", ".shp", ".geojson", ".kml",
    }

    for path in outputs.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in image_suffixes:
            artifacts.append(make_artifact(str(path), "image", name=path.name))
        elif suffix in file_suffixes:
            artifacts.append(make_artifact(str(path), "file", name=path.name))

    return artifacts


@tool
def run_opencode_admin_workspace(params: str | dict | None = None):
    """
    Run OpenCode-generated admin Python code.

    This is the high-capability replacement for the old restricted analysis sandbox.

    Params:
      - session_id: str
      - code: str
      - runtime_results: dict
      - input_key: str
      - timeout_seconds: int, default 300, max 900
    """
    parsed = _parse_params(params)
    session_id = str(parsed.get("session_id") or "default").strip() or "default"
    code = str(parsed.get("code") or "").strip()
    input_key = str(parsed.get("input_key") or "").strip()
    runtime_results = parsed.get("runtime_results")
    if not isinstance(runtime_results, dict):
        runtime_results = {}

    if not code:
        return _result(False, "OpenCode Admin 没有生成可执行代码。")

    timeout_seconds = int(parsed.get("timeout_seconds") or 300)
    timeout_seconds = max(1, min(timeout_seconds, 900))

    workspace, outputs, logs = _make_workspace(session_id)
    context_path = workspace / "runtime_results.json"
    result_path = workspace / "opencode_result.json"
    script_path = workspace / "opencode_admin_task.py"

    context_path.write_text(
        json.dumps(runtime_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    wrapped = f'''
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

workspace_dir = {str(workspace)!r}
outputs_dir = {str(outputs)!r}
data_root = str((Path.cwd() / "data").resolve())
input_key = {input_key!r}

WORKSPACE = Path(workspace_dir)
OUTPUTS = Path(outputs_dir)
DATA_ROOT = Path(data_root)
RUNTIME_CONTEXT_PATH = Path({str(context_path)!r})
RESULT_PATH = Path({str(result_path)!r})

runtime_results = json.loads(RUNTIME_CONTEXT_PATH.read_text(encoding="utf-8"))

message_holder = {{"message": "OpenCode Admin task executed."}}
data_holder = {{}}

def set_message(text):
    message_holder["message"] = str(text or "").strip() or message_holder["message"]

def set_data(key, value):
    data_holder[str(key)] = value

def output_path(name):
    filename = str(name or "").strip()
    if not filename:
        filename = "opencode_output.txt"
    path = (OUTPUTS / filename).resolve()
    path.relative_to(OUTPUTS.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _to_data_relative_path(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("/api/artifacts/"):
        text = text[len("/api/artifacts/"):]
    if text.startswith("data/"):
        text = text[len("data/"):]
    return text

def resolve_data_path(path_or_key):
    token = str(path_or_key or "").strip()
    if not token:
        raise ValueError("empty path_or_key")

    if token in runtime_results:
        token = str(runtime_results.get(token) or "").strip()

    rel = _to_data_relative_path(token)
    path = Path(token)

    if not path.is_absolute():
        if token.startswith("data/"):
            path = Path(token)
        else:
            path = DATA_ROOT / rel

    path = path.resolve()

    try:
        path.relative_to(DATA_ROOT)
    except ValueError:
        raise ValueError(f"resolve_data_path only resolves paths under data/: {{path}}")

    if not path.exists():
        raise FileNotFoundError(str(path))

    return str(path)

def read_csv(path_or_key):
    import pandas as pd
    return pd.read_csv(resolve_data_path(path_or_key))

def read_json(path_or_key):
    return json.loads(Path(resolve_data_path(path_or_key)).read_text(encoding="utf-8"))

def save_csv(name, obj):
    import pandas as pd
    path = output_path(name)
    if not str(path).lower().endswith(".csv"):
        path = Path(str(path) + ".csv")
    if hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False)
    else:
        pd.DataFrame(obj).to_csv(path, index=False)
    return str(path)

def save_json(name, obj):
    path = output_path(name)
    if not str(path).lower().endswith(".json"):
        path = Path(str(path) + ".json")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)

def save_text(name, text):
    path = output_path(name)
    path.write_text(str(text), encoding="utf-8")
    return str(path)

def save_plot(name="figure.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = output_path(name)
    if not path.suffix:
        path = Path(str(path) + ".png")
    plt.gcf().savefig(path, dpi=220, bbox_inches="tight")
    plt.close(plt.gcf())
    return str(path)

def shell(command, timeout=300):
    completed = subprocess.run(
        str(command),
        shell=True,
        cwd=str(WORKSPACE),
        text=True,
        capture_output=True,
        timeout=int(timeout),
    )
    return {{
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": str(command),
    }}

def pip_install(package, timeout=600):
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(package)],
        cwd=str(WORKSPACE),
        text=True,
        capture_output=True,
        timeout=int(timeout),
    )
    return {{
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "package": str(package),
    }}

def http_get(url, timeout=60):
    import requests
    response = requests.get(str(url), timeout=int(timeout))
    response.raise_for_status()
    return response.text

# User generated code follows

{code}

# User generated code ends

RESULT_PATH.write_text(
    json.dumps(
        {{
            "message": message_holder["message"],
            "data": data_holder,
        }},
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
'''

    script_path.write_text(textwrap.dedent(wrapped), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    env["MPLBACKEND"] = "Agg"
    env.setdefault("GMT_SESSION_NAME", f"quakecore_{session_id}")

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _result(
            False,
            f"OpenCode Admin 执行超时（{timeout_seconds}s）。",
            data={"generated_code": code, "timeout_seconds": timeout_seconds},
        )
    except Exception as exc:
        return _result(False, f"OpenCode Admin 执行异常：{exc}")

    (logs / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (logs / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    (logs / "script.py").write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts = _collect_artifacts(outputs)

    result_payload: dict[str, Any] = {}
    if result_path.exists():
        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result_payload = {}

    if completed.returncode != 0:
        return _result(
            False,
            f"OpenCode Admin 执行失败：{(completed.stderr or completed.stdout)[-3000:]}",
            data={
                "generated_code": code,
                "workspace": str(workspace),
                "outputs": str(outputs),
                "stdout": (completed.stdout or "")[-3000:],
                "stderr": (completed.stderr or "")[-3000:],
            },
            artifacts=artifacts,
        )

    return _result(
        True,
        str(result_payload.get("message") or "OpenCode Admin 执行完成。"),
        data={
            **dict(result_payload.get("data") or {}),
            "generated_code": code,
            "workspace": str(workspace),
            "outputs": str(outputs),
        },
        artifacts=artifacts,
    )
