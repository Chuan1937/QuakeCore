from __future__ import annotations

from pathlib import Path


def to_data_relative_path(value: str | None) -> str:
    normalized = str(value or "").replace("\\", "/").strip()

    if normalized.startswith("./"):
        normalized = normalized[2:]

    if normalized.startswith("/api/artifacts/"):
        normalized = normalized[len("/api/artifacts/"):]

    if "/data/" in normalized:
        normalized = normalized.split("/data/", 1)[1]

    if normalized.startswith("data/"):
        normalized = normalized[5:]

    return normalized.lstrip("/")


def make_artifact(path: str, artifact_type: str = "file", name: str | None = None) -> dict:
    rel = to_data_relative_path(path)
    return {
        "type": artifact_type,
        "name": name or Path(rel).name,
        "path": rel,
        "url": f"/api/artifacts/{rel}",
    }
