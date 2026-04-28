"""Artifact download/view route."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services.router_service import RouterService

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])
_data_root = Path("data")
_router_service = RouterService()


@router.get("/{artifact_path:path}")
def get_artifact(artifact_path: str):
    resolved = _router_service.resolve_artifact_path(_data_root, artifact_path)
    if resolved is None or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=resolved)

