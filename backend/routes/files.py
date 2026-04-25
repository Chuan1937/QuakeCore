"""File upload route."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.schemas import FileUploadResponse
from backend.services.file_service import FileService

router = APIRouter(prefix="/api/files", tags=["files"])


def get_file_service() -> FileService:
    return FileService()


@router.post("/upload", response_model=FileUploadResponse)
def upload_file(
    file: UploadFile = File(...),
    file_service: FileService = Depends(get_file_service),
):
    try:
        uploaded = file_service.save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": uploaded.filename,
        "path": uploaded.path,
        "file_type": uploaded.file_type,
    }
