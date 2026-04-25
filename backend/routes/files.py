"""File upload route."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.schemas import FileUploadResponse
from backend.services.file_service import FileService, bind_uploaded_file_to_agent

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

    bound_to_agent = bind_uploaded_file_to_agent(uploaded.path, uploaded.file_type)

    return {
        "filename": uploaded.filename,
        "path": uploaded.path,
        "file_type": uploaded.file_type,
        "bound_to_agent": bound_to_agent,
    }
