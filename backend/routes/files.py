"""File upload route."""

from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.schemas import FileUploadResponse
from backend.services.file_service import FileService, bind_uploaded_file_to_agent
from backend.services.session_store import SessionStore, get_session_store

router = APIRouter(prefix="/api/files", tags=["files"])


def get_file_service() -> FileService:
    return FileService()


def get_files_session_store() -> SessionStore:
    return get_session_store()


@router.post("/upload", response_model=FileUploadResponse)
def upload_file(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    file_service: FileService = Depends(get_file_service),
    session_store: SessionStore = Depends(get_files_session_store),
):
    final_session_id = session_id or uuid4().hex

    try:
        uploaded = file_service.save_upload(file, final_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_store.add_file(final_session_id, uploaded.path)
    session_store.set_current_file(final_session_id, uploaded.path)

    bound_to_agent = bind_uploaded_file_to_agent(uploaded.path, uploaded.file_type)

    return {
        "session_id": final_session_id,
        "filename": uploaded.filename,
        "path": uploaded.path,
        "file_type": uploaded.file_type,
        "bound_to_agent": bound_to_agent,
    }
