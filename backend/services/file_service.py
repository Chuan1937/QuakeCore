"""File upload persistence and type inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


FILE_TYPE_BY_SUFFIX = {
    ".mseed": "miniseed",
    ".miniseed": "miniseed",
    ".sgy": "segy",
    ".segy": "segy",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".sac": "sac",
    ".npy": "npy",
    ".npz": "npz",
}


def bind_uploaded_file_to_agent(path: str, file_type: str) -> bool:
    if file_type == "segy":
        from agent.tools import set_current_segy_path

        set_current_segy_path(path)
        return True
    if file_type == "miniseed":
        from agent.tools import set_current_miniseed_path

        set_current_miniseed_path(path)
        return True
    if file_type == "hdf5":
        from agent.tools import set_current_hdf5_path

        set_current_hdf5_path(path)
        return True
    if file_type == "sac":
        from agent.tools import set_current_sac_path

        set_current_sac_path(path)
        return True
    return False


@dataclass(frozen=True)
class UploadedFileInfo:
    filename: str
    path: str
    file_type: str


class FileService:
    def __init__(self, upload_dir: str | Path = "data/uploads"):
        self.upload_dir = Path(upload_dir)

    @staticmethod
    def infer_file_type(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return FILE_TYPE_BY_SUFFIX.get(suffix, "unknown")

    def save_upload(self, file: UploadFile, session_id: str) -> UploadedFileInfo:
        original_name = Path(file.filename or "").name
        if not original_name:
            raise ValueError("Filename is required")

        file_type = self.infer_file_type(original_name)
        session_upload_dir = self.upload_dir / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{uuid4().hex}_{original_name}"
        destination = session_upload_dir / safe_name

        with destination.open("wb") as handle:
            handle.write(file.file.read())

        return UploadedFileInfo(
            filename=original_name,
            path=str(destination),
            file_type=file_type,
        )
