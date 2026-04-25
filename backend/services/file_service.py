"""File upload persistence and type inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


FILE_TYPE_BY_SUFFIX = {
    ".mseed": "miniseed",
    ".sgy": "segy",
    ".segy": "segy",
    ".h5": "hdf5",
    ".sac": "sac",
}


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
        file_type = FILE_TYPE_BY_SUFFIX.get(suffix)
        if file_type is None:
            raise ValueError(f"Unsupported file type: {suffix or '<none>'}")
        return file_type

    def save_upload(self, file: UploadFile) -> UploadedFileInfo:
        original_name = Path(file.filename or "").name
        if not original_name:
            raise ValueError("Filename is required")

        file_type = self.infer_file_type(original_name)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{uuid4().hex}_{original_name}"
        destination = self.upload_dir / safe_name

        with destination.open("wb") as handle:
            handle.write(file.file.read())

        return UploadedFileInfo(
            filename=original_name,
            path=str(destination),
            file_type=file_type,
        )

