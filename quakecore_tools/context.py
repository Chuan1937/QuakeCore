"""Session-scoped file context — replaces module-level CURRENT_* globals.

Usage:
    from quakecore_tools.context import FileContext

    ctx = FileContext.current()
    if ctx.miniseed_path:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileContext:
    """Holds the currently loaded file paths for one session/request.

    In a single-user dev setup the module-level singleton is used.
    For multi-user, pass a per-request ``FileContext`` instance.
    """

    segy_path: str | None = None
    miniseed_path: str | None = None
    miniseed_paths: list[str] = field(default_factory=list)
    hdf5_path: str | None = None
    sac_path: str | None = None
    uploaded_files: list[str] = field(default_factory=list)
    lang: str = "en"
    picks: list[Any] | None = None
    stations: list[Any] | None = None
    location: dict[str, Any] | None = None
    demo_mode: bool = False

    @property
    def current_type(self) -> str | None:
        if self.segy_path:
            return "segy"
        if self.miniseed_paths:
            return "miniseed"
        if self.hdf5_path:
            return "hdf5"
        if self.sac_path:
            return "sac"
        return None

    @property
    def active_path(self) -> str | None:
        """Return the most recently loaded file path regardless of type."""
        return (
            self.segy_path
            or self.miniseed_path
            or self.hdf5_path
            or self.sac_path
        )

    def set_uploaded_files(self, paths: list[str]) -> None:
        """Replace the uploaded-file list and rebuild per-type pointers."""
        normalized: list[str] = []
        for raw in paths or []:
            p = str(raw or "").strip()
            if p and p not in normalized:
                normalized.append(p)

        self.uploaded_files = normalized
        self.segy_path = None
        self.miniseed_path = None
        self.miniseed_paths = []
        self.hdf5_path = None
        self.sac_path = None

        for p in normalized:
            lowered = p.lower()
            if lowered.endswith((".mseed", ".miniseed")):
                self.miniseed_paths.append(p)
            elif lowered.endswith((".segy", ".sgy")):
                self.segy_path = p
            elif lowered.endswith((".h5", ".hdf5")):
                self.hdf5_path = p
            elif lowered.endswith(".sac"):
                self.sac_path = p

        if self.miniseed_paths:
            self.miniseed_path = self.miniseed_paths[-1]

    def add_miniseed_path(self, path: str) -> None:
        if path and path not in self.miniseed_paths:
            self.miniseed_paths.append(path)
            if not self.miniseed_path:
                self.miniseed_path = path

    def clear_miniseed_paths(self) -> None:
        self.miniseed_paths = []
        self.miniseed_path = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_type": self.current_type,
            "segy_path": self.segy_path,
            "miniseed_path": self.miniseed_path,
            "miniseed_paths": self.miniseed_paths,
            "num_miniseed_files": len(self.miniseed_paths),
            "hdf5_path": self.hdf5_path,
            "sac_path": self.sac_path,
            "has_picks": self.picks is not None and len(self.picks) > 0,
            "num_stations_with_coords": len(self.stations) if self.stations else 0,
        }


# ──────────────────────────────────────────────
# Module-level singleton (backward compatible)
# ──────────────────────────────────────────────

_global_ctx = FileContext()


def get_context() -> FileContext:
    """Return the global singleton context."""
    return _global_ctx


def sync_to_agent_tools() -> None:
    """Copy context state to agent.tools globals.

    Call this after updating context to ensure legacy tools see the changes.
    """
    try:
        import agent.tools

        agent.tools.CURRENT_SEGY_PATH = _global_ctx.segy_path
        agent.tools.CURRENT_MINISEED_PATH = _global_ctx.miniseed_path
        agent.tools.CURRENT_MINISEED_PATHS = list(_global_ctx.miniseed_paths)
        agent.tools.CURRENT_HDF5_PATH = _global_ctx.hdf5_path
        agent.tools.CURRENT_SAC_PATH = _global_ctx.sac_path
        agent.tools.CURRENT_UPLOADED_FILES = list(_global_ctx.uploaded_files)
        agent.tools.CURRENT_LANG = _global_ctx.lang
    except ImportError:
        pass


# Backward-compatible setters that mirror the old CURRENT_* globals.
# Each setter updates both the context AND the agent.tools globals.

def set_current_segy_path(path: str | None) -> None:
    _global_ctx.segy_path = path
    sync_to_agent_tools()


def set_current_miniseed_path(path: str | None) -> None:
    _global_ctx.miniseed_path = path
    if path and path not in _global_ctx.miniseed_paths:
        _global_ctx.miniseed_paths.append(path)
    sync_to_agent_tools()


def set_current_hdf5_path(path: str | None) -> None:
    _global_ctx.hdf5_path = path
    sync_to_agent_tools()


def set_current_sac_path(path: str | None) -> None:
    _global_ctx.sac_path = path
    sync_to_agent_tools()


def set_current_lang(lang: str) -> None:
    _global_ctx.lang = lang
    sync_to_agent_tools()


def set_demo_mode(enabled: bool = True) -> None:
    _global_ctx.demo_mode = enabled


def set_current_uploaded_files(paths: list[str]) -> None:
    _global_ctx.set_uploaded_files(paths)
    sync_to_agent_tools()


def get_current_uploaded_files() -> list[str]:
    return list(_global_ctx.uploaded_files)
