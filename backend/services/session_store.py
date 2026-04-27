"""Session-aware runtime state for chat agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable


@dataclass
class AgentSession:
    session_id: str
    lang: str = "en"
    agent: Any = None
    skill_context: str = ""
    uploaded_files: list[str] = field(default_factory=list)
    current_file: str | None = None
    runtime_results: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.last_active_at = datetime.now(timezone.utc)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, AgentSession] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str, factory: Callable[[], AgentSession]) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = factory()
                self._sessions[session_id] = session
            session.touch()
            return session

    def ensure_session(self, session_id: str) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentSession(session_id=session_id)
                self._sessions[session_id] = session
            session.touch()
            return session

    def add_file(self, session_id: str, path: str) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentSession(session_id=session_id)
                self._sessions[session_id] = session
            session.uploaded_files.append(path)
            session.touch()
            return session

    def set_current_file(self, session_id: str, path: str | None) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentSession(session_id=session_id)
                self._sessions[session_id] = session
            session.current_file = path
            session.touch()
            return session

    def get_current_file(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.touch()
            return session.current_file

    def get_uploaded_files(self, session_id: str) -> list[str]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.touch()
            return list(session.uploaded_files)

    def clear_files(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.uploaded_files.clear()
            session.current_file = None
            session.touch()

    def set_runtime_result(self, session_id: str, key: str, value: Any) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentSession(session_id=session_id)
                self._sessions[session_id] = session
            session.runtime_results[str(key)] = value
            session.touch()
            return session

    def update_runtime_results(self, session_id: str, values: dict[str, Any]) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentSession(session_id=session_id)
                self._sessions[session_id] = session
            for key, value in (values or {}).items():
                session.runtime_results[str(key)] = value
            session.touch()
            return session

    def get_runtime_results(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {}
            session.touch()
            return dict(session.runtime_results)

    def clear_runtime_results(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.runtime_results.clear()
            session.touch()


_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _session_store
