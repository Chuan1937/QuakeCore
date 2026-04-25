"""Session-aware runtime state for chat agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class AgentSession:
    session_id: str
    lang: str
    agent: Any
    skill_context: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.last_active_at = datetime.now(timezone.utc)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, AgentSession] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str, factory) -> AgentSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = factory()
                self._sessions[session_id] = session
            session.touch()
            return session

