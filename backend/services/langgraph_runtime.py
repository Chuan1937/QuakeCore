"""LangGraph preparation layer (disabled by default)."""

from __future__ import annotations

from backend.services.tool_result import ToolResult


class LangGraphRuntime:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def invoke(self, session_id: str, message: str, lang: str) -> ToolResult:
        if not self.enabled:
            raise RuntimeError("LangGraph runtime is disabled")
        raise NotImplementedError("LangGraph runtime is not wired yet")

