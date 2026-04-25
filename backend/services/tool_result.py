"""Common result container for tool/agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    raw: Any = None

    @classmethod
    def from_response(cls, response: Any) -> "ToolResult":
        if isinstance(response, dict):
            return cls(ok=True, output=str(response.get("output", "")), raw=response)
        return cls(ok=True, output=str(response), raw=response)

    @classmethod
    def from_error(cls, error: Exception, raw: Any = None) -> "ToolResult":
        return cls(ok=False, output="", error=str(error), raw=raw)

