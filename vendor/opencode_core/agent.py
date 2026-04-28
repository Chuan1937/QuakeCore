from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .extraction import extract_code
from .prompts import build_admin_codegen_prompt


@dataclass
class OpenCodeAdminResult:
    code: str
    raw: str


class OpenCodeAdminAgent:
    def __init__(self, model_client: Any):
        self.model_client = model_client

    def generate_code(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
        input_key: str,
        lang: str,
        previous_error: str = "",
        previous_code: str = "",
    ) -> OpenCodeAdminResult:
        prompt = build_admin_codegen_prompt(
            message=message,
            runtime_results=runtime_results,
            input_key=input_key,
            lang=lang,
            previous_error=previous_error,
            previous_code=previous_code,
        )
        out = self.model_client.invoke(prompt)
        raw = str(getattr(out, "content", out) or "")
        code = extract_code(raw)
        return OpenCodeAdminResult(code=code, raw=raw)
