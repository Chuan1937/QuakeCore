from __future__ import annotations

import re


def extract_code(text: str) -> str:
    content = str(text or "").strip()
    if not content:
        return ""

    block = re.search(r"```(?:python)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if block:
        return block.group(1).strip()

    return content
