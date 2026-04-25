"""Build skill context text injected into agent prompts."""

from __future__ import annotations

from backend.services.skills_service import SkillsService


class SkillsPromptService:
    def __init__(self, skills_service: SkillsService | None = None):
        self._skills_service = skills_service or SkillsService()

    def build_skill_context(self, max_skills: int = 8) -> str:
        skills = self._skills_service.list_skills()
        if not skills:
            return ""

        lines: list[str] = []
        for item in skills[:max_skills]:
            try:
                detail = self._skills_service.get_skill(item.name)
            except FileNotFoundError:
                continue
            first_line = ""
            for line in detail["content"].splitlines():
                stripped = line.strip()
                if stripped:
                    first_line = stripped
                    break
            summary = first_line or "No summary."
            lines.append(f"- {item.name}: {summary}")
        return "\n".join(lines)

