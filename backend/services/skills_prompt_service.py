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
            content = str(detail.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 2000:
                content = content[:2000]
            lines.append(f"## Skill: {item.name}\n{content}")
        return "\n".join(lines)
