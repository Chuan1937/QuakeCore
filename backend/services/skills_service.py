"""Skill markdown discovery and reading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSummary:
    name: str
    path: str


class SkillsService:
    def __init__(self, skills_dir: str | Path = "skills"):
        self.skills_dir = Path(skills_dir)

    def list_skills(self) -> list[SkillSummary]:
        if not self.skills_dir.exists():
            return []

        items: list[SkillSummary] = []
        for path in sorted(self.skills_dir.glob("*.md")):
            items.append(
                SkillSummary(
                    name=path.stem,
                    path=str(path),
                )
            )
        return items

    def get_skill(self, name: str) -> dict:
        candidate = Path(name).stem if name else ""
        if not candidate:
            raise FileNotFoundError("Skill not found")

        path = self.skills_dir / f"{candidate}.md"
        if not path.is_file():
            raise FileNotFoundError("Skill not found")

        return {
            "name": candidate,
            "path": str(path),
            "content": path.read_text(encoding="utf-8"),
        }

