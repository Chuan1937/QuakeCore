from backend.services.skills_prompt_service import SkillsPromptService
from backend.services.skills_service import SkillsService


def test_build_skill_context_uses_full_content(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    body = "# Demo Skill\nline1\nline2\nline3"
    (skills_dir / "demo.md").write_text(body, encoding="utf-8")

    service = SkillsPromptService(skills_service=SkillsService(skills_dir))
    context = service.build_skill_context(max_skills=4)

    assert "## Skill: demo" in context
    assert "line2" in context


def test_build_skill_context_truncates_long_content(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    long_text = "# Long\n" + ("a" * 2600)
    (skills_dir / "long.md").write_text(long_text, encoding="utf-8")

    service = SkillsPromptService(skills_service=SkillsService(skills_dir))
    context = service.build_skill_context(max_skills=4)

    assert len(context) < 2300
