"""Skills discovery API routes."""

from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import SkillDetailResponse, SkillListItem, SkillListResponse
from backend.services.skills_service import SkillsService

router = APIRouter(prefix="/api/skills", tags=["skills"])

_skills_service = SkillsService()


def get_skills_service() -> SkillsService:
    return _skills_service


@router.get("", response_model=SkillListResponse)
def list_skills(skills_service: SkillsService = Depends(get_skills_service)):
    items = skills_service.list_skills()
    return {"skills": [{"name": item.name, "path": item.path} for item in items]}


@router.get("/{name}", response_model=SkillDetailResponse)
def get_skill(name: str, skills_service: SkillsService = Depends(get_skills_service)):
    try:
        return skills_service.get_skill(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Skill not found") from exc

