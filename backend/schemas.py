"""Shared API schemas for the backend."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class FileUploadResponse(BaseModel):
    filename: str
    path: str
    file_type: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    lang: Literal["en", "zh"] = "en"


class ArtifactResponse(BaseModel):
    type: str
    url: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    error: str | None = None
    route: str
    artifacts: list[ArtifactResponse] = []


class LlmConfigRequest(BaseModel):
    provider: Literal["deepseek", "ollama"]
    model_name: str
    api_key: str | None = None
    base_url: str | None = None


class LlmConfigResponse(LlmConfigRequest):
    pass


class ConfigDefaultsResponse(BaseModel):
    providers: list[str]
    default_llm_config: LlmConfigResponse
    provider_defaults: dict


class SkillListItem(BaseModel):
    name: str
    path: str


class SkillListResponse(BaseModel):
    skills: list[SkillListItem]


class SkillDetailResponse(BaseModel):
    name: str
    path: str
    content: str
