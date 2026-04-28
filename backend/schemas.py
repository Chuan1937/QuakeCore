"""Shared API schemas for the backend."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class FileUploadResponse(BaseModel):
    session_id: str
    filename: str
    path: str
    file_type: str
    bound_to_agent: bool = False


class ChatAttachmentRequest(BaseModel):
    name: str
    path: str
    file_type: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    lang: Literal["en", "zh"] = "en"
    language: Literal["en", "zh"] | None = None
    attachments: list[ChatAttachmentRequest] | None = None


class ArtifactResponse(BaseModel):
    type: str
    name: str
    path: str
    url: str


class WorkflowStepResponse(BaseModel):
    name: str
    status: str
    required: bool
    message: str
    error: str | None = None
    data: dict = Field(default_factory=dict)
    artifacts: list[dict] = Field(default_factory=list)
    duration_ms: int


class WorkflowResultResponse(BaseModel):
    status: str
    summary: str | None = None
    message: str | None = None
    steps: list[WorkflowStepResponse] = Field(default_factory=list)
    location: dict = Field(default_factory=dict)
    artifacts: list[dict] = Field(default_factory=list)
    error: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    error: str | None = None
    route: str
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    workflow: WorkflowResultResponse | None = None


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


class LocationWorkflowRunRequest(BaseModel):
    session_id: str | None = None
