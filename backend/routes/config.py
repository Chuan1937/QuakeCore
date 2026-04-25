"""Configuration API routes."""

from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import ConfigDefaultsResponse, LlmConfigRequest, LlmConfigResponse
from backend.services.config_service import ConfigService, LlmConfig

router = APIRouter(prefix="/api/config", tags=["config"])

_config_service = ConfigService()


def get_config_service() -> ConfigService:
    return _config_service


@router.get("/defaults", response_model=ConfigDefaultsResponse)
def get_defaults(config_service: ConfigService = Depends(get_config_service)):
    return config_service.get_defaults()


@router.get("/llm", response_model=LlmConfigResponse)
def get_llm_config(config_service: ConfigService = Depends(get_config_service)):
    return config_service.get_llm_config()


@router.post("/llm", response_model=LlmConfigResponse)
def save_llm_config(
    payload: LlmConfigRequest,
    config_service: ConfigService = Depends(get_config_service),
):
    try:
        return config_service.save_llm_config(LlmConfig(**payload.model_dump()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

