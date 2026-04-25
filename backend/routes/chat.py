"""Chat route backed by the existing AgentExecutor."""

from fastapi import APIRouter, Depends

from backend.schemas import ChatRequest, ChatResponse
from backend.services.agent_service import AgentService

router = APIRouter(prefix="/api", tags=["chat"])

_agent_service = AgentService()


def get_agent_service() -> AgentService:
    return _agent_service


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    resolved_lang = payload.language or payload.lang
    result = agent_service.chat(
        message=payload.message,
        session_id=payload.session_id,
        lang=resolved_lang,
    )
    return {
        "session_id": result.session_id,
        "answer": result.answer,
        "error": result.error,
        "route": result.route,
        "artifacts": [
            {"type": item.type, "name": item.name, "path": item.path, "url": item.url}
            for item in result.artifacts
        ],
    }
