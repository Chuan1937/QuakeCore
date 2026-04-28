"""Chat route backed by the existing AgentExecutor."""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

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
        attachments=[item.path for item in payload.attachments or [] if item.path],
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
        "workflow": result.workflow,
    }


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    resolved_lang = payload.language or payload.lang

    def event_stream():
        yield ": stream-start\n\n"
        for event in agent_service.chat_stream(
            message=payload.message,
            session_id=payload.session_id,
            lang=resolved_lang,
            attachments=[item.path for item in payload.attachments or [] if item.path],
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
