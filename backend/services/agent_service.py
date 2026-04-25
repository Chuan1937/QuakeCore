"""Agent wrapper service for backend chat API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

from agent.core import get_agent_executor
from agent.tools import set_current_lang
from backend.services.config_service import ConfigService
from backend.services.langgraph_runtime import LangGraphRuntime
from backend.services.router_service import ArtifactItem, RouterService
from backend.services.session_store import AgentSession, SessionStore
from backend.services.skills_prompt_service import SkillsPromptService
from backend.services.tool_result import ToolResult


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    answer: str
    error: str | None
    route: str
    artifacts: list[ArtifactItem]


class AgentService:
    def __init__(self):
        self._sessions = SessionStore()
        self._router_service = RouterService()
        self._config_service = ConfigService()
        self._skills_prompt_service = SkillsPromptService()
        self._langgraph_runtime = LangGraphRuntime(
            enabled=os.getenv("QUAKECORE_USE_LANGGRAPH", "0") == "1"
        )

    @staticmethod
    def _normalize_lang(lang: str | None) -> str:
        return "zh" if str(lang or "").lower().startswith("zh") else "en"

    def _build_agent_session(self, session_id: str, lang: str) -> AgentSession:
        llm_config = self._config_service.get_llm_config()
        skill_context = self._skills_prompt_service.build_skill_context()
        set_current_lang(lang)
        agent = get_agent_executor(
            provider=llm_config.get("provider", "deepseek"),
            model_name=llm_config.get("model_name", "deepseek-v4-flash"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
            lang=lang,
            skill_context=skill_context,
        )
        return AgentSession(
            session_id=session_id,
            lang=lang,
            agent=agent,
            skill_context=skill_context,
        )

    def _get_or_create_session(self, session_id: str, lang: str) -> AgentSession:
        return self._sessions.get_or_create(
            session_id=session_id,
            factory=lambda: self._build_agent_session(session_id, lang),
        )

    def chat(self, message: str, session_id: str | None = None, lang: str | None = "en") -> ChatResult:
        final_session_id = session_id or uuid4().hex
        final_lang = self._normalize_lang(lang)
        route = self._router_service.route_intent(message)

        try:
            session = self._get_or_create_session(final_session_id, final_lang)
            set_current_lang(final_lang)
            if self._langgraph_runtime.enabled:
                tool_result = self._langgraph_runtime.invoke(
                    session_id=final_session_id,
                    message=message,
                    lang=final_lang,
                )
            else:
                response = session.agent.invoke({"input": message})
                tool_result = ToolResult.from_response(response)

            answer = tool_result.output
            artifacts = self._router_service.extract_artifacts(answer)
            return ChatResult(
                session_id=final_session_id,
                answer=answer,
                error=tool_result.error,
                route=route,
                artifacts=artifacts,
            )
        except Exception as exc:
            tool_error = ToolResult.from_error(exc)
            return ChatResult(
                session_id=final_session_id,
                answer=tool_error.output,
                error=tool_error.error,
                route=route,
                artifacts=[],
            )
