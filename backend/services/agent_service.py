"""Agent wrapper service for backend chat API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent.core import get_agent_executor
from agent.tools import set_current_lang
from backend.services.config_service import ConfigService
from backend.services.file_service import FileService, bind_uploaded_file_to_agent
from backend.services.langgraph_runtime import LangGraphRuntime
from backend.services.router_service import ArtifactItem, RouterService
from backend.services.session_store import AgentSession, SessionStore, get_session_store
from backend.services.skills_prompt_service import SkillsPromptService
from backend.services.tool_result import NormalizedToolResult, normalize_tool_output


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    answer: str
    error: str | None
    route: str
    artifacts: list[ArtifactItem]


class AgentService:
    def __init__(self, session_store: SessionStore | None = None):
        self._sessions = session_store or get_session_store()
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
        session = self._sessions.get_or_create(
            session_id=session_id,
            factory=lambda: self._build_agent_session(session_id, lang),
        )
        if session.agent is None:
            built = self._build_agent_session(session_id, lang)
            session.agent = built.agent
            session.lang = built.lang
            session.skill_context = built.skill_context
        return session

    def _inject_session_file_context(self, session_id: str) -> None:
        current_file = self._sessions.get_current_file(session_id)
        if not current_file:
            return

        file_type = FileService.infer_file_type(Path(current_file).name)
        try:
            bind_uploaded_file_to_agent(current_file, file_type)
        except Exception:
            # Keep backward compatibility: file-context injection must never block chat.
            return

    def _build_chat_artifacts(self, normalized: NormalizedToolResult) -> list[ArtifactItem]:
        structured: list[ArtifactItem] = []
        for artifact in normalized.artifacts:
            if not isinstance(artifact, dict):
                continue
            url = str(artifact.get("url", "")).strip()
            if not url:
                continue
            artifact_type = str(artifact.get("type", "image"))
            path = str(artifact.get("path", "")).strip()
            name = str(artifact.get("name", "")).strip()
            if not path and url.startswith("/api/artifacts/"):
                path = url[len("/api/artifacts/"):]
            if not name:
                name = Path(path or url).name
            structured.append(
                ArtifactItem(
                    type=artifact_type,
                    name=name,
                    path=path or name or url,
                    url=url,
                )
            )

        if structured:
            return structured
        return self._router_service.extract_artifacts(normalized.message)

    def chat(self, message: str, session_id: str | None = None, lang: str | None = "en") -> ChatResult:
        final_session_id = session_id or uuid4().hex
        final_lang = self._normalize_lang(lang)
        route = self._router_service.route_intent(message)

        try:
            session = self._get_or_create_session(final_session_id, final_lang)
            set_current_lang(final_lang)
            self._inject_session_file_context(final_session_id)
            if self._langgraph_runtime.enabled:
                raw_result = self._langgraph_runtime.invoke(
                    session_id=final_session_id,
                    message=message,
                    lang=final_lang,
                )
            else:
                raw_result = session.agent.invoke({"input": message})

            normalized = normalize_tool_output(raw_result)
            answer = normalized.message
            artifacts = self._build_chat_artifacts(normalized)
            return ChatResult(
                session_id=final_session_id,
                answer=answer,
                error=normalized.error,
                route=route,
                artifacts=artifacts,
            )
        except Exception as exc:
            normalized = normalize_tool_output(exc)
            return ChatResult(
                session_id=final_session_id,
                answer=normalized.message,
                error=normalized.error,
                route=route,
                artifacts=[],
            )
