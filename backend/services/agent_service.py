"""Agent wrapper service for backend chat API."""

from __future__ import annotations

import json
import os
import re
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
from backend.workflows.location_workflow import run_location_workflow


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    answer: str
    error: str | None
    route: str
    artifacts: list[ArtifactItem]
    workflow: dict | None = None


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
        provider = llm_config.get("provider", "deepseek")
        api_key = llm_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        base_url = llm_config.get("base_url")
        model_name = llm_config.get("model_name", "deepseek-v4-flash")
        if provider == "deepseek" and not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for provider=deepseek.")
        skill_context = self._skills_prompt_service.build_skill_context()
        set_current_lang(lang)
        agent = get_agent_executor(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
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

    def _inject_session_file_context(
        self,
        session_id: str,
        attachments: list[str] | None = None,
    ) -> None:
        uploaded_files = [item for item in (attachments or self._sessions.get_uploaded_files(session_id)) if item]
        try:
            from agent.tools import set_current_uploaded_files

            set_current_uploaded_files(uploaded_files)
        except Exception:
            # Keep backward compatibility: file-context injection must never block chat.
            pass

        if not uploaded_files:
            self._sessions.set_current_file(session_id, None)
            return

        current_file = uploaded_files[-1]
        self._sessions.set_current_file(session_id, current_file)
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

    def _artifacts_from_payload(self, payload: list[dict]) -> list[ArtifactItem]:
        structured: list[ArtifactItem] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            item_type = str(item.get("type", "image")).strip() or "image"
            path = str(item.get("path", "")).strip()
            name = str(item.get("name", "")).strip()
            if not path and url.startswith("/api/artifacts/"):
                path = url[len("/api/artifacts/"):]
            if not name:
                name = Path(path or url).name
            structured.append(
                ArtifactItem(
                    type=item_type,
                    url=url,
                    name=name,
                    path=path or name or url,
                )
            )
        return structured

    def _artifacts_from_intermediate_steps(self, raw_result: object) -> list[ArtifactItem]:
        artifacts: list[ArtifactItem] = []
        if not isinstance(raw_result, dict):
            return artifacts

        for step in raw_result.get("intermediate_steps", []) or []:
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                continue
            _, observation = step
            payload = None
            if isinstance(observation, dict):
                payload = observation
            elif isinstance(observation, str):
                try:
                    payload = json.loads(observation)
                except Exception:
                    payload = None

            if not isinstance(payload, dict):
                continue
            items = payload.get("artifacts")
            if isinstance(items, list):
                artifacts.extend(self._artifacts_from_payload(items))

        seen: set[str] = set()
        unique: list[ArtifactItem] = []
        for item in artifacts:
            key = item.url
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _clean_public_answer(self, answer: str, route: str) -> str:
        if route != "continuous_monitoring":
            return answer

        text = answer or ""
        text = re.sub(
            r"\*\*官方目录对比\*\*：[\s\S]*?(?=\n\*\*主要活动区域\*\*|\n\*\*最佳定位事件|\n三视图：|$)",
            "",
            text,
        )
        text = re.sub(
            r"\*\*最佳定位事件（经目录校正后）\*\*：.*?(?=\n|$)",
            "",
            text,
        )
        text = text.replace("经目录校正后", "")
        text = re.sub(r"三视图：\s*!\[[^\]]*\]\([^)]+\)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        lang: str | None = "en",
        attachments: list[str] | None = None,
    ) -> ChatResult:
        final_session_id = session_id or uuid4().hex
        final_lang = self._normalize_lang(lang)
        route = self._router_service.route_intent(message)

        try:
            self._inject_session_file_context(final_session_id, attachments)
            if route == "earthquake_location":
                workflow_result = run_location_workflow(final_session_id)
                workflow_status = str(workflow_result.get("status", "failed"))
                workflow_steps = workflow_result.get("steps", [])
                if workflow_steps:
                    if workflow_status in {"success", "partial_success"}:
                        answer = str(workflow_result.get("summary") or workflow_result.get("message") or "")
                    else:
                        answer = str(
                            workflow_result.get("summary")
                            or workflow_result.get("message")
                            or "地震定位工作流执行失败，但已完成部分步骤。请查看步骤详情。"
                        )
                    return ChatResult(
                        session_id=final_session_id,
                        answer=answer,
                        error=workflow_result.get("error"),
                        route=route,
                        artifacts=self._artifacts_from_payload(
                            workflow_result.get("artifacts", [])
                        ),
                        workflow={
                            "status": workflow_result.get("status"),
                            "summary": workflow_result.get("summary"),
                            "message": workflow_result.get("message"),
                            "steps": workflow_steps,
                            "location": workflow_result.get("location", {}),
                            "artifacts": workflow_result.get("artifacts", []),
                            "error": workflow_result.get("error"),
                        },
                    )

            session = self._get_or_create_session(final_session_id, final_lang)
            set_current_lang(final_lang)
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
            tool_artifacts = self._artifacts_from_intermediate_steps(raw_result)
            artifacts = tool_artifacts or self._build_chat_artifacts(normalized)
            answer = self._clean_public_answer(answer, route)
            return ChatResult(
                session_id=final_session_id,
                answer=answer,
                error=normalized.error,
                route=route,
                artifacts=artifacts,
                workflow=None,
            )
        except Exception as exc:
            normalized = normalize_tool_output(exc)
            raw_error = str(normalized.error or "")
            if "Invalid Format" in raw_error or "Missing 'Action:' after 'Thought:'" in raw_error:
                return ChatResult(
                    session_id=final_session_id,
                    answer="工具执行已完成，但 Agent 输出格式异常。请查看已生成结果或重试。",
                    error=raw_error,
                    route=route,
                    artifacts=[],
                    workflow=None,
                )
            return ChatResult(
                session_id=final_session_id,
                answer=normalized.message,
                error=normalized.error,
                route=route,
                artifacts=[],
                workflow=None,
            )
