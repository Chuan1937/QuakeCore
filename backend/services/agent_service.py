"""Agent wrapper service for backend chat API."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.core import get_agent_executor
from agent.tools import set_current_lang
from agent.tools_facade import (
    convert_hdf5_to_excel,
    convert_hdf5_to_numpy,
    convert_miniseed_to_hdf5,
    convert_miniseed_to_numpy,
    convert_miniseed_to_sac,
    convert_sac_to_hdf5,
    convert_sac_to_miniseed,
    convert_sac_to_numpy,
    convert_segy_to_excel,
    convert_segy_to_hdf5,
    convert_segy_to_numpy,
    get_file_structure,
    pick_first_arrivals,
    plot_location_map,
    read_file_trace,
)
from backend.services.artifact_utils import to_data_relative_path
from backend.services.config_service import ConfigService
from backend.services.file_service import FileService, bind_uploaded_file_to_agent
from backend.services.langgraph_runtime import LangGraphRuntime
from backend.services.router_service import ArtifactItem, RouterService
from backend.services.session_store import AgentSession, SessionStore, get_session_store
from backend.services.skills_prompt_service import SkillsPromptService
from backend.services.tool_planner import ToolPlan, ToolPlanner
from backend.services.tool_result import NormalizedToolResult, ToolResult, normalize_tool_output
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
        self._tool_planner = ToolPlanner()
        self._langgraph_runtime = LangGraphRuntime(
            enabled=os.getenv("QUAKECORE_USE_LANGGRAPH", "1") == "1"
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
        text = answer or ""
        text = re.sub(
            r"Invalid Format: Missing 'Action:' after 'Thought:'",
            "",
            text,
        )
        text = re.sub(
            r"Invalid Format:\s*Missing ['\"]Action:['\"] after ['\"]Thought:['\"]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"Got unsupported early_stopping_method `?generate`?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if route != "continuous_monitoring":
            return text

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

    @staticmethod
    def _extract_trace_number_from_message(message: str) -> int | None:
        text = str(message or "")
        patterns = [
            r"trace\s*#?\s*(\d+)",
            r"第\s*(\d+)\s*(?:条|个)?\s*(?:trace|轨迹|道)",
            r"(?:轨迹|道)\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return max(1, int(match.group(1)))
            except Exception:
                return None
        return None

    @staticmethod
    def _is_trace_pick_request(message: str, route: str) -> bool:
        if route != "phase_picking":
            return False
        text = str(message or "").lower()
        has_trace = any(token in text for token in ("trace", "轨迹", "道", "第"))
        has_pick = any(token in text for token in ("pick", "phase", "拾取", "震相"))
        return has_trace and has_pick

    def _try_fast_path_trace_pick(
        self,
        *,
        message: str,
        session_id: str,
        route: str,
    ) -> ChatResult | None:
        if not self._is_trace_pick_request(message, route):
            return None
        trace_number = self._extract_trace_number_from_message(message)
        if trace_number is None:
            return None
        try:
            raw = pick_first_arrivals.invoke({"params": {"trace_number": trace_number}})
        except Exception:
            return None
        normalized = normalize_tool_output(raw)
        artifacts = self._build_chat_artifacts(normalized)
        self._persist_runtime_updates(
            session_id,
            self._extract_runtime_updates(route=route, data=normalized.data, artifacts=artifacts),
        )
        return ChatResult(
            session_id=session_id,
            answer=self._summarize_direct_tool_result(
                message=message,
                route=route,
                normalized=normalized,
                artifacts=artifacts,
                lang="zh",
            ),
            error=normalized.error,
            route=route,
            artifacts=artifacts,
            workflow=None,
        )

    @staticmethod
    def _is_advanced_picking_request(message: str) -> bool:
        text = str(message or "").lower()
        return any(
            token in text
            for token in (
                "method",
                "methods",
                "方法",
                "traditional",
                "传统",
                "sta_lta",
                "sta/lta",
                "aic",
                "phasenet",
                "eqtransformer",
                "gpd",
            )
        )

    @staticmethod
    def _extract_trace_index_zero_based(message: str) -> int:
        trace_number = AgentService._extract_trace_number_from_message(message)
        if trace_number is not None:
            return max(0, trace_number - 1)
        return 0

    @staticmethod
    def _wants_plot(message: str) -> bool:
        text = str(message or "").lower()
        return any(token in text for token in ("plot", "draw", "图", "绘", "画"))

    @staticmethod
    def _invoke_direct_tool(tool_obj: Any, payload: Any | None = None) -> Any:
        if hasattr(tool_obj, "invoke"):
            if payload is None:
                return tool_obj.invoke({})
            return tool_obj.invoke(payload)
        if callable(tool_obj):
            if payload is None:
                return tool_obj()
            return tool_obj(payload)
        raise TypeError(f"Unsupported direct tool object: {tool_obj!r}")

    def _summarize_direct_tool_result(
        self,
        *,
        message: str,
        route: str,
        normalized: NormalizedToolResult,
        artifacts: list[ArtifactItem],
        lang: str,
    ) -> str:
        raw_text = normalized.message or ""
        data = normalized.data or {}
        if not raw_text and data:
            try:
                raw_text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            except Exception:
                raw_text = str(data)
        raw_text = raw_text[:8000]
        artifact_brief = [
            {"type": item.type, "name": item.name, "path": item.path}
            for item in artifacts
        ]
        prompt = (
            "你是 QuakeCore 的结果总结器。你不能调用工具，只能根据给定工具结果生成简洁中文回答。\n"
            "要求：\n"
            "1. 不要编造结果。\n"
            "2. 不要重复长路径。\n"
            "3. 不要把图片 markdown 写进正文，因为前端会单独显示 artifact。\n"
            "4. 如果有 CSV/PNG/HDF5 等 artifact，只在文字里简要说明结果文件已生成。\n"
            "5. 回答适合聊天窗口展示，优先总结关键结论。\n\n"
            f"用户问题：{message}\n"
            f"路由：{route}\n"
            f"工具原始结果：\n{raw_text}\n\n"
            f"artifacts：{json.dumps(artifact_brief, ensure_ascii=False)}\n"
        )
        try:
            from agent.core import _build_llm

            llm_config = self._config_service.get_llm_config()
            llm = _build_llm(
                provider=llm_config.get("provider", "deepseek"),
                model_name=llm_config.get("model_name", "deepseek-v4-flash"),
                api_key=llm_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
                base_url=llm_config.get("base_url"),
                streaming=False,
            )
            out = llm.invoke(prompt)
            content = getattr(out, "content", out)
            return self._clean_public_answer(str(content or raw_text), route)
        except Exception:
            return self._clean_public_answer(raw_text, route)

    def _select_conversion_tool(self, message: str, session_id: str) -> tuple[Any | None, dict[str, Any] | None]:
        text = str(message or "").lower()
        current_file = self._sessions.get_current_file(session_id)
        uploaded = self._sessions.get_uploaded_files(session_id)
        path = current_file or (uploaded[-1] if uploaded else None)
        if not path:
            return None, None

        suffix = Path(path).suffix.lower()
        target = None
        if any(k in text for k in ("hdf5", "h5")):
            target = "hdf5"
        elif any(k in text for k in ("numpy", "npy")):
            target = "numpy"
        elif any(k in text for k in ("excel", "xlsx")):
            target = "excel"
        elif "sac" in text:
            target = "sac"
        elif "miniseed" in text or "mseed" in text:
            target = "miniseed"

        payload = {"params": {"path": path}}
        if suffix in {".mseed", ".miniseed"}:
            if target == "hdf5":
                return convert_miniseed_to_hdf5, payload
            if target == "numpy":
                return convert_miniseed_to_numpy, payload
            if target == "sac":
                return convert_miniseed_to_sac, payload
        if suffix == ".sac":
            if target == "hdf5":
                return convert_sac_to_hdf5, payload
            if target == "numpy":
                return convert_sac_to_numpy, payload
            if target == "miniseed":
                return convert_sac_to_miniseed, payload
        if suffix in {".segy", ".sgy"}:
            if target == "hdf5":
                return convert_segy_to_hdf5, payload
            if target == "numpy":
                return convert_segy_to_numpy, payload
            if target == "excel":
                return convert_segy_to_excel, payload
        if suffix in {".h5", ".hdf5"}:
            if target == "numpy":
                return convert_hdf5_to_numpy, payload
            if target == "excel":
                return convert_hdf5_to_excel, payload
        return None, None

    def _try_fast_path_deterministic(
        self,
        *,
        message: str,
        session_id: str,
        route: str,
    ) -> ChatResult | None:
        tool_obj = None
        payload: Any | None = None
        # Keep specialized trace-pick handling in its own fast path.
        if route == "phase_picking" and self._is_trace_pick_request(message, route):
            return None
        if route == "phase_picking" and not self._is_advanced_picking_request(message):
            has_session_file = bool(self._sessions.get_current_file(session_id) or self._sessions.get_uploaded_files(session_id))
            if not has_session_file:
                return None
            tool_obj = pick_first_arrivals
            payload = {"params": {}}
        elif route == "file_structure":
            tool_obj = get_file_structure
            payload = {}
        elif route == "waveform_reading":
            tool_obj = read_file_trace
            payload = {
                "params": {
                    "trace_index": self._extract_trace_index_zero_based(message),
                    "plot": self._wants_plot(message),
                }
            }
        elif route == "map_plotting":
            tool_obj = plot_location_map
            payload = {"params": {}}
        elif route == "format_conversion":
            tool_obj, payload = self._select_conversion_tool(message, session_id)
            if tool_obj is None:
                return None
        else:
            return None

        try:
            raw = self._invoke_direct_tool(tool_obj, payload)
        except Exception:
            return None
        normalized = normalize_tool_output(raw)
        artifacts = self._build_chat_artifacts(normalized)
        self._persist_runtime_updates(
            session_id,
            self._extract_runtime_updates(route=route, data=normalized.data, artifacts=artifacts),
        )
        return ChatResult(
            session_id=session_id,
            answer=self._summarize_direct_tool_result(
                message=message,
                route=route,
                normalized=normalized,
                artifacts=artifacts,
                lang="zh",
            ),
            error=normalized.error,
            route=route,
            artifacts=artifacts,
            workflow=None,
        )

    @staticmethod
    def _to_runtime_path(value: str | None) -> str:
        return to_data_relative_path(value)

    def _build_runtime_context_suffix(self, session_id: str, lang: str) -> str:
        runtime_results = self._sessions.get_runtime_results(session_id)
        if not runtime_results:
            return ""
        try:
            serialized = json.dumps(runtime_results, ensure_ascii=False, default=str)
        except Exception:
            serialized = str(runtime_results)
        if len(serialized) > 6000:
            serialized = serialized[:6000]
        title = "当前会话已有结果上下文" if lang == "zh" else "Current session runtime results"
        return f"\n\n【{title}】\n{serialized}"

    def _build_message_with_runtime_context(self, message: str, session_id: str, lang: str) -> str:
        suffix = self._build_runtime_context_suffix(session_id, lang)
        return f"{message}{suffix}" if suffix else message

    def _extract_runtime_updates(
        self,
        *,
        route: str,
        data: dict[str, Any] | None,
        artifacts: list[ArtifactItem],
    ) -> dict[str, Any]:
        payload = dict(data or {})
        updates: dict[str, Any] = {"last_route": route}

        artifact_payload = [
            {
                "type": item.type,
                "name": item.name,
                "path": self._to_runtime_path(item.path),
                "url": item.url,
            }
            for item in artifacts
            if item.url
        ]
        if artifact_payload:
            updates["last_artifacts"] = artifact_payload

        # Result analysis artifacts must NOT overwrite main workflow context
        if route == "result_analysis":
            updates["last_analysis_artifacts"] = artifact_payload
            for item in artifacts:
                path = self._to_runtime_path(item.path or item.url)
                if path:
                    updates.setdefault("last_analysis_files", []).append(path)
            return updates

        key_mapping = {
            "picks_csv": "last_picks_csv",
            "picks_image": "last_picks_image",
            "plot_path": "last_picks_image",
            "catalog_csv": "last_catalog_csv",
            "catalog_json": "last_catalog_json",
            "location_map": "last_location_image",
            "location_3view": "last_location_image",
            "catalog_3view": "last_location_image",
        }
        for source_key, target_key in key_mapping.items():
            if source_key not in payload:
                continue
            normalized = self._to_runtime_path(str(payload.get(source_key, "") or ""))
            if normalized:
                updates[target_key] = normalized

        for item in artifacts:
            path = self._to_runtime_path(item.path or item.url)
            lowered = path.lower()

            if item.type == "image" and path:
                if "pick" in lowered:
                    updates.setdefault("last_picks_image", path)
                elif "location" in lowered or "catalog" in lowered:
                    updates.setdefault("last_location_image", path)

            if item.type != "file" or not path:
                continue

            if lowered.endswith(".csv") and "pick" in lowered:
                updates.setdefault("last_picks_csv", path)

            if lowered.endswith(".csv") and "catalog" in lowered:
                updates.setdefault("last_catalog_csv", path)

            if lowered.endswith(".json") and "catalog" in lowered:
                updates.setdefault("last_catalog_json", path)

        if route == "format_conversion":
            for item in artifacts:
                path = self._to_runtime_path(item.path or item.url)
                if not path:
                    continue
                lowered = path.lower()
                if item.type == "file" and lowered.endswith((".h5", ".hdf5", ".npy", ".sac", ".mseed", ".miniseed", ".xlsx")):
                    updates["last_converted_file"] = path
                    break
            for item in artifacts:
                path = self._to_runtime_path(item.path or item.url)
                if item.type == "image" and path:
                    updates["last_conversion_image"] = path
                    break

        if route == "continuous_monitoring" and payload:
            updates["last_continuous_monitoring"] = payload
        if route == "result_analysis":
            # Result analysis should not replace core runtime anchors.
            for key in ("last_picks_csv", "last_catalog_csv", "last_catalog_json", "last_location_image"):
                updates.pop(key, None)
            if artifact_payload:
                updates["last_analysis_artifacts"] = artifact_payload
                updates["last_analysis_files"] = [item.get("path", "") for item in artifact_payload if item.get("path")]

        return updates

    def _persist_runtime_updates(self, session_id: str, updates: dict[str, Any]) -> None:
        if not updates:
            return

        current_file = self._sessions.get_current_file(session_id)
        uploaded_files = self._sessions.get_uploaded_files(session_id)
        normalized_uploaded = [self._to_runtime_path(path) for path in uploaded_files if str(path or "").strip()]
        if normalized_uploaded:
            updates.setdefault("last_uploaded_files", normalized_uploaded)

        if current_file:
            normalized_current = self._to_runtime_path(current_file)
            if normalized_current:
                updates.setdefault("last_current_file", normalized_current)
            suffix = Path(str(current_file)).suffix.lower()
            if suffix in {".mseed", ".miniseed"}:
                updates.setdefault("last_miniseed_file", normalized_current)

        self._sessions.update_runtime_results(session_id, updates)

    def _persist_workflow_runtime(self, session_id: str, workflow_result: dict[str, Any]) -> None:
        artifacts = self._artifacts_from_payload(workflow_result.get("artifacts", []))
        updates = self._extract_runtime_updates(
            route="earthquake_location",
            data=workflow_result.get("location", {}),
            artifacts=artifacts,
        )
        updates["last_location_workflow"] = {
            "status": workflow_result.get("status"),
            "summary": workflow_result.get("summary"),
            "error": workflow_result.get("error"),
        }
        self._persist_runtime_updates(session_id, updates)

    @staticmethod
    def _decode_embedded_payload(value: str | None) -> dict[str, Any] | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _try_execute_tool_plan(
        self,
        *,
        plan: ToolPlan,
        message: str,
        session_id: str,
        lang: str,
    ) -> ChatResult | None:
        tool_obj = None
        payload: Any | None = None
        route = plan.route
        tool = plan.tool
        params = dict(plan.params or {})

        if tool == "pick_first_arrivals":
            has_session_file = bool(self._sessions.get_current_file(session_id) or self._sessions.get_uploaded_files(session_id))
            if not has_session_file:
                return None
            tool_obj = pick_first_arrivals
            payload = {"params": params}
        elif tool == "get_file_structure":
            tool_obj = get_file_structure
            payload = {}
        elif tool == "read_file_trace":
            tool_obj = read_file_trace
            payload = {"params": params}
        elif tool == "plot_location_map":
            tool_obj = plot_location_map
            payload = {"params": params}
        elif tool in {
            "convert_miniseed_to_hdf5",
            "convert_miniseed_to_numpy",
            "convert_miniseed_to_sac",
            "convert_sac_to_hdf5",
            "convert_sac_to_numpy",
            "convert_sac_to_miniseed",
            "convert_segy_to_hdf5",
            "convert_segy_to_numpy",
            "convert_segy_to_excel",
            "convert_hdf5_to_numpy",
            "convert_hdf5_to_excel",
        }:
            mapping = {
                "convert_miniseed_to_hdf5": convert_miniseed_to_hdf5,
                "convert_miniseed_to_numpy": convert_miniseed_to_numpy,
                "convert_miniseed_to_sac": convert_miniseed_to_sac,
                "convert_sac_to_hdf5": convert_sac_to_hdf5,
                "convert_sac_to_numpy": convert_sac_to_numpy,
                "convert_sac_to_miniseed": convert_sac_to_miniseed,
                "convert_segy_to_hdf5": convert_segy_to_hdf5,
                "convert_segy_to_numpy": convert_segy_to_numpy,
                "convert_segy_to_excel": convert_segy_to_excel,
                "convert_hdf5_to_numpy": convert_hdf5_to_numpy,
                "convert_hdf5_to_excel": convert_hdf5_to_excel,
            }
            current_file = self._sessions.get_current_file(session_id)
            uploaded = self._sessions.get_uploaded_files(session_id)
            path = params.get("path") or current_file or (uploaded[-1] if uploaded else None)
            if not path:
                return None
            params["path"] = path
            tool_obj = mapping[tool]
            payload = {"params": params}
            route = "format_conversion"
        elif tool in {
            "picks_trace_plot",
            "picks_trace_detail",
            "picks_summary",
            "picks_by_station",
            "catalog_magnitude_hist",
            "catalog_depth_hist",
            "catalog_time_series",
            "catalog_mag_depth_scatter",
            "catalog_event_index",
        }:
            # For pick-analysis requests without existing picks context, bootstrap
            # one pass of deterministic picking first, then future follow-ups can
            # be handled by code-first analysis runtime.
            if tool.startswith("picks_"):
                runtime = self._sessions.get_runtime_results(session_id)
                if not runtime.get("last_picks_csv"):
                    has_session_file = bool(
                        self._sessions.get_current_file(session_id) or self._sessions.get_uploaded_files(session_id)
                    )
                    if not has_session_file:
                        return None
                    trace_index = int(params.get("trace_index", 0) or 0)
                    tool_obj = pick_first_arrivals
                    payload = {"params": {"trace_number": max(1, trace_index + 1)}}
                    route = "phase_picking"
                else:
                    return None
            else:
                # Route analysis requests to LangGraphRuntime so it can do code-first,
                # template-fallback execution consistently.
                return None
        elif tool == "result_explanation" or not tool:
            return None
        else:
            return None

        try:
            raw = self._invoke_direct_tool(tool_obj, payload)
        except Exception:
            return None
        normalized = normalize_tool_output(raw)
        artifacts = self._build_chat_artifacts(normalized)
        self._persist_runtime_updates(
            session_id,
            self._extract_runtime_updates(route=route, data=normalized.data, artifacts=artifacts),
        )
        return ChatResult(
            session_id=session_id,
            answer=self._summarize_direct_tool_result(
                message=message,
                route=route,
                normalized=normalized,
                artifacts=artifacts,
                lang=lang,
            ),
            error=normalized.error,
            route=route,
            artifacts=artifacts,
            workflow=None,
        )

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
            runtime_results = self._sessions.get_runtime_results(final_session_id)
            uploaded_files = self._sessions.get_uploaded_files(final_session_id)
            current_file = self._sessions.get_current_file(final_session_id)
            plan = self._tool_planner.plan(
                message=message,
                route=route,
                runtime_results=runtime_results,
                uploaded_files=uploaded_files,
                current_file=current_file,
                lang=final_lang,
            )
            planned_result = self._try_execute_tool_plan(
                plan=plan,
                message=message,
                session_id=final_session_id,
                lang=final_lang,
            )
            if planned_result is not None:
                return planned_result
            fast_path_result = self._try_fast_path_trace_pick(
                message=message,
                session_id=final_session_id,
                route=route,
            )
            if fast_path_result is not None:
                return fast_path_result
            deterministic_result = self._try_fast_path_deterministic(
                message=message,
                session_id=final_session_id,
                route=route,
            )
            if deterministic_result is not None:
                return deterministic_result
            if route == "earthquake_location":
                workflow_result = run_location_workflow(final_session_id)
                self._persist_workflow_runtime(final_session_id, workflow_result)
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
            message_with_context = self._build_message_with_runtime_context(
                message=message,
                session_id=final_session_id,
                lang=final_lang,
            )
            if self._langgraph_runtime.enabled:
                raw_result = self._langgraph_runtime.invoke(
                    session_id=final_session_id,
                    message=message_with_context,
                    lang=final_lang,
                    fallback_agent=session.agent,
                )
            else:
                raw_result = session.agent.invoke({"input": message_with_context})

            normalized = normalize_tool_output(raw_result)
            answer = normalized.message
            decoded_payload = self._decode_embedded_payload(normalized.raw or normalized.message)
            runtime_data = normalized.data
            if not runtime_data and isinstance(decoded_payload, dict):
                embedded_data = decoded_payload.get("data")
                if isinstance(embedded_data, dict):
                    runtime_data = embedded_data
            raw_for_steps: object = raw_result.raw if isinstance(raw_result, ToolResult) else raw_result
            tool_artifacts = self._artifacts_from_intermediate_steps(raw_for_steps)
            artifacts = tool_artifacts or self._build_chat_artifacts(normalized)
            if not artifacts and isinstance(decoded_payload, dict):
                embedded_artifacts = decoded_payload.get("artifacts")
                if isinstance(embedded_artifacts, list):
                    artifacts = self._artifacts_from_payload(embedded_artifacts)
                if decoded_payload.get("message"):
                    answer = str(decoded_payload.get("message"))
            self._persist_runtime_updates(
                final_session_id,
                self._extract_runtime_updates(
                    route=route,
                    data=runtime_data,
                    artifacts=artifacts,
                ),
            )
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
