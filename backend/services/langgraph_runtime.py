"""LangGraph preparation layer (disabled by default)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.core import _build_llm
from backend.services.artifact_utils import make_artifact, to_data_relative_path
from backend.services.config_service import ConfigService
from backend.services.router_service import RouterService
from backend.services.tool_result import ToolResult
from quakecore_tools.analysis_tools import run_analysis_sandbox


class LangGraphRuntime:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._router = RouterService()

    @staticmethod
    def _extract_runtime_context(message: str) -> dict[str, Any]:
        text = str(message or "")
        markers = [
            "【当前会话已有结果上下文】\n",
            "【Current session runtime results】\n",
        ]
        for marker in markers:
            idx = text.rfind(marker)
            if idx < 0:
                continue
            payload = text[idx + len(marker) :].strip()
            try:
                parsed = json.loads(payload)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _extract_event_index(message: str) -> int:
        text = str(message or "")
        patterns = [
            r"第\s*(\d+)\s*个事件",
            r"event\s*#?\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return max(1, int(match.group(1)))
                except Exception:
                    return 1
        return 1

    @staticmethod
    def _extract_trace_index(message: str) -> int | None:
        text = str(message or "")

        zh = re.search(r"第\s*(\d+)\s*(?:道|条|个)?\s*(?:trace|轨迹|拾取)?", text)
        if zh and any(k in text for k in ("道", "trace", "轨迹", "拾取")):
            try:
                return max(0, int(zh.group(1)) - 1)
            except Exception:
                return None

        en = re.search(r"trace\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
        if en:
            try:
                return max(0, int(en.group(1)))
            except Exception:
                return None

        return None

    @staticmethod
    def _select_template(message: str, route: str) -> str:
        text = str(message or "").lower()

        # Trace-specific picks detail takes priority over generic image display
        if ("trace" in text or "轨迹" in text or "道" in text) and ("拾取" in text or "pick" in text):
            return "picks_trace_detail"

        if any(k in text for k in ("图像", "图", "图片", "plot", "image")) and any(
            k in text for k in ("拾取", "pick", "phase")
        ):
            return "picks_existing_image"
        if "第3个事件" in text or "第 3 个事件" in text or "第三个事件" in text or "event" in text and "#" in text:
            return "catalog_event_index"
        if "p/s" in text or "p 波" in text or "s 波" in text:
            return "picks_summary"
        if "台站" in text or "station" in text:
            return "picks_by_station"
        if "震级-深度" in text or "mag-depth" in text or "scatter" in text:
            return "catalog_mag_depth_scatter"
        if "深度分布" in text or "depth distribution" in text:
            return "catalog_depth_hist"
        if "震级分布" in text or "magnitude distribution" in text or "直方图" in text or "histogram" in text:
            return "catalog_magnitude_hist"
        if "时间序列" in text or "time series" in text:
            return "catalog_time_series"
        if route == "result_explanation":
            return ""
        return ""

    @staticmethod
    def _template_input_key(template: str) -> str:
        if template.startswith("picks_"):
            return "last_picks_csv"
        return "last_catalog_csv"

    @staticmethod
    def _decode_payload(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return {"success": False, "message": str(raw), "error": str(raw)}
            return parsed if isinstance(parsed, dict) else {"success": False, "message": str(raw), "error": str(raw)}
        return {"success": False, "message": str(raw), "error": str(raw)}

    @staticmethod
    def _code_input_key(message: str, runtime_results: dict[str, Any]) -> str:
        text = str(message or "").lower()

        if any(token in text for token in ("p波", "s波", "pick", "phase", "拾取", "震相", "台站", "station", "trace", "轨迹", "道")):
            if runtime_results.get("last_picks_csv"):
                return "last_picks_csv"

        if any(token in text for token in ("catalog", "目录", "事件", "震级", "深度", "定位")):
            if runtime_results.get("last_catalog_csv"):
                return "last_catalog_csv"

        if runtime_results.get("last_catalog_csv"):
            return "last_catalog_csv"

        if runtime_results.get("last_picks_csv"):
            return "last_picks_csv"

        return "last_catalog_csv"

    @staticmethod
    def _extract_code(text: str) -> str:
        content = str(text or "").strip()
        if not content:
            return ""
        block = re.search(r"```(?:python)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if block:
            return str(block.group(1)).strip()
        return content

    @staticmethod
    def _build_code_prompt(
        *,
        message: str,
        runtime_results: dict[str, Any],
        input_key: str,
        lang: str,
    ) -> str:
        if str(lang).lower().startswith("zh"):
            return (
                "你是地震分析代码生成器。请输出一段可执行 Python 代码（仅代码，不要解释，不要 markdown）。\n"
                "目标：完成用户提出的分析请求。\n"
                "输入数据：rows(list[dict])、columns(list[str])，来自 input_artifact_key。\n"
                "可用 helper：save_csv(name, table), save_plot(name), set_message(text), set_data(key, value), np, pd, plt(可选)。\n"
                "约束：禁止 import、禁止 open/exec/eval、禁止网络与系统调用。\n"
                "输出要求：\n"
                "1) 至少设置 set_message。\n"
                "2) 若有统计结果，调用 set_data。\n"
                "3) 若有表格或图，调用 save_csv/save_plot。\n\n"
                f"用户请求：{message}\n"
                f"推荐输入：{input_key}\n"
                f"runtime_results keys: {sorted(runtime_results.keys())}\n"
            )
        return (
            "You are a seismic analysis code generator. Output executable Python code only.\n"
            "Goal: satisfy the user's analysis request.\n"
            "Input data: rows(list[dict]), columns(list[str]) from input_artifact_key.\n"
            "Helpers: save_csv(name, table), save_plot(name), set_message(text), set_data(key, value), np, pd, plt(optional).\n"
            "Constraints: no imports, no open/exec/eval, no network/system calls.\n"
            "Output rules:\n"
            "1) call set_message.\n"
            "2) call set_data for key metrics.\n"
            "3) call save_csv/save_plot when useful.\n\n"
            f"User request: {message}\n"
            f"Recommended input key: {input_key}\n"
            f"runtime_results keys: {sorted(runtime_results.keys())}\n"
        )

    def _generate_analysis_code(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
        input_key: str,
        lang: str,
    ) -> str:
        cfg = ConfigService().get_llm_config()
        provider = str(cfg.get("provider", "deepseek"))
        model_name = str(cfg.get("model_name", "deepseek-v4-flash"))
        llm = _build_llm(
            provider=provider,  # type: ignore[arg-type]
            model_name=model_name,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            streaming=False,
        )
        prompt = self._build_code_prompt(
            message=message,
            runtime_results=runtime_results,
            input_key=input_key,
            lang=lang,
        )
        out = llm.invoke(prompt)
        content = getattr(out, "content", out)
        return self._extract_code(str(content or ""))

    @staticmethod
    def _run_picks_existing_image(message: str) -> dict[str, Any]:
        runtime = LangGraphRuntime._extract_runtime_context(message)
        image_path = runtime.get("last_picks_image")
        csv_path = runtime.get("last_picks_csv")

        artifacts = []

        if image_path:
            rel = to_data_relative_path(image_path)
            artifacts.append(make_artifact(rel, "image"))

        if csv_path:
            rel = to_data_relative_path(csv_path)
            artifacts.append(make_artifact(rel, "file"))

        return {
            "success": True,
            "message": "已找到最近一次初至拾取的图像和结果文件。" if artifacts else "没有找到已保存的拾取图像。请先运行初至拾取。",
            "artifacts": artifacts,
            "data": {
                "last_picks_image": image_path if isinstance(image_path, str) else "",
                "last_picks_csv": csv_path if isinstance(csv_path, str) else "",
            },
        }

    @staticmethod
    def _run_template_analysis(
        *,
        session_id: str,
        message: str,
        template: str,
    ) -> dict[str, Any]:
        if template == "picks_existing_image":
            return LangGraphRuntime._run_picks_existing_image(message)

        params: dict[str, Any] = {
            "template": template,
            "session_id": session_id,
            "input_artifact_key": LangGraphRuntime._template_input_key(template),
        }
        if template == "catalog_event_index":
            params["event_index"] = LangGraphRuntime._extract_event_index(message)
        if template == "picks_trace_detail":
            trace_index = LangGraphRuntime._extract_trace_index(message)
            if trace_index is not None:
                params["trace_index"] = trace_index
        raw = run_analysis_sandbox.invoke({"params": params})
        return LangGraphRuntime._decode_payload(raw)

    def _run_code_analysis(
        self,
        *,
        session_id: str,
        message: str,
        runtime_results: dict[str, Any],
        lang: str,
    ) -> dict[str, Any]:
        input_key = self._code_input_key(message, runtime_results)
        last_error = ""
        generated_code = ""
        for _attempt in range(2):
            prompt_message = message
            if last_error:
                prompt_message = (
                    f"{message}\n\n"
                    f"上一次生成代码执行失败，错误如下：{last_error}\n"
                    "请修复代码。"
                )
            generated_code = self._generate_analysis_code(
                message=prompt_message,
                runtime_results=runtime_results,
                input_key=input_key,
                lang=lang,
            )
            if not generated_code:
                last_error = "empty_generated_code"
                continue
            raw = run_analysis_sandbox.invoke(
                {
                    "params": {
                        "session_id": session_id,
                        "input_artifact_key": input_key,
                        "allow_code": True,
                        "code": generated_code,
                        "timeout_seconds": 8,
                    }
                }
            )
            payload = self._decode_payload(raw)
            if payload.get("success") is True:
                data = payload.get("data")
                if not isinstance(data, dict):
                    data = {}
                data["generated_code"] = generated_code
                data["input_key"] = input_key
                payload["data"] = data
                return payload
            last_error = str(payload.get("error") or payload.get("message") or "unknown_error")
        return {
            "success": False,
            "message": f"黑箱分析失败：{last_error}",
            "error": last_error,
            "data": {
                "generated_code": generated_code,
                "input_key": input_key,
            },
            "artifacts": [],
        }

    @staticmethod
    def _build_explanation_payload(runtime_results: dict[str, Any], lang: str) -> dict[str, Any]:
        monitoring = runtime_results.get("last_continuous_monitoring")
        if isinstance(monitoring, dict):
            detected = monitoring.get("n_events_detected")
            picks = monitoring.get("n_picks")
            stations = monitoring.get("n_stations")
            if str(lang).lower().startswith("zh"):
                message = (
                    f"已记录最近一次连续监测结果：识别事件 {detected} 个，"
                    f"拾取 {picks} 条，台站 {stations} 个。"
                )
            else:
                message = (
                    f"Latest continuous monitoring result is available: "
                    f"{detected} events, {picks} picks, {stations} stations."
                )
        else:
            if str(lang).lower().startswith("zh"):
                message = "已读取会话结果上下文。可继续让我做事件筛选、P/S 统计或分布绘图。"
            else:
                message = "Session runtime results are available. I can run event filtering, P/S counts, or distribution plots."
        artifacts = runtime_results.get("last_artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        return {
            "success": True,
            "message": message,
            "data": {
                "runtime_keys": sorted(runtime_results.keys()),
            },
            "artifacts": artifacts,
        }

    def invoke(
        self,
        session_id: str,
        message: str,
        lang: str,
        fallback_agent: Any | None = None,
    ) -> ToolResult:
        if not self.enabled:
            raise RuntimeError("LangGraph runtime is disabled")
        route = self._router.route_intent(message)
        if route not in {"result_analysis", "result_explanation"}:
            if fallback_agent is None:
                raise NotImplementedError("LangGraph runtime fallback is not configured")
            raw = fallback_agent.invoke({"input": message})
            return ToolResult.from_response(raw)

        runtime_results = self._extract_runtime_context(message)
        if route == "result_explanation":
            return ToolResult.from_response(self._build_explanation_payload(runtime_results, lang))

        template = self._select_template(message, route)
        try:
            if template:
                template_payload = self._run_template_analysis(
                    session_id=session_id,
                    message=message,
                    template=template,
                )
                # Template first. On success return directly; on failure, try code fallback.
                if template_payload.get("success") is True:
                    return ToolResult.from_response(template_payload)
            code_payload = self._run_code_analysis(
                session_id=session_id,
                message=message,
                runtime_results=runtime_results,
                lang=lang,
            )
            if code_payload.get("success") is True:
                return ToolResult.from_response(code_payload)
            if template:
                template_payload = self._run_template_analysis(
                    session_id=session_id,
                    message=message,
                    template=template,
                )
                if template_payload.get("success") is True:
                    return ToolResult.from_response(template_payload)
            return ToolResult.from_response(code_payload)
        except Exception as exc:
            return ToolResult.from_response(
                {
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                    "data": {"route": route, "template": template or ""},
                    "artifacts": [],
                }
            )
