"""LangGraph preparation layer (disabled by default)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agent.core import _build_llm
from backend.services.artifact_utils import make_artifact, to_data_relative_path
from backend.services.config_service import ConfigService
from backend.services.router_service import RouterService
from backend.services.session_store import get_session_store
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

        zh_word = re.search(r"第\s*([一二三四五六七八九十])\s*(?:道|条|个)?", text)
        if zh_word:
            zh_num = {
                "一": 1,
                "二": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
            }
            try:
                return max(0, int(zh_num.get(zh_word.group(1), 1)) - 1)
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
    def _is_trace_pick_image_request(message: str) -> bool:
        text = str(message or "").lower()
        has_trace = any(token in text for token in ("trace", "轨迹", "道", "第"))
        has_pick = any(token in text for token in ("pick", "phase", "拾取", "震相"))
        has_image = any(token in text for token in ("图", "图像", "图片", "plot", "image", "波形"))
        return has_trace and has_pick and has_image

    @staticmethod
    def _build_trace_pick_waveform_code(trace_index: int) -> str:
        return (
            f"trace_index = {int(trace_index)}\n"
            "picks_csv = runtime_results.get('last_picks_csv') or get_runtime_artifact_path('picks_csv')\n"
            "mseed_path = runtime_results.get('last_miniseed_file') or get_runtime_file_path('miniseed')\n"
            "if not picks_csv:\n"
            "    set_message('缺少 last_picks_csv，无法绘制指定道拾取图像。')\n"
            "elif not mseed_path:\n"
            "    set_message('缺少 last_miniseed_file，无法绘制指定道拾取图像。')\n"
            "else:\n"
            "    picks_obj = read_csv(picks_csv)\n"
            "    if isinstance(picks_obj, list):\n"
            "        picks_rows = picks_obj\n"
            "    else:\n"
            "        picks_rows = picks_obj.to_dict('records') if hasattr(picks_obj, 'to_dict') else []\n"
            "    stream = read_waveform(mseed_path)\n"
            "    if trace_index < 0 or trace_index >= len(stream):\n"
            "        set_message(f'trace_index={trace_index} 超出范围，可用道数={{len(stream)}}。')\n"
            "    else:\n"
            "        tr = stream[trace_index]\n"
            "        data = tr.data\n"
            "        sr = float(tr.stats.sampling_rate) if float(tr.stats.sampling_rate) > 0 else 1.0\n"
            "        x = [i / sr for i in range(len(data))]\n"
            "        trace_col = ''\n"
            "        phase_col = ''\n"
            "        sample_col = ''\n"
            "        for key in (picks_rows[0].keys() if picks_rows else []):\n"
            "            lk = str(key).lower()\n"
            "            if not trace_col and lk in ('trace_index', 'trace', 'trace_id', 'index'):\n"
            "                trace_col = key\n"
            "            if not phase_col and lk in ('phase', 'phase_type', 'type'):\n"
            "                phase_col = key\n"
            "            if not sample_col and lk in ('sample_index', 'sample', 'index'):\n"
            "                sample_col = key\n"
            "        selected = []\n"
            "        if trace_col and sample_col:\n"
            "            for row in picks_rows:\n"
            "                ti_raw = str(row.get(trace_col, '-1')).strip()\n"
            "                ti_norm = ti_raw.replace('-', '', 1).replace('.', '', 1)\n"
            "                if not ti_norm.isdigit():\n"
            "                    continue\n"
            "                ti = int(float(ti_raw))\n"
            "                if int(ti) == int(trace_index):\n"
            "                    selected.append(row)\n"
            "        fig = plt.figure(figsize=(12, 4))\n"
            "        ax = fig.add_subplot(111)\n"
            "        ax.plot(x, data, color='#1f77b4', linewidth=0.8)\n"
            "        for row in selected:\n"
            "            sx_raw = str(row.get(sample_col, '-1')).strip()\n"
            "            sx_norm = sx_raw.replace('-', '', 1).replace('.', '', 1)\n"
            "            if not sx_norm.isdigit():\n"
            "                continue\n"
            "            sx = int(float(sx_raw))\n"
            "            phase = str(row.get(phase_col, 'P')).upper() if phase_col else 'P'\n"
            "            color = '#2ca02c' if phase.startswith('P') else '#d62728'\n"
            "            ax.axvline(sx / sr, color=color, linestyle='--', linewidth=1.0, alpha=0.85)\n"
            "        ax.set_title(f'Trace {{trace_index}} Waveform with Picks')\n"
            "        ax.set_xlabel('Time (s)')\n"
            "        ax.set_ylabel('Amplitude')\n"
            "        fig.tight_layout()\n"
            "        save_plot(f'trace_{trace_index}_waveform_picks.png')\n"
            "        if selected:\n"
            "            save_csv(f'trace_{trace_index}_picks.csv', selected)\n"
            "        set_data('trace_index', trace_index)\n"
            "        set_data('pick_count', len(selected))\n"
            "        set_message(f'已生成第{trace_index+1}道拾取结果图。')\n"
        )

    @staticmethod
    def _select_fallback_template(message: str, route: str) -> str:
        text = str(message or "").lower()

        # Fallback only: image-class trace pick requests should stay code-first.
        if ("trace" in text or "轨迹" in text or "道" in text) and ("拾取" in text or "pick" in text):
            if any(token in text for token in ("图", "图像", "图片", "plot", "image")):
                return ""
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
    def _sanitize_generated_code(code: str) -> str:
        text = str(code or "")
        if not text.strip():
            return ""
        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            cleaned_lines.append(raw_line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned

    @staticmethod
    def _build_code_prompt(
        *,
        message: str,
        runtime_results: dict[str, Any],
        input_key: str,
        lang: str,
    ) -> str:
        try:
            runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)
        except Exception:
            runtime_preview = str(runtime_results)
        runtime_preview = runtime_preview[:5000]
        if str(lang).lower().startswith("zh"):
            return (
                "你是 QuakeCore 的受限 Python 黑箱分析器。只输出 Python 代码，不要 markdown，不要解释。\n"
                "目标：基于当前会话已有结果做小分析、小统计、小绘图或结果解释。\n"
                "你需要像数据分析师一样自行决策读取哪些数据与如何绘图，不要套固定模板。\n"
                "不要重新运行完整 workflow。\n\n"
                "默认输入：\n"
                "- rows: list[dict]，来自 input_artifact_key 对应 CSV\n"
                "- columns: list[str]\n"
                "- input_path: 默认 CSV 路径\n"
                "- runtime_results: 当前会话已登记结果\n\n"
                "可用 helper：\n"
                "- set_message(text)\n"
                "- set_data(key, value)\n"
                "- save_csv(name, table)\n"
                "- save_plot(name)\n"
                "- resolve_data_path(path_or_key)\n"
                "- get_runtime_artifact_path(kind)\n"
                "- get_runtime_file_path(kind)\n"
                "- read_csv(path_or_key)\n"
                "- read_json(path_or_key)\n"
                "- read_waveform(path_or_key)\n\n"
                "强规则：\n"
                "1) 中文“第N道”表示 trace_index=N-1；英文 trace N 表示 trace_index=N。\n"
                "2) 用户要求某道拾取图像时，应画 waveform 并用竖线标出 P/S 到时，不要只画 sample-score 散点图。\n"
                "3) rows/columns 可能为空。为空时不要失败，主动通过 runtime_results 或 get_runtime_artifact_path() 查找文件。\n"
                "4) 用户问题与默认 rows 不匹配时，主动 read_csv/read_json/read_waveform 读取更合适的 runtime key。\n"
                "5) read_csv(path_or_key) 优先返回 pandas DataFrame；若返回 list，可用 pd.DataFrame(list_obj) 转换。\n"
                "6) 拾取图像常用：picks_csv = runtime_results.get('last_picks_csv') or get_runtime_artifact_path('picks_csv')；"
                "mseed = runtime_results.get('last_miniseed_file') or get_runtime_file_path('miniseed')。\n"
                "7) 图像必须 save_plot，表格必须 save_csv，且必须 set_message。\n"
                "8) 禁止 import、open、exec、eval、网络与系统调用。\n"
                "输出要求：\n"
                "1) 至少设置 set_message。\n"
                "2) 若有统计结果，调用 set_data。\n"
                "3) 若有表格或图，调用 save_csv/save_plot。\n\n"
                f"用户请求：{message}\n"
                f"推荐输入：{input_key}\n"
                f"runtime_results：{runtime_preview}\n"
            )
        return (
            "You are QuakeCore's restricted Python analysis sandbox generator. Output Python code only.\n"
            "Goal: analyze existing session artifacts/results for lightweight stats/plots/explanations.\n"
            "Do not re-run full workflows.\n"
            "Input data: rows(list[dict]), columns(list[str]), input_path, runtime_results.\n"
            "Helpers: set_message, set_data, save_csv, save_plot, resolve_data_path, get_runtime_artifact_path, get_runtime_file_path, read_csv, read_json, read_waveform.\n"
            "Rules:\n"
            "1) Chinese 第N道 => trace_index=N-1; English trace N => trace_index=N.\n"
            "2) For trace pick image requests, plot waveform with P/S vertical markers (not sample-score scatter only).\n"
            "3) rows/columns may be empty; do not fail and read better files from runtime_results/get_runtime_artifact_path.\n"
            "4) read_csv should be treated as DataFrame-first; if list fallback appears, convert with pd.DataFrame.\n"
            "5) If last_miniseed_file is empty, use get_runtime_file_path('miniseed').\n"
            "6) Must call set_message; call save_plot/save_csv when producing image/table.\n"
            "7) No imports/open/exec/eval/network/system calls.\n"
            "Output rules:\n"
            "1) call set_message.\n"
            "2) call set_data for key metrics.\n"
            "3) call save_csv/save_plot when useful.\n\n"
            f"User request: {message}\n"
            f"Recommended input key: {input_key}\n"
            f"runtime_results: {runtime_preview}\n"
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
            api_key=cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
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
        if template in {"picks_trace_detail", "picks_trace_plot"}:
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
        timeout_seconds = 60 if self._is_trace_pick_image_request(message) else 8
        max_attempts = 3 if self._is_trace_pick_image_request(message) else 2
        last_error = ""
        generated_code = ""
        for _attempt in range(max_attempts):
            prompt_message = message
            if last_error:
                prompt_message = (
                    f"{message}\n\n"
                    f"上一次生成代码执行失败，错误如下：{last_error}\n"
                    "请修复代码。"
                )
            try:
                generated_code = self._generate_analysis_code(
                    message=prompt_message,
                    runtime_results=runtime_results,
                    input_key=input_key,
                    lang=lang,
                )
            except Exception as exc:
                if self._is_trace_pick_image_request(message):
                    trace_index = self._extract_trace_index(message) or 0
                    generated_code = self._build_trace_pick_waveform_code(trace_index)
                    last_error = f"llm_codegen_failed: {exc}"
                else:
                    last_error = str(exc)
                    continue
            generated_code = self._sanitize_generated_code(generated_code)
            if not generated_code:
                last_error = "empty_generated_code"
                continue
            raw = run_analysis_sandbox.invoke(
                {
                    "params": {
                        "session_id": session_id,
                        "input_artifact_key": input_key,
                        "runtime_results": runtime_results,
                        "allow_code": True,
                        "code": generated_code,
                        "timeout_seconds": timeout_seconds,
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
            if self._is_trace_pick_image_request(message):
                trace_index = self._extract_trace_index(message) or 0
                fallback_code = self._build_trace_pick_waveform_code(trace_index)
                raw_fallback = run_analysis_sandbox.invoke(
                    {
                        "params": {
                            "session_id": session_id,
                            "input_artifact_key": input_key,
                            "runtime_results": runtime_results,
                            "allow_code": True,
                            "code": fallback_code,
                            "timeout_seconds": timeout_seconds,
                        }
                    }
                )
                payload_fallback = self._decode_payload(raw_fallback)
                if payload_fallback.get("success") is True:
                    data = payload_fallback.get("data")
                    if not isinstance(data, dict):
                        data = {}
                    data["generated_code"] = fallback_code
                    data["input_key"] = input_key
                    data["fallback_codegen"] = "builtin_trace_pick_waveform"
                    payload_fallback["data"] = data
                    return payload_fallback
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

        store_runtime = get_session_store().get_runtime_results(session_id)
        message_runtime = self._extract_runtime_context(message)
        runtime_results = dict(store_runtime)
        runtime_results.update(message_runtime)
        if route == "result_explanation":
            return ToolResult.from_response(self._build_explanation_payload(runtime_results, lang))

        fallback_template = self._select_fallback_template(message, route)
        try:
            code_payload = self._run_code_analysis(
                session_id=session_id,
                message=message,
                runtime_results=runtime_results,
                lang=lang,
            )
            if code_payload.get("success") is True:
                return ToolResult.from_response(code_payload)
            return ToolResult.from_response(code_payload)
        except Exception as exc:
            return ToolResult.from_response(
                {
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                    "data": {"route": route, "fallback_template": fallback_template or "", "template_disabled": True},
                    "artifacts": [],
                }
            )
