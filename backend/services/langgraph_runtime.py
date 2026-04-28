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
from backend.services.opencode_admin_runtime import OpenCodeAdminRuntime
from backend.services.router_service import RouterService
from backend.services.session_store import get_session_store
from backend.services.tool_result import ToolResult


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
            "        ax.set_title(f'Trace {trace_index} Waveform with Picks')\n"
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
        # OpenCode Admin allows imports, pip, shell, network — no sanitization needed.
        return str(code or "").strip()

    @staticmethod
    def _extract_json_dict(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        if block:
            raw = block.group(1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _generate_analysis_plan(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
        lang: str,
    ) -> dict[str, Any]:
        cfg = ConfigService().get_llm_config()
        llm = _build_llm(
            provider=str(cfg.get("provider", "deepseek")),
            model_name=str(cfg.get("model_name", "deepseek-v4-flash")),
            api_key=cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
            base_url=cfg.get("base_url"),
            streaming=False,
        )
        try:
            runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)[:5000]
        except Exception:
            runtime_preview = str(runtime_results)[:5000]
        prompt = (
            "你是 QuakeCore 的分析规划器。只输出 JSON。\n"
            f"用户请求：{message}\n"
            f"runtime_results：{runtime_preview}\n"
            "规则：\n"
            "1) “所有/整体/全部/总体/拾取情况”优先 picks_summary，不是单道图。\n"
            "2) 只有“第N道/trace N/某道图像”才 picks_trace_plot。\n"
            "3) “第二张图/第N张图”使用 artifact_view，并给 image_index。\n"
            "4) 中文第N道 => trace_index=N-1。\n"
            "输出：\n"
            "{\n"
            '  "task_type":"picks_summary|picks_trace_plot|picks_distribution|catalog_summary|catalog_plot|artifact_view|conversion_analysis|custom",\n'
            '  "target_file":"",\n'
            '  "input_keys":[],\n'
            '  "trace_index":null,\n'
            '  "image_index":null,\n'
            '  "need_waveform":false,\n'
            '  "need_csv":true,\n'
            '  "need_plot":true,\n'
            '  "expected_outputs":[],\n'
            '  "user_intent":""\n'
            "}"
        )
        out = llm.invoke(prompt)
        content = getattr(out, "content", out)
        parsed = self._extract_json_dict(str(content or ""))
        if isinstance(parsed, dict):
            return parsed
        return {"task_type": "custom", "input_keys": [], "user_intent": message}

    @staticmethod
    def _build_code_prompt(
        *,
        message: str,
        runtime_results: dict[str, Any],
        input_key: str,
        lang: str,
        plan: dict[str, Any] | None = None,
    ) -> str:
        try:
            runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)
        except Exception:
            runtime_preview = str(runtime_results)
        runtime_preview = runtime_preview[:5000]
        try:
            plan_preview = json.dumps(plan or {}, ensure_ascii=False, default=str)
        except Exception:
            plan_preview = str(plan or {})
        plan_preview = plan_preview[:2000]
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
                "- get_visible_artifact(index, artifact_type='')\n"
                "- get_active_file_record()\n"
                "- get_file_record(name)\n"
                "- plot_trace_picks(trace_index, picks_key='last_picks_csv', waveform_key='last_miniseed_file')\n"
                "- read_csv(path_or_key)\n"
                "- read_json(path_or_key)\n"
                "- read_waveform(path_or_key)\n\n"
                f"分析计划：\n{plan_preview}\n\n"
                "强规则：\n"
                "1) 中文“第N道”表示 trace_index=N-1；英文 trace N 表示 trace_index=N。\n"
                "2) 用户要求某道拾取图像时，优先调用 plot_trace_picks(trace_index)。\n"
                "2.1) 如果 task_type 是 picks_summary 或用户说所有/整体/全部/总体/拾取情况，不要调用 plot_trace_picks，优先 summarize_picks。\n"
                "3) rows/columns 可能为空。为空时不要失败，主动通过 runtime_results 或 get_runtime_artifact_path() 查找文件。\n"
                "4) 用户问题与默认 rows 不匹配时，主动 read_csv/read_json/read_waveform 读取更合适的 runtime key。\n"
                "5) read_csv(path_or_key) 优先返回 pandas DataFrame；若返回 list，可用 pd.DataFrame(list_obj) 转换。\n"
                "6) 拾取图像常用：picks_csv = runtime_results.get('last_picks_csv') or get_runtime_artifact_path('picks_csv')；"
                "mseed = runtime_results.get('last_miniseed_file') or get_runtime_file_path('miniseed')。\n"
                "7) 用户说“这个文件”时优先 get_active_file_record()；说具体文件名时优先 get_file_record(name)。\n"
                "8) 用户说“第二张图”等序号图时，使用 get_visible_artifact(2, 'image')。\n"
                "9) 图像必须 save_plot，表格必须 save_csv，且必须 set_message。\n"
                "10) 禁止 import、open、exec、eval、网络与系统调用。\n"
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
            f"Plan: {plan_preview}\n"
            "Helpers: set_message, set_data, save_csv, save_plot, resolve_data_path, get_runtime_artifact_path, get_runtime_file_path, get_visible_artifact, get_active_file_record, get_file_record, plot_trace_picks, read_csv, read_json, read_waveform.\n"
            "Rules:\n"
            "1) Chinese 第N道 => trace_index=N-1; English trace N => trace_index=N.\n"
            "2) For trace pick image requests, prefer plot_trace_picks(trace_index).\n"
            "3) rows/columns may be empty; do not fail and read better files from runtime_results/get_runtime_artifact_path.\n"
            "4) read_csv should be treated as DataFrame-first; if list fallback appears, convert with pd.DataFrame.\n"
            "5) If last_miniseed_file is empty, use get_runtime_file_path('miniseed').\n"
            "6) For 'this file', use get_active_file_record(); for specific filename, use get_file_record(name).\n"
            "7) For 'second image', use get_visible_artifact(2, 'image').\n"
            "8) Must call set_message; call save_plot/save_csv when producing image/table.\n"
            "9) No imports/open/exec/eval/network/system calls.\n"
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
        plan: dict[str, Any] | None = None,
        previous_error: str = "",
        previous_code: str = "",
    ) -> str:
        return OpenCodeAdminRuntime().generate_code(
            message=message,
            runtime_results=runtime_results,
            input_key=input_key,
            lang=lang,
            previous_error=previous_error,
            previous_code=previous_code,
        )

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
        timeout_seconds = int(os.getenv("QUAKECORE_OPENCODE_ADMIN_TIMEOUT", "300"))

        result = OpenCodeAdminRuntime().execute(
            message=message,
            runtime_results=runtime_results,
            model="deepseek/deepseek-v4-flash",
            timeout_seconds=timeout_seconds,
        )

        result["data"] = result.get("data") or {}
        result["data"]["input_key"] = self._code_input_key(message, runtime_results)
        result.setdefault("opencode_admin", True)
        result.setdefault("artifacts", [])
        result.setdefault("error", "" if result.get("success") else result.get("message", ""))

        return result

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

        try:
            code_payload = self._run_code_analysis(
                session_id=session_id,
                message=message,
                runtime_results=runtime_results,
                lang=lang,
            )
            return ToolResult.from_response(code_payload)
        except Exception as exc:
            return ToolResult.from_response(
                {
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                    "data": {"route": route, "opencode_admin": True},
                    "artifacts": [],
                }
            )
