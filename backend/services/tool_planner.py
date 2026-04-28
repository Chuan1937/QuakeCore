"""Lightweight planner for deterministic tool execution."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent.core import _build_llm
from backend.services.config_service import ConfigService


@dataclass
class ToolPlan:
    route: str
    tool: str
    params: dict[str, Any]
    need_rerun: bool
    confidence: float = 0.0


class ToolPlanner:
    def __init__(self):
        self._config = ConfigService()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
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

    @staticmethod
    def _rule_trace_index(message: str) -> int | None:
        text = str(message or "")
        zh_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        zh = re.search(r"第\s*(\d+)\s*(?:道|条|个)?\s*(?:trace|轨迹|波形|拾取|结果图|图像)?", text, flags=re.IGNORECASE)
        if zh and any(k in text for k in ("第", "道", "trace", "轨迹", "波形", "拾取", "图")):
            try:
                return max(0, int(zh.group(1)) - 1)
            except Exception:
                return 0
        zh_word = re.search(r"第\s*([一二三四五六七八九十])\s*(?:道|条|个)?", text)
        if zh_word:
            value = zh_num.get(zh_word.group(1), 1)
            return max(0, value - 1)
        en = re.search(r"trace\s*#?\s*(\d+)", text, flags=re.IGNORECASE)
        if en:
            try:
                return max(0, int(en.group(1)))
            except Exception:
                return 0
        return None

    @staticmethod
    def _rule_time_window(message: str) -> dict[str, str]:
        text = str(message or "")
        patterns = [
            r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日(?:的)?\s*"
            r"(?P<h1>\d{1,2})\s*(?:点|时)?\s*(?:到|至|-|—|~)\s*(?P<h2>\d{1,2})\s*(?:点|时)?",
            r"(?P<date>\d{4}-\d{1,2}-\d{1,2})\s+(?P<h1>\d{1,2})\s*(?:点|时)?\s*(?:到|至|-|—|~)\s*(?P<h2>\d{1,2})\s*(?:点|时)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                if match.groupdict().get("date"):
                    year, month, day = [int(part) for part in str(match.group("date")).split("-")]
                else:
                    year = int(match.group("year"))
                    month = int(match.group("month"))
                    day = int(match.group("day"))
                h1 = int(match.group("h1"))
                h2 = int(match.group("h2"))
                start_dt = datetime(year, month, day, h1, 0, 0)
                end_dt = datetime(year, month, day, h2, 0, 0)
            except Exception:
                continue
            return {
                "start": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "end": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        return {}

    @staticmethod
    def _rule_region(message: str) -> str | None:
        text = str(message or "").lower()
        candidates = [
            (("南加州", "southern california", "socal"), "南加州"),
            (("北加州", "northern california"), "北加州"),
            (("中加州", "central california"), "中加州"),
            (("加州", "california"), "加州"),
            (("日本", "japan"), "日本"),
            (("新西兰", "new zealand"), "新西兰"),
        ]
        for aliases, normalized in candidates:
            if any(alias in text for alias in aliases):
                return normalized
        return None

    def _rule_fallback(
        self,
        *,
        message: str,
        route: str,
        trace_index: int | None,
    ) -> ToolPlan:
        text = str(message or "").lower()
        if route == "file_structure":
            return ToolPlan(route=route, tool="get_file_structure", params={}, need_rerun=False, confidence=0.9)
        if route == "waveform_reading":
            params = {
                "trace_index": trace_index if trace_index is not None else 0,
                "plot": any(k in text for k in ("plot", "draw", "图", "画", "绘")),
            }
            return ToolPlan(route=route, tool="read_file_trace", params=params, need_rerun=False, confidence=0.85)
        if route == "phase_picking":
            return ToolPlan(route=route, tool="pick_first_arrivals", params={}, need_rerun=True, confidence=0.85)
        if route == "map_plotting":
            return ToolPlan(route=route, tool="plot_location_map", params={}, need_rerun=False, confidence=0.8)
        if route == "format_conversion":
            return ToolPlan(route=route, tool="", params={}, need_rerun=True, confidence=0.6)
        if route == "continuous_monitoring":
            params: dict[str, Any] = {}
            params.update(self._rule_time_window(message))
            region = self._rule_region(message)
            if region:
                params["region"] = region
            return ToolPlan(
                route=route,
                tool="run_continuous_monitoring",
                params=params,
                need_rerun=True,
                confidence=0.9 if params.get("start") and params.get("end") else 0.4,
            )
        if route == "result_explanation":
            return ToolPlan(route=route, tool="result_explanation", params={}, need_rerun=False, confidence=0.8)
        if route == "result_analysis":
            if ("trace" in text or "轨迹" in text or "道" in text) and ("拾取" in text or "pick" in text):
                if any(k in text for k in ("图", "图像", "图片", "plot", "image")):
                    return ToolPlan(
                        route=route,
                        tool="picks_trace_plot",
                        params={"trace_index": trace_index if trace_index is not None else 0},
                        need_rerun=False,
                        confidence=0.85,
                    )
                return ToolPlan(
                    route=route,
                    tool="picks_trace_detail",
                    params={"trace_index": trace_index if trace_index is not None else 0},
                    need_rerun=False,
                    confidence=0.85,
                )
            return ToolPlan(route=route, tool="", params={}, need_rerun=False, confidence=0.2)
        return ToolPlan(route=route, tool="", params={}, need_rerun=False, confidence=0.0)

    def plan(
        self,
        *,
        message: str,
        route: str,
        runtime_results: dict[str, Any],
        uploaded_files: list[str],
        current_file: str | None,
        lang: str,
    ) -> ToolPlan:
        trace_index = self._rule_trace_index(message)
        fallback = self._rule_fallback(message=message, route=route, trace_index=trace_index)

        # Skip LLM planning for non-tool or weakly-defined routes.
        if route in {"general_chat", "seismo_qa", "settings"}:
            return fallback

        try:
            cfg = self._config.get_llm_config()
            llm = _build_llm(
                provider=cfg.get("provider", "deepseek"),
                model_name=cfg.get("model_name", "deepseek-v4-flash"),
                api_key=cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
                base_url=cfg.get("base_url"),
                streaming=False,
            )
            prompt = f"""
你是 QuakeCore 的工具参数规划器。只输出 JSON，不要解释。

用户消息：
{message}

已识别 route：
{route}

当前文件：
{current_file or ""}

已上传文件数量：
{len(uploaded_files)}

runtime_results keys：
{sorted(runtime_results.keys())}

候选工具：
- pick_first_arrivals
- read_file_trace
- get_file_structure
- convert_miniseed_to_hdf5
- convert_miniseed_to_numpy
- convert_miniseed_to_sac
- convert_sac_to_hdf5
- convert_sac_to_numpy
- convert_sac_to_miniseed
- convert_segy_to_hdf5
- convert_segy_to_numpy
- convert_segy_to_excel
- convert_hdf5_to_numpy
- convert_hdf5_to_excel
- download_continuous_waveforms
- run_continuous_picking
- associate_continuous_events
- run_continuous_monitoring
- locate_earthquake
- locate_uploaded_data_nearseismic
- locate_place_data_nearseismic
- picks_trace_plot
- picks_trace_detail
- picks_summary
- picks_by_station
- catalog_magnitude_hist
- catalog_depth_hist
- catalog_time_series
- catalog_mag_depth_scatter
- catalog_event_index
- result_explanation

规则：
1. 已有结果查看/解释不要重跑 pick/location。
2. 只有“重新/重新计算/rerun”才 need_rerun=true。
3. 中文“第N道”按 1-based -> trace_index=N-1。
4. 英文 trace N 按 0-based。
5. 含“图/plot/image”优先 picks_trace_plot，不要 picks_trace_detail。
6. continuous_monitoring 必须从自然语言中解析 start/end 或 date+hours。
7. “2019年7月4日17到18点”或“2019年7月4日的17到18点” => start=2019-07-04T17:00:00, end=2019-07-04T18:00:00。
8. “加州” => region="加州"。
9. 如果用户只说日期和小时范围，优先输出 start/end，不要只输出 date/hours。
10. earthquake_location 如果用户说“定位当前文件/这个文件”，优先使用当前上传文件上下文，不要要求用户重复路径。
11. 仅输出 JSON。

输出格式：
{{
  "route": "...",
  "tool": "...",
  "params": {{}},
  "need_rerun": false,
  "confidence": 0.0
}}
"""
            out = llm.invoke(prompt)
            content = getattr(out, "content", out)
            parsed = self._extract_json(str(content or "")) or {}
            tool = str(parsed.get("tool") or "").strip()
            params = parsed.get("params")
            params = dict(params) if isinstance(params, dict) else {}
            planned_route = str(parsed.get("route") or route).strip() or route
            need_rerun = bool(parsed.get("need_rerun", False))
            confidence = float(parsed.get("confidence") or 0.0)

            if trace_index is not None and tool in {"picks_trace_plot", "picks_trace_detail", "read_file_trace"}:
                params["trace_index"] = trace_index

            if tool == "picks_trace_detail" and any(k in str(message).lower() for k in ("图", "图像", "图片", "plot", "image")):
                tool = "picks_trace_plot"

            if not tool:
                return fallback
            if route == "continuous_monitoring":
                params = {**fallback.params, **params}
            return ToolPlan(
                route=planned_route,
                tool=tool,
                params=params,
                need_rerun=need_rerun,
                confidence=confidence,
            )
        except Exception:
            return fallback
