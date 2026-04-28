from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.skills_prompt_service import SkillsPromptService


OPENCODE_BIN = os.getenv("OPENCODE_BIN", "opencode")
PLOT_HELPERS_FILENAME = "quakecore_plot_helpers.py"
PLOT_HELPERS_CONTENT = """from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read


def pick_value(row, candidates, default=None):
    for key in candidates:
        if key in row and row[key] not in ("", None):
            return row[key]
    lowered = {str(k).lower(): k for k in row.keys()}
    for key in candidates:
        real = lowered.get(str(key).lower())
        if real is not None and row[real] not in ("", None):
            return row[real]
    return default


def parse_pick_row(row):
    method = str(pick_value(row, ["method", "model", "picker", "algorithm"], "unknown"))
    phase = str(pick_value(row, ["phase", "phase_type", "phase_name", "label", "type"], "")).upper()
    score_raw = pick_value(
        row,
        ["score", "confidence", "probability", "prob", "peak_value", "phase_score", "p_score", "s_score"],
        None,
    )
    sample_raw = pick_value(
        row,
        ["sample", "sample_index", "arrival_sample", "pick_sample", "index", "sample_id"],
        None,
    )
    time_raw = pick_value(row, ["time", "arrival_time", "relative_time", "time_sec", "seconds"], None)

    try:
        score = float(score_raw)
    except Exception:
        score = None

    try:
        sample = int(float(sample_raw))
    except Exception:
        sample = None

    try:
        relative_time = float(time_raw)
    except Exception:
        relative_time = None

    if not phase:
        row_text = " ".join(f"{k}={v}" for k, v in row.items()).lower()
        if "phase" in row_text and "p" in row_text:
            phase = "P"
        elif "phase" in row_text and "s" in row_text:
            phase = "S"

    return {
        "method": method,
        "phase": phase,
        "score": score,
        "sample": sample,
        "relative_time": relative_time,
    }


def plot_trace_picks_standard(picks_csv, waveform_path, trace_index, output_png):
    df = pd.read_csv(picks_csv)
    st = read(waveform_path)
    if len(st) <= int(trace_index):
        raise IndexError(f"trace_index {trace_index} out of range for waveform stream of size {len(st)}")

    tr = st[int(trace_index)]
    sr = float(tr.stats.sampling_rate)
    y = np.asarray(tr.data, dtype=float)
    x = np.arange(len(y)) / sr

    trace_df = df[df[\"trace_index\"].astype(int) == int(trace_index)].copy()
    if trace_df.empty:
        raise ValueError(f"no picks found for trace_index={trace_index}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={\"height_ratios\": [3, 1]},
        constrained_layout=True,
    )

    ax1.plot(x, y, color=\"black\", linewidth=0.7)
    ax1.axhline(0, color=\"0.75\", linewidth=0.7)

    for _, row in trace_df.iterrows():
        parsed = parse_pick_row(row)
        phase = parsed["phase"]
        method = parsed["method"]
        score = parsed["score"]
        sample = parsed["sample"]
        relative_time = parsed["relative_time"]
        if sample is not None:
            t = sample / sr if sr else 0.0
        elif relative_time is not None:
            t = float(relative_time)
        else:
            continue

        color = \"red\" if phase.startswith(\"P\") else \"blue\"
        marker = \"^\" if phase.startswith(\"P\") else \"o\"
        alpha = 0.9 if score is not None and score >= 0.5 else 0.4
        linestyle = \"--\" if \"eqtransformer\" in method.lower() else \":\"
        label_score = "NA" if score is None else f"{score:.2f}"

        ax1.axvline(t, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.1)
        ax1.scatter([t], [0], color=color, marker=marker, s=48, alpha=alpha, zorder=5)
        ax2.scatter([t], [0 if score is None else score], color=color, marker=marker, s=64, alpha=alpha, label=f\"{method}-{phase or 'UNK'} ({label_score})\")

    ax1.set_ylabel(\"Amplitude (counts)\")
    ax1.set_title(f\"Trace {trace_index} ({tr.id}) - Phase Picking\")
    ax1.grid(True, alpha=0.22)

    ax2.axhline(0.5, color=\"0.55\", linestyle=\"--\", linewidth=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel(\"Confidence\")
    ax2.set_xlabel(\"Time relative to trace start (s)\")
    ax2.grid(True, alpha=0.22)

    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc=\"upper right\", fontsize=8)

    fig.savefig(output_png, dpi=220, bbox_inches=\"tight\")
    plt.close(fig)
"""


@dataclass(frozen=True)
class WorkspaceContext:
    project_root: str
    data_root: str
    outputs_dir: str
    runtime_results_path: str
    context_path: str
    plot_helper_path: str


class OpenCodeAdminRuntime:
    """
    Invokes the real opencode CLI (https://github.com/anomalyco/opencode)
    as a subprocess for high-capability post-processing and analysis.
    """

    def execute(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any] | None = None,
        model: str = "deepseek/deepseek-v4-flash",
        timeout_seconds: int = 600,
        workdir: str | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        import time as _time

        runtime_results = runtime_results or {}
        workspace = Path(workdir) if workdir else self._make_workspace(session_id)
        compact = self._compact_runtime_results(runtime_results)
        context = self._prepare_workspace(workspace, compact)
        max_attempts = max(1, int(os.getenv("QUAKECORE_OPENCODE_MAX_ATTEMPTS", "2")))
        retry_reason = ""
        last_result = {
            "success": False,
            "message": self._public_label("QuakeCore did not run"),
            "data": {},
            "artifacts": [],
            "opencode_admin": True,
        }

        for attempt in range(1, max_attempts + 1):
            prompt = self._build_opencode_prompt(
                message=message,
                runtime_results=compact,
                context=context,
                is_workspace=workdir is None,
                retry_reason=retry_reason,
            )
            env = os.environ.copy()
            env["OPENCODE_USE_BUILTIN_TOOLS"] = "1"
            scan_start = _time.time()

            try:
                proc = subprocess.run(
                    [
                        OPENCODE_BIN,
                        "run",
                        "--format",
                        "json",
                        "--model",
                        model,
                        "--dangerously-skip-permissions",
                        prompt,
                    ],
                    cwd=str(workspace),
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "message": self._public_label(f"QuakeCore execution timed out after {timeout_seconds}s."),
                    "data": {"timeout_seconds": timeout_seconds, "attempt": attempt},
                    "artifacts": [],
                    "opencode_admin": True,
                }
            except Exception as exc:
                return {
                    "success": False,
                    "message": self._public_label(f"QuakeCore invocation failed: {exc}"),
                    "data": {"attempt": attempt},
                    "artifacts": [],
                    "opencode_admin": True,
                }

            last_result = self._parse_opencode_output(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                scan_since=scan_start,
                workspace=workspace,
                message=message,
            )
            last_result.setdefault("data", {})
            last_result["data"]["attempt"] = attempt
            if last_result.get("success") is True:
                return last_result
            retry_reason = str(last_result.get("message") or "validation failed")

        return last_result

    def stream_execute(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any] | None = None,
        model: str = "deepseek/deepseek-v4-flash",
        timeout_seconds: int = 600,
        session_id: str = "default",
    ):
        import time as _time

        runtime_results = runtime_results or {}
        workspace = self._make_workspace(session_id)
        compact = self._compact_runtime_results(runtime_results)
        context = self._prepare_workspace(workspace, compact)
        max_attempts = max(1, int(os.getenv("QUAKECORE_OPENCODE_MAX_ATTEMPTS", "2")))
        retry_reason = ""
        last_result = {
            "success": False,
            "message": self._public_label("QuakeCore did not run"),
            "data": {},
            "artifacts": [],
            "opencode_admin": True,
        }

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                yield {
                    "type": "progress",
                    "event": {
                        "type": "retry",
                        "status": "running",
                        "summary": self._compact_summary(f"QuakeCore retry {attempt}/{max_attempts}"),
                        "detail": self._compact_detail(retry_reason),
                        "timestamp": int(_time.time() * 1000),
                    },
                }

            prompt = self._build_opencode_prompt(
                message=message,
                runtime_results=compact,
                context=context,
                is_workspace=True,
                retry_reason=retry_reason,
            )
            env = os.environ.copy()
            env["OPENCODE_USE_BUILTIN_TOOLS"] = "1"
            scan_start = _time.time()

            try:
                debug_enabled = os.getenv("QUAKECORE_DEBUG_OPENCODE_STREAM") == "1"
                debug_stream = None
                if debug_enabled:
                    debug_stream = (workspace / "stream_debug.jsonl").open("a", encoding="utf-8")
                proc = subprocess.Popen(
                    [
                        OPENCODE_BIN,
                        "run",
                        "--format",
                        "json",
                        "--model",
                        model,
                        "--dangerously-skip-permissions",
                        prompt,
                    ],
                    cwd=str(workspace),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    bufsize=1,
                )
            except Exception as exc:
                yield {"type": "error", "message": self._public_label(f"QuakeCore invocation failed: {exc}")}
                return

            final_message = ""
            total_tokens = 0
            total_cost = 0.0
            written_files: set[str] = set()
            progress_events: list[dict[str, Any]] = []

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_event = json.loads(line)
                    except Exception:
                        if debug_stream is not None:
                            debug_stream.write(json.dumps({"raw_line": line, "parse_error": "invalid_json"}, ensure_ascii=False) + "\n")
                            debug_stream.flush()
                        continue
                    if debug_stream is not None:
                        debug_stream.write(json.dumps(raw_event, ensure_ascii=False, default=str) + "\n")
                        debug_stream.flush()

                    parsed = self._parse_single_event(raw_event, written_files=written_files)
                    if parsed is None:
                        continue
                    tokens_info = parsed.pop("_tokens", None)
                    if tokens_info:
                        total_tokens += int(tokens_info.get("total", 0))
                        total_cost += float(tokens_info.get("cost", 0.0))
                    text_content = parsed.pop("_text", None)
                    if text_content:
                        final_message = text_content
                    progress_events.append(parsed)
                    yield {"type": "progress", "event": parsed}

                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                yield {"type": "error", "message": self._public_label(f"QuakeCore execution timed out after {timeout_seconds}s.")}
                return
            except Exception as exc:
                proc.kill()
                yield {"type": "error", "message": self._public_label(f"QuakeCore streaming failed: {exc}")}
                return
            finally:
                if debug_stream is not None:
                    debug_stream.close()

            artifacts = self._scan_artifacts(written_files, scan_start, workspace=workspace)
            final_message = self._clean_image_contradiction(final_message, artifacts)
            last_result = {
                "success": proc.returncode == 0,
                "message": self._public_label(final_message or ("QuakeCore 已完成分析。" if proc.returncode == 0 else "QuakeCore exited with error")),
                "data": {
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "progress_events": progress_events,
                    "event_count": len(progress_events),
                    "attempt": attempt,
                },
                "artifacts": artifacts,
                "opencode_admin": True,
            }
            if proc.returncode != 0:
                last_result["data"]["returncode"] = proc.returncode

            is_valid, validation_reason = self._validate_artifacts_for_request(
                message=message,
                artifacts=artifacts,
                workspace=workspace,
            )
            if not is_valid:
                last_result["success"] = False
                last_result["message"] = validation_reason
                last_result["data"]["validation_error"] = validation_reason

            if last_result.get("success") is True:
                yield {"type": "final", "result": last_result}
                return

            retry_reason = str(last_result.get("message") or "validation failed")

        yield {"type": "final", "result": last_result}

    def _prepare_workspace(self, workspace: Path, runtime_results: dict[str, Any]) -> WorkspaceContext:
        workspace.mkdir(parents=True, exist_ok=True)
        outputs_dir = workspace / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        runtime_results_path = workspace / "runtime_results.json"
        runtime_results_path.write_text(
            json.dumps(runtime_results, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        plot_helper_path = workspace / PLOT_HELPERS_FILENAME
        plot_helper_path.write_text(PLOT_HELPERS_CONTENT, encoding="utf-8")
        context_path = workspace / "quakecore_context.json"
        context = WorkspaceContext(
            project_root=str(Path.cwd().resolve()),
            data_root=str((Path.cwd() / "data").resolve()),
            outputs_dir=str(outputs_dir.resolve()),
            runtime_results_path=str(runtime_results_path.resolve()),
            context_path=str(context_path.resolve()),
            plot_helper_path=str(plot_helper_path.resolve()),
        )
        context_path.write_text(
            json.dumps(
                {
                    "project_root": context.project_root,
                    "data_root": context.data_root,
                    "outputs_dir": context.outputs_dir,
                    "runtime_results_path": context.runtime_results_path,
                    "plot_helper_path": context.plot_helper_path,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return context

    @staticmethod
    def _make_workspace(session_id: str) -> Path:
        import time as _time

        ts = str(int(_time.time() * 1_000_000))
        ws = Path.cwd() / "data" / "opencode_runtime" / session_id / ts
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    @staticmethod
    def _compact_runtime_results(runtime_results: dict[str, Any]) -> dict[str, Any]:
        keep_keys = [
            "active_file",
            "last_current_file",
            "last_miniseed_file",
            "last_picks_csv",
            "last_picks_image",
            "last_catalog_csv",
            "last_catalog_json",
            "last_location_image",
            "last_converted_file",
            "last_uploaded_files",
            "last_artifacts",
            "last_visible_artifacts",
            "files",
        ]
        compact = {k: runtime_results[k] for k in keep_keys if k in runtime_results}
        for key in ("last_artifacts", "last_visible_artifacts"):
            if isinstance(compact.get(key), list):
                compact[key] = compact[key][-8:]
        return compact

    def _build_opencode_prompt(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
        context: WorkspaceContext,
        is_workspace: bool = False,
        retry_reason: str = "",
    ) -> str:
        runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)[:4000]
        try:
            skill_context = SkillsPromptService().build_skill_context(max_skills=8)
        except Exception:
            skill_context = ""

        task_hint = self._build_task_hint(message)
        skill_block = f"\n\n【QuakeCore Skills — 必须遵守】\n{skill_context}" if skill_context else ""

        if is_workspace:
            workspace_guide = (
                "你当前在任务 workspace 中，不要搜索整个项目。\n"
                f"必须先读取 {Path(context.context_path).name}。\n"
                f"runtime_results 在 {Path(context.runtime_results_path).name}。\n"
                f"所有图像、CSV、JSON、PDF 输出必须写入 outputs_dir 绝对路径：{context.outputs_dir}\n"
                "不要写相对路径 data/plots/xxx.png，也不要把结果留在 workspace 根目录。\n"
                f"单道/多道拾取可视化优先 import {Path(context.plot_helper_path).stem} 并调用 plot_trace_picks_standard。\n"
                "如果是拾取图，必须读取 waveform，不要只画 picks。\n"
            )
        else:
            workspace_guide = "重要：所有输出文件请写到 data/ 目录下，例如 data/analysis/ 或 data/plots/。\n"

        retry_block = ""
        if retry_reason:
            retry_block = (
                "\n\n上一次结果校验失败，必须修复后重试：\n"
                f"- {retry_reason}\n"
                "- 如果用户要求图像，必须生成 image artifact。\n"
                "- 如果是拾取图，必须读取 waveform，不要只画 picks。\n"
                "- 图片不能是空白图，不能只有坐标轴或图例。\n"
            )

        return f"""你是 QuakeCore 地震数据处理平台的后处理 Agent。

你的任务是基于已有的运行结果，完成用户提出的后处理请求。
这些后处理可能包括：统计分析、绘图、表格生成、文件转换、GMT/PyGMT 制图、结果解释等。

{workspace_guide}
你有完全的文件系统访问权限：
- 使用 Python、bash 或任何已安装的工具
- 安装缺失的 Python 包（pip install）
- 生成 PNG、CSV、JSON、PDF 等输出文件
- 使用 matplotlib、obspy、pandas、pygmt 等库
{skill_block}
当前会话的已有运行结果（runtime_results）：
{runtime_preview}
工作目录上下文：
- project_root: {context.project_root}
- data_root: {context.data_root}
- outputs_dir: {context.outputs_dir}
- runtime_results_path: {context.runtime_results_path}
- plot_helper_path: {context.plot_helper_path}

请根据这些路径读取已有的数据文件，完成用户请求。
{task_hint}
{retry_block}
用户请求：
{message}

请直接完成任务，完成后说明你做了什么。

注意：如果你生成了图片，不要说"无法显示图像"。前端会自动显示 artifact。你只需说"图像已生成，见下方结果"。不要在正文中重复本地绝对路径。""".strip()

    @staticmethod
    def _build_task_hint(message: str) -> str:
        text = str(message or "").lower()
        if ("第二道" in message or "trace" in text or "轨迹" in text or "道" in text) and (
            "拾取" in text or "pick" in text
        ):
            return "这是单道拾取可视化任务，优先使用 last_picks_csv 和 last_miniseed_file，不要搜索整个项目。"
        return ""

    @staticmethod
    def _parse_single_event(
        raw_event: dict[str, Any],
        *,
        written_files: set[str] | None = None,
    ) -> dict[str, Any] | None:
        event_type = raw_event.get("type", "")
        timestamp = raw_event.get("timestamp", 0)
        part = raw_event.get("part", {})

        if event_type == "step_start":
            return {
                "type": "step",
                "status": "running",
                "summary": OpenCodeAdminRuntime._compact_summary("Agent is thinking..."),
                "timestamp": timestamp,
            }

        if event_type == "step_finish":
            reason = part.get("reason", "")
            tokens = part.get("tokens", {})
            return {
                "type": "step",
                "status": "completed",
                "summary": OpenCodeAdminRuntime._compact_summary(f"Step completed ({reason})"),
                "detail": OpenCodeAdminRuntime._compact_detail(
                    f"tokens={tokens.get('total', 0)}, cost=${tokens.get('cost', 0):.6f}" if tokens.get("total") else ""
                ),
                "timestamp": timestamp,
                "_tokens": tokens,
            }

        if event_type == "text":
            text = part.get("text", "")
            if text:
                return {
                    "type": "text",
                    "icon": "message",
                    "status": "completed",
                    "summary": OpenCodeAdminRuntime._compact_summary(text),
                    "timestamp": timestamp,
                    "_text": text,
                }
            return None

        if event_type == "tool_use":
            tool = part.get("tool", "")
            state = part.get("state", {})
            tool_input = state.get("input", {})
            tool_output = state.get("output", "")
            tool_title = state.get("title", "")
            icon_map = {
                "bash": "terminal",
                "read": "file-eye",
                "write": "file-edit",
                "edit": "file-edit",
                "glob": "search",
                "grep": "search",
                "codesearch": "search",
                "webfetch": "globe",
                "websearch": "globe",
                "task": "cog",
                "question": "help-circle",
                "todo": "list",
                "todowrite": "list",
            }
            icon = icon_map.get(tool, "tool")

            if tool == "bash":
                summary = tool_input.get("description", "") or tool_input.get("command", "")[:120]
            elif tool == "write":
                filepath = tool_input.get("filePath", "")
                summary = f"Save: {Path(filepath).name}" if filepath else "Save file"
                if filepath and written_files is not None:
                    written_files.add(filepath)
            elif tool == "read":
                filepath = tool_input.get("filePath", "")
                summary = f"Read: {Path(filepath).name}" if filepath else "Read file"
            elif tool == "grep":
                pattern = tool_input.get("pattern", "")
                summary = f"Search: {pattern[:80]}"
            elif tool == "glob":
                pattern = tool_input.get("pattern", "")
                summary = f"Find: {pattern[:80]}"
            else:
                summary = tool_title or f"{tool}"

            return {
                "type": "tool_use",
                "tool": tool,
                "icon": icon,
                "status": "completed",
                "summary": OpenCodeAdminRuntime._compact_summary(summary),
                "detail": OpenCodeAdminRuntime._compact_detail(tool_output or ""),
                "timestamp": timestamp,
            }

        return {
            "type": "raw",
            "status": "completed",
            "summary": OpenCodeAdminRuntime._compact_summary(f"OpenCode event: {event_type or 'unknown'}"),
            "detail": OpenCodeAdminRuntime._compact_detail(json.dumps(raw_event, ensure_ascii=False, default=str)),
            "timestamp": timestamp,
        }

    @staticmethod
    def _compact_summary(summary: str) -> str:
        return re.sub(r"\s+", " ", OpenCodeAdminRuntime._public_label(str(summary or ""))).strip()[:160]

    @staticmethod
    def _compact_detail(detail: str) -> str:
        return OpenCodeAdminRuntime._public_label(str(detail or ""))[:500]

    @staticmethod
    def _public_label(text: str) -> str:
        value = str(text or "")
        replacements = {
            "OpenCode Admin": "QuakeCore",
            "OpenCode": "QuakeCore",
            "opencode_admin": "quakecore_runtime",
            "opencode": "QuakeCore",
        }
        for src, dst in replacements.items():
            value = value.replace(src, dst)
        return value

    def _parse_opencode_output(
        self,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        scan_since: float | None = None,
        workspace: Path | None = None,
        message: str = "",
    ) -> dict[str, Any]:
        final_message = ""
        total_tokens = 0
        total_cost = 0.0
        written_files: set[str] = set()
        progress_events: list[dict[str, Any]] = []

        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                raw_event = json.loads(line)
            except Exception:
                continue

            parsed = self._parse_single_event(raw_event, written_files=written_files)
            if parsed is None:
                continue

            tokens_info = parsed.pop("_tokens", None)
            if tokens_info:
                total_tokens += int(tokens_info.get("total", 0))
                total_cost += float(tokens_info.get("cost", 0.0))

            text_content = parsed.pop("_text", None)
            if text_content:
                final_message = text_content

            progress_events.append(parsed)

        artifacts = self._scan_artifacts(written_files, scan_since, workspace=workspace)
        final_message = self._clean_image_contradiction(final_message, artifacts)
        result: dict[str, Any]
        data = {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "progress_events": progress_events,
            "event_count": len(progress_events),
        }

        if returncode != 0:
            result = {
                "success": False,
                "message": self._public_label(stderr[-3000:] or "QuakeCore exited with error"),
                "data": {**data, "returncode": returncode},
                "artifacts": artifacts,
                "opencode_admin": True,
            }
        else:
            result = {
                "success": True,
                "message": self._public_label(final_message or "QuakeCore 已完成分析。"),
                "data": data,
                "artifacts": artifacts,
                "opencode_admin": True,
            }

        is_valid, validation_reason = self._validate_artifacts_for_request(
            message=message,
            artifacts=artifacts,
            workspace=workspace,
        )
        if not is_valid:
            result["success"] = False
            result["message"] = validation_reason
            result.setdefault("data", {})["validation_error"] = validation_reason
        return result

    @staticmethod
    def _scan_artifacts(
        written_files: set[str],
        scan_since: float | None = None,
        workspace: Path | None = None,
    ) -> list[dict[str, str]]:
        from backend.services.artifact_utils import make_artifact

        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
        file_suffixes = {
            ".csv", ".json", ".txt", ".xlsx", ".xls", ".pdf", ".ps", ".eps",
            ".nc", ".grd", ".h5", ".hdf5", ".npy", ".npz", ".mseed",
            ".miniseed", ".sac", ".shp", ".geojson", ".kml",
        }
        artifacts: list[dict[str, str]] = []
        seen: set[str] = set()

        for fpath in written_files:
            real_path = Path(fpath)
            if not real_path.is_absolute():
                base = workspace if workspace else Path.cwd()
                real_path = base / fpath
            if not real_path.exists():
                continue
            key = str(real_path)
            if key in seen:
                continue
            suffix = real_path.suffix.lower()
            art_type = "image" if suffix in image_suffixes else "file" if suffix in file_suffixes else None
            if art_type:
                seen.add(key)
                artifacts.append(make_artifact(str(real_path), art_type))

        if scan_since is not None:
            scan_roots: list[Path] = []
            if workspace and workspace.exists():
                scan_roots.append(workspace)
            if workspace and (workspace / "outputs").exists():
                scan_roots.append(workspace / "outputs")
            scan_roots.extend([Path.cwd() / "data" / "plots", Path.cwd() / "data" / "analysis"])

            for scan_root in scan_roots:
                if not scan_root.exists():
                    continue
                iterator = scan_root.iterdir() if workspace and scan_root == workspace else scan_root.rglob("*")
                for path in iterator:
                    if not path.is_file():
                        continue
                    if path.name in {"runtime_results.json", "quakecore_context.json", PLOT_HELPERS_FILENAME, "stream_debug.jsonl"}:
                        continue
                    if path.stat().st_mtime < scan_since:
                        continue
                    fpath = str(path)
                    if fpath in seen:
                        continue
                    suffix = path.suffix.lower()
                    art_type = "image" if suffix in image_suffixes else "file" if suffix in file_suffixes else None
                    if art_type:
                        seen.add(fpath)
                        artifacts.append(make_artifact(fpath, art_type))

        return artifacts

    def _run_streaming_attempt(
        self,
        *,
        workspace: Path,
        prompt: str,
        model: str,
        timeout_seconds: int,
        env: dict[str, str],
        scan_start: float,
        message: str,
    ) -> dict[str, Any]:
        debug_enabled = os.getenv("QUAKECORE_DEBUG_OPENCODE_STREAM") == "1"
        debug_stream = None
        try:
            if debug_enabled:
                debug_stream = (workspace / "stream_debug.jsonl").open("a", encoding="utf-8")
            proc = subprocess.Popen(
                [
                    OPENCODE_BIN,
                    "run",
                    "--format",
                    "json",
                    "--model",
                    model,
                    "--dangerously-skip-permissions",
                    prompt,
                ],
                cwd=str(workspace),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
            )
        except Exception:
            if debug_stream is not None:
                debug_stream.close()
            raise

        final_message = ""
        total_tokens = 0
        total_cost = 0.0
        written_files: set[str] = set()
        progress_events: list[dict[str, Any]] = []

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_event = json.loads(line)
                except Exception:
                    if debug_stream is not None:
                        debug_stream.write(json.dumps({"raw_line": line, "parse_error": "invalid_json"}, ensure_ascii=False) + "\n")
                        debug_stream.flush()
                    continue
                if debug_stream is not None:
                    debug_stream.write(json.dumps(raw_event, ensure_ascii=False, default=str) + "\n")
                    debug_stream.flush()
                parsed = self._parse_single_event(raw_event, written_files=written_files)
                if parsed is None:
                    continue
                tokens_info = parsed.pop("_tokens", None)
                if tokens_info:
                    total_tokens += int(tokens_info.get("total", 0))
                    total_cost += float(tokens_info.get("cost", 0.0))
                text_content = parsed.pop("_text", None)
                if text_content:
                    final_message = text_content
                progress_events.append(parsed)
            proc.wait(timeout=timeout_seconds)
        finally:
            if debug_stream is not None:
                debug_stream.close()

        artifacts = self._scan_artifacts(written_files, scan_start, workspace=workspace)
        final_message = self._clean_image_contradiction(final_message, artifacts)
        result: dict[str, Any]
        if proc.returncode != 0:
            result = {
                "success": False,
                "message": self._public_label(final_message[-3000:] or "QuakeCore exited with error"),
                "data": {
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "progress_events": progress_events,
                    "event_count": len(progress_events),
                    "returncode": proc.returncode,
                },
                "artifacts": artifacts,
                "opencode_admin": True,
            }
        else:
            result = {
                "success": True,
                "message": self._public_label(final_message or "QuakeCore 已完成分析。"),
                "data": {
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "progress_events": progress_events,
                    "event_count": len(progress_events),
                },
                "artifacts": artifacts,
                "opencode_admin": True,
            }

        is_valid, validation_reason = self._validate_artifacts_for_request(
            message=message,
            artifacts=artifacts,
            workspace=workspace,
        )
        if not is_valid:
            result["success"] = False
            result["message"] = validation_reason
            result["data"]["validation_error"] = validation_reason
        return result

    def _validate_artifacts_for_request(
        self,
        *,
        message: str,
        artifacts: list[dict[str, str]],
        workspace: Path | None,
    ) -> tuple[bool, str]:
        text = str(message or "").lower()
        wants_plot = any(token in text for token in ("图", "绘", "plot", "可视化", "拾取情况"))
        image_artifacts = [item for item in artifacts if item.get("type") == "image"]

        if wants_plot and not image_artifacts:
            return False, "用户要求图像，但没有生成 image artifact。"

        for item in image_artifacts:
            real_path = self._artifact_to_real_path(item, workspace=workspace)
            if real_path is None or not real_path.exists():
                return False, f"图片 artifact 路径不存在: {item.get('path') or item.get('name') or 'unknown'}"
            if real_path.stat().st_size < 20_000:
                return False, f"图片文件过小，可能为空图: {real_path.name}"
            try:
                from PIL import Image, ImageStat  # type: ignore

                img = Image.open(real_path).convert("L")
                stat = ImageStat.Stat(img)
                if stat.stddev and stat.stddev[0] < 3:
                    return False, f"图片接近空白: {real_path.name}"
            except Exception:
                pass

        return True, ""

    @staticmethod
    def _artifact_to_real_path(item: dict[str, str], *, workspace: Path | None) -> Path | None:
        raw = str(item.get("path") or item.get("url") or item.get("name") or "")
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        candidate = Path.cwd() / "data" / raw
        if candidate.exists():
            return candidate
        if workspace is not None:
            ws_candidate = workspace / raw
            if ws_candidate.exists():
                return ws_candidate
            outputs_candidate = workspace / "outputs" / Path(raw).name
            if outputs_candidate.exists():
                return outputs_candidate
        return candidate

    @staticmethod
    def _clean_image_contradiction(message: str, artifacts: list[dict[str, str]]) -> str:
        text = str(message or "").strip()
        has_image = any(item.get("type") == "image" for item in artifacts)
        if not has_image:
            return text

        patterns = [
            r"该模型无法显示图像[，,。；;]?\s*",
            r"模型无法显示图像[，,。；;]?\s*",
            r"无法直接显示图像[，,。；;]?\s*",
            r"不能直接显示图像[，,。；;]?\s*",
            r"但我已生成并保存了图片[，,。；;]?\s*",
            r"图片已保存至\s*`?[^`，。；;\n]+`?[，,。；;]?\s*",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
