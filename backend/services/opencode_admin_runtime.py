from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from backend.services.skills_prompt_service import SkillsPromptService


OPENCODE_BIN = os.getenv("OPENCODE_BIN", "opencode")


class OpenCodeAdminRuntime:
    """
    Invokes the real opencode CLI (https://github.com/anomalyco/opencode)
    as a subprocess for high-capability post-processing and analysis.

    opencode is an AI coding agent that can read files, run bash commands,
    write outputs, generate plots, and more — all within the QuakeCore project
    working directory.
    """

    def execute(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any] | None = None,
        model: str = "deepseek/deepseek-v4-flash",
        timeout_seconds: int = 300,
        workdir: str | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        Run opencode to handle a QuakeCore post-processing task.

        Returns a dict with keys:
          - success: bool
          - message: str
          - data: dict
          - artifacts: list of {type, name, path, url}
        """
        import time as _time
        runtime_results = runtime_results or {}

        # Create lightweight per-task workspace so opencode doesn't scan the whole repo
        if workdir:
            cwd = workdir
            workspace = Path(cwd)
        else:
            workspace = self._make_workspace(session_id)
            cwd = str(workspace)

        # Write compact runtime_results to workspace
        compact = self._compact_runtime_results(runtime_results)
        runtime_json_path = workspace / "runtime_results.json"
        runtime_json_path.write_text(
            json.dumps(compact, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (workspace / "outputs").mkdir(parents=True, exist_ok=True)

        prompt = self._build_opencode_prompt(
            message=message,
            runtime_results=compact,
            is_workspace=workdir is None,
        )

        env = os.environ.copy()
        env["OPENCODE_USE_BUILTIN_TOOLS"] = "1"

        _scan_start = _time.time()

        try:
            proc = subprocess.run(
                [
                    OPENCODE_BIN, "run",
                    "--format", "json",
                    "--model", model,
                    "--dangerously-skip-permissions",
                    prompt,
                ],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"opencode execution timed out after {timeout_seconds}s.",
                "data": {"timeout_seconds": timeout_seconds},
                "artifacts": [],
                "opencode_admin": True,
            }
        except Exception as exc:
            return {
                "success": False,
                "message": f"opencode invocation failed: {exc}",
                "data": {},
                "artifacts": [],
                "opencode_admin": True,
            }

        return self._parse_opencode_output(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            scan_since=_scan_start,
            workspace=workspace if not workdir else None,
        )

    def stream_execute(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any] | None = None,
        model: str = "deepseek/deepseek-v4-flash",
        timeout_seconds: int = 300,
        session_id: str = "default",
    ):
        """Stream opencode execution events as they arrive.

        Yields dicts with keys:
          - type: "progress" | "final" | "error"
          - For progress: event (the parsed progress event dict)
          - For final: result (the full result dict)
          - For error: message (error string)
        """
        import time as _time
        runtime_results = runtime_results or {}

        workspace = self._make_workspace(session_id)
        compact = self._compact_runtime_results(runtime_results)
        runtime_json_path = workspace / "runtime_results.json"
        runtime_json_path.write_text(
            json.dumps(compact, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (workspace / "outputs").mkdir(parents=True, exist_ok=True)

        prompt = self._build_opencode_prompt(
            message=message,
            runtime_results=compact,
            is_workspace=True,
        )

        env = os.environ.copy()
        env["OPENCODE_USE_BUILTIN_TOOLS"] = "1"

        _scan_start = _time.time()

        try:
            debug_enabled = os.getenv("QUAKECORE_DEBUG_OPENCODE_STREAM") == "1"
            debug_stream = None
            if debug_enabled:
                debug_stream = (workspace / "stream_debug.jsonl").open("a", encoding="utf-8")
            proc = subprocess.Popen(
                [
                    OPENCODE_BIN, "run",
                    "--format", "json",
                    "--model", model,
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
            yield {"type": "error", "message": f"opencode invocation failed: {exc}"}
            return

        final_message = ""
        total_tokens = 0
        total_cost = 0.0
        written_files: set[str] = set()
        progress_events: list[dict[str, Any]] = []

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_event = json.loads(line)
                except Exception:
                    if debug_stream is not None:
                        debug_stream.write(
                            json.dumps({"raw_line": line, "parse_error": "invalid_json"}, ensure_ascii=False) + "\n"
                        )
                        debug_stream.flush()
                    continue
                if debug_stream is not None:
                    debug_stream.write(json.dumps(raw_event, ensure_ascii=False, default=str) + "\n")
                    debug_stream.flush()

                parsed = self._parse_single_event(raw_event, written_files=written_files)
                if parsed is None:
                    continue

                # Track tokens and final message
                tokens_info = parsed.pop("_tokens", None)
                if tokens_info:
                    total_tokens += int(tokens_info.get("total", 0))
                    total_cost += float(tokens_info.get("cost", 0.0))

                text_content = parsed.pop("_text", None)
                if text_content:
                    final_message = text_content

                progress_events.append(parsed)
                yield {"type": "progress", "event": parsed}

            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            yield {
                "type": "error",
                "message": f"opencode execution timed out after {timeout_seconds}s.",
            }
            return
        except Exception as exc:
            proc.kill()
            yield {"type": "error", "message": f"opencode streaming failed: {exc}"}
            return
        finally:
            if debug_stream is not None:
                debug_stream.close()

        # Build final result
        artifacts = self._scan_artifacts(written_files, _scan_start, workspace=workspace)
        final_message = self._clean_image_contradiction(final_message, artifacts)

        if proc.returncode != 0:
            result = {
                "success": False,
                "message": final_message[-3000:] or "opencode exited with error",
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
                "message": final_message or "opencode 已完成分析。",
                "data": {
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "progress_events": progress_events,
                    "event_count": len(progress_events),
                },
                "artifacts": artifacts,
                "opencode_admin": True,
            }

        yield {"type": "final", "result": result}

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

        # artifacts 只保留最近 8 个
        for key in ("last_artifacts", "last_visible_artifacts"):
            if isinstance(compact.get(key), list):
                compact[key] = compact[key][-8:]

        return compact

    def _build_opencode_prompt(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
        is_workspace: bool = False,
    ) -> str:
        runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)[:4000]

        # Inject QuakeCore skills into the OpenCode prompt
        try:
            skill_context = SkillsPromptService().build_skill_context(max_skills=8)
        except Exception:
            skill_context = ""

        # Provide task-specific hints to avoid unnecessary exploration
        task_hint = self._build_task_hint(message)

        skill_block = f"\n\n【QuakeCore Skills — 必须遵守】\n{skill_context}" if skill_context else ""

        if is_workspace:
            workspace_guide = (
                "你当前在任务 workspace 中，不要搜索整个项目。\n"
                "runtime_results 在 runtime_results.json。\n"
                "输出必须写到 outputs/ 目录。\n"
                "如需读取原始数据文件，根据 runtime_results 中的路径，"
                "通过 ../ 访问项目根目录下的 data/ 文件。\n"
            )
        else:
            workspace_guide = (
                "重要：所有输出文件请写到 data/ 目录下，例如 data/analysis/ 或 data/plots/。\n"
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

请根据这些路径读取已有的数据文件，完成用户请求。
{task_hint}
用户请求：
{message}

请直接完成任务，完成后说明你做了什么。

注意：如果你生成了图片，不要说"无法显示图像"。前端会自动显示 artifact。你只需说"图像已生成，见下方结果"。不要在正文中重复图片路径。""".strip()

    @staticmethod
    def _build_task_hint(message: str) -> str:
        text = str(message or "").lower()
        if "第二道" in message or "trace" in text or "轨迹" in text or "道" in text:
            if "拾取" in text or "pick" in text:
                return "这是单道拾取可视化任务，优先使用 last_picks_csv 和 last_miniseed_file，不要搜索整个项目。"
        return ""

    @staticmethod
    def _parse_single_event(
        raw_event: dict[str, Any],
        *,
        written_files: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Parse one opencode JSON-line event into a normalized progress event dict.
        Returns None if the raw event should be skipped.
        Mutates written_files in-place when a write tool is detected.
        """
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
                if filepath and "data/" in filepath and written_files is not None:
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
        return re.sub(r"\s+", " ", str(summary or "")).strip()[:160]

    @staticmethod
    def _compact_detail(detail: str) -> str:
        return str(detail or "")[:500]

    def _parse_opencode_output(
        self,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        scan_since: float | None = None,
        workspace: Path | None = None,
    ) -> dict[str, Any]:
        final_message = ""
        data: dict[str, Any] = {}
        total_tokens = 0
        total_cost = 0.0
        written_files: set[str] = set()
        progress_events: list[dict[str, Any]] = []

        lines = stdout.strip().split("\n")
        for line in lines:
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

            # Accumulate tokens from step_finish events
            tokens_info = parsed.pop("_tokens", None)
            if tokens_info:
                total_tokens += int(tokens_info.get("total", 0))
                total_cost += float(tokens_info.get("cost", 0.0))

            # Track final message from text events
            text_content = parsed.pop("_text", None)
            if text_content:
                final_message = text_content

            progress_events.append(parsed)

        # Scan for artifacts generated by opencode
        artifacts = self._scan_artifacts(written_files, scan_since, workspace=workspace)
        final_message = self._clean_image_contradiction(final_message, artifacts)

        data["total_tokens"] = total_tokens
        data["total_cost_usd"] = round(total_cost, 6)
        data["progress_events"] = progress_events
        data["event_count"] = len(progress_events)

        if returncode != 0:
            return {
                "success": False,
                "message": stderr[-3000:] or "opencode exited with error",
                "data": {**data, "returncode": returncode},
                "artifacts": artifacts,
                "opencode_admin": True,
            }

        return {
            "success": True,
            "message": final_message or "opencode 已完成分析。",
            "data": data,
            "artifacts": artifacts,
            "opencode_admin": True,
        }

    @staticmethod
    def _scan_artifacts(
        written_files: set[str],
        scan_since: float | None = None,
        workspace: Path | None = None,
    ) -> list[dict[str, str]]:
        from backend.services.artifact_utils import make_artifact

        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}
        file_suffixes = {".csv", ".json", ".txt", ".xlsx", ".xls", ".pdf", ".ps", ".eps",
                         ".nc", ".grd", ".h5", ".hdf5", ".npy", ".npz", ".mseed",
                         ".miniseed", ".sac", ".shp", ".geojson", ".kml"}
        artifacts: list[dict[str, str]] = []
        seen: set[str] = set()

        for fpath in written_files:
            if fpath in seen:
                continue
            real_path = Path(fpath)
            if not real_path.is_absolute():
                # If workspace is set, resolve relative to workspace; else cwd
                base = workspace if workspace else Path.cwd()
                real_path = base / fpath
            if not real_path.exists():
                continue
            seen.add(fpath)
            suffix = real_path.suffix.lower()
            art_type = "image" if suffix in image_suffixes else "file" if suffix in file_suffixes else None
            if art_type:
                artifacts.append(make_artifact(str(real_path), art_type))

        # Scan for recently modified files; scope to workspace/outputs first, then data/plots
        if scan_since is not None:
            scan_roots: list[Path] = []
            if workspace and (workspace / "outputs").exists():
                scan_roots.append(workspace / "outputs")
            scan_roots.extend([
                Path.cwd() / "data" / "plots",
                Path.cwd() / "data" / "analysis",
            ])

            for scan_root in scan_roots:
                if not scan_root.exists():
                    continue
                for path in scan_root.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.stat().st_mtime < scan_since:
                        continue
                    fpath = str(path)
                    if fpath in seen:
                        continue
                    suffix = path.suffix.lower()[:10]
                    art_type = "image" if suffix in image_suffixes else "file" if suffix in file_suffixes else None
                    if art_type:
                        seen.add(fpath)
                        artifacts.append(make_artifact(fpath, art_type))

        return artifacts

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
        text = text.strip()

        return text
