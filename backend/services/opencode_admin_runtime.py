from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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
        cwd = workdir or str(Path.cwd())

        prompt = self._build_opencode_prompt(
            message=message,
            runtime_results=runtime_results,
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
        )

    def _build_opencode_prompt(
        self,
        *,
        message: str,
        runtime_results: dict[str, Any],
    ) -> str:
        runtime_preview = json.dumps(runtime_results, ensure_ascii=False, default=str)[:8000]

        return f"""你是 QuakeCore 地震数据处理平台的后处理 Agent。

你的任务是基于已有的运行结果，完成用户提出的后处理请求。
这些后处理可能包括：统计分析、绘图、表格生成、文件转换、GMT/PyGMT 制图、结果解释等。

你有完全的文件系统访问权限，可以：
- 读取 QuakeCore 项目的任何文件
- 使用 Python、bash 或任何已安装的工具
- 安装缺失的 Python 包（pip install）
- 生成 PNG、CSV、JSON、PDF 等输出文件
- 使用 matplotlib、obspy、pandas、pygmt 等库

重要：所有输出文件请写到 data/ 目录下，例如 data/analysis/ 或 data/plots/。

当前会话的已有运行结果（runtime_results）：
{runtime_preview}

请根据这些路径读取已有的数据文件，完成用户请求。

用户请求：
{message}

请直接完成任务，完成后说明你做了什么。""".strip()

    def _parse_opencode_output(
        self,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
        scan_since: float | None = None,
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
                event = json.loads(line)
            except Exception:
                continue

            event_type = event.get("type", "")
            timestamp = event.get("timestamp", 0)
            part = event.get("part", {})

            if event_type == "step_start":
                progress_events.append({
                    "type": "step",
                    "status": "running",
                    "summary": "Agent is thinking...",
                    "timestamp": timestamp,
                })

            if event_type == "step_finish":
                reason = part.get("reason", "")
                tokens = part.get("tokens", {})
                total_tokens += int(tokens.get("total", 0))
                total_cost += float(tokens.get("cost", 0.0))
                progress_events.append({
                    "type": "step",
                    "status": "completed",
                    "summary": f"Step completed ({reason})",
                    "detail": f"tokens={tokens.get('total', 0)}, cost=${tokens.get('cost', 0):.6f}" if tokens.get("total") else "",
                    "timestamp": timestamp,
                })

            if event_type == "text":
                text = part.get("text", "")
                if text:
                    final_message = text or final_message
                    progress_events.append({
                        "type": "text",
                        "icon": "message",
                        "status": "completed",
                        "summary": text[:200],
                        "timestamp": timestamp,
                    })

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
                    if filepath and "data/" in filepath:
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

                progress_events.append({
                    "type": "tool_use",
                    "tool": tool,
                    "icon": icon,
                    "status": "completed",
                    "summary": summary[:200],
                    "detail": (tool_output or "")[:500],
                    "timestamp": timestamp,
                })

        # Scan data/ for artifacts generated by opencode
        artifacts = self._scan_artifacts(written_files, scan_since)

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
    def _scan_artifacts(written_files: set[str], scan_since: float | None = None) -> list[dict[str, str]]:
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
                real_path = Path.cwd() / fpath
            if not real_path.exists():
                continue
            seen.add(fpath)
            suffix = real_path.suffix.lower()
            art_type = "image" if suffix in image_suffixes else "file" if suffix in file_suffixes else None
            if art_type:
                artifacts.append(make_artifact(str(real_path), art_type))

        # Also scan data/ for recently modified files not captured via write tool
        if scan_since is not None:
            import time as _time
            data_dir = Path.cwd() / "data"
            for path in data_dir.rglob("*"):
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
