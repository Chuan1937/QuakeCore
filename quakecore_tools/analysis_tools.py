"""Analysis sandbox tool for lightweight post-workflow statistics and plots."""

from __future__ import annotations

import ast
import csv
import json
import multiprocessing as mp
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.tools import tool

from backend.services.artifact_utils import make_artifact, to_data_relative_path
from backend.services.session_store import get_session_store


DEFAULT_ANALYSIS_ROOT = Path("data/analysis")
_CODE_BLOCKED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "shutil",
    "socket",
    "requests",
}


def _parse_params(params: str | dict | None) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    text = str(params).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_relative_data_path(value: str | None) -> str:
    return to_data_relative_path(value)


def _resolve_input_path(parsed: dict[str, Any]) -> Path | None:
    input_path = str(parsed.get("input_path") or "").strip()
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = Path.cwd() / input_path
            if not path.exists():
                fallback = Path.cwd() / "data" / _to_relative_data_path(input_path)
                path = fallback
        return path.resolve()

    key = str(parsed.get("input_artifact_key") or parsed.get("input") or "").strip()
    session_id = str(parsed.get("session_id") or "").strip()
    if not key or not session_id:
        return None

    runtime = get_session_store().get_runtime_results(session_id)
    candidate = runtime.get(key)
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    rel = _to_relative_data_path(candidate)
    return (Path.cwd() / "data" / rel).resolve()


def _make_output_dir(session_id: str | None) -> Path:
    sid = str(session_id or "default").strip() or "default"
    output_dir = (DEFAULT_ANALYSIS_ROOT / sid).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _artifact(path: Path, artifact_type: str) -> dict[str, str]:
    return make_artifact(str(path), artifact_type, name=path.name)


def _plot_or_error() -> tuple[Any | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:
        return None, f"matplotlib unavailable: {exc}"


def _result(success: bool, message: str, data: dict[str, Any] | None = None, artifacts: list[dict[str, str]] | None = None) -> str:
    payload = {
        "success": bool(success),
        "message": message,
        "data": data or {},
        "artifacts": artifacts or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _code_execution_enabled(parsed: dict[str, Any]) -> bool:
    if str(parsed.get("allow_code", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        return True
    return str(os.getenv("QUAKECORE_ANALYSIS_ALLOW_CODE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_restricted_code(source: str) -> str | None:
    if len(source) > 6000:
        return "Code is too long (max 6000 chars)."
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return f"Code syntax error: {exc}"

    blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.Try,
        ast.Raise,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.While,
    )
    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            return f"Blocked syntax in code: {type(node).__name__}"
        if isinstance(node, ast.Attribute):
            if str(node.attr).startswith("__"):
                return "Dunder attribute access is blocked."
        if isinstance(node, ast.Name):
            name = str(node.id)
            if name.startswith("__") or name in _CODE_BLOCKED_NAMES:
                return f"Blocked name in code: {name}"
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                name = str(fn.id)
                if name in _CODE_BLOCKED_NAMES:
                    return f"Blocked function call: {name}"
    return None


def _run_restricted_code(
    *,
    code: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    input_path: Path,
    output_dir: Path,
    runtime_results: dict[str, Any] | None = None,
) -> str:
    error = _validate_restricted_code(code)
    if error:
        return _result(False, error)

    artifacts: list[dict[str, str]] = []
    data: dict[str, Any] = {}
    message_holder = {"message": "Analysis code executed."}
    runtime_payload = dict(runtime_results or {})
    data_root = (Path.cwd() / "data").resolve()

    def save_csv(name: str, table: list[dict[str, Any]] | list[list[Any]] | dict[str, Any]) -> str:
        filename = str(name or "").strip() or f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        out = (output_dir / filename).resolve()
        try:
            out.relative_to(output_dir.resolve())
        except ValueError:
            raise ValueError("save_csv path must stay within output_dir")

        if isinstance(table, dict):
            table = [table]
        if isinstance(table, list) and table and isinstance(table[0], dict):
            keys = list(table[0].keys())
            with out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in table:
                    writer.writerow(row)
        elif isinstance(table, list):
            with out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for row in table:
                    if isinstance(row, (list, tuple)):
                        writer.writerow(row)
                    else:
                        writer.writerow([row])
        else:
            raise ValueError("Unsupported table type for save_csv")

        artifacts.append(_artifact(out, "file"))
        return str(out)

    def save_plot(name: str = "analysis_plot.png") -> str:
        plt, err = _plot_or_error()
        if plt is None:
            raise RuntimeError(err or "matplotlib unavailable")
        filename = str(name or "").strip() or "analysis_plot.png"
        if not any(filename.lower().endswith(sfx) for sfx in (".png", ".jpg", ".jpeg", ".svg", ".webp")):
            filename += ".png"
        out = (output_dir / filename).resolve()
        try:
            out.relative_to(output_dir.resolve())
        except ValueError:
            raise ValueError("save_plot path must stay within output_dir")
        plt.gcf().savefig(out, dpi=180)
        plt.close(plt.gcf())
        artifacts.append(_artifact(out, "image"))
        return str(out)

    def set_message(text: str) -> None:
        message_holder["message"] = str(text or "").strip() or message_holder["message"]

    def set_data(key: str, value: Any) -> None:
        data[str(key)] = value

    def resolve_data_path(path_or_key: str) -> str:
        token = str(path_or_key or "").strip()
        if not token:
            raise ValueError("Empty path_or_key.")
        if token in runtime_payload:
            token = str(runtime_payload.get(token) or "").strip()
        if not token:
            raise ValueError("Runtime key is empty.")

        rel = _to_relative_data_path(token)
        resolved = (data_root / rel).resolve()
        try:
            resolved.relative_to(data_root)
        except ValueError as exc:
            raise ValueError("Path must stay inside data directory.") from exc
        if not resolved.exists():
            raise FileNotFoundError(f"File does not exist: {resolved}")
        return str(resolved)

    def read_csv(path_or_key: str) -> list[dict[str, Any]]:
        resolved = Path(resolve_data_path(path_or_key))
        with resolved.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def read_json(path_or_key: str) -> Any:
        resolved = Path(resolve_data_path(path_or_key))
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_waveform(path_or_key: str) -> Any:
        resolved = resolve_data_path(path_or_key)
        try:
            from obspy import read
        except Exception as exc:
            raise RuntimeError(f"ObsPy unavailable: {exc}") from exc
        return read(resolved)

    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "round": round,
        "float": float,
        "int": int,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": print,
    }

    local_ctx: dict[str, Any] = {
        "rows": rows,
        "columns": columns,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "runtime_results": runtime_payload,
        "resolve_data_path": resolve_data_path,
        "read_csv": read_csv,
        "read_json": read_json,
        "read_waveform": read_waveform,
        "save_csv": save_csv,
        "save_plot": save_plot,
        "set_message": set_message,
        "set_data": set_data,
    }
    try:
        import numpy as np

        local_ctx["np"] = np
    except Exception:
        pass
    try:
        plt, _ = _plot_or_error()
        if plt is not None:
            local_ctx["plt"] = plt
    except Exception:
        pass
    try:
        import pandas as pd

        local_ctx["pd"] = pd
    except Exception:
        pass

    try:
        exec(compile(code, "<analysis_sandbox>", "exec"), {"__builtins__": safe_builtins}, local_ctx)
    except Exception as exc:
        return _result(False, f"Code execution failed: {exc}")

    return _result(True, message_holder["message"], data=data, artifacts=artifacts)


def _restricted_code_worker(payload: dict[str, Any], queue: Any) -> None:
    try:
        result = _run_restricted_code(
            code=str(payload["code"]),
            rows=list(payload["rows"]),
            columns=list(payload["columns"]),
            input_path=Path(str(payload["input_path"])),
            output_dir=Path(str(payload["output_dir"])),
            runtime_results=dict(payload.get("runtime_results") or {}),
        )
    except Exception as exc:
        result = _result(False, f"Code worker failed: {exc}")
    queue.put(result)


def _run_restricted_code_with_timeout(
    *,
    code: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    input_path: Path,
    output_dir: Path,
    runtime_results: dict[str, Any] | None = None,
    timeout_seconds: int = 8,
) -> str:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    payload = {
        "code": code,
        "rows": rows[:20000],
        "columns": columns,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "runtime_results": dict(runtime_results or {}),
    }
    proc = ctx.Process(target=_restricted_code_worker, args=(payload, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        return _result(False, f"Analysis code timed out after {timeout_seconds}s.")
    try:
        return queue.get_nowait()
    except Exception:
        return _result(False, "Analysis code finished without result.")


@tool
def run_analysis_sandbox(params: str | dict | None = None):
    """
    Run lightweight analysis templates against existing session artifacts.

    Params:
      - template: picks_summary | picks_by_station | catalog_magnitude_hist |
                  catalog_depth_hist | catalog_time_series |
                  catalog_mag_depth_scatter | catalog_event_index |
                  picks_trace_detail
      - code: optional restricted Python snippet for custom analysis (disabled by default)
      - allow_code: true/false override to enable code mode in this call
      - timeout_seconds: code mode timeout in seconds, default 8, max 30
      - input_artifact_key: runtime key such as last_picks_csv / last_catalog_csv
      - input_path: explicit path (relative to repo or data/)
      - session_id: required when using input_artifact_key
      - event_index: required for catalog_event_index (1-based)
    """

    parsed = _parse_params(params)
    template = str(parsed.get("template") or "").strip()
    code = str(parsed.get("code") or "").strip()
    if not template and not code:
        return _result(False, "Missing template.")

    input_path = _resolve_input_path(parsed)
    if input_path is None:
        return _result(False, "Input artifact path not found. Provide input_path or (session_id + input_artifact_key).")
    if not input_path.exists() or not input_path.is_file():
        return _result(False, f"Input file does not exist: {input_path}")

    output_dir = _make_output_dir(parsed.get("session_id"))
    rows, columns = _read_csv_rows(input_path)
    if not rows:
        return _result(False, "Input CSV has no rows.")

    if code:
        if not _code_execution_enabled(parsed):
            return _result(
                False,
                "Code mode is disabled. Set allow_code=true (or QUAKECORE_ANALYSIS_ALLOW_CODE=1) to enable.",
            )
        session_id = str(parsed.get("session_id") or "").strip()
        runtime_results = get_session_store().get_runtime_results(session_id) if session_id else {}
        timeout_seconds = int(parsed.get("timeout_seconds", 8) or 8)
        timeout_seconds = max(2, min(timeout_seconds, 30))
        return _run_restricted_code_with_timeout(
            code=code,
            rows=rows,
            columns=columns,
            input_path=input_path,
            output_dir=output_dir,
            runtime_results=runtime_results,
            timeout_seconds=timeout_seconds,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts: list[dict[str, str]] = []

    if template == "picks_trace_detail":
        raw_trace_index = parsed.get("trace_index")
        raw_trace_number = parsed.get("trace_number")
        if raw_trace_index is not None:
            try:
                trace_index = int(raw_trace_index)
            except Exception:
                trace_index = 0
        elif raw_trace_number is not None:
            try:
                trace_index = max(0, int(raw_trace_number) - 1)
            except Exception:
                trace_index = 0
        else:
            trace_index = 0

        trace_col = _find_column(columns, ["trace_index", "trace", "trace_id", "index"])
        if not trace_col:
            return _result(False, "Trace index column not found.")

        selected: list[dict[str, Any]] = []
        for row in rows:
            try:
                value = int(float(row.get(trace_col, -1)))
            except Exception:
                continue
            if value == trace_index:
                selected.append(row)

        if not selected:
            return _result(False, f"No picks found for trace {trace_index}.")

        output_csv = output_dir / f"trace_{trace_index}_picks_{timestamp}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in selected:
                writer.writerow(row)
        artifacts.append(_artifact(output_csv, "file"))
        phase_col = _find_column(columns, ["phase", "phase_type"])
        method_col = _find_column(columns, ["method", "pick_method"])
        score_col = _find_column(columns, ["score", "probability", "confidence", "normalized_score"])
        return _result(
            True,
            f"Trace {trace_index} has {len(selected)} picks.",
            data={
                "trace_index": trace_index,
                "pick_count": len(selected),
                "rows": selected[:20],
                "phase_column": phase_col,
                "method_column": method_col,
                "score_column": score_col,
            },
            artifacts=artifacts,
        )

    if template == "picks_trace_plot":
        raw_trace_index = parsed.get("trace_index")
        raw_trace_number = parsed.get("trace_number")
        if raw_trace_index is not None:
            try:
                trace_index = int(raw_trace_index)
            except Exception:
                trace_index = 0
        elif raw_trace_number is not None:
            try:
                trace_index = max(0, int(raw_trace_number) - 1)
            except Exception:
                trace_index = 0
        else:
            trace_index = 0

        trace_col = _find_column(columns, ["trace_index", "trace", "trace_id", "index"])
        if not trace_col:
            return _result(False, "Trace index column not found.")
        phase_col = _find_column(columns, ["phase", "phase_type", "type"])
        method_col = _find_column(columns, ["method", "pick_method"])
        sample_col = _find_column(columns, ["sample_index", "sample", "index"])
        score_col = _find_column(columns, ["normalized_score", "score", "probability", "confidence"])
        if not sample_col:
            return _result(False, "Sample index column not found.")

        selected: list[dict[str, Any]] = []
        for row in rows:
            try:
                value = int(float(row.get(trace_col, -1)))
            except Exception:
                continue
            if value == trace_index:
                selected.append(row)
        if not selected:
            return _result(False, f"No picks found for trace {trace_index}.")

        output_csv = output_dir / f"trace_{trace_index}_picks_{timestamp}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in selected:
                writer.writerow(row)
        artifacts.append(_artifact(output_csv, "file"))

        plt, err = _plot_or_error()
        if plt is None:
            return _result(False, err or "matplotlib unavailable")

        xs: list[float] = []
        ys: list[float] = []
        colors: list[str] = []
        labels: list[str] = []
        for row in selected:
            try:
                sample = float(row.get(sample_col, 0.0))
            except Exception:
                continue
            phase = str(row.get(phase_col, "P") or "P").upper() if phase_col else "P"
            method = str(row.get(method_col, "") or "") if method_col else ""
            try:
                score = float(row.get(score_col, 0.0)) if score_col else 0.0
            except Exception:
                score = 0.0
            xs.append(sample)
            ys.append(score)
            colors.append("#1B9E77" if phase == "P" else "#C2185B")
            labels.append(f"{phase}/{method}" if method else phase)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys, c=colors, s=40, alpha=0.9)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (xs[i], ys[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title(f"Trace {trace_index} Picks")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=min(-0.05, min(ys) - 0.05), top=max(1.05, max(ys) + 0.05))
        fig.tight_layout()
        output_png = output_dir / f"trace_{trace_index}_picks_{timestamp}.png"
        fig.savefig(output_png, dpi=180)
        plt.close(fig)
        artifacts.append(_artifact(output_png, "image"))

        return _result(
            True,
            f"Trace {trace_index} pick plot generated.",
            data={
                "trace_index": trace_index,
                "pick_count": len(selected),
                "phase_column": phase_col,
                "method_column": method_col,
                "score_column": score_col,
            },
            artifacts=artifacts,
        )

    if template == "picks_summary":
        phase_col = _find_column(columns, ["phase", "phase_type", "type"])
        if not phase_col:
            return _result(False, "Phase column not found.")
        counts = Counter(str(row.get(phase_col, "")).strip().upper() for row in rows)
        output_csv = output_dir / f"picks_summary_{timestamp}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "count"])
            for phase in sorted(counts.keys()):
                writer.writerow([phase, counts[phase]])
        artifacts.append(_artifact(output_csv, "file"))
        return _result(True, f"Picks summary completed ({len(rows)} rows).", data={"counts": counts}, artifacts=artifacts)

    if template == "picks_by_station":
        station_col = _find_column(columns, ["station", "station_id", "sta", "station_code"])
        if not station_col:
            return _result(False, "Station column not found.")
        counts = Counter(str(row.get(station_col, "")).strip() for row in rows)
        top_items = counts.most_common(20)

        output_csv = output_dir / f"picks_by_station_{timestamp}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["station", "count"])
            for station, count in top_items:
                writer.writerow([station, count])
        artifacts.append(_artifact(output_csv, "file"))

        plt, err = _plot_or_error()
        if plt is not None:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            labels = [item[0] or "UNKNOWN" for item in top_items]
            values = [item[1] for item in top_items]
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title("Picks by Station (Top 20)")
            ax.set_ylabel("Pick Count")
            fig.tight_layout()
            output_png = output_dir / f"picks_by_station_{timestamp}.png"
            fig.savefig(output_png, dpi=180)
            plt.close(fig)
            artifacts.append(_artifact(output_png, "image"))
        elif err:
            return _result(False, err)

        return _result(True, "Station pick distribution generated.", data={"top_count": len(top_items)}, artifacts=artifacts)

    if template in {"catalog_magnitude_hist", "catalog_depth_hist", "catalog_mag_depth_scatter", "catalog_time_series", "catalog_event_index"}:
        mag_col = _find_column(columns, ["magnitude_pred", "magnitude", "mag"])
        depth_col = _find_column(columns, ["depth_km", "depth"])
        time_col = _find_column(columns, ["time", "origin_time"])

        if template == "catalog_event_index":
            raw_index = parsed.get("event_index", parsed.get("index", 1))
            try:
                idx = max(1, int(raw_index))
            except Exception:
                idx = 1
            if idx > len(rows):
                return _result(False, f"event_index out of range: {idx} > {len(rows)}")
            event = rows[idx - 1]
            output_csv = output_dir / f"catalog_event_{idx}_{timestamp}.csv"
            with output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(event.keys()))
                writer.writeheader()
                writer.writerow(event)
            artifacts.append(_artifact(output_csv, "file"))
            return _result(True, f"Selected event #{idx} from catalog.", data={"event_index": idx, "event": event}, artifacts=artifacts)

        if template == "catalog_time_series":
            if not time_col:
                return _result(False, "Time column not found.")
            buckets: Counter[str] = Counter()
            for row in rows:
                raw = str(row.get(time_col, "")).strip()
                if not raw:
                    continue
                try:
                    normalized = raw.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(normalized)
                    buckets[dt.strftime("%Y-%m-%d %H:00")] += 1
                except Exception:
                    buckets[raw[:13]] += 1

            ordered = sorted(buckets.items(), key=lambda item: item[0])
            output_csv = output_dir / f"catalog_time_series_{timestamp}.csv"
            with output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_bucket", "event_count"])
                for key, value in ordered:
                    writer.writerow([key, value])
            artifacts.append(_artifact(output_csv, "file"))

            plt, err = _plot_or_error()
            if plt is None:
                return _result(False, err or "matplotlib unavailable")
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.plot([item[0] for item in ordered], [item[1] for item in ordered], marker="o")
            ax.set_title("Catalog Time Series")
            ax.set_ylabel("Event Count")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            output_png = output_dir / f"catalog_time_series_{timestamp}.png"
            fig.savefig(output_png, dpi=180)
            plt.close(fig)
            artifacts.append(_artifact(output_png, "image"))
            return _result(True, "Catalog time series generated.", data={"points": len(ordered)}, artifacts=artifacts)

        if template in {"catalog_magnitude_hist", "catalog_mag_depth_scatter"} and not mag_col:
            return _result(False, "Magnitude column not found.")
        if template in {"catalog_depth_hist", "catalog_mag_depth_scatter"} and not depth_col:
            return _result(False, "Depth column not found.")

        def _to_float(value: Any) -> float | None:
            try:
                return float(value)
            except Exception:
                return None

        mags = [_to_float(row.get(mag_col)) for row in rows] if mag_col else []
        mags = [item for item in mags if item is not None]
        depths = [_to_float(row.get(depth_col)) for row in rows] if depth_col else []
        depths = [item for item in depths if item is not None]

        plt, err = _plot_or_error()
        if plt is None:
            return _result(False, err or "matplotlib unavailable")

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        output_png = output_dir / f"{template}_{timestamp}.png"

        if template == "catalog_magnitude_hist":
            ax.hist(mags, bins=int(parsed.get("bins", 12)), edgecolor="black")
            ax.set_xlabel("Magnitude")
            ax.set_ylabel("Count")
            ax.set_title("Catalog Magnitude Distribution")
        elif template == "catalog_depth_hist":
            ax.hist(depths, bins=int(parsed.get("bins", 12)), edgecolor="black")
            ax.set_xlabel("Depth (km)")
            ax.set_ylabel("Count")
            ax.set_title("Catalog Depth Distribution")
        else:
            paired = [
                (_to_float(row.get(mag_col)), _to_float(row.get(depth_col)))
                for row in rows
            ]
            paired = [(m, d) for m, d in paired if m is not None and d is not None]
            ax.scatter([item[0] for item in paired], [item[1] for item in paired], s=18, alpha=0.8)
            ax.set_xlabel("Magnitude")
            ax.set_ylabel("Depth (km)")
            ax.set_title("Magnitude vs Depth")

        fig.tight_layout()
        fig.savefig(output_png, dpi=180)
        plt.close(fig)
        artifacts.append(_artifact(output_png, "image"))

        return _result(True, f"{template} generated.", data={"row_count": len(rows)}, artifacts=artifacts)

    return _result(False, f"Unsupported template: {template}")
