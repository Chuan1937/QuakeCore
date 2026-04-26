import csv
import glob
from dataclasses import asdict
from datetime import datetime

from langchain.tools import tool
from utils.segy_handler import SegyHandler
from utils.miniseed_handler import MiniSEEDHandler
from utils.hdf5_handler import HDF5Handler
from utils.phase_picker import pick_phases, summarize_pick_results, load_traces, plot_waveform_with_picks
from utils.phase_picker import TraceRecord, PickResult
from utils.sac_handler import SACHandler
from utils.continuous_data import (
    download_continuous_data, estimate_continuous_download, load_station_data, fetch_catalog,
    haversine_km, get_default_data_dir,
    get_region, resolve_named_place, get_chunk_dir, load_chunk_stations_cache,
)
from utils.continuous_picking import continuous_picking, clear_model_cache
from utils.association import associate_multiple_events
from utils.catalog_matcher import match_catalogs, compute_detection_stats, print_detection_summary
import json
from typing import Union
import numpy as np
import os
import h5py
import importlib.util
import shutil

# Global variable to store the current file path being analyzed
# In a multi-user web app, this should be handled via session state or context
CURRENT_SEGY_PATH = None
CURRENT_MINISEED_PATH = None
CURRENT_MINISEED_PATHS = []  # Support multiple MiniSEED files for multi-station location
CURRENT_HDF5_PATH = None
CURRENT_SAC_PATH = None
CURRENT_UPLOADED_FILES = []
CURRENT_LANG = "en"  # Current UI language, set from app.py
_VALIDATION_MODULE_CACHE = {}


def set_current_lang(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang


DEFAULT_CONVERT_DIR = "data/convert"
DEFAULT_STRUCTURE_DIR = "data/structure"
DEFAULT_PICKS_DIR = "data/picks"
DEFAULT_LOCATION_DIR = "data/location"

# Plot font configuration
PLOT_FONT_FAMILY = "Times New Roman"
PLOT_FONT_FALLBACK = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def _build_artifact_response(
    result: dict,
    message: Union[str, None] = None,
    output_path: Union[str, None] = None,
    input_path: Union[str, None] = None,
) -> str:
    """Wrap conversion result dict into structured format with artifacts and return JSON."""
    result = dict(result)
    plot_path = result.pop("plot_path", None)
    saved_to = result.pop("saved_to", None) or output_path

    if not message:
        input_basename = os.path.basename(input_path) if input_path else ""
        output_basename = os.path.basename(saved_to) if saved_to else ""
        message = f'文件 "{input_basename}" 已成功转换为 {output_basename}'
        if "trace_count" in result:
            message += f"，共包含 {result["trace_count"]} 条迹"

    response = {"success": True, "message": message}
    artifacts = []

    if saved_to:
        rel = saved_to.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("data/"):
            rel = rel[5:]
        rel = rel.lstrip("/")
        if rel:
            artifacts.append({
                "type": "file",
                "name": os.path.basename(saved_to),
                "path": rel,
                "url": f"/api/artifacts/{rel}",
            })

    if plot_path:
        rel = plot_path.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("data/"):
            rel = rel[5:]
        rel = rel.lstrip("/")
        if rel:
            artifacts.append({
                "type": "image",
                "name": os.path.basename(plot_path),
                "path": rel,
                "url": f"/api/artifacts/{rel}",
            })

    response["artifacts"] = artifacts
    if result:
        response["data"] = result

    return json.dumps(response, indent=2, ensure_ascii=False)

def configure_plot_fonts():
    """Configure matplotlib to use Times New Roman with fallbacks."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Try Times New Roman first, fall back to others
        font_list = [PLOT_FONT_FAMILY] + PLOT_FONT_FALLBACK
        mpl.rcParams['font.family'] = font_list
        mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

# Apply font configuration on module load
configure_plot_fonts()


# 设置当前 SEGY 文件路径的辅助函数
def set_current_segy_path(path):
    global CURRENT_SEGY_PATH
    CURRENT_SEGY_PATH = path

# 设置当前 MiniSEED 文件路径的辅助函数
def set_current_miniseed_path(path):
    global CURRENT_MINISEED_PATH, CURRENT_MINISEED_PATHS
    CURRENT_MINISEED_PATH = path
    # Also add to list if not already present
    if path and path not in CURRENT_MINISEED_PATHS:
        CURRENT_MINISEED_PATHS.append(path)


def add_miniseed_path(path):
    """Add a MiniSEED file path to the list of loaded files."""
    global CURRENT_MINISEED_PATHS, CURRENT_MINISEED_PATH
    if path and path not in CURRENT_MINISEED_PATHS:
        CURRENT_MINISEED_PATHS.append(path)
        # Set as current if it's the first one
        if not CURRENT_MINISEED_PATH:
            CURRENT_MINISEED_PATH = path


def clear_miniseed_paths():
    """Clear all stored MiniSEED paths."""
    global CURRENT_MINISEED_PATHS, CURRENT_MINISEED_PATH
    CURRENT_MINISEED_PATHS = []
    CURRENT_MINISEED_PATH = None


def set_current_uploaded_files(paths):
    """
    Replace current uploaded-file context and rebuild per-type current pointers.
    """
    global CURRENT_UPLOADED_FILES, CURRENT_SEGY_PATH, CURRENT_MINISEED_PATH
    global CURRENT_MINISEED_PATHS, CURRENT_HDF5_PATH, CURRENT_SAC_PATH

    normalized = []
    for raw_path in paths or []:
        path = str(raw_path or "").strip()
        if path and path not in normalized:
            normalized.append(path)

    CURRENT_UPLOADED_FILES = normalized
    CURRENT_SEGY_PATH = None
    CURRENT_MINISEED_PATH = None
    CURRENT_MINISEED_PATHS = []
    CURRENT_HDF5_PATH = None
    CURRENT_SAC_PATH = None

    for path in normalized:
        lowered = path.lower()
        if lowered.endswith(".mseed") or lowered.endswith(".miniseed"):
            CURRENT_MINISEED_PATHS.append(path)
        elif lowered.endswith(".segy") or lowered.endswith(".sgy"):
            CURRENT_SEGY_PATH = path
        elif lowered.endswith(".h5") or lowered.endswith(".hdf5"):
            CURRENT_HDF5_PATH = path
        elif lowered.endswith(".sac"):
            CURRENT_SAC_PATH = path

    if CURRENT_MINISEED_PATHS:
        CURRENT_MINISEED_PATH = CURRENT_MINISEED_PATHS[-1]


def get_current_uploaded_files():
    return list(CURRENT_UPLOADED_FILES)

# 设置当前 HDF5 文件路径的辅助函数
def set_current_hdf5_path(path):
    global CURRENT_HDF5_PATH
    CURRENT_HDF5_PATH = path

# 设置当前 SAC 文件路径的辅助函数
def set_current_sac_path(path):
    global CURRENT_SAC_PATH
    CURRENT_SAC_PATH = path

# 解析文件路径的辅助函数
def _resolve_file_path(path: str) -> str:
    """Resolve file path, checking data/ directory if relative path doesn't exist."""
    """解析文件路径，如果相对路径不存在则检查 data/ 目录。"""
    if not path:
        return path
    # If absolute path and exists, return as is
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # If relative path and exists, return as is
    if os.path.exists(path):
        return path
    # Try in data/ directory
    data_path = os.path.join("data", path)
    if os.path.exists(data_path):
        return data_path
    # Try with absolute path from cwd
    cwd_data_path = os.path.join(os.getcwd(), "data", path)
    if os.path.exists(cwd_data_path):
        return cwd_data_path
    return path  # Return original path even if not found (will error later)


# 解析参数字典的辅助函数
def _parse_param_dict(raw_params):
    """Normalize incoming tool arguments to a dict."""
    """归一化传入的工具参数为字典。"""
    if raw_params is None:
        return {}
    if isinstance(raw_params, dict):
        result = raw_params
    elif isinstance(raw_params, str):
        candidate = raw_params.strip()
        if not candidate:
            return {}
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            params = {}
            for chunk in candidate.split(","):
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                    params[key.strip()] = value.strip()
            result = params
    else:
        return {}

    # Auto-resolve path parameter if present
    if "path" in result and result["path"]:
        result["path"] = _resolve_file_path(result["path"])
    return result

# 将值强制转换为整数的辅助函数
def _coerce_int(value, *, allow_none=False, default=None, field_name="value"):
    if value is None:
        if allow_none:
            return None
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if allow_none and lowered in {"none", "null", ""}:
            return None
        return int(lowered)
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _coerce_float(value, *, allow_none=False, default=None, field_name="value"):
    if value is None:
        if allow_none:
            return default
        if default is not None:
            return float(default)
        raise ValueError(f"{field_name} must be provided")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip()
        if allow_none and lowered.lower() in {"none", "null", ""}:
            return default
        return float(lowered)
    raise ValueError(f"{field_name} must be a float, got {value!r}")


def _normalize_method_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None
        if candidate.startswith("["):
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return [str(item).strip() for item in data if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in candidate.split(",") if part.strip()]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return None


def _pick_weighted_score(pick: PickResult) -> float:
    """Weighted score used for selecting best P/S pick for plotting."""
    method_weights = {
        "phasenet": 1.2,
        "eqtransformer": 1.2,
        "gpd": 1.1,
        "sta_lta": 1.0,
        "aic": 0.9,
        "pai_k": 0.9,
        "pai_s": 0.9,
        "s_phase": 0.8,
        "frequency_ratio": 0.8,
        "autocorr": 0.7,
        "feature_threshold": 0.7,
        "ar_model": 0.7,
        "template_correlation": 0.8,
    }
    base = float(pick.normalized_score) if pick.normalized_score is not None else 0.0
    return base * method_weights.get(pick.method, 0.8)


def _select_best_p_s_for_trace(picks: list[PickResult], trace_index: int) -> list[PickResult]:
    """Force plotting to include best P and best S (if detected) for the target trace."""
    trace_picks = [p for p in picks if int(p.trace_index) == int(trace_index)]
    p_candidates = [p for p in trace_picks if (p.phase_type or "P").upper() == "P"]
    s_candidates = [p for p in trace_picks if (p.phase_type or "P").upper() == "S"]

    selected = []
    if p_candidates:
        selected.append(max(p_candidates, key=_pick_weighted_score))
    if s_candidates:
        selected.append(max(s_candidates, key=_pick_weighted_score))
    return selected


def _trace_channel_priority(trace) -> int:
    """Prefer vertical channels when selecting a representative trace for plotting."""
    metadata = getattr(trace, "metadata", {}) or {}
    channel = str(metadata.get("channel", "")).upper()
    if channel.endswith("Z"):
        return 0
    if channel.endswith(("N", "E")):
        return 1
    return 2


def _parse_method_params(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def _normalize_source_type(value: str | None):
    if not value:
        return None
    normalized = value.lower().strip()
    alias = {
        "miniseed": "mseed",
        "mseed": "mseed",
        "segy": "segy",
        "sgy": "segy",
        "hdf5": "hdf5",
        "h5": "hdf5",
        "npy": "npy",
        "npz": "npz",
        "sac": "sac",
    }
    return alias.get(normalized, normalized)


def _infer_file_type_from_path(path: str | None):
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    mapping = {
        ".segy": "segy",
        ".sgy": "segy",
        ".mseed": "mseed",
        ".miniseed": "mseed",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".npy": "npy",
        ".npz": "npz",
        ".sac": "sac",
    }
    return mapping.get(ext)


def _resolve_source_path(path: str | None, source_type: str | None):
    normalized_type = _normalize_source_type(source_type)
    if path:
        inferred = normalized_type or _infer_file_type_from_path(path)
        return path, inferred

    candidates = [
        ("segy", CURRENT_SEGY_PATH),
        ("mseed", CURRENT_MINISEED_PATH),
        ("hdf5", CURRENT_HDF5_PATH),
    ]

    if normalized_type:
        for ctype, cpath in candidates:
            if ctype == normalized_type and cpath:
                return cpath, ctype
        return None, normalized_type

    for ctype, cpath in candidates:
        if cpath:
            return cpath, ctype
    return None, None


def _resolve_output_path(output_path: str | None, *, default_filename: str, base_dir: str | None = None) -> str:
    """Resolve output file path.

    Rules:
    - If output_path is empty -> {base_dir}/{default_filename}
    - If output_path is filename only -> {base_dir}/{output_path}
    - If output_path contains a directory (relative/absolute) -> keep as is
    Also ensures the parent directory exists.
    """
    if base_dir is None:
        base_dir = DEFAULT_CONVERT_DIR

    if not output_path or not str(output_path).strip():
        final_path = os.path.join(base_dir, default_filename)
    else:
        output_path = str(output_path).strip()
        if os.path.dirname(output_path) == "":
            final_path = os.path.join(base_dir, output_path)
        else:
            final_path = output_path

    parent = os.path.dirname(final_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return final_path

# tool to read basic SEGY structure info
# 读取 SEGY 结构信息的工具
@tool
def get_segy_structure(params: Union[str, dict, None] = None):
    """
    Reads the currently loaded SEGY file and returns a summary of its structure,
    including trace count, sample rate, and sample count.
    Use this tool when the user asks about the 'structure', 'basic info', or 'overview' of the SEGY file.
    Args: plot (bool, default True) - whether to generate waveform plot.
    """
    """读取当前加载的 SEGY 文件并返回其结构摘要，
    包括道数、采样率和采样点数。
    当用户询问 SEGY 文件的"结构"、"基本信息"或"概览"时使用此工具。
    参数：plot（bool，默认 True）- 是否生成波形图。
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded. Please ask the user to upload a file first."

    handler = SegyHandler(CURRENT_SEGY_PATH)
    info = handler.get_basic_info()

    # Parse params for plot option
    parsed = _parse_param_dict(params)
    plot = parsed.get("plot", True)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    plot_markdown = None
    if plot:
        try:
            record_data = handler.get_trace_record_data(trace_index=0)
            if "error" not in record_data:
                tr = TraceRecord(
                    data=record_data["data"],
                    sampling_rate=record_data["sampling_rate"],
                    start_time=record_data.get("start_time"),
                    metadata={"trace_index": 0, "type": "segy"}
                )
                output_filename = "segy_structure_plot.png"
                output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
                os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
                plot_path = plot_waveform_with_picks([tr], [], output_path)
                info["plot_path"] = plot_path
                if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                    plot_markdown = f"\n\n![波形图]({plot_path})"
        except Exception as e:
            info["plot_error"] = str(e)

    if plot_markdown:
        return json.dumps(info, indent=2, ensure_ascii=False) + plot_markdown
    return json.dumps(info, indent=2, ensure_ascii=False)

# tool to read SEGY text header
# 读取 SEGY 文本头的工具
@tool
def get_segy_text_header():
    """
    Reads the EBCDIC text header of the SEGY file.
    Use this tool when the user asks to see the 'text header', 'EBCDIC header', or 'notes' in the file header.
    """
    """读取 SEGY 文件的 EBCDIC 文本头。
    当用户要求查看文件头中的“文本头”、“EBCDIC 头”或“注释”时使用此工具。
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    header = handler.get_text_header()
    return header

# tool to read SEGY binary header
# 读取 SEGY 二进制头的工具
@tool
def get_segy_binary_header():
    """
    Reads the Binary header of the SEGY file.
    Use this tool when the user asks for 'binary header' information like format code, job id, etc.
    """
    """读取 SEGY 文件的二进制头。
    当用户询问“二进制头”信息（如格式代码、作业 ID 等）时使用此工具。
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    header = handler.get_binary_header()
    return json.dumps(header, indent=2)

# tool to read a specific trace sample
# 读取特定地震道样本的工具
@tool
def read_trace_sample(trace_index: Union[int, str] = 0, plot: bool = False):
    """
    Reads data from a specific trace index.
    Args:
        trace_index: The 0-based index of the trace to read. Defaults to 0.
        plot: Whether to generate a plot of the waveform. Defaults to False.
    Use this tool to inspect actual seismic data values.
    """
    """读取特定地震道索引的数据。
    参数:
        trace_index: 要读取的地震道的 0 基索引。默认为 0。
        plot: 是否生成波形图。默认为 False。
    使用此工具检查实际的地震数据值。
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    # Handle potential JSON string input from ReAct agents
    # 应对 ReAct 代理可能的 JSON 字符串输入
    if isinstance(trace_index, str):
        raw_value = trace_index.strip()
        if raw_value.lower().startswith("trace_index"):
            parts = raw_value.split("=", 1)
            if len(parts) == 2:
                raw_value = parts[1].strip()
        try:
            # Try to parse as integer first
            # 尝试先解析为整数
            trace_index = int(raw_value)
        except ValueError:
            try:
                # Try to parse as JSON or plain integer string
                # 尝试解析为 JSON 或纯整数字符串
                data = json.loads(raw_value)
                if isinstance(data, dict) and "trace_index" in data:
                    trace_index = int(data["trace_index"])
                    if "plot" in data:
                        plot = data["plot"]
                elif isinstance(data, int):
                    trace_index = data
            except Exception:
                return "Trace index must be an integer, e.g. 0 or 10."

    if not isinstance(trace_index, int):
        return "Trace index must be an integer, e.g. 0 or 10."

    handler = SegyHandler(CURRENT_SEGY_PATH)
    # Read just one trace for inspection
    # 仅读取一个地震道以供检查
    data = handler.get_trace_data(trace_index=trace_index, count=1)
    
    # Summarize data to avoid context overflow
    # 总结数据以避免上下文溢出
    if "data" in data:
        trace_vals = data["data"][0]
        summary = {
            "trace_index": trace_index,
            "min_value": float(np.min(trace_vals)),
            "max_value": float(np.max(trace_vals)),
            "mean_value": float(np.mean(trace_vals)),
            "first_10_samples": trace_vals[:10],
            "total_samples": len(trace_vals)
        }
        
        plot_markdown = None
        if plot:
            record_data = handler.get_trace_record_data(trace_index=trace_index)
            if "error" not in record_data:
                tr = TraceRecord(
                    data=record_data["data"],
                    sampling_rate=record_data["sampling_rate"],
                    start_time=record_data["start_time"],
                    metadata={"trace_index": trace_index, "type": "segy"}
                )
                output_filename = f"trace_plot_segy_{trace_index}.png"
                output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
                os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
                
                plot_path = plot_waveform_with_picks([tr], [], output_path)
                summary["plot_path"] = plot_path
                if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                    plot_markdown = f"![波形图]({plot_path})"
        
        if plot_markdown:
            return json.dumps(summary, indent=2) + "\n\n" + plot_markdown
        return json.dumps(summary, indent=2)
    return json.dumps(data)


# tool to convert SEGY to NumPy
# 将 SEGY 转换为 NumPy 的工具
@tool
def convert_segy_to_numpy(params: Union[str, dict, None] = None):
    """convert SEGY data to NumPy, supporting start_trace/count/output_path parameters."""
    """将 SEGY 数据转换为 NumPy，支持 start_trace/count/output_path 参数。"""
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."

    parsed = _parse_param_dict(params)
    try:
        start_trace = _coerce_int(parsed.get("start_trace"), default=0, field_name="start_trace")
        count = _coerce_int(parsed.get("count"), allow_none=True, default=None, field_name="count")
    except ValueError as exc:
        return str(exc)

    output_path = _resolve_output_path(
        parsed.get("output_path"),
        default_filename="segy_data.npy",
    )

    handler = SegyHandler(CURRENT_SEGY_PATH)
    result = handler.to_numpy(start_trace=start_trace, count=count, output_path=output_path)

    if "error" in result:
        return json.dumps(result, indent=2)

    data = result.pop("array", None)
    summary = result
    summary["saved_to"] = output_path
    if data is not None and data.size:
        first_trace = data[0] if data.ndim > 1 else data
        summary["samples_per_trace"] = int(first_trace.shape[0])
        summary["preview_first_trace_first_10"] = first_trace[:10].tolist()
        summary["dtype"] = str(data.dtype)

        # Plot first trace after conversion
        try:
            sr = summary.get("sample_rate", 100.0) * 1000  # convert ms to Hz
            tr = TraceRecord(
                data=first_trace,
                sampling_rate=float(sr),
                start_time=None,
                metadata={"trace_index": 0, "type": "segy"}
            )
            base_name = os.path.splitext(os.path.basename(CURRENT_SEGY_PATH))[0]
            plot_filename = f"{base_name}_numpy_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                summary["plot_path"] = plot_result
        except Exception as e:
            summary["plot_error"] = str(e)
    return _build_artifact_response(summary, input_path=CURRENT_SEGY_PATH, output_path=output_path)


# tool to convert SEGY to Excel
# 将 SEGY 转换为 Excel 的工具
@tool
def convert_segy_to_excel(params: Union[str, dict, None] = None):
    """convert specified range of traces to Excel. Args: output_path, start_trace, count."""
    """将指定范围的地震道导出为 Excel。参数: output_path, start_trace, count。"""
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."

    parsed = _parse_param_dict(params)
    output_path = _resolve_output_path(
        parsed.get("output_path"),
        default_filename="output_segy_data.xlsx",
    )

    try:
        start_trace = _coerce_int(parsed.get("start_trace"), default=0, field_name="start_trace")
        # 默认导出全部道
        # 默认全量导出：count=None
        count = _coerce_int(parsed.get("count"), allow_none=True, default=None, field_name="count")
    except ValueError as exc:
        return str(exc)

    handler = SegyHandler(CURRENT_SEGY_PATH)
    result = handler.to_excel(output_path=output_path, start_trace=start_trace, count=count)

    if isinstance(result, dict) and "error" not in result:
        result["saved_to"] = output_path
        # Plot first trace after conversion
        try:
            segy_result = handler.to_numpy(start_trace=start_trace, count=min(count or 1, 1), output_path=None)
            if "error" not in segy_result and "array" in segy_result:
                data = segy_result["array"]
                if data is not None and data.size:
                    first_trace = data[0] if data.ndim > 1 else data
                    sr = segy_result.get("sample_rate", 100.0) * 1000
                    tr = TraceRecord(
                        data=first_trace,
                        sampling_rate=float(sr),
                        start_time=None,
                        metadata={"trace_index": 0, "type": "segy"}
                    )
                    base_name = os.path.splitext(os.path.basename(CURRENT_SEGY_PATH))[0]
                    plot_filename = f"{base_name}_excel_plot.png"
                    plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
                    os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
                    plot_result = plot_waveform_with_picks([tr], [], plot_path)
                    if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                        result["plot_path"] = plot_result
        except Exception as e:
            result["plot_error"] = str(e)
        return _build_artifact_response(result, input_path=CURRENT_SEGY_PATH, output_path=output_path)
    return result


# tool to convert SEGY to HDF5
# 将 SEGY 转换为 HDF5 的工具
@tool
def convert_segy_to_hdf5(params: Union[str, dict, None] = None):
    """convert specified range of traces to HDF5. Args: output_path, start_trace, count, compression."""
    """将数据写入 HDF5。参数: output_path, start_trace, count, compression。"""
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."

    parsed = _parse_param_dict(params)
    output_path = _resolve_output_path(
        parsed.get("output_path"),
        default_filename="output_segy_data.h5",
    )

    try:
        start_trace = _coerce_int(parsed.get("start_trace"), default=0, field_name="start_trace")
        count = _coerce_int(parsed.get("count"), allow_none=True, default=None, field_name="count")
    except ValueError as exc:
        return str(exc)

    compression = parsed.get("compression", "gzip")
    if isinstance(compression, str) and compression.lower() in {"none", "null", ""}:
        compression = None

    handler = SegyHandler(CURRENT_SEGY_PATH)
    result = handler.to_hdf5(
        output_path=output_path,
        start_trace=start_trace,
        count=count,
        compression=compression,
    )

    if isinstance(result, dict) and "error" not in result:
        result["saved_to"] = output_path
        # Plot first trace after conversion
        try:
            handler_h5 = HDF5Handler(output_path)
            record_data = handler_h5.get_trace_record_data(trace_index=0, dataset="traces")
            if "error" not in record_data:
                tr = TraceRecord(
                    data=record_data["data"],
                    sampling_rate=record_data["sampling_rate"],
                    start_time=record_data.get("start_time"),
                    metadata={"trace_index": 0, "type": "hdf5"}
                )
                base_name = os.path.splitext(os.path.basename(CURRENT_SEGY_PATH))[0]
                plot_filename = f"{base_name}_hdf5_plot.png"
                plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
                os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
                plot_result = plot_waveform_with_picks([tr], [], plot_path)
                if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                    result["plot_path"] = plot_result
        except Exception as e:
            result["plot_error"] = str(e)
        return _build_artifact_response(result, input_path=CURRENT_SEGY_PATH, output_path=output_path)
    return result


# 读取 MiniSEED 基本结构信息的工具
@tool
def get_miniseed_structure(params: Union[str, dict, None] = None):
    """
    Read MiniSEED file basic structure info. Args: path (optional), plot (bool, default True).
    """
    """读取 MiniSEED 文件的基本结构信息。参数：path（可选）、plot（bool，默认 True）。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_MINISEED_PATH
    if not path:
        return "No MiniSEED file is currently loaded."
    handler = MiniSEEDHandler(path)
    info = handler.get_basic_info()

    # Default plot=True for better UX
    plot = parsed.get("plot", True)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    plot_markdown = None
    if plot:
        try:
            record_data = handler.get_trace_record_data(trace_index=0)
            if "error" not in record_data:
                tr = TraceRecord(
                    data=record_data["data"],
                    sampling_rate=record_data["sampling_rate"],
                    start_time=record_data.get("start_time"),
                    metadata={"trace_index": 0, "type": "miniseed"}
                )
                output_filename = "miniseed_structure_plot.png"
                output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
                os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
                plot_path = plot_waveform_with_picks([tr], [], output_path)
                info["plot_path"] = plot_path
                if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                    plot_markdown = f"\n\n![波形图]({plot_path})"
        except Exception as e:
            info["plot_error"] = str(e)

    if plot_markdown:
        return json.dumps(info, indent=2, ensure_ascii=False) + plot_markdown
    return json.dumps(info, indent=2, ensure_ascii=False)

# 读取 MiniSEED 指定道/轨迹数据的工具
@tool
def read_miniseed_trace(params: Union[str, dict, None] = None):
    """
    Read one MiniSEED trace by 0-based index. Args: path, trace_index, plot (bool).
    """
    """按 0 基索引读取一条 MiniSEED 轨迹。参数：path，trace_index, plot (bool)。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_MINISEED_PATH
    if not path:
        return "No MiniSEED file is currently loaded."
    try:
        trace_index = _coerce_int(parsed.get("trace_index"), default=0, field_name="trace_index")
    except ValueError as exc:
        return str(exc)
    
    plot = parsed.get("plot", False)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    handler = MiniSEEDHandler(path)
    result = handler.get_trace_data(trace_index=trace_index)
    
    plot_markdown = None
    if plot and "error" not in result:
        record_data = handler.get_trace_record_data(trace_index=trace_index)
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data["start_time"],
                metadata={"trace_index": trace_index, "type": "miniseed"}
            )
            output_filename = f"trace_plot_miniseed_{trace_index}.png"
            output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
            os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
            
            plot_path = plot_waveform_with_picks([tr], [], output_path)
            result["plot_path"] = plot_path
            if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                plot_markdown = f"![波形图]({plot_path})"

    if plot_markdown:
        return json.dumps(result, indent=2) + "\n\n" + plot_markdown
    return json.dumps(result, indent=2)

# 返回当前已加载的文件上下文（类型与路径）
@tool
def get_loaded_context():
    """
    Return currently loaded file type and paths.
    """
    """返回当前已加载的文件类型与路径。"""
    current_type = None
    if CURRENT_SEGY_PATH:
        current_type = "segy"
    elif CURRENT_MINISEED_PATHS:
        current_type = "miniseed"
    elif CURRENT_HDF5_PATH:
        current_type = "hdf5"
    elif CURRENT_SAC_PATH:
        current_type = "sac"
    return json.dumps(
        {
            "current_type": current_type,
            "segy_path": CURRENT_SEGY_PATH,
            "miniseed_path": CURRENT_MINISEED_PATH,
            "miniseed_paths": CURRENT_MINISEED_PATHS,
            "num_miniseed_files": len(CURRENT_MINISEED_PATHS),
            "hdf5_path": CURRENT_HDF5_PATH,
            "sac_path": CURRENT_SAC_PATH,
            "has_picks": CURRENT_PICKS is not None and len(CURRENT_PICKS) > 0,
            "num_stations_with_coords": len(CURRENT_STATIONS) if CURRENT_STATIONS else 0,
        },
        indent=2,
        ensure_ascii=False,
    )

# 泛化的结构读取工具：根据已加载文件类型自动选择
@tool
def get_file_structure():
    """
    Read structure of currently loaded file (SEGY/MiniSEED/HDF5/SAC).
    """
    """读取当前已加载文件（SEGY/MiniSEED/HDF5/SAC）的结构信息。"""
    if CURRENT_SEGY_PATH:
        handler = SegyHandler(CURRENT_SEGY_PATH)
        info = handler.get_basic_info()
        return json.dumps(info, indent=2)
    if CURRENT_MINISEED_PATH:
        handler = MiniSEEDHandler(CURRENT_MINISEED_PATH)
        info = handler.get_basic_info()
        return json.dumps(info, indent=2)
    if CURRENT_HDF5_PATH:
        handler = HDF5Handler(CURRENT_HDF5_PATH)
        info = handler.get_basic_info()
        return json.dumps(info, indent=2)
    if CURRENT_SAC_PATH:
        handler = SACHandler(CURRENT_SAC_PATH)
        info = handler.get_basic_info()
        return json.dumps(info, indent=2)
    return "No data file is currently loaded."

# 泛化的轨迹读取工具：根据已加载文件类型自动选择
@tool
def read_file_trace(params: Union[str, dict, None] = None):
    """
    Read one trace from currently loaded file. Args: trace_index, plot (bool).
    """
    """读取当前已加载文件的一条轨迹。参数：trace_index, plot (bool)。"""
    parsed = _parse_param_dict(params)
    try:
        trace_index = _coerce_int(parsed.get("trace_index"), default=0, field_name="trace_index")
    except ValueError as exc:
        return str(exc)
    
    plot = parsed.get("plot", False)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    handler = None
    file_type = None
    
    if CURRENT_SEGY_PATH:
        handler = SegyHandler(CURRENT_SEGY_PATH)
        file_type = "segy"
    elif CURRENT_MINISEED_PATH:
        handler = MiniSEEDHandler(CURRENT_MINISEED_PATH)
        file_type = "miniseed"
    elif CURRENT_HDF5_PATH:
        handler = HDF5Handler(CURRENT_HDF5_PATH)
        file_type = "hdf5"
    elif CURRENT_SAC_PATH:
        handler = SACHandler(CURRENT_SAC_PATH)
        file_type = "sac"
    
    if not handler:
        return "No data file is currently loaded."

    # Get data/summary
    # 获取数据/摘要
    if file_type == "segy":
        # SEGY handler returns full data, we need to summarize it
        # SEGY handler 返回完整数据，我们需要对其进行摘要
        data = handler.get_trace_data(trace_index=trace_index, count=1)
        if "data" in data:
            trace_vals = data["data"][0]
            result = {
                "trace_index": trace_index,
                "min_value": float(np.min(trace_vals)),
                "max_value": float(np.max(trace_vals)),
                "mean_value": float(np.mean(trace_vals)),
                "first_10_samples": trace_vals[:10],
                "total_samples": len(trace_vals),
                "type": "segy",
            }
        else:
            result = data
    else:
        # Other handlers return summary by default
        # 其他 handler 默认返回摘要
        result = handler.get_trace_data(trace_index=trace_index)
        if isinstance(result, dict):
            result["type"] = file_type

    if "error" in result:
        return json.dumps(result, indent=2)

    # Handle plotting
    # 处理绘图
    plot_markdown = None
    if plot:
        record_data = handler.get_trace_record_data(trace_index=trace_index)
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data["start_time"],
                metadata={"trace_index": trace_index, "type": file_type}
            )
            output_filename = f"trace_plot_{file_type}_{trace_index}.png"
            output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            
            plot_path = plot_waveform_with_picks([tr], [], output_path)
            result["plot_path"] = plot_path
            if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                plot_markdown = f"![波形图]({plot_path})"

    if plot_markdown:
        return json.dumps(result, indent=2) + "\n\n" + plot_markdown
    return json.dumps(result, indent=2)

# 工具：读取 HDF5 结构信息
@tool
def get_hdf5_structure(params: Union[str, dict, None] = None):
    """
    Read HDF5 file structure info. Args: path (optional), dataset (optional).
    """
    """读取 HDF5 文件的结构信息。参数：path（可选）、dataset（可选）。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."
    handler = HDF5Handler(path)
    info = handler.get_basic_info(dataset=parsed.get("dataset"))
    return json.dumps(info, indent=2)

# 工具：读取 HDF5 一条轨迹
@tool
def read_hdf5_trace(params: Union[str, dict, None] = None):
    """
    Read one trace from HDF5 file. Args: path (optional), dataset (optional), trace_index, plot (bool).
    """
    """从 HDF5 文件读取一条轨迹。参数：path（可选）、dataset（可选）、trace_index, plot (bool)。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."
    try:
        trace_index = _coerce_int(parsed.get("trace_index"), default=0, field_name="trace_index")
    except ValueError as exc:
        return str(exc)
    
    plot = parsed.get("plot", False)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    handler = HDF5Handler(path)
    result = handler.get_trace_data(trace_index=trace_index, dataset=parsed.get("dataset"))

    plot_markdown = None
    if plot and "error" not in result:
        record_data = handler.get_trace_record_data(trace_index=trace_index, dataset=parsed.get("dataset"))
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data["start_time"],
                metadata={"trace_index": trace_index, "type": "hdf5"}
            )
            output_filename = f"trace_plot_hdf5_{trace_index}.png"
            output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
            os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)

            plot_path = plot_waveform_with_picks([tr], [], output_path)
            result["plot_path"] = plot_path
            if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                plot_markdown = f"![波形图]({plot_path})"

    if plot_markdown:
        return json.dumps(result, indent=2) + "\n\n" + plot_markdown
    return json.dumps(result, indent=2)

# 工具：HDF5 转 NumPy
@tool
def convert_hdf5_to_numpy(params: Union[str, dict, None] = None):
    """
    Convert HDF5 dataset to NumPy. Args: path (optional), dataset (optional), output_path.
    """
    """将 HDF5 的数据集转换为 NumPy。参数：path（可选）、dataset（可选）、output_path。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="hdf5_data.npy")
    handler = HDF5Handler(path)
    result = handler.to_numpy(output_path=output_path, dataset=parsed.get("dataset"))
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = result.get("output_path", output_path)

    # Plot first trace after conversion
    try:
        data = np.load(output_path) if output_path.endswith(".npy") else None
        if data is not None and data.size:
            first_trace = data[0] if data.ndim > 1 else data
            tr = TraceRecord(
                data=first_trace,
                sampling_rate=float(result.get("sampling_rate", 100.0)),
                start_time=None,
                metadata={"trace_index": 0, "type": "hdf5"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_numpy_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)
    return _build_artifact_response(result, input_path=path, output_path=output_path)


# 工具：HDF5 转 Excel
@tool
def convert_hdf5_to_excel(params: Union[str, dict, None] = None):
    """
    Convert HDF5 dataset to Excel. Args: path (optional), dataset (optional), output_path, start_trace, count.
    """
    """将 HDF5 的数据集转换为 Excel。参数：path（可选）、dataset（可选）、output_path、start_trace、count。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="hdf5_data.xlsx")
    try:
        start_trace = _coerce_int(parsed.get("start_trace"), default=0, field_name="start_trace")
        count = _coerce_int(parsed.get("count"), allow_none=True, default=None, field_name="count")
    except ValueError as exc:
        return str(exc)
    handler = HDF5Handler(path)
    result = handler.to_excel(output_path=output_path, start_trace=start_trace, count=count, dataset=parsed.get("dataset"))
    if isinstance(result, dict):
        return _build_artifact_response(result, input_path=path, output_path=output_path)
    return result

# 工具：HDF5 ZFP 压缩
@tool
def compress_hdf5_to_zfp(params: Union[str, dict, None] = None):
    """
    Compress a dataset in HDF5 using ZFP. Args: path (optional), dataset (optional), output_path, mode (rate|accuracy), value.
    """
    """对 HDF5 数据集使用 ZFP 压缩。参数：path（可选）、dataset（可选）、output_path、mode（rate|accuracy）、value。"""

    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."

    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="hdf5_zfp.h5")
    dataset_name = parsed.get("dataset")
    mode = (parsed.get("mode") or "rate").strip().lower()
    value = parsed.get("value")

    try:
        import hdf5plugin  # type: ignore
    except Exception as exc:
        return f"需要先安装 hdf5plugin 才能使用 ZFP 压缩: {exc}"

    def _pick_dataset(h5f):
        if dataset_name:
            node = h5f.get(dataset_name)
            if node is not None:
                return node
            raise ValueError(f"Dataset {dataset_name} not found in file")
        for preferred in getattr(HDF5Handler, "_PREFERRED_DATASETS", ()):  # type: ignore
            node = h5f.get(preferred)
            if node is not None:
                return node
        # Fallback: first dataset encountered
        chosen = None
        def visitor(name, node):
            nonlocal chosen
            if chosen is None and isinstance(node, h5py.Dataset):
                chosen = node
        h5f.visititems(visitor)
        if chosen is None:
            raise ValueError("No dataset found to compress")
        return chosen

    compression_kwargs = None
    mode_desc = None
    try:
        if mode == "rate":
            rate = float(value) if value is not None else 8.0
            compression_kwargs = hdf5plugin.Zfp(rate=rate)
            mode_desc = f"rate={rate}"
        elif mode == "accuracy":
            accuracy = float(value) if value is not None else 1e-3
            compression_kwargs = hdf5plugin.Zfp(accuracy=accuracy)
            mode_desc = f"accuracy={accuracy}"
        else:
            return "mode 仅支持 rate 或 accuracy"
    except Exception as exc:
        return f"ZFP 参数解析失败: {exc}"

    with h5py.File(path, "r") as src:
        try:
            source_dset = _pick_dataset(src)
        except Exception as exc:
            return str(exc)

        data = source_dset[()]
        source_name = source_dset.name or "/dataset"

        with h5py.File(output_path, "w") as dst:
            group = dst
            parent = os.path.dirname(source_name)
            if parent and parent not in {"", "/"}:
                group = dst.require_group(parent.lstrip("/"))

            target_name = os.path.basename(source_name) or "dataset"
            try:
                new_dset = group.create_dataset(target_name, data=data, **compression_kwargs)
            except Exception as exc:
                return f"创建压缩数据集失败: {exc}"

            # 复制属性
            for key, val in source_dset.attrs.items():
                new_dset.attrs[key] = val

            dst.attrs["source_file"] = os.path.abspath(path)
            dst.attrs["compression"] = "zfp"
            dst.attrs["zfp_mode"] = mode
            dst.attrs["zfp_param"] = mode_desc

    return json.dumps(
        {
            "source": os.path.abspath(path),
            "dataset": source_name,
            "output_path": os.path.abspath(output_path),
            "compression": "zfp",
            "mode": mode_desc,
            "shape": getattr(data, "shape", None),
            "dtype": str(getattr(data, "dtype", "")),
        },
        indent=2,
    )

# 工具：列出 HDF5 键
@tool
def get_hdf5_keys(params: Union[str, dict, None] = None):
    """
    List all groups and datasets with shapes and dtypes. Args: path (optional).
    """
    """列出所有分组与数据集及其形状和类型。参数：path（可选）。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_HDF5_PATH
    if not path:
        return "No HDF5 file is currently loaded."
    handler = HDF5Handler(path)
    result = handler.list_keys()
    return json.dumps(result, indent=2)

# 将 MiniSEED 转换为 NumPy
@tool
def convert_miniseed_to_numpy(params: Union[str, dict, None] = None):
    """
    Convert MiniSEED data to NumPy. Args: path (optional), output_path.
    """
    """将 MiniSEED 数据转换为 NumPy。参数：path（可选）、output_path。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_MINISEED_PATH
    if not path:
        return "No MiniSEED file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="miniseed_data.npz")
    handler = MiniSEEDHandler(path)
    result = handler.to_numpy(output_path=output_path)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = output_path

    # Plot first trace after conversion
    try:
        if output_path.endswith(".npz"):
            npz = np.load(output_path)
            # Get first array from npz
            keys = list(npz.keys())
            if keys:
                first_trace = npz[keys[0]]
        elif output_path.endswith(".npy"):
            data = np.load(output_path)
            if data is not None and data.size:
                first_trace = data[0] if data.ndim > 1 else data
        else:
            first_trace = None
        if first_trace is not None and first_trace.size:
            tr = TraceRecord(
                data=first_trace,
                sampling_rate=float(result.get("sampling_rate", 100.0)),
                start_time=None,
                metadata={"trace_index": 0, "type": "miniseed"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_numpy_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 将 MiniSEED 写入 HDF5
@tool
def convert_miniseed_to_hdf5(params: Union[str, dict, None] = None):
    """
    Convert MiniSEED data to HDF5. Args: path (optional), output_path, compression.
    """
    """将 MiniSEED 数据写入 HDF5。参数：path（可选）、output_path、compression。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_MINISEED_PATH
    if not path:
        return "No MiniSEED file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="miniseed_data.h5")
    compression = parsed.get("compression", "gzip")
    if isinstance(compression, str) and compression.lower() in {"none", "null", ""}:
        compression = None
    handler = MiniSEEDHandler(path)
    result = handler.to_hdf5(output_path=output_path, compression=compression)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = output_path

    # Plot first trace after conversion
    try:
        handler_h5 = HDF5Handler(output_path)
        record_data = handler_h5.get_trace_record_data(trace_index=0, dataset="traces")
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data.get("start_time"),
                metadata={"trace_index": 0, "type": "hdf5"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_hdf5_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 将 MiniSEED 导出为 SAC（每轨迹一个文件）
@tool
def convert_miniseed_to_sac(params: Union[str, dict, None] = None):
    """
    Export MiniSEED traces to SAC files. Args: path (optional), output_dir.
    """
    """将 MiniSEED 轨迹导出为 SAC 文件。参数：path（可选）、output_dir。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_MINISEED_PATH
    if not path:
        return "No MiniSEED file is currently loaded."
    output_dir = parsed.get("output_dir")
    if not output_dir or not str(output_dir).strip():
        output_dir = os.path.join(DEFAULT_CONVERT_DIR, "miniseed_sac")
    os.makedirs(output_dir, exist_ok=True)
    handler = MiniSEEDHandler(path)
    result = handler.to_sac(output_dir=output_dir)
    if "error" in result:
        return json.dumps(result, indent=2)

    # Plot first SAC file after conversion
    try:
        sac_files = [f for f in os.listdir(output_dir) if f.endswith(".sac")]
        if sac_files:
            first_sac = os.path.join(output_dir, sorted(sac_files)[0])
            handler_sac = SACHandler(first_sac)
            sac_data = handler_sac.get_trace_data(trace_index=0)
            if "error" not in sac_data:
                tr = TraceRecord(
                    data=sac_data["data"],
                    sampling_rate=sac_data["sampling_rate"],
                    start_time=sac_data.get("start_time"),
                    metadata={"trace_index": 0, "type": "sac"}
                )
                base_name = os.path.splitext(os.path.basename(path))[0]
                plot_filename = f"{base_name}_sac_plot.png"
                plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
                os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
                plot_result = plot_waveform_with_picks([tr], [], plot_path)
                if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                    result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 读取 SAC 结构信息
@tool
def get_sac_structure(params: Union[str, dict, None] = None):
    """
    Read SAC file basic structure info. Args: path (optional), plot (bool, default True).
    """
    """读取 SAC 文件的基本结构信息。参数：path（可选）、plot（bool，默认 True）。"""
    parsed = _parse_param_dict(params)
    path = _resolve_file_path(parsed.get("path")) if parsed.get("path") else CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    handler = SACHandler(path)
    info = handler.get_basic_info()

    # Default plot=True for better UX
    plot = parsed.get("plot", True)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    plot_markdown = None
    if plot:
        try:
            record_data = handler.get_trace_record_data(trace_index=0)
            if "error" not in record_data:
                tr = TraceRecord(
                    data=record_data["data"],
                    sampling_rate=record_data["sampling_rate"],
                    start_time=record_data["start_time"],
                    metadata={"trace_index": 0, "type": "sac"}
                )
                output_filename = "sac_structure_plot.png"
                output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
                os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
                plot_path = plot_waveform_with_picks([tr], [], output_path)
                info["plot_path"] = plot_path
                if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                    plot_markdown = f"\n\n![波形图]({plot_path})"
        except Exception as e:
            info["plot_error"] = str(e)

    if plot_markdown:
        return json.dumps(info, indent=2, ensure_ascii=False) + plot_markdown
    return json.dumps(info, indent=2, ensure_ascii=False)

# 读取 SAC 轨迹
@tool
def read_sac_trace(params: Union[str, dict, None] = None):
    """
    Read SAC trace by 0-based index. Args: path (optional), trace_index, plot (bool).
    """
    """按 0 基索引读取 SAC 轨迹。参数：path（可选）、trace_index, plot (bool)。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    try:
        trace_index = _coerce_int(parsed.get("trace_index"), default=0, field_name="trace_index")
    except ValueError as exc:
        return str(exc)
    
    plot = parsed.get("plot", False)
    if isinstance(plot, str):
        plot = plot.lower() in ("true", "yes", "1")

    handler = SACHandler(path)
    result = handler.get_trace_data(trace_index=trace_index)

    plot_markdown = None
    if plot and "error" not in result:
        record_data = handler.get_trace_record_data(trace_index=trace_index)
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data["start_time"],
                metadata={"trace_index": trace_index, "type": "sac"}
            )
            output_filename = f"trace_plot_sac_{trace_index}.png"
            output_path = os.path.join(DEFAULT_STRUCTURE_DIR, output_filename)
            os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)

            plot_path = plot_waveform_with_picks([tr], [], output_path)
            result["plot_path"] = plot_path
            if isinstance(plot_path, str) and plot_path.lower().endswith(".png"):
                plot_markdown = f"![波形图]({plot_path})"

    if plot_markdown:
        return json.dumps(result, indent=2) + "\n\n" + plot_markdown
    return json.dumps(result, indent=2)

# 将 SAC 转换为 NumPy
@tool
def convert_sac_to_numpy(params: Union[str, dict, None] = None):
    """
    Convert SAC data to NumPy. Args: path (optional), output_path.
    """
    """将 SAC 数据转换为 NumPy。参数：path（可选）、output_path。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="sac_data.npy")
    handler = SACHandler(path)
    result = handler.to_numpy(output_path=output_path)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = result.get("output_path", output_path)

    # Plot first trace after conversion
    try:
        data = np.load(output_path) if output_path.endswith(".npy") else None
        if data is not None and data.size:
            first_trace = data[0] if data.ndim > 1 else data
            tr = TraceRecord(
                data=first_trace,
                sampling_rate=float(result.get("sampling_rate", 100.0)),
                start_time=None,
                metadata={"trace_index": 0, "type": "sac"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_numpy_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 将 SAC 转换为 HDF5
@tool
def convert_sac_to_hdf5(params: Union[str, dict, None] = None):
    """
    Convert SAC data to HDF5. Args: path (optional), output_path, compression.
    """
    """将 SAC 数据写入 HDF5。参数：path（可选）、output_path、compression。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="sac_data.h5")
    compression = parsed.get("compression", "gzip")
    if isinstance(compression, str) and compression.lower() in {"none", "null", ""}:
        compression = None
    handler = SACHandler(path)
    result = handler.to_hdf5(output_path=output_path, compression=compression)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = result.get("output_path", output_path)

    # Plot first trace after conversion
    try:
        handler_h5 = HDF5Handler(output_path)
        record_data = handler_h5.get_trace_record_data(trace_index=0, dataset="traces")
        if "error" not in record_data:
            tr = TraceRecord(
                data=record_data["data"],
                sampling_rate=record_data["sampling_rate"],
                start_time=record_data.get("start_time"),
                metadata={"trace_index": 0, "type": "hdf5"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_hdf5_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 将 SAC 转换为 MiniSEED
@tool
def convert_sac_to_miniseed(params: Union[str, dict, None] = None):
    """
    Convert SAC traces to MiniSEED. Args: path (optional), output_path.
    """
    """将 SAC 数据转换为 MiniSEED。参数：path（可选）、output_path。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="sac_data.mseed")
    handler = SACHandler(path)
    result = handler.to_miniseed(output_path=output_path)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = result.get("output_path", output_path)

    # Plot first trace after conversion
    try:
        handler_ms = MiniSEEDHandler(output_path)
        ms_data = handler_ms.get_trace_record_data(trace_index=0)
        if "error" not in ms_data:
            tr = TraceRecord(
                data=ms_data["data"],
                sampling_rate=ms_data["sampling_rate"],
                start_time=ms_data.get("start_time"),
                metadata={"trace_index": 0, "type": "miniseed"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_miniseed_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)

# 将 SAC 转换为 Excel
@tool
def convert_sac_to_excel(params: Union[str, dict, None] = None):
    """
    Convert SAC data to Excel. Args: path (optional), output_path.
    """
    """将 SAC 数据转换为 Excel。参数：path（可选）、output_path。"""
    parsed = _parse_param_dict(params)
    path = parsed.get("path") or CURRENT_SAC_PATH
    if not path:
        return "No SAC file is currently loaded."
    output_path = _resolve_output_path(parsed.get("output_path"), default_filename="sac_data.xlsx")
    handler = SACHandler(path)
    result = handler.to_excel(output_path=output_path)
    if "error" in result:
        return json.dumps(result, indent=2)
    result["saved_to"] = result.get("output_path", output_path)

    # Plot first trace after conversion
    try:
        handler_sac = SACHandler(path)
        sac_data = handler_sac.get_trace_data(trace_index=0)
        if "error" not in sac_data:
            tr = TraceRecord(
                data=sac_data["data"],
                sampling_rate=sac_data["sampling_rate"],
                start_time=sac_data.get("start_time"),
                metadata={"trace_index": 0, "type": "sac"}
            )
            base_name = os.path.splitext(os.path.basename(path))[0]
            plot_filename = f"{base_name}_excel_plot.png"
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, plot_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_result = plot_waveform_with_picks([tr], [], plot_path)
            if isinstance(plot_result, str) and plot_result.lower().endswith(".png"):
                result["plot_path"] = plot_result
    except Exception as e:
        result["plot_error"] = str(e)

    return _build_artifact_response(result, input_path=path, output_path=output_path)


@tool
def pick_first_arrivals(params: Union[str, dict, None] = None):
    """
    Pick first arrivals (phases) on the currently loaded file.

    By default, uses deep learning methods (EQTransformer and PhaseNet) for picking.
    Traditional methods are only used when explicitly requested via the `methods` parameter.

    Available Methods:
    - eqtransformer: Deep learning P/S picker using EQTransformer (SeisBench). [DEFAULT]
    - phasenet: Deep learning P/S picker using PhaseNet (SeisBench). [DEFAULT]
    - gpd: Deep learning P/S picker using GPD (SeisBench).
    - sta_lta: Classic Short-Term/Long-Term Average ratio trigger.
    - aic: Akaike Information Criterion for precise onset refinement.
    - frequency_ratio: Detection based on spectral content changes.
    - autocorr: Autocorrelation-based detection.
    - feature_threshold: Simple amplitude/envelope thresholding.
    - ar_model: Auto-Regressive model prediction error.
    - template_correlation: Matched filtering with a waveform template.
    - pai_k: PAI-K kurtosis-based picker.
    - pai_s: PAI-S skewness-based picker.
    - s_phase: Heuristic S-wave picker (STA/LTA max after P-wave delay).

    Args:
      path (optional): File path override.
      file_type (optional): 'hdf5', 'sac', 'segy', 'miniseed'.
      dataset (optional): For HDF5, specific dataset name.
      methods (optional): Comma-separated list of methods (e.g. 'sta_lta,aic').
          If not specified, defaults to 'eqtransformer,phasenet' (deep learning only).
      method_params (optional): JSON string or dict of params for methods.
    """
    parsed = _parse_param_dict(params)
    
    # Determine source path and type
    path = parsed.get("path")
    file_type = parsed.get("file_type")
    
    if not path:
        if CURRENT_HDF5_PATH:
            path = CURRENT_HDF5_PATH
            if not file_type: file_type = "hdf5"
        elif CURRENT_SAC_PATH:
            path = CURRENT_SAC_PATH
            if not file_type: file_type = "sac"
        elif CURRENT_SEGY_PATH:
            path = CURRENT_SEGY_PATH
            if not file_type: file_type = "segy"
        elif CURRENT_MINISEED_PATH:
            path = CURRENT_MINISEED_PATH
            if not file_type: file_type = "miniseed"
            
    if not path:
        return "No file is currently loaded. Please load a file first."

    dataset = parsed.get("dataset")
    
    # Parse methods
    methods_str = parsed.get("methods")
    methods = [m.strip() for m in methods_str.split(",")] if methods_str else None
    
    # Parse method_params
    method_params_raw = parsed.get("method_params")
    method_params = None
    if method_params_raw:
        if isinstance(method_params_raw, dict):
            method_params = method_params_raw
        elif isinstance(method_params_raw, str):
            try:
                method_params = json.loads(method_params_raw)
            except json.JSONDecodeError:
                pass # Or handle error appropriately

    try:
        # 1. Pick phases
        picks = pick_phases(
            source=path,
            file_type=file_type,
            dataset=dataset,
            methods=methods,
            method_params=method_params
        )

        # Store picks for later use by locate_earthquake
        global CURRENT_PICKS
        CURRENT_PICKS = picks

        summary = summarize_pick_results(picks)
        traces = load_traces(
            source=path,
            file_type=file_type,
            dataset=dataset
        )

        # Find best trace:
        # 1) Prefer traces with both P and S picks
        # 2) Then choose highest total confidence score
        best_trace_idx = 0
        best_trace_score = -1.0
        best_has_both = False
        for item in summary:
            trace_idx = int(item.get("trace_index", 0))
            best_p = item.get("best_p") or {}
            best_s = item.get("best_s") or {}
            phase_scores = []
            for best_pick in (best_p, best_s):
                if isinstance(best_pick, dict):
                    score = best_pick.get("weighted_score")
                    if score is None:
                        score = best_pick.get("score")
                    if score is not None:
                        phase_scores.append(float(score))
            total_score = sum(phase_scores) if phase_scores else float(item.get("average_score") or 0.0)
            has_both = bool(best_p and best_s)
            if (
                (has_both and not best_has_both)
                or (has_both == best_has_both and total_score > best_trace_score)
                or (
                    has_both == best_has_both
                    and total_score == best_trace_score
                    and _trace_channel_priority(traces[trace_idx] if trace_idx < len(traces) else traces[0])
                    < _trace_channel_priority(traces[best_trace_idx] if best_trace_idx < len(traces) else traces[0])
                )
            ):
                best_trace_score = total_score
                best_trace_idx = trace_idx
                best_has_both = has_both

        # Filter to best trace only
        best_traces = [t for t in traces if t.metadata.get("trace_index", 0) == best_trace_idx]
        if not best_traces and traces:
            best_traces = [traces[best_trace_idx] if best_trace_idx < len(traces) else traces[0]]

        best_picks = _select_best_p_s_for_trace(picks, best_trace_idx)

        # 3. Generate plot with filename-based naming
        station_name = os.path.splitext(os.path.basename(path))[0]
        plot_filename = f"{station_name}_picks.png"
        plot_path = os.path.join(DEFAULT_PICKS_DIR, plot_filename)
        os.makedirs(DEFAULT_PICKS_DIR, exist_ok=True)
        configure_plot_fonts()
        plot_waveform_with_picks(best_traces, best_picks, plot_path)

        # 3.5. Save picks to CSV
        csv_path = None
        if picks:
            csv_path = os.path.join(DEFAULT_PICKS_DIR, f"{station_name}_picks.csv")
            os.makedirs(DEFAULT_PICKS_DIR, exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['station', 'trace_index', 'phase_type', 'method', 'sample_index', 'absolute_time', 'normalized_score'])
                for pick in picks:
                    writer.writerow([
                        station_name,
                        pick.trace_index,
                        pick.phase_type or 'P',
                        pick.method,
                        pick.sample_index,
                        pick.absolute_time or '',
                        pick.normalized_score if pick.normalized_score is not None else ''
                    ])

        # 4. Build a brief Chinese summary (best P / best S per trace)
        best_by_trace: dict[int, dict[str, dict]] = {}
        for item in summary:
            trace_index = int(item.get("trace_index", 0))
            best_by_trace.setdefault(trace_index, {})
            for m in item.get("methods", []):
                phase = (m.get("phase_type") or "P").upper()
                score = m.get("normalized_score")
                if score is None:
                    continue
                current = best_by_trace[trace_index].get(phase)
                if current is None or (current.get("normalized_score") is None) or (score > current.get("normalized_score")):
                    best_by_trace[trace_index][phase] = m
        
        # Format as a Markdown output (language-aware)
        _l = CURRENT_LANG
        lines = []
        lines.append(f"![{'拾取结果图' if _l == 'zh' else 'Picking Results'}]({plot_path})")
        lines.append("")

        lines.append(f"{'初至拾取已完成，图表已保存至' if _l == 'zh' else 'Phase picking completed, plot saved to'}：`{plot_path}`")
        if csv_path:
            lines.append(f"{'拾取结果CSV已保存至' if _l == 'zh' else 'Picks CSV saved to'}：`{csv_path}`")
        for trace_index in sorted(best_by_trace.keys()):
            best_p = best_by_trace[trace_index].get("P")
            best_s = best_by_trace[trace_index].get("S")
            parts = []
            if best_p:
                if _l == "zh":
                    parts.append(f"最佳P波：方法 {best_p.get('method')}，样本点 {best_p.get('sample_index')}，时间 {best_p.get('absolute_time')}，评分 {best_p.get('normalized_score'):.4f}")
                else:
                    parts.append(f"Best P: {best_p.get('method')}, sample {best_p.get('sample_index')}, time {best_p.get('absolute_time')}, score {best_p.get('normalized_score'):.4f}")
            else:
                parts.append("最佳P波：未检出" if _l == "zh" else "Best P: not detected")
            if best_s:
                if _l == "zh":
                    parts.append(f"最佳S波：方法 {best_s.get('method')}，样本点 {best_s.get('sample_index')}，时间 {best_s.get('absolute_time')}，评分 {best_s.get('normalized_score'):.4f}")
                else:
                    parts.append(f"Best S: {best_s.get('method')}, sample {best_s.get('sample_index')}, time {best_s.get('absolute_time')}, score {best_s.get('normalized_score'):.4f}")
            else:
                parts.append("最佳S波：未检出" if _l == "zh" else "Best S: not detected")
            sep = "；" if _l == "zh" else "; "
            lines.append(f"{'摘要' if _l == 'zh' else 'Summary'}（Trace {trace_index}）：{sep.join(parts)}")
        lines.append("")

        # Table headers
        hdr_phase = "相位" if _l == "zh" else "Phase"
        hdr_method = "方法" if _l == "zh" else "Method"
        hdr_sample = "样本点" if _l == "zh" else "Sample"
        hdr_score = "评分" if _l == "zh" else "Score"
        hdr_time = "绝对时间" if _l == "zh" else "Absolute Time"

        METHOD_DISPLAY = {
            "zh": {
                "sta_lta": "STA/LTA", "aic": "AIC",
                "frequency_ratio": "频率比", "autocorr": "自相关",
                "feature_threshold": "特征阈值", "ar_model": "AR 模型",
                "template_correlation": "模板相关", "s_phase": "S 波拾取",
                "pai_k": "PAI-K", "pai_s": "PAI-S",
            },
            "en": {},  # use original method names
        }

        for item in summary:
            lines.append(f"### Trace {item['trace_index']}")
            lines.append(f"| {hdr_phase} | {hdr_method} | {hdr_sample} | {hdr_score} | {hdr_time} |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")

            # Sort methods by score descending
            sorted_methods = sorted(item['methods'], key=lambda x: x['normalized_score'] or 0, reverse=True)
            for m in sorted_methods:
                score = f"{m['normalized_score']:.4f}" if m['normalized_score'] is not None else "N/A"
                phase = m.get('phase_type', 'P')
                method_key = m['method']
                method_display = METHOD_DISPLAY[_l].get(method_key, method_key)
                lines.append(f"| {phase} | {method_display} | {m['sample_index']} | {score} | {m['absolute_time']} |")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error picking phases: {str(e)}"


@tool
def pick_all_miniseed_files(params: Union[str, dict, None] = None):
    """
    Pick first arrivals (phases) on ALL loaded MiniSEED files.

    Use this tool when you have uploaded multiple MiniSEED files (one per station)
    and want to pick phases on all of them for earthquake location.

    By default, uses deep learning methods (EQTransformer and PhaseNet) for picking.
    Traditional methods are only used when explicitly requested via the `methods` parameter.

    This tool will:
    1. Pick P and S phases on each loaded MiniSEED file
    2. Store all picks for later use by locate_earthquake
    3. Generate only one best-result plot (full results are saved in CSV)

    Args:
        params: Dictionary with optional parameters:
            - methods: Comma-separated list of methods (default: 'eqtransformer,phasenet')
            - method_params: JSON string or dict of params for methods
    """
    global CURRENT_PICKS, CURRENT_MINISEED_PATHS

    parsed = _parse_param_dict(params)

    if not CURRENT_MINISEED_PATHS:
        _l = CURRENT_LANG
        return json.dumps({
            "error": "No MiniSEED files loaded.",
            "hint": "请先上传 MiniSEED 文件。" if _l == "zh" else "Please upload MiniSEED files first."
        }, ensure_ascii=False, indent=2)

    # Parse methods
    methods_str = parsed.get("methods")
    methods = [m.strip() for m in methods_str.split(",")] if methods_str else None

    # Parse method_params
    method_params_raw = parsed.get("method_params")
    method_params = None
    if method_params_raw:
        if isinstance(method_params_raw, dict):
            method_params = method_params_raw
        elif isinstance(method_params_raw, str):
            try:
                method_params = json.loads(method_params_raw)
            except json.JSONDecodeError:
                pass

    all_picks = []
    all_traces = []
    file_summaries = []
    picks_by_station = []  # List of (station_name, picks) tuples for CSV
    plot_candidates = []  # Collect per-file best trace/picks, then plot only global best

    try:
        for filepath in CURRENT_MINISEED_PATHS:
            try:
                # Pick phases on this file
                file_picks = pick_phases(
                    source=filepath,
                    file_type="miniseed",
                    methods=methods,
                    method_params=method_params
                )

                # Load traces for plotting
                file_traces = load_traces(
                    source=filepath,
                    file_type="miniseed"
                )

                all_picks.extend(file_picks)
                all_traces.extend(file_traces)

                # Get station name from filename
                filename = os.path.basename(filepath)
                station_name = os.path.splitext(filename)[0]

                # Store picks with station name for CSV, attach filepath to each pick
                for pick in file_picks:
                    pick.metadata['filepath'] = filepath
                picks_by_station.append((station_name, file_picks))

                # Get summary to find best trace
                file_summary = summarize_pick_results(file_picks)

                # Find best trace:
                # 1) Prefer traces with both P and S picks
                # 2) Then choose highest total confidence score
                best_trace_idx = 0
                best_trace_score = -1.0
                best_has_both = False
                for item in file_summary:
                    trace_idx = int(item.get("trace_index", 0))
                    best_p = item.get("best_p") or {}
                    best_s = item.get("best_s") or {}
                    phase_scores = []
                    for best_pick in (best_p, best_s):
                        if isinstance(best_pick, dict):
                            score = best_pick.get("weighted_score")
                            if score is None:
                                score = best_pick.get("score")
                            if score is not None:
                                phase_scores.append(float(score))
                    total_score = sum(phase_scores) if phase_scores else float(item.get("average_score") or 0.0)
                    has_both = bool(best_p and best_s)
                    if (
                        (has_both and not best_has_both)
                        or (has_both == best_has_both and total_score > best_trace_score)
                        or (
                            has_both == best_has_both
                            and total_score == best_trace_score
                            and _trace_channel_priority(file_traces[trace_idx] if trace_idx < len(file_traces) else file_traces[0])
                            < _trace_channel_priority(file_traces[best_trace_idx] if best_trace_idx < len(file_traces) else file_traces[0])
                        )
                    ):
                        best_trace_score = total_score
                        best_trace_idx = trace_idx
                        best_has_both = has_both

                # Filter to best trace only
                best_traces = [t for t in file_traces if t.metadata.get("trace_index", 0) == best_trace_idx]
                if not best_traces and file_traces:
                    best_traces = [file_traces[best_trace_idx] if best_trace_idx < len(file_traces) else file_traces[0]]
                best_picks = _select_best_p_s_for_trace(file_picks, best_trace_idx)

                file_summaries.append({
                    "file": filename,
                    "station": station_name,
                    "traces": len(file_traces),
                    "picks": len(file_picks),
                    "best_trace": best_trace_idx
                })

                if best_traces and best_picks:
                    plot_candidates.append({
                        "station": station_name,
                        "best_score": best_trace_score,
                        "traces": best_traces,
                        "picks": best_picks,
                    })

            except Exception as e:
                file_summaries.append({
                    "file": os.path.basename(filepath),
                    "error": str(e)
                })

        # Store all picks for location
        CURRENT_PICKS = all_picks

        # Save all picks to CSV
        csv_path = None
        if picks_by_station:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(DEFAULT_PICKS_DIR, f"picks_{timestamp}.csv")
            os.makedirs(DEFAULT_PICKS_DIR, exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['original_file', 'station', 'trace_index', 'phase_type', 'method', 'sample_index', 'absolute_time', 'normalized_score'])
                for station_name, station_picks in picks_by_station:
                    for pick in station_picks:
                        writer.writerow([
                            os.path.basename(pick.metadata.get('filepath', '')),
                            station_name,
                            pick.trace_index,
                            pick.phase_type or 'P',
                            pick.method,
                            pick.sample_index,
                            pick.absolute_time or '',
                            pick.normalized_score if pick.normalized_score is not None else ''
                        ])

        # Generate ONE plot only: global best result among all files/traces
        plot_path = None
        if plot_candidates:
            best_item = max(plot_candidates, key=lambda x: x.get("best_score", -1))
            station_name = best_item["station"]
            plot_path = os.path.join(DEFAULT_PICKS_DIR, f"{station_name}_picks.png")
            os.makedirs(DEFAULT_PICKS_DIR, exist_ok=True)
            configure_plot_fonts()
            plot_waveform_with_picks(best_item["traces"], best_item["picks"], plot_path)

        # Build output (language-aware)
        _l = CURRENT_LANG
        lines = []

        if plot_path:
            lines.append(f"![{'拾取结果图' if _l == 'zh' else 'Picking Results'}]({plot_path})")
            lines.append("")

        lines.append(f"**{'初至拾取完成' if _l == 'zh' else 'Phase picking completed'}**")
        lines.append(f"- {'处理文件数' if _l == 'zh' else 'Files processed'}: {len(CURRENT_MINISEED_PATHS)}")
        lines.append(f"- {'总拾取数' if _l == 'zh' else 'Total picks'}: {len(all_picks)}")
        lines.append(f"- {'总轨迹数' if _l == 'zh' else 'Total traces'}: {len(all_traces)}")
        if plot_path:
            lines.append(f"- {'最佳结果图' if _l == 'zh' else 'Best result plot'}: `{plot_path}`")
        if csv_path:
            lines.append(f"- {'拾取结果CSV' if _l == 'zh' else 'Picks CSV'}: `{csv_path}`")
        lines.append("")

        # Show per-file summary
        lines.append(f"### {'各文件拾取结果' if _l == 'zh' else 'Per-file Results'}")
        for fs in file_summaries:
            if "error" in fs:
                lines.append(f"- **{fs['file']}**: {'错误' if _l == 'zh' else 'error'} - {fs['error']}")
            else:
                if _l == "zh":
                    lines.append(f"- **{fs['file']}**: {fs['picks']} 个拾取, {fs['traces']} 条轨迹, {'最佳道' if _l == 'zh' else 'best trace'} #{fs.get('best_trace', 0)}")
                else:
                    lines.append(f"- **{fs['file']}**: {fs['picks']} picks, {fs['traces']} traces, best trace #{fs.get('best_trace', 0)}")

        lines.append("")
        if _l == "zh":
            lines.append("拾取结果已保存，可使用 `locate_earthquake` 工具进行定位。")
            lines.append("如需添加台站坐标，请使用 `add_station_coordinates` 工具。")
        else:
            lines.append("Picks saved. Use `locate_earthquake` for hypocenter determination.")
            lines.append("Use `add_station_coordinates` to add station coordinates.")

        return "\n".join(lines)

    except Exception as e:
        _l = CURRENT_LANG
        return f"{'批量拾取失败' if _l == 'zh' else 'Batch picking failed'}: {str(e)}"


# ==================== Earthquake Location Tools ====================

# Store picks for location
CURRENT_PICKS = None
CURRENT_STATIONS = None
CURRENT_LOCATION = None  # Store last location result for re-plotting


def set_current_picks(picks):
    """Store current picks for location."""
    global CURRENT_PICKS
    CURRENT_PICKS = picks


def set_current_stations(stations):
    """Store current station metadata for location."""
    global CURRENT_STATIONS
    CURRENT_STATIONS = stations


def _load_validation_module(module_key: str):
    """Dynamically load validation modules under Validation/near-seismic-location."""
    if module_key in _VALIDATION_MODULE_CACHE:
        return _VALIDATION_MODULE_CACHE[module_key]

    mapping = {
        "regular": "nearseismic_location_validation.py",
        "blind": "nearseismic_location_blind_validation.py",
        "continuous": "continuous_monitoring_validation.py",
        "catalog_plot": "plot_catalog_debug.py",
    }
    filename = mapping.get(module_key)
    if not filename:
        raise ValueError(f"Unknown validation module key: {module_key}")

    file_path = os.path.join(
        os.getcwd(), "Validation", "near-seismic-location", filename
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Validation script not found: {file_path}")

    module_name = f"quakecore_{module_key}_validation"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validation module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VALIDATION_MODULE_CACHE[module_key] = module
    return module


def _ensure_nearseismic_taup_assets(mode: str = "regular"):
    """Ensure near-seismic TauP assets exist and are ready.

    Uses centralized assets under resources/taup.
    Migrates from legacy Validation/near-seismic-location/taup if needed.
    If any required file is missing, triggers validation module builders.
    """
    module_key = "blind" if mode in {"blind", "blind_nearseismic"} else "regular"
    module = _load_validation_module(module_key)

    project_root = os.getcwd()
    taup_dir = os.path.join(project_root, "resources", "taup")
    legacy_taup_dir = os.path.join(project_root, "Validation", "near-seismic-location", "taup")
    os.makedirs(taup_dir, exist_ok=True)

    # Migrate existing cache/model files from legacy location once.
    for fname in ("socal.npz", "socal.tvel", "socal.nd", "tt_interp_cache_v2.npz"):
        src = os.path.join(legacy_taup_dir, fname)
        dst = os.path.join(taup_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Force validation modules to use centralized TAUP directory.
    previous_taup_dir = getattr(module, "TAUP_DIR", None)
    module.TAUP_DIR = taup_dir
    if hasattr(module, "_TT_CACHE_FILE"):
        module._TT_CACHE_FILE = os.path.join(taup_dir, "tt_interp_cache_v2.npz")

    # Reset in-module singleton caches only when path switches.
    if previous_taup_dir != taup_dir:
        if hasattr(module, "_taup_model"):
            module._taup_model = None
        if hasattr(module, "_tt_interp"):
            module._tt_interp = None

    required = [
        os.path.join(taup_dir, "socal.npz"),
        os.path.join(taup_dir, "tt_interp_cache_v2.npz"),
    ]

    existed_before = {os.path.basename(p): os.path.exists(p) for p in required}

    # Build/load TauP model and travel-time interpolation cache.
    # Validation modules encapsulate the correct build strategy.
    module._get_taup()
    module._get_tt_interp()

    existed_after = {os.path.basename(p): os.path.exists(p) for p in required}

    return {
        "taup_dir": taup_dir,
        "required_files": [os.path.basename(p) for p in required],
        "existed_before": existed_before,
        "existed_after": existed_after,
    }


def _to_nearseismic_station_dict(stations_dict):
    """Convert station objects/dicts to near-seismic script station format."""
    converted = {}
    for sta_id, sta in stations_dict.items():
        if hasattr(sta, "latitude"):
            net = getattr(sta, "network", "XX")
            name = getattr(sta, "station", "UNK")
            latitude = float(getattr(sta, "latitude", 0.0))
            longitude = float(getattr(sta, "longitude", 0.0))
            elevation = float(getattr(sta, "elevation", 0.0) or 0.0)
        elif isinstance(sta, dict):
            net = sta.get("network", "XX")
            name = sta.get("station", sta.get("name", "UNK"))
            latitude = float(sta.get("latitude", sta.get("lat", 0.0)) or 0.0)
            longitude = float(sta.get("longitude", sta.get("lon", 0.0)) or 0.0)
            elevation = float(sta.get("elevation", sta.get("elev", 0.0)) or 0.0)
        else:
            continue

        key = f"{net}.{name}"
        if sta_id and isinstance(sta_id, str) and "." in sta_id:
            key = sta_id

        if latitude == 0.0 and longitude == 0.0:
            continue

        converted[key] = {
            "network": net,
            "station": name,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
        }
    return converted


def _to_nearseismic_pick_list(valid_picks):
    """Convert PhasePick list to near-seismic script pick format."""
    picks = []
    for p in valid_picks:
        sta_id = f"{p.station.network}.{p.station.station}"
        phase = str(getattr(p, "phase_type", "P") or "P").upper()
        if phase not in {"P", "S"}:
            continue
        score = float(getattr(p, "weight", 1.0) or 1.0)
        picks.append({
            "station_id": sta_id,
            "network": p.station.network,
            "station": p.station.station,
            "phase": phase,
            "time_str": str(getattr(p, "arrival_time", "")),
            "score": score,
            "method": str((getattr(p, "metadata", {}) or {}).get("method", "unknown")),
            "time": float(getattr(p, "arrival_time", 0.0)),
        })
    return picks


@tool
def prepare_nearseismic_taup_cache(params: Union[str, dict, None] = None):
    """Prepare/reuse near-seismic TauP cache files for QuakeCore location.

    Args:
        params:
            - mode: "regular" or "blind" (default: "regular")
    """
    parsed = _parse_param_dict(params)
    mode = str(parsed.get("mode", "regular")).strip().lower()
    try:
        info = _ensure_nearseismic_taup_assets(mode=mode)
        return json.dumps({
            "success": True,
            "mode": mode,
            "taup_dir": info["taup_dir"],
            "required_files": info["required_files"],
            "existed_before": info["existed_before"],
            "existed_after": info["existed_after"],
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "mode": mode,
            "error": str(e),
        }, ensure_ascii=False, indent=2)


@tool
def locate_earthquake(params: Union[str, dict, None] = None):
    """
    Locate earthquake hypocenter using phase picks from multiple stations.

    This tool requires:
    1. Phase picks with arrival times (from pick_first_arrivals)
    2. Station coordinates (latitude, longitude)

    The location algorithm supports three families:
    - Default locator: grid search + Geiger with IASP91/TauP
    - Near-seismic validator locator: 2-stage EDT + P-S depth search
    - Blind near-seismic locator: REAL-Lite association + EDT grid search

    Args:
        params: Dictionary with optional parameters:
            - method: "auto", "grid_search", "geiger", "nearseismic", or "blind_nearseismic" (default: "auto")
            - grid_center: [lat, lon] center of search grid
            - grid_size_deg: Search grid half-width in degrees (default: 2.0)
            - depth_range_km: [min, max] depth range in km (default: [0, 50])
            - seed_depth_km: Initial depth hint for nearseismic mode (default: 10)
            - stations: List of station metadata with lat/lon coordinates

    Returns:
        JSON with hypocenter location (lat, lon, depth, origin_time) and statistics.
    """
    """
    使用多台站的震相拾取结果定位震源。

    此工具需要：
    1. 震相到时（来自 pick_first_arrivals）
    2. 台站坐标（纬度、经度）

    定位算法使用：
    - 网格搜索获取初始估计
    - 盖格法（最小二乘）进行精修
    - IASP91 速度模型计算走时

    参数：
        params: 可选参数字典：
            - method: "auto"、"grid_search" 或 "geiger"（默认："auto"）
            - grid_center: [纬度, 经度] 搜索网格中心
            - grid_size_deg: 搜索网格半宽度（度）（默认：2.0）
            - depth_range_km: [最小, 最大] 深度范围（公里）（默认：[0, 50]）
            - stations: 包含经纬度坐标的台站元数据列表

    返回：
        包含震源位置（纬度、经度、深度、发震时刻）和统计信息的 JSON。
    """
    from utils.locator import (
        EarthquakeLocator,
        PhasePick,
        Station,
        picks_from_pick_results,
        locate_earthquake as do_locate,
        OBSPY_AVAILABLE
    )

    parsed = _parse_param_dict(params)

    # Check if we have stored picks
    if CURRENT_PICKS is None or len(CURRENT_PICKS) == 0:
        return json.dumps({
            "error": "No picks available. Please run pick_first_arrivals first.",
            "hint": "使用 pick_first_arrivals 工具进行震相拾取后再定位。"
        }, ensure_ascii=False, indent=2)

    # Get station metadata
    stations_dict = {}
    stations_input = parsed.get("stations", CURRENT_STATIONS)

    if isinstance(stations_input, dict):
        # It's a dict with station_id as keys
        for sta_id, sta_info in stations_input.items():
            if isinstance(sta_info, dict):
                net = sta_info.get("network", "XX")
                name = sta_info.get("station", sta_info.get("name", "UNK"))
                lat = sta_info.get("latitude", sta_info.get("lat"))
                lon = sta_info.get("longitude", sta_info.get("lon"))
                elev = sta_info.get("elevation", sta_info.get("elev", 0))

                if lat is not None and lon is not None:
                    stations_dict[sta_id] = Station(
                        network=net,
                        station=name,
                        latitude=float(lat),
                        longitude=float(lon),
                        elevation=float(elev) if elev else 0.0,
                        metadata=sta_info
                    )
    elif isinstance(stations_input, list):
        # It's a list of station dicts
        for sta in stations_input:
            if isinstance(sta, dict):
                net = sta.get("network", "XX")
                name = sta.get("station", sta.get("name", "UNK"))
                lat = sta.get("latitude", sta.get("lat"))
                lon = sta.get("longitude", sta.get("lon"))
                elev = sta.get("elevation", sta.get("elev", 0))

                if lat is not None and lon is not None:
                    sta_id = f"{net}.{name}"
                    stations_dict[sta_id] = Station(
                        network=net,
                        station=name,
                        latitude=float(lat),
                        longitude=float(lon),
                        elevation=float(elev) if elev else 0.0,
                        metadata=sta
                    )

    # Convert picks to PhasePick objects
    phase_picks = picks_from_pick_results(CURRENT_PICKS, stations_dict)

    # Check for valid picks with station coordinates
    valid_picks = [
        p for p in phase_picks
        if p.station.latitude != 0 or p.station.longitude != 0
    ]

    if len(valid_picks) < 3:
        # Need station coordinates
        missing_stations = list(set(
            f"{p.station.network}.{p.station.station}"
            for p in phase_picks
            if p.station.latitude == 0 and p.station.longitude == 0
        ))

        return json.dumps({
            "error": f"Need station coordinates for location. Only {len(valid_picks)}/{len(phase_picks)} picks have valid coordinates.",
            "missing_stations": missing_stations,
            "hint": "请提供台站坐标信息。在参数中添加 stations 列表，包含每个台站的 network, station, latitude, longitude。",
            "example": {
                "stations": [
                    {"network": "IU", "station": "ANMO", "latitude": 34.9459, "longitude": -106.4572},
                    {"network": "IU", "station": "BJI", "latitude": 40.0409, "longitude": 116.1754}
                ]
            }
        }, ensure_ascii=False, indent=2)

    # Get location parameters
    method = parsed.get("method", "auto")
    grid_center = parsed.get("grid_center")
    grid_size = parsed.get("grid_size_deg", 2.0)
    depth_range = parsed.get("depth_range_km", [0.0, 50.0])

    # Prepare kwargs
    kwargs = {}
    if grid_center:
        kwargs["grid_center"] = tuple(grid_center)
    if grid_size:
        kwargs["grid_size_deg"] = float(grid_size)
    if depth_range:
        kwargs["grid_depth_range"] = tuple(depth_range)

    # Perform location
    try:
        if method in {"nearseismic", "grid_search_edt_local"}:
            taup_status = _ensure_nearseismic_taup_assets(mode="regular")
            module = _load_validation_module("regular")
            near_stations = _to_nearseismic_station_dict(stations_dict)
            near_picks = _to_nearseismic_pick_list(valid_picks)

            if len(near_stations) < 3:
                return json.dumps({
                    "error": "Need at least 3 stations with valid coordinates for nearseismic location.",
                    "num_stations": len(near_stations),
                }, ensure_ascii=False, indent=2)

            if len(near_picks) < 4:
                return json.dumps({
                    "error": "Need at least 4 valid picks for nearseismic location.",
                    "num_picks": len(near_picks),
                }, ensure_ascii=False, indent=2)

            center = parsed.get("grid_center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                seed_lat, seed_lon = float(center[0]), float(center[1])
            else:
                lat_arr = np.array([s["latitude"] for s in near_stations.values()])
                lon_arr = np.array([s["longitude"] for s in near_stations.values()])
                seed_lat, seed_lon = float(np.mean(lat_arr)), float(np.mean(lon_arr))

            seed_depth = _coerce_float(
                parsed.get("seed_depth_km"), allow_none=True, default=10.0, field_name="seed_depth_km"
            )

            loc = module.locate_grid_search_local(
                near_picks,
                near_stations,
                seed_lat,
                seed_lon,
                float(seed_depth),
                cache_dir=None,
            )
            if not loc:
                return json.dumps({
                    "error": "Near-seismic EDT location failed.",
                    "method": "nearseismic",
                }, ensure_ascii=False, indent=2)

            result = {
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "depth_km": float(loc.get("depth", 0.0)),
                "origin_time": float(loc.get("ot", 0.0) or 0.0),
                "origin_time_iso": str(loc.get("ot", "")),
                "rms_residual": float(loc.get("rms", -1.0) or -1.0),
                "azimuthal_gap": float(loc.get("gap", 360.0) or 360.0),
                "num_picks": int(loc.get("num_picks", len(near_picks))),
                "num_stations": int(len(near_stations)),
                "method": "nearseismic_edt",
                "picks": loc.get("clean_picks", near_picks),
                "taup_cache": taup_status,
            }

        elif method in {"blind_nearseismic", "real_lite_edt"}:
            taup_status = _ensure_nearseismic_taup_assets(mode="blind")
            module = _load_validation_module("blind")
            near_stations = _to_nearseismic_station_dict(stations_dict)
            near_picks = _to_nearseismic_pick_list(valid_picks)

            if len(near_stations) < 3:
                return json.dumps({
                    "error": "Need at least 3 stations with valid coordinates for blind nearseismic location.",
                    "num_stations": len(near_stations),
                }, ensure_ascii=False, indent=2)

            if len(near_picks) < 4:
                return json.dumps({
                    "error": "Need at least 4 picks for blind nearseismic location.",
                    "num_picks": len(near_picks),
                }, ensure_ascii=False, indent=2)

            tt_interp = module._get_tt_interp()
            associated_picks, init_est = module.associate_by_origin_time(
                near_picks,
                near_stations,
                tt_interp,
                time_tolerance=1.5,
                min_picks=4,
            )

            if not associated_picks or not init_est:
                return json.dumps({
                    "error": "Blind association failed: no reliable event cluster.",
                    "method": "blind_nearseismic",
                }, ensure_ascii=False, indent=2)

            loc = module.locate_grid_search_local_blind(
                associated_picks,
                near_stations,
                float(init_est["latitude"]),
                float(init_est["longitude"]),
                float(init_est.get("depth", 10.0)),
            )

            if not loc:
                return json.dumps({
                    "error": "Blind near-seismic EDT location failed.",
                    "method": "blind_nearseismic",
                }, ensure_ascii=False, indent=2)

            result = {
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "depth_km": float(loc.get("depth", 0.0)),
                "origin_time": float(init_est.get("time", 0.0) or 0.0),
                "origin_time_iso": str(init_est.get("time", "")),
                "rms_residual": -1.0,
                "azimuthal_gap": 360.0,
                "num_picks": int(loc.get("num_picks", len(associated_picks))),
                "num_stations": int(len(near_stations)),
                "method": "blind_real_lite_edt",
                "picks": associated_picks,
                "taup_cache": taup_status,
            }

        else:
            result = do_locate(valid_picks, method=method, **kwargs)

        if "error" in result:
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Format output
        output = {
            "success": True,
            "hypocenter": {
                "latitude": round(result["latitude"], 4),
                "longitude": round(result["longitude"], 4),
                "depth_km": round(result["depth_km"], 2),
                "origin_time": result.get("origin_time_iso", result["origin_time"]),
            },
            "quality": {
                "rms_residual_s": round(result["rms_residual"], 3),
                "azimuthal_gap_deg": round(result["azimuthal_gap"], 1),
                "num_picks": result["num_picks"],
                "num_stations": result["num_stations"],
                "method": result["method"],
            },
            "picks_used": result.get("picks", []),
        }

        # Add quality assessment
        gap = result["azimuthal_gap"]
        rms = result["rms_residual"]

        if gap > 180:
            output["quality"]["warning"] = "Azimuthal gap > 180°, location may have large uncertainty."
        elif gap > 270:
            output["quality"]["warning"] = "Azimuthal gap > 270°, location unreliable."

        if rms > 1.0:
            output["quality"]["warning"] = f"RMS residual is high ({rms:.2f}s), check pick quality."

        # Store location result for re-plotting
        global CURRENT_LOCATION
        station_list_for_store = []
        if stations_dict:
            for sta in stations_dict.values():
                if hasattr(sta, "latitude"):
                    station_list_for_store.append({
                        "network": sta.network,
                        "station": sta.station,
                        "latitude": sta.latitude,
                        "longitude": sta.longitude,
                        "elevation": sta.elevation,
                    })
                elif isinstance(sta, dict):
                    station_list_for_store.append(sta)
        if not station_list_for_store:
            station_list_for_store = [
                {
                    "latitude": p.station.latitude,
                    "longitude": p.station.longitude,
                    "station": p.station.station,
                    "network": p.station.network,
                }
                for p in valid_picks[:20]
            ]
        CURRENT_LOCATION = {
            "hypocenter": output["hypocenter"],
            "stations": station_list_for_store,
        }

        # Generate location map using Cartopy (with matplotlib fallback)
        try:
            from utils.locator import plot_location_map

            # Get station list for plotting (reuse stored dict list)
            station_list = station_list_for_store
            if not station_list:
                # Create station list from valid picks
                station_list = [
                    {
                        "latitude": p.station.latitude,
                        "longitude": p.station.longitude,
                        "station": p.station.station,
                        "network": p.station.network,
                    }
                    for p in valid_picks[:20]  # Limit for plotting
                ]

            # Generate map
            map_path = os.path.join(DEFAULT_LOCATION_DIR, "earthquake_location_map.png")
            os.makedirs(DEFAULT_LOCATION_DIR, exist_ok=True)

            plot_result = plot_location_map(
                hypocenter=output["hypocenter"],
                stations=station_list,
                output_path=map_path,
            )

            if plot_result and os.path.exists(map_path):
                output["location_map"] = map_path
        except Exception as plot_error:
            print(f"Failed to generate location map: {plot_error}")

        return json.dumps(output, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Location failed: {str(e)}",
            "valid_picks": len(valid_picks),
            "total_picks": len(phase_picks)
        }, ensure_ascii=False, indent=2)


@tool
def add_station_coordinates(params: Union[str, dict, None] = None):
    """
    Add station coordinates for earthquake location.

    Use this tool to provide station coordinates when they are not
    included in the waveform file metadata.

    Args:
        params: Dictionary with:
            - stations: List of station info, each with:
                - network: Network code (e.g., "IU")
                - station: Station code (e.g., "ANMO")
                - latitude: Station latitude in degrees
                - longitude: Station longitude in degrees
                - elevation: Station elevation in meters (optional)

    Example:
        {"stations": [
            {"network": "IU", "station": "ANMO", "latitude": 34.9459, "longitude": -106.4572},
            {"network": "IU", "station": "BJI", "latitude": 40.0409, "longitude": 116.1754}
        ]}
    """
    """
    添加台站坐标用于地震定位。

    当波形文件元数据中不包含台站坐标时，使用此工具提供。

    参数：
        params: 包含以下内容的字典：
            - stations: 台站信息列表，每个台站包含：
                - network: 台网代码（如 "IU"）
                - station: 台站代码（如 "ANMO"）
                - latitude: 台站纬度（度）
                - longitude: 台站经度（度）
                - elevation: 台站高程（米，可选）

    示例：
        {"stations": [
            {"network": "IU", "station": "ANMO", "latitude": 34.9459, "longitude": -106.4572},
            {"network": "IU", "station": "BJI", "latitude": 40.0409, "longitude": 116.1754}
        ]}
    """
    global CURRENT_STATIONS

    from utils.locator import Station

    parsed = _parse_param_dict(params)
    stations_input = parsed.get("stations", [])

    # Auto-load from stations.json if no stations provided
    if not stations_input:
        for candidate in [
            os.path.join(os.getcwd(), "data", "stations.json"),
            os.path.join(os.getcwd(), "example_data", "stations.json"),
            os.path.join(os.getcwd(), "data", "fdsn", "stations.json"),
        ]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r") as f:
                        sjson = json.load(f)
                    # Format 1: {"stations": [...]}
                    stations_input = sjson.get("stations", [])
                    # Format 2: flat dict keyed by filename, e.g. {"CI.ADO..BHZ.mseed": {"latitude": ..., ...}}
                    if not stations_input and isinstance(sjson, dict):
                        for key, val in sjson.items():
                            if isinstance(val, dict) and "latitude" in val:
                                # Parse "CI.ADO..BHZ.mseed" -> network=CI, station=ADO
                                parts = key.replace(".mseed", "").replace(".miniseed", "").split(".")
                                net = parts[0] if len(parts) >= 1 else "XX"
                                name = parts[1] if len(parts) >= 2 else "UNK"
                                entry = dict(val)
                                entry.setdefault("network", net)
                                entry.setdefault("station", name)
                                stations_input.append(entry)
                    if stations_input:
                        break
                except Exception:
                    pass

    if not stations_input:
        return json.dumps({
            "error": "No stations provided.",
            "usage": "Provide a list of stations with network, station, latitude, longitude.",
            "hint": "Or place a stations.json file in data/ or example_data/."
        }, ensure_ascii=False, indent=2)

    if CURRENT_STATIONS is None:
        CURRENT_STATIONS = {}

    added = []
    for sta in stations_input:
        if isinstance(sta, dict):
            net = sta.get("network", "XX")
            name = sta.get("station", sta.get("name", "UNK"))
            lat = sta.get("latitude", sta.get("lat"))
            lon = sta.get("longitude", sta.get("lon"))
            elev = sta.get("elevation", sta.get("elev", 0))

            if lat is None or lon is None:
                continue

            sta_id = f"{net}.{name}"
            CURRENT_STATIONS[sta_id] = {
                "network": net,
                "station": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "elevation": float(elev) if elev else 0.0
            }
            added.append(sta_id)

    return json.dumps({
        "success": True,
        "stations_added": added,
        "total_stations": len(CURRENT_STATIONS),
        "message": f"Added {len(added)} stations. Total: {len(CURRENT_STATIONS)} stations stored."
    }, ensure_ascii=False, indent=2)


@tool
def plot_location_map(params: Union[str, dict, None] = None):
    """
    Plot earthquake location and station positions on a map using Cartopy.

    Use this tool after locating an earthquake to visualize the result.
    It reads the last location result and station coordinates automatically.

    Args:
        params: Dictionary with optional parameters:
            - region: [west, east, south, north] custom map region in degrees
            - title: Custom map title
            - output: Custom output file path (default: data/location/earthquake_location_map.png)

    Returns:
        JSON with the path to the saved map image.
    """
    """
    使用 Cartopy 将地震定位结果和台站位置绘制在地图上。

    在完成地震定位后使用此工具可视化结果。
    自动读取上一次定位结果和台站坐标。

    参数：
        params: 可选参数字典：
            - region: [西, 东, 南, 北] 自定义地图范围（度）
            - title: 自定义地图标题
            - output: 自定义输出文件路径（默认：data/convert/earthquake_location_map.png）

    返回：
        包含保存的地图图像路径的 JSON。
    """
    from utils.locator import plot_location_map as do_plot

    parsed = _parse_param_dict(params)
    region = parsed.get("region")
    title = parsed.get("title")
    output_path = parsed.get("output", os.path.join(DEFAULT_LOCATION_DIR, "earthquake_location_map.png"))

    # Get data from stored location result
    if CURRENT_LOCATION is None:
        return json.dumps({
            "error": "No location result available. Please run locate_earthquake first.",
            "hint": "请先运行 locate_earthquake 完成定位。"
        }, ensure_ascii=False, indent=2)

    hypocenter = CURRENT_LOCATION["hypocenter"]
    stations = CURRENT_LOCATION["stations"]

    if not stations:
        return json.dumps({
            "error": "No station data available for plotting.",
            "hint": "请先使用 add_station_coordinates 添加台站坐标。"
        }, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result_path = do_plot(
            hypocenter=hypocenter,
            stations=stations,
            output_path=output_path,
            region=region,
            title=title,
        )

        if result_path and os.path.exists(result_path):
            return f"![Earthquake Location Map]({result_path})\nLocation map successfully generated and saved to {result_path}."
        else:
            return json.dumps({
                "error": "Failed to generate map. Cartopy/matplotlib may not be available.",
                "hint": "请确保 Cartopy 或 matplotlib 已正确安装。"
            }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Map generation failed: {str(e)}",
            "hint": "地图生成失败，请检查 Cartopy/matplotlib 安装。"
        }, ensure_ascii=False, indent=2)
from typing import Union
from langchain.tools import tool
import os

@tool
def load_local_data(params: Union[str, dict, None] = None):
    """
    Load a local directory or file into the current workspace.
    
    This tool should be used when the user provides a local file path or directory path
    (e.g., '/path/to/data' or 'example_data/') to perform analysis on local files.
    It automatically scans the directory for supported seismic data formats (.mseed, .miniseed, .segy, .sgy, .h5, .hdf5, .sac)
    and loads them into the current session.

    Args:
        params: Dictionary with optional parameters:
            - path: The local directory or file path to load
            
    Returns:
        String describing the files that were successfully loaded.
    """
    """
    加载本地目录或文件到当前工作空间。
    
    当用户提供本地文件路径或目录路径（如'/path/to/data'或'example_data/'）时，应使用此工具。
    它会自动扫描目录中支持的地震数据格式（.mseed, .miniseed, .segy, .sgy, .h5, .hdf5, .sac）并加载到当前会话中。
    
    参数：
        params: 包含可选参数的字典：
            - path: 要加载的本地目录或文件路径
            
    返回：
        描述成功加载的文件的字符串。
    """
    from agent.tools import _parse_param_dict, _resolve_file_path, add_miniseed_path, set_current_miniseed_path, set_current_segy_path, set_current_hdf5_path, set_current_sac_path
    params_dict = _parse_param_dict(params) if params else {}
    path = params_dict.get("path")
    if not path:
        # If user didn't specify path but provided one as string directly
        if isinstance(params, str):
            path = params.strip()
        else:
            return "Error: Please provide a valid path parameter."
        
    resolved_path = _resolve_file_path(path)
    if not resolved_path or not os.path.exists(resolved_path):
        return f"Error: Path '{path}' not found."
        
    loaded_files = {'miniseed': [], 'segy': [], 'hdf5': [], 'sac': []}
    
    if os.path.isfile(resolved_path):
        files_to_check = [resolved_path]
    else:
        files_to_check = [os.path.join(resolved_path, f) for f in os.listdir(resolved_path) if os.path.isfile(os.path.join(resolved_path, f))]
        
    for f_path in files_to_check:
        ext = f_path.lower()
        if ext.endswith('.mseed') or ext.endswith('.miniseed'):
            add_miniseed_path(f_path)
            loaded_files['miniseed'].append(os.path.basename(f_path))
        elif ext.endswith('.segy') or ext.endswith('.sgy'):
            set_current_segy_path(f_path)
            loaded_files['segy'].append(os.path.basename(f_path))
        elif ext.endswith('.h5') or ext.endswith('.hdf5'):
            set_current_hdf5_path(f_path)
            loaded_files['hdf5'].append(os.path.basename(f_path))
        elif ext.endswith('.sac'):
            set_current_sac_path(f_path)
            loaded_files['sac'].append(os.path.basename(f_path))
            
    summary = []
    if loaded_files['miniseed']:
        summary.append(f"Loaded {len(loaded_files['miniseed'])} MiniSEED files: {', '.join(loaded_files['miniseed'])}")
    if loaded_files['segy']:
        summary.append(f"Loaded {len(loaded_files['segy'])} SEGY files: {', '.join(loaded_files['segy'])}")
    if loaded_files['hdf5']:
        summary.append(f"Loaded {len(loaded_files['hdf5'])} HDF5 files: {', '.join(loaded_files['hdf5'])}")
    if loaded_files['sac']:
        summary.append(f"Loaded {len(loaded_files['sac'])} SAC files: {', '.join(loaded_files['sac'])}")
        
    if not summary:
        return f"No supported seismic data files (.mseed, .miniseed, .segy, .sgy, .h5, .hdf5, .sac) found in '{path}'."

    return "\n".join(summary)


@tool
def download_seismic_data(params: Union[str, dict, None] = None):
    """
    Download seismic waveform data and event information from FDSN web services (e.g., IRIS).

    This tool can:
    1. Search for recent earthquakes in a region
    2. Download waveform data (MiniSEED) from nearby stations
    3. Automatically save station coordinates to stations.json

    Args:
        params: Dictionary with parameters:
            - latitude: Center latitude in degrees
            - longitude: Center longitude in degrees
            - radius_km: Search radius in kilometers (default: 200)
            - starttime: Start time in ISO format (default: 30 days ago)
            - endtime: End time in ISO format (default: now)
            - minmagnitude: Minimum earthquake magnitude (default: 4.0)
            - network: FDSN network code filter (e.g., "IU", "AK")
            - channel: Channel code filter (e.g., "BH?", "HH?")
            - output_dir: Directory to save downloaded data (default: data/fdsn/)
            - max_stations: Maximum number of stations to download (default: 10)

    Example:
        {"latitude": 61.2, "longitude": -150.0, "radius_km": 500, "minmagnitude": 5.0}
        {"latitude": 29.6, "longitude": 102.1, "radius_km": 300}
    """
    """
    从 FDSN 网络服务（如 IRIS）下载地震波形数据和事件信息。

    此工具可以：
    1. 搜索某个区域最近的地震
    2. 从附近的台站下载波形数据（MiniSEED）
    3. 自动将台站坐标保存到 stations.json

    参数：
        params: 包含参数的字典：
            - latitude: 中心纬度（度）
            - longitude: 中心经度（度）
            - radius_km: 搜索半径（公里），默认 200
            - starttime: 起始时间，ISO 格式（默认：30 天前）
            - endtime: 结束时间，ISO 格式（默认：现在）
            - minmagnitude: 最小震级（默认：4.0）
            - network: FDSN 台网代码（如 "IU"、"AK"）
            - channel: 通道代码（如 "BH?"、"HH?"）
            - output_dir: 保存下载数据的目录（默认：data/fdsn/）
            - max_stations: 最大下载台站数（默认：10）

    示例：
        {"latitude": 61.2, "longitude": -150.0, "radius_km": 500, "minmagnitude": 5.0}
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from obspy.clients.fdsn import Client as FDSNClient
    from obspy import UTCDateTime
    from obspy.geodetics import locations2degrees

    parsed = _parse_param_dict(params)
    _l = CURRENT_LANG
    local_only = str(parsed.get("local_only", parsed.get("use_local_only", "false"))).strip().lower() in {"1", "true", "yes", "y", "on"}
    local_only = str(parsed.get("local_only", parsed.get("use_local_only", "false"))).strip().lower() in {"1", "true", "yes", "y", "on"}
    local_only = str(parsed.get("local_only", parsed.get("use_local_only", "false"))).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Parameters
    lat = _coerce_float(parsed.get("latitude"), allow_none=True, field_name="latitude")
    lon = _coerce_float(parsed.get("longitude"), allow_none=True, field_name="longitude")
    radius_km = _coerce_float(parsed.get("radius_km"), default=200.0, field_name="radius_km")
    minmagnitude = _coerce_float(parsed.get("minmagnitude"), default=4.0, field_name="minmagnitude")
    max_stations = _coerce_int(parsed.get("max_stations"), default=10, field_name="max_stations")
    network = parsed.get("network", None)
    channel = parsed.get("channel", "BH?,HH?")
    output_dir = parsed.get("output_dir", "data/fdsn/")
    provider = str(parsed.get("provider", "auto")).lower().strip()
    max_workers = _coerce_int(parsed.get("max_workers"), default=8, field_name="max_workers")
    preprocess_raw = parsed.get("preprocess", True)
    if isinstance(preprocess_raw, str):
        preprocess = preprocess_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        preprocess = bool(preprocess_raw)

    # Time range
    endtime = parsed.get("endtime")
    starttime = parsed.get("starttime")
    if endtime:
        endtime = UTCDateTime(endtime)
    else:
        endtime = UTCDateTime.now()
    if starttime:
        starttime = UTCDateTime(starttime)
    else:
        starttime = endtime - 30 * 86400  # 30 days ago

    if lat is None or lon is None:
        return json.dumps({
            "error": "latitude and longitude are required.",
            "hint": "请提供 latitude 和 longitude 参数。" if _l == "zh" else "Please provide latitude and longitude.",
            "example": {"latitude": 61.2, "longitude": -150.0, "radius_km": 500}
        }, ensure_ascii=False, indent=2)

    os.makedirs(output_dir, exist_ok=True)

    # Build candidate FDSN providers: SCEDC first (near-seismic), fallback to IRIS.
    if provider == "auto":
        provider_candidates = ["SCEDC", "IRIS"]
    elif provider in {"scedc", "iris"}:
        provider_candidates = [provider.upper()]
    else:
        provider_candidates = ["SCEDC", "IRIS"]

    def _provider_order(*prefer):
        out = []
        seen = set()
        for item in list(prefer) + provider_candidates:
            if not item:
                continue
            name = str(item).upper()
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    event_search_errors = []
    catalog = None
    client_name = None
    client = None
    event_search_s = 0.0

    # Event search with provider fallback.
    for provider_name in _provider_order():
        t0_evt = time.perf_counter()
        try:
            trial_client = FDSNClient(provider_name, timeout=120)
            trial_catalog = trial_client.get_events(
                starttime=starttime,
                endtime=endtime,
                latitude=lat,
                longitude=lon,
                maxradius=radius_km / 111.0,
                minmagnitude=minmagnitude,
                orderby="magnitude",
            )
            event_search_s = time.perf_counter() - t0_evt
            client_name = provider_name
            client = trial_client
            catalog = trial_catalog
            break
        except Exception as e:
            event_search_errors.append(f"{provider_name}: {e}")

    if catalog is None or client is None or client_name is None:
        return json.dumps({
            "error": "Event search failed on all providers.",
            "providers": provider_candidates,
            "details": event_search_errors[:5],
            "hint": "无法连接 FDSN 服务，请检查网络。" if _l == "zh" else "Cannot connect to FDSN service.",
        }, ensure_ascii=False, indent=2)

    lines = []

    # Step 1: Search for events
    lines.append(f"### {'地震事件搜索' if _l == 'zh' else 'Event Search'}")
    lines.append(f"- {'数据源' if _l == 'zh' else 'Provider'}: {client_name}")
    lines.append(f"- {'搜索范围' if _l == 'zh' else 'Region'}: ({lat}, {lon}), {'半径' if _l == 'zh' else 'radius'} {radius_km} km")
    lines.append(f"- {'时间范围' if _l == 'zh' else 'Time range'}: {starttime} ~ {endtime}")
    lines.append(f"- {'最小震级' if _l == 'zh' else 'Min magnitude'}: {minmagnitude}")
    lines.append("")

    lines.append(f"- {'事件检索耗时' if _l == 'zh' else 'Event search time'}: {event_search_s:.2f}s")
    if event_search_errors:
        lines.append(f"- {'回退记录' if _l == 'zh' else 'Fallback log'}: {' | '.join(event_search_errors[:2])}")

    if len(catalog) == 0:
        lines.append(f"{'未找到符合条件的地震事件' if _l == 'zh' else 'No events found matching criteria'}.")
        lines.append(f"{'尝试减小最小震级或扩大搜索范围' if _l == 'zh' else 'Try reducing minmagnitude or increasing radius_km'}.")
        return "\n".join(lines)

    lines.append(f"{'找到' if _l == 'zh' else 'Found'} {len(catalog)} {'个事件' if _l == 'zh' else 'events'}.")
    lines.append("")

    # Show top events
    for i, event in enumerate(catalog[:5]):
        origin = event.preferred_origin() or event.origins[0]
        mag = event.preferred_magnitude() or event.magnitudes[0] if event.magnitudes else None
        mag_val = mag.mag if mag else "?"
        mag_type = mag.magnitude_type if mag else ""
        t = origin.time
        lines.append(f"{i+1}. {t} | {abs(origin.latitude):.2f}{'N' if origin.latitude >= 0 else 'S'}, "
                     f"{abs(origin.longitude):.2f}{'E' if origin.longitude >= 0 else 'W'} | "
                     f"{'深度' if _l == 'zh' else 'depth'} {origin.depth/1000:.1f} km | "
                     f"M{mag_type} {mag_val}")
    if len(catalog) > 5:
        lines.append(f"... ({'还有' if _l == 'zh' else 'and'} {len(catalog)-5} {'个' if _l == 'zh' else 'more'})")
    lines.append("")

    # Step 2: Use the largest event for waveform download
    event = catalog[0]
    origin = event.preferred_origin() or event.origins[0]
    evt_lat = origin.latitude
    evt_lon = origin.longitude
    evt_time = origin.time
    evt_mag = None
    preferred_mag = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
    if preferred_mag is not None and getattr(preferred_mag, "mag", None) is not None:
        evt_mag = float(preferred_mag.mag)

    lines.append(f"### {'下载波形数据' if _l == 'zh' else 'Downloading Waveform Data'}")
    lines.append(f"{'目标事件' if _l == 'zh' else 'Target event'}: {evt_time}, ({evt_lat:.2f}, {evt_lon:.2f})")
    lines.append("")

    # Search for stations (response-level metadata for fast downstream preprocessing)
    station_kwargs = {
        "starttime": evt_time - 3600,
        "endtime": evt_time + 7200,
        "latitude": evt_lat,
        "longitude": evt_lon,
        "maxradius": radius_km / 111.0,
        "level": "response",
    }
    if network:
        station_kwargs["network"] = network
    if channel:
        station_kwargs["channel"] = channel

    inventory = None
    station_provider = None
    station_search_s = 0.0
    station_errors = []
    for provider_name in _provider_order(client_name):
        t0_sta = time.perf_counter()
        try:
            trial_client = FDSNClient(provider_name, timeout=120)
            trial_inventory = trial_client.get_stations(**station_kwargs)
            station_search_s = time.perf_counter() - t0_sta
            inventory = trial_inventory
            station_provider = provider_name
            break
        except Exception as e:
            station_errors.append(f"{provider_name}: {e}")

    if inventory is None or station_provider is None:
        lines.append(f"{'台站搜索失败' if _l == 'zh' else 'Station search failed'}")
        if station_errors:
            lines.append(f"- {'错误' if _l == 'zh' else 'Errors'}: {' | '.join(station_errors[:3])}")
        return "\n".join(lines)

    lines.append(f"- {'台站检索数据源' if _l == 'zh' else 'Station provider'}: {station_provider}")
    lines.append(f"- {'台站检索耗时' if _l == 'zh' else 'Station search time'}: {station_search_s:.2f}s")

    # Collect stations with distance info
    stations_info = []
    for net in inventory:
        for sta in net:
            dist_deg = locations2degrees(evt_lat, evt_lon, sta.latitude, sta.longitude)
            dist_km = dist_deg * 111.195
            if dist_km <= radius_km:
                stations_info.append({
                    "network": net.code,
                    "station": sta.code,
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation": sta.elevation if sta.elevation else 0,
                    "distance_km": dist_km,
                })

    stations_info.sort(key=lambda x: x["distance_km"])
    stations_info = stations_info[:max_stations]

    if not stations_info:
        lines.append(f"{'未找到符合条件的台站' if _l == 'zh' else 'No stations found matching criteria'}.")
        return "\n".join(lines)

    lines.append(f"{'找到' if _l == 'zh' else 'Found'} {len(stations_info)} {'个台站' if _l == 'zh' else 'stations'}:")
    for s in stations_info:
        lines.append(f"- {s['network']}.{s['station']} ({s['latitude']:.2f}, {s['longitude']:.2f}), {'距离' if _l == 'zh' else 'dist'} {s['distance_km']:.0f} km")
    lines.append("")

    # Download window and filter strategy follows validation scripts.
    if evt_mag is not None and evt_mag >= 5.5:
        duration_sec = 300
    else:
        duration_sec = 180
    pad_sec = 30
    t_start_dl = evt_time - 30 - pad_sec
    t_end_dl = evt_time + duration_sec + pad_sec
    t_start_save = evt_time - 30
    t_end_save = evt_time + duration_sec
    pre_filt = (0.5, 1.0, 30.0, 40.0)

    lines.append(f"- {'下载窗口' if _l == 'zh' else 'Download window'}: {t_start_dl} ~ {t_end_dl}")
    lines.append(f"- {'保存窗口' if _l == 'zh' else 'Saved window'}: {t_start_save} ~ {t_end_save}")
    lines.append(f"- {'并行线程' if _l == 'zh' else 'Parallel workers'}: {max_workers}")
    lines.append("")

    # Step 3: Download waveform data in parallel
    downloaded_files = []
    failed_stations = []

    waveform_provider_order = _provider_order(station_provider, client_name)
    provider_stats = {
        p: {"ok": 0, "fail": 0, "time_s": 0.0} for p in waveform_provider_order
    }

    def _download_one(sta_info):
        sta_code = f"{sta_info['network']}.{sta_info['station']}"
        filename = f"{sta_info['network']}.{sta_info['station']}.mseed"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            return {"ok": True, "filepath": filepath, "station": sta_code, "cached": True}

        waveform_kwargs = {
            "network": sta_info["network"],
            "station": sta_info["station"],
            "location": "*",
            "channel": channel,
            "starttime": t_start_dl,
            "endtime": t_end_dl,
        }

        errors = []
        attempts = []
        for provider_name in waveform_provider_order:
            t0_wf = time.perf_counter()
            try:
                local_client = FDSNClient(provider_name, timeout=60)
                st = local_client.get_waveforms(**waveform_kwargs)
                if preprocess:
                    try:
                        if provider_name == station_provider:
                            inv_single = inventory.select(network=sta_info["network"], station=sta_info["station"])
                        else:
                            inv_single = local_client.get_stations(
                                network=sta_info["network"],
                                station=sta_info["station"],
                                channel=channel,
                                starttime=t_start_dl,
                                endtime=t_end_dl,
                                level="response",
                            )
                        st.detrend("demean")
                        st.detrend("linear")
                        st.taper(max_percentage=0.05)
                        if hasattr(inv_single, "networks") and len(inv_single.networks) > 0:
                            st.remove_response(inventory=inv_single, output="VEL", pre_filt=pre_filt)
                        st.trim(starttime=t_start_save, endtime=t_end_save)
                        st.interpolate(sampling_rate=100.0)
                    except Exception:
                        pass
                st.write(filepath, format="MSEED")
                elapsed = time.perf_counter() - t0_wf
                return {
                    "ok": True,
                    "filepath": filepath,
                    "station": sta_code,
                    "cached": False,
                    "provider": provider_name,
                    "elapsed_s": elapsed,
                    "attempts": attempts,
                }
            except Exception as e:
                elapsed = time.perf_counter() - t0_wf
                errors.append(f"{provider_name}: {e}")
                attempts.append({"provider": provider_name, "ok": False, "elapsed_s": elapsed})

        return {
            "ok": False,
            "station": sta_code,
            "error": " | ".join(errors[:3]),
            "attempts": attempts,
        }

    t0_wave_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(_download_one, s) for s in stations_info]
        for fut in as_completed(futures):
            out = fut.result()
            for att in out.get("attempts", []):
                p_att = att.get("provider")
                if p_att in provider_stats:
                    provider_stats[p_att]["fail"] += 1
                    provider_stats[p_att]["time_s"] += float(att.get("elapsed_s", 0.0) or 0.0)

            if out.get("ok"):
                downloaded_files.append(out["filepath"])
                add_miniseed_path(out["filepath"])
                p = out.get("provider")
                if p in provider_stats:
                    provider_stats[p]["ok"] += 1
                    provider_stats[p]["time_s"] += float(out.get("elapsed_s", 0.0) or 0.0)
            else:
                failed_stations.append(f"{out.get('station')}: {out.get('error')}")
    waveform_wall_s = time.perf_counter() - t0_wave_all

    # Step 4: Save stations.json
    if downloaded_files:
        stations_json_path = os.path.join(output_dir, "stations.json")
        stations_dict = {}
        for s in stations_info:
            target_name = f"{s['network']}.{s['station']}.mseed"
            for f in downloaded_files:
                fname = os.path.basename(f)
                if fname == target_name:
                    stations_dict[fname] = {
                        "latitude": s["latitude"],
                        "longitude": s["longitude"],
                        "elevation": s["elevation"],
                    }
                    break

        with open(stations_json_path, "w") as f:
            json.dump(stations_dict, f, indent=2)

        lines.append(f"### {'下载结果' if _l == 'zh' else 'Download Results'}")
        lines.append(f"- {'成功下载' if _l == 'zh' else 'Downloaded'}: {len(downloaded_files)} {'个文件' if _l == 'zh' else 'files'}")
        lines.append(f"- {'台站坐标条目' if _l == 'zh' else 'Station entries'}: {len(stations_dict)}")
        if failed_stations:
            lines.append(f"- {'失败' if _l == 'zh' else 'Failed'}: {len(failed_stations)} {'个台站' if _l == 'zh' else 'stations'}")
            for fs in failed_stations[:5]:
                lines.append(f"  - {fs}")
        lines.append(f"- {'台站坐标已保存至' if _l == 'zh' else 'Station coordinates saved to'}: {stations_json_path}")
        lines.append(f"- {'数据保存目录' if _l == 'zh' else 'Data saved to'}: {output_dir}")
        lines.append("")
        lines.append(f"### {'下载性能统计' if _l == 'zh' else 'Download Performance'}")
        lines.append(f"- {'波形并行总耗时' if _l == 'zh' else 'Waveform wall time'}: {waveform_wall_s:.2f}s")
        for provider_name in waveform_provider_order:
            stat = provider_stats.get(provider_name, {"ok": 0, "fail": 0, "time_s": 0.0})
            lines.append(
                f"- {provider_name}: "
                f"{'成功' if _l == 'zh' else 'ok'} {stat['ok']}, "
                f"{'失败' if _l == 'zh' else 'fail'} {stat['fail']}, "
                f"{'累计耗时' if _l == 'zh' else 'cum time'} {stat['time_s']:.2f}s"
            )
        lines.append("")
        lines.append(f"{'下一步可以使用 pick_all_miniseed_files 进行震相拾取，然后定位' if _l == 'zh' else 'Next: use pick_all_miniseed_files for phase picking, then locate_earthquake'}.")

    if not downloaded_files:
        lines.append(f"{'未下载到可用波形' if _l == 'zh' else 'No waveforms downloaded'}.")
        lines.append(f"- {'波形并行总耗时' if _l == 'zh' else 'Waveform wall time'}: {waveform_wall_s:.2f}s")

    return "\n".join(lines)


@tool
def locate_uploaded_data_nearseismic(params: Union[str, dict, None] = None):
    """
    Locate an event from user-uploaded data using near-seismic EDT locator.

    Typical workflow:
    1. load_local_data
    2. pick_all_miniseed_files
    3. add_station_coordinates
    4. locate_uploaded_data_nearseismic

    Args:
        params: Optional dict, supports:
            - grid_center: [lat, lon] search center hint
            - seed_depth_km: initial depth hint (default 10)
            - stations: optional station metadata override
    """
    parsed = _parse_param_dict(params)
    payload = {
        "method": "nearseismic",
        "grid_center": parsed.get("grid_center"),
        "seed_depth_km": parsed.get("seed_depth_km", 10.0),
        "stations": parsed.get("stations", CURRENT_STATIONS),
    }
    return locate_earthquake.invoke(payload)


@tool
def locate_place_data_nearseismic(params: Union[str, dict, None] = None):
    """
    Download waveform data for a user-specified place and run near-seismic location.

    This tool orchestrates the complete workflow for place-based requests:
    1) download data around a place
    2) pick phases for all downloaded stations
    3) load station coordinates
    4) locate event with near-seismic/blind-nearseismic engine

    Args:
        params: Dictionary supports:
            - latitude (required)
            - longitude (required)
            - radius_km (default 200)
            - minmagnitude (default 4.0)
            - starttime/endtime (optional)
            - network/channel/max_stations/output_dir (optional)
            - method: "nearseismic" or "blind_nearseismic" (default "blind_nearseismic")
            - grid_center/seed_depth_km (optional for nearseismic)
    """
    parsed = _parse_param_dict(params)
    _l = CURRENT_LANG

    latitude = parsed.get("latitude")
    longitude = parsed.get("longitude")
    if (latitude is None or longitude is None) and all(k in parsed for k in ("min_lat", "max_lat", "min_lon", "max_lon")):
        try:
            latitude = (float(parsed["min_lat"]) + float(parsed["max_lat"])) / 2.0
            longitude = (float(parsed["min_lon"]) + float(parsed["max_lon"])) / 2.0
        except Exception:
            pass
    if latitude is None or longitude is None:
        return json.dumps({
            "error": "latitude and longitude are required.",
            "hint": "请提供 latitude 和 longitude 参数。" if _l == "zh" else "Please provide latitude and longitude.",
        }, ensure_ascii=False, indent=2)

    download_payload = {
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": parsed.get("radius_km", 200),
        "minmagnitude": parsed.get("minmagnitude", 4.0),
        "starttime": parsed.get("starttime"),
        "endtime": parsed.get("endtime"),
        "network": parsed.get("network"),
        "channel": parsed.get("channel", "BH?"),
        "max_stations": parsed.get("max_stations", 10),
        "output_dir": parsed.get("output_dir", "data/fdsn/"),
    }

    download_text = download_seismic_data.invoke(download_payload)
    if isinstance(download_text, str) and ("\"error\"" in download_text.lower() or "failed" in download_text.lower()):
        return download_text

    pick_text = pick_all_miniseed_files.invoke({})
    if isinstance(pick_text, str) and ("failed" in pick_text.lower() or "错误" in pick_text):
        return f"{download_text}\n\n{pick_text}"

    station_text = add_station_coordinates.invoke({})

    method = str(parsed.get("method", "blind_nearseismic"))
    if method not in {"nearseismic", "blind_nearseismic"}:
        method = "blind_nearseismic"

    locate_payload = {
        "method": method,
        "grid_center": parsed.get("grid_center", [latitude, longitude]),
        "seed_depth_km": parsed.get("seed_depth_km", 10.0),
    }
    location_text = locate_earthquake.invoke(locate_payload)

    lines = []
    lines.append("### Place-based Near-Seismic Workflow")
    lines.append(download_text)
    lines.append("")
    lines.append(pick_text)
    lines.append("")
    lines.append(station_text)
    lines.append("")
    lines.append(location_text)
    return "\n".join(lines)


# =================== Continuous Monitoring Tools ===================

# Global state for continuous monitoring
_CURRENT_CONTINUOUS_STREAMS = None
_CURRENT_CONTINUOUS_STATIONS = None
_CURRENT_CONTINUOUS_PICKS = None
_CURRENT_TT_INTERP = None


def _parse_param_dict(params):
    """Parse params into dict."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        try:
            return json.loads(params)
        except Exception:
            return {}
    return {}


def _resolve_continuous_time_window(parsed):
    """Resolve start/end for continuous workflows from explicit or relative inputs."""
    from obspy import UTCDateTime

    start_raw = parsed.get("start") or parsed.get("starttime")
    end_raw = parsed.get("end") or parsed.get("endtime")
    date_raw = parsed.get("date") or parsed.get("day")
    hours_raw = parsed.get("hours", parsed.get("duration_hours"))

    def _parse_time(value):
        if value is None:
            return None
        try:
            return UTCDateTime(value)
        except Exception:
            return None

    start_time = _parse_time(start_raw)
    end_time = _parse_time(end_raw)

    hours = None
    if hours_raw is not None:
        try:
            hours = float(hours_raw)
        except Exception:
            hours = None

    if start_time and end_time:
        return start_time, end_time

    if date_raw and hours is not None:
        date_text = str(date_raw).strip()
        if "T" in date_text:
            base_end = _parse_time(date_text)
            if base_end is None:
                base_end = UTCDateTime.now()
        else:
            try:
                base_end = UTCDateTime(f"{date_text}T23:59:59")
            except Exception:
                base_end = UTCDateTime.now()
        return base_end - hours * 3600.0, base_end

    if start_time and hours is not None and end_time is None:
        return start_time, start_time + hours * 3600.0

    if end_time and hours is not None and start_time is None:
        return end_time - hours * 3600.0, end_time

    if hours is not None and start_time is None and end_time is None:
        end_time = UTCDateTime.now()
        return end_time - hours * 3600.0, end_time

    return start_time, end_time


def _resolve_continuous_region(parsed):
    """Resolve region bounds and network defaults for continuous monitoring."""
    place_name = parsed.get("place") or parsed.get("location") or parsed.get("site")
    region_name = parsed.get("region")
    region = get_region(str(region_name)) if region_name else None
    place = resolve_named_place(str(place_name)) if place_name else None

    def _place_bbox(lat, lon, radius_km):
        lat_delta = float(radius_km) / 111.0
        lon_delta = float(radius_km) / max(1e-6, 111.0 * np.cos(np.radians(float(lat))))
        return {
            "min_lat": round(float(lat) - lat_delta, 3),
            "max_lat": round(float(lat) + lat_delta, 3),
            "min_lon": round(float(lon) - lon_delta, 3),
            "max_lon": round(float(lon) + lon_delta, 3),
        }

    if place:
        bbox = _place_bbox(place["latitude"], place["longitude"], place.get("radius_km", 40.0))
        min_lat = float(parsed.get("min_lat", bbox["min_lat"]))
        max_lat = float(parsed.get("max_lat", bbox["max_lat"]))
        min_lon = float(parsed.get("min_lon", bbox["min_lon"]))
        max_lon = float(parsed.get("max_lon", bbox["max_lon"]))
        network = str(parsed.get("network", place.get("network", "CI")))
        client_name = str(parsed.get("client", place.get("client", "SCEDC")))
        catalog_name = str(parsed.get("catalog", place.get("catalog", "SCEDC")))
        region_label = f"{place['name']} ({place['en_name']})"
        return {
            "region": region_label,
            "place": place,
            "mode": "place",
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "network": network,
            "client": client_name,
            "catalog": catalog_name,
        }

    if region:
        min_lat = float(parsed.get("min_lat", region["min_lat"]))
        max_lat = float(parsed.get("max_lat", region["max_lat"]))
        min_lon = float(parsed.get("min_lon", region["min_lon"]))
        max_lon = float(parsed.get("max_lon", region["max_lon"]))
        network = str(parsed.get("network", region.get("network", "CI")))
        client_name = str(parsed.get("client", region.get("client", "SCEDC")))
        catalog_name = str(parsed.get("catalog", region.get("catalog", "SCEDC")))
        region_label = str(region_name)

        return {
            "region": region_label,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "network": network,
            "client": client_name,
            "catalog": catalog_name,
            "mode": "region",
        }

    if place is None and region_name:
        place = resolve_named_place(str(region_name))
        if place:
            bbox = _place_bbox(place["latitude"], place["longitude"], place.get("radius_km", 40.0))
            min_lat = float(parsed.get("min_lat", bbox["min_lat"]))
            max_lat = float(parsed.get("max_lat", bbox["max_lat"]))
            min_lon = float(parsed.get("min_lon", bbox["min_lon"]))
            max_lon = float(parsed.get("max_lon", bbox["max_lon"]))
            network = str(parsed.get("network", place.get("network", "CI")))
            client_name = str(parsed.get("client", place.get("client", "SCEDC")))
            catalog_name = str(parsed.get("catalog", place.get("catalog", "SCEDC")))
            region_label = f"{place['name']} ({place['en_name']})"
            return {
                "region": region_label,
                "place": place,
                "mode": "place",
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "network": network,
                "client": client_name,
                "catalog": catalog_name,
            }

    if place_name or (region_name and str(region_name).strip()):
        return {
            "error": f"Unable to resolve place or region: {place_name or region_name}",
            "mode": "unresolved_place",
            "requested_name": place_name or region_name,
        }

    return {
        "region": "南加州",
        "mode": "region",
        "min_lat": 32.0,
        "max_lat": 36.5,
        "min_lon": -120.0,
        "max_lon": -115.0,
        "network": "CI",
        "client": "SCEDC",
        "catalog": "SCEDC",
    }


def _continuous_download_requires_confirmation(estimate: dict, parsed: dict) -> bool:
    if not estimate or estimate.get("error"):
        return False
    station_count = int(estimate.get("station_count", 0))
    duration_hours = float(estimate.get("duration_seconds", 0.0)) / 3600.0
    estimated_gb = float(estimate.get("estimated_gb", 0.0))

    explicit_confirm = str(parsed.get("force", parsed.get("confirm", ""))).strip().lower() in {"1", "true", "yes", "y"}
    if explicit_confirm:
        return False

    return station_count >= 80 or duration_hours >= 2.0 or estimated_gb >= 1.0


def _continuous_task_summary(region: dict, start_time, end_time) -> str:
    if region.get("mode") == "place" and region.get("place"):
        place = region["place"]
        label = f"{place['name']}（{place['en_name']}）周边"
    else:
        label = region.get("region") or "自定义区域"
    return (
        f"{str(start_time)} 至 {str(end_time)}（UTC），"
        f"{label}，网络{region['network']}，"
        f"官方目录{region['catalog']}"
    )


def _continuous_monitoring_recommendation(region: dict, estimate: dict, start_time=None, end_time=None, channel=None) -> dict:
    station_count = int(estimate.get("station_count", 0) or 0)
    duration_hours = round(float(estimate.get("duration_seconds", 0.0)) / 3600.0, 3)
    estimated_mb = round(float(estimate.get("estimated_mb", 0.0)), 2)
    estimated_gb = round(float(estimate.get("estimated_gb", 0.0)), 3)
    duration_minutes = round(float(estimate.get("duration_seconds", 0.0)) / 60.0, 1)

    if duration_hours < 1.0:
        duration_label = f"{duration_minutes:g}分钟"
    else:
        duration_label = f"{duration_hours:g}小时"

    if region.get("mode") == "place" and region.get("place"):
        place = region["place"]
        area_text = f"{place['name']}（{place['en_name']}）周边"
    elif region.get("region"):
        area_text = region["region"]
    else:
        lon_min = f"{abs(region['min_lon']):g}°{'W' if region['min_lon'] < 0 else 'E'}"
        lon_max = f"{abs(region['max_lon']):g}°{'W' if region['max_lon'] < 0 else 'E'}"
        area_text = f"{region['min_lat']:g}°N–{region['max_lat']:g}°N, {lon_min}–{lon_max}"

    reasons = []
    if station_count >= 80:
        reasons.append(f"{area_text}覆盖台站约 {station_count} 个，需要分批下载")
    if duration_hours >= 2.0:
        reasons.append(f"时间窗约 {duration_hours} 小时，数据累计较大")
    if estimated_gb >= 1.0:
        reasons.append(f"预估数据量约 {estimated_gb} GB")
    if not reasons:
        reasons.append(f"预估数据量约 {estimated_mb} MB")

    lat_center = (float(region["min_lat"]) + float(region["max_lat"])) / 2.0
    lon_center = (float(region["min_lon"]) + float(region["max_lon"])) / 2.0
    lat_span = abs(float(region["max_lat"]) - float(region["min_lat"]))
    lon_span = abs(float(region["max_lon"]) - float(region["min_lon"]))
    focused_lat_half = max(0.25, lat_span * 0.25)
    focused_lon_half = max(0.25, lon_span * 0.25)
    focused_region = {
        "min_lat": round(lat_center - focused_lat_half, 3),
        "max_lat": round(lat_center + focused_lat_half, 3),
        "min_lon": round(lon_center - focused_lon_half, 3),
        "max_lon": round(lon_center + focused_lon_half, 3),
    }

    adjustment_options = [
        {
            "label": "缩小到核心子区",
            "example_params": (
                f"min_lat={focused_region['min_lat']}, max_lat={focused_region['max_lat']}, "
                f"min_lon={focused_region['min_lon']}, max_lon={focused_region['max_lon']}"
            ),
            "why": "减少台站覆盖范围，最直接地降低下载时间与关联复杂度。",
        },
        {
            "label": "仅保留垂直分量",
            "example_params": "channel=BHZ,HHZ",
            "why": "显著减少要下载的分量数，适合先做快速监测和定位。",
        },
        {
            "label": "只做核心监测",
            "example_params": "compare_with_catalog=false",
            "why": "如果目标是先看是否有事件，先跳过目录对比会更快。",
        },
    ]

    if region.get("mode") == "place" and region.get("place"):
        place = region["place"]
        radius_km = float(place.get("radius_km", 20.0))
        channel_text = str(channel or "BH?,HH?")
        channel_hint = "，仅保留Z分量（BHZ/HHZ）" if "Z" not in channel_text.upper() else f"，通道{channel_text}"
        suggested_request = (
            f"以{place['name']}（{place['en_name']}）为中心、半径约{radius_km:g} km的最近{duration_label}连续地震监测，"
            f"网络{region['network']}{channel_hint}，先完成事件检测与定位，目录对比可后续补做"
        )
    else:
        channel_text = str(channel or "BH?,HH?")
        channel_hint = "，仅保留Z分量（BHZ/HHZ）" if "Z" not in channel_text.upper() else f"，通道{channel_text}"
        suggested_request = (
            f"{area_text}，网络{region['network']}，"
            f"最近{duration_label}连续地震监测{channel_hint}，"
            f"先完成事件检测与定位，目录对比可后续补做"
        )

    return {
        "reason": "；".join(reasons),
        "suggested_request": suggested_request,
        "risk_factors": reasons,
        "adjustment_options": adjustment_options,
        "focused_region": focused_region,
        "mode": region.get("mode", "region"),
    }


@tool
def download_continuous_waveforms(params: Union[str, dict, None] = None):
    """
    Download continuous waveform data for a time window and region.

    This is used for continuous monitoring scenarios where you want to:
    - Monitor a region over a time period
    - Detect multiple events within the time window

    Args:
        params: Dictionary supports:
            - start: Start time in ISO format (e.g., "2019-07-04T17:00:00")
            - end: End time in ISO format (e.g., "2019-07-04T18:00:00")
            - date: Date in YYYY-MM-DD or ISO datetime format
            - hours/duration_hours: Relative duration when start/end are omitted
            - region: Named region such as "南加州", "北加州", or "加州"
            - min_lat: Minimum latitude (default 32.0 for Southern California)
            - max_lat: Maximum latitude (default 36.5)
            - min_lon: Minimum longitude (default -120.0)
            - max_lon: Maximum longitude (default -115.0)
            - network: Network code (default "CI" for SCEDC)
            - channel: Channel codes (default "BH?,HH?")
            - data_dir: Optional directory to store data

    Returns:
        JSON string with download status, number of stations, and data location
    """
    from obspy import UTCDateTime

    parsed = _parse_param_dict(params)
    _l = CURRENT_LANG

    start_time, end_time = _resolve_continuous_time_window(parsed)
    if start_time is None or end_time is None:
        return json.dumps({
            "error": "start/end or date+hours are required.",
            "hint": "请提供 start/end，或 date + hours 参数。" if _l == "zh"
                else "Please provide start/end, or date + hours.",
        }, ensure_ascii=False, indent=2)

    region = _resolve_continuous_region(parsed)
    if region.get("error"):
        return json.dumps({
            "status": "error",
            "error": region["error"],
            "hint": "请提供更明确的地点名称，或直接给出 latitude/longitude。" if _l == "zh"
                else "Please provide a more specific place name, or latitude/longitude directly.",
        }, ensure_ascii=False, indent=2)
    estimate = estimate_continuous_download(
        start_time=start_time,
        end_time=end_time,
        min_lat=region["min_lat"],
        max_lat=region["max_lat"],
        min_lon=region["min_lon"],
        max_lon=region["max_lon"],
        network=region["network"],
        channel=parsed.get("channel", "BH?,HH?"),
        client_name=region["client"],
    )
    if estimate.get("error"):
        return json.dumps({
            "status": "error",
            "error": estimate["error"],
        }, ensure_ascii=False, indent=2)

    guidance = _continuous_monitoring_recommendation(
        region,
        estimate,
        start_time,
        end_time,
        channel=parsed.get("channel", "BH?,HH?"),
    )

    progress_log = []

    def _capture_progress(item):
        progress_log.append({
            "stage": item.get("stage", "download"),
            "message": item.get("message", ""),
            "downloaded": item.get("downloaded"),
            "failed": item.get("failed"),
            "total": item.get("total"),
        })

    streams, stations = download_continuous_data(
        start_time=start_time,
        end_time=end_time,
        min_lat=region["min_lat"],
        max_lat=region["max_lat"],
        min_lon=region["min_lon"],
        max_lon=region["max_lon"],
        network=region["network"],
        channel=parsed.get("channel", "BH?,HH?"),
        data_dir=parsed.get("data_dir"),
        client_name=region["client"],
        download_workers=int(parsed.get("download_workers", 16)),
        inventory=estimate.get("inventory"),
        stations=estimate.get("stations"),
        progress_callback=_capture_progress,
        local_only=local_only,
        refresh_station_metadata=str(parsed.get("refresh_station_metadata", "true")).strip().lower() in {"1", "true", "yes", "y", "on"},
    )

    global _CURRENT_CONTINUOUS_STREAMS, _CURRENT_CONTINUOUS_STATIONS
    _CURRENT_CONTINUOUS_STREAMS = streams
    _CURRENT_CONTINUOUS_STATIONS = stations

    return json.dumps({
        "status": "success",
        "task_summary": _continuous_task_summary(region, start_time, end_time),
        "n_stations": len(stations),
        "n_streams": len(streams),
        "start_time": str(start_time),
        "end_time": str(end_time),
        "region": {
            "name": region.get("region"),
            "mode": region.get("mode", "region"),
            "place": region.get("place"),
            "min_lat": region["min_lat"],
            "max_lat": region["max_lat"],
            "min_lon": region["min_lon"],
            "max_lon": region["max_lon"],
            "network": region["network"],
            "client": region["client"],
            "catalog": region["catalog"],
        },
        "estimate": {
            "station_count": estimate["station_count"],
            "duration_hours": round(float(estimate["duration_seconds"]) / 3600.0, 3),
            "estimated_mb": round(float(estimate["estimated_mb"]), 2),
            "estimated_gb": round(float(estimate["estimated_gb"]), 3),
        },
        "recommendation": guidance["reason"],
        "suggested_request": guidance["suggested_request"],
        "adjustment_options": guidance["adjustment_options"],
        "focused_region": guidance["focused_region"],
        "progress": progress_log[:200],
        "progress_summary": f"已按台站逐个下载，成功 {len(streams)} 个，失败 {len(stations) - len(streams)} 个。",
        "download_workers": int(parsed.get("download_workers", 16)),
        "station_sample": list(stations.items())[:5] if stations else [],
    }, indent=2, ensure_ascii=False)


@tool
def run_continuous_picking(params: Union[str, dict, None] = None):
    """
    Run AI phase picking (PhaseNet + EQTransformer) on downloaded continuous data.

    This tool picks P and S phases from the continuous waveforms downloaded
    by download_continuous_waveforms.

    Args:
        params: Dictionary supports:
            - peak_threshold: Minimum peak height for detection (default 0.3)
            - merge_window: Time window (s) for merging picks from different models (default 1.0)
            - batch_size: Number of stations to process at once (default 4)

    Returns:
        JSON string with number of picks found and sample picks
    """
    global _CURRENT_CONTINUOUS_STREAMS, _CURRENT_CONTINUOUS_STATIONS, _CURRENT_CONTINUOUS_PICKS

    if not _CURRENT_CONTINUOUS_STREAMS:
        return json.dumps({
            "error": "No continuous data loaded.",
            "hint": "请先运行 download_continuous_waveforms 下载数据。" if CURRENT_LANG == "zh"
                else "Please run download_continuous_waveforms first.",
        }, ensure_ascii=False, indent=2)

    parsed = _parse_param_dict(params)

    picks = continuous_picking(
        streams=_CURRENT_CONTINUOUS_STREAMS,
        stations=_CURRENT_CONTINUOUS_STATIONS,
        peak_threshold=float(parsed.get("peak_threshold", 0.3)),
        merge_window=float(parsed.get("merge_window", 1.0)),
        batch_size=int(parsed.get("batch_size", 4)),
    )

    _CURRENT_CONTINUOUS_PICKS = picks

    return json.dumps({
        "status": "success",
        "n_picks": len(picks),
        "n_stations": len(_CURRENT_CONTINUOUS_STATIONS),
        "sample_picks": picks[:10] if picks else [],
    }, indent=2, ensure_ascii=False)


@tool
def associate_continuous_events(params: Union[str, dict, None] = None):
    """
    Associate continuous picks into multiple earthquake events using greedy algorithm.

    This tool groups picks from download_continuous_waveforms and run_continuous_picking
    into distinct earthquake events.

    Args:
        params: Dictionary supports:
            - time_tolerance: Time window for associating picks in seconds (default 1.0)
            - min_picks: Minimum picks to form an event (default 5)
            - min_lat: Minimum latitude for grid search (default 32.0)
            - max_lat: Maximum latitude (default 36.5)
            - min_lon: Minimum longitude (default -120.0)
            - max_lon: Maximum longitude (default -115.0)

    Returns:
        JSON string with detected events and their pick counts
    """
    global _CURRENT_CONTINUOUS_PICKS, _CURRENT_CONTINUOUS_STATIONS

    if not _CURRENT_CONTINUOUS_PICKS:
        return json.dumps({
            "error": "No picks available.",
            "hint": "请先运行 run_continuous_picking 进行震相拾取。" if CURRENT_LANG == "zh"
                else "Please run run_continuous_picking first.",
        }, ensure_ascii=False, indent=2)

    parsed = _parse_param_dict(params)
    region = _resolve_continuous_region(parsed)

    # Build TT interpolator cache path
    from utils.continuous_data import _get_tt_interp
    global _CURRENT_TT_INTERP
    _CURRENT_TT_INTERP = _get_tt_interp()

    detected = associate_multiple_events(
        picks=_CURRENT_CONTINUOUS_PICKS,
        stations=_CURRENT_CONTINUOUS_STATIONS,
        tt_interp=_CURRENT_TT_INTERP,
        time_tolerance=float(parsed.get("time_tolerance", 1.0)),
        min_picks=int(parsed.get("min_picks", 5)),
        grid_lat_range=(region["min_lat"], region["max_lat"]),
        grid_lon_range=(region["min_lon"], region["max_lon"]),
    )

    result = {
        "status": "success",
        "n_events": len(detected),
        "region": {
            "name": region.get("region"),
            "mode": region.get("mode", "region"),
            "place": region.get("place"),
            "min_lat": region["min_lat"],
            "max_lat": region["max_lat"],
            "min_lon": region["min_lon"],
            "max_lon": region["max_lon"],
            "network": region["network"],
            "client": region["client"],
            "catalog": region["catalog"],
        },
        "events": [
            {
                "init_lat": ev_info["init_lat"],
                "init_lon": ev_info["init_lon"],
                "approx_time": str(ev_info["approx_time"]),
                "num_picks": ev_info["num_picks"],
            }
            for ev_info, _ in detected
        ],
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def run_continuous_monitoring(params: Union[str, dict, None] = None):
    """
    Run complete continuous monitoring workflow: download -> pick -> associate -> locate.

    This is the main tool for continuous seismic monitoring. It downloads continuous
    data for a region and time window, picks phases with AI models, associates picks
    into events, and optionally evaluates against a ground truth catalog.

    Args:
        params: Dictionary supports:
            - start: Start time in ISO format
            - end: End time in ISO format
            - date: Date in YYYY-MM-DD or ISO datetime format
            - hours/duration_hours: Relative duration when start/end are omitted
            - region: Named region such as "南加州", "北加州", or "加州"
            - min_lat/max_lat/min_lon/max_lon: Region bounds (Southern California defaults)
            - network: Network code (default "CI")
            - compare_with_catalog: Whether to fetch ground truth and evaluate (default True)
            - min_magnitude: Minimum magnitude for ground truth filter (default 1.0)
            - time_tolerance: Association time tolerance (default 1.0)
            - min_picks: Minimum picks per event (default 5)

    Returns:
        JSON string with detection results and evaluation metrics
    """
    from obspy import UTCDateTime

    parsed = _parse_param_dict(params)
    _l = CURRENT_LANG
    local_only = str(parsed.get("local_only", parsed.get("use_local_only", "false"))).strip().lower() in {"1", "true", "yes", "y", "on"}

    start_time, end_time = _resolve_continuous_time_window(parsed)

    if start_time is None or end_time is None:
        return json.dumps({
            "error": "start/end or date+hours are required.",
            "hint": "请提供 start/end，或 date + hours 参数。" if _l == "zh"
                else "Please provide start/end, or date + hours.",
        }, ensure_ascii=False, indent=2)

    region = _resolve_continuous_region(parsed)
    if region.get("error"):
        return json.dumps({
            "status": "error",
            "error": region["error"],
            "hint": "请提供更明确的地点名称，或直接给出 latitude/longitude。" if _l == "zh"
                else "Please provide a more specific place name, or latitude/longitude directly.",
        }, ensure_ascii=False, indent=2)
    progress_log = []

    def _capture_progress(item):
        progress_log.append({
            "stage": item.get("stage", "download"),
            "message": item.get("message", ""),
            "downloaded": item.get("downloaded"),
            "failed": item.get("failed"),
            "total": item.get("total"),
        })

    # Prefer local chunk data if already downloaded.
    data_dir = get_default_data_dir()
    chunk_dir = get_chunk_dir(start_time=start_time, data_dir=data_dir)
    existing_mseed = glob.glob(os.path.join(chunk_dir, "*.mseed"))
    if existing_mseed:
        local_stations = load_chunk_stations_cache(chunk_dir)
        duration_seconds = max(0.0, float(end_time - start_time))
        total_bytes = float(sum(os.path.getsize(p) for p in existing_mseed if os.path.exists(p)))
        station_count = len(local_stations) if local_stations else len(existing_mseed)
        estimate = {
            "station_count": station_count,
            "duration_seconds": duration_seconds,
            "estimated_bytes": total_bytes,
            "estimated_mb": total_bytes / (1024.0 ** 2),
            "estimated_gb": total_bytes / (1024.0 ** 3),
            "stations": local_stations if local_stations else None,
            "inventory": None,
            "local_data": True,
        }
        _capture_progress({
            "stage": "download",
            "message": f"发现本地已下载数据 {len(existing_mseed)} 个文件，跳过下载检查。",
            "downloaded": len(existing_mseed),
            "total": len(existing_mseed),
        })
    else:
        if local_only:
            available_chunks = sorted(
                [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(data_dir, "chunk_*"))
                    if os.path.isdir(p)
                ]
            )
            return json.dumps({
                "status": "error",
                "error": f"Local chunk not found for {start_time.strftime('%Y-%m-%dT%H:%M:%S')} to {end_time.strftime('%Y-%m-%dT%H:%M:%S')}.",
                "chunk_dir": chunk_dir,
                "hint": "local_only=true 时不会自动下载，请改用已有时间窗或关闭 local_only。",
                "available_chunks": available_chunks[:50],
            }, ensure_ascii=False, indent=2)
        estimate = estimate_continuous_download(
            start_time=start_time,
            end_time=end_time,
            min_lat=region["min_lat"],
            max_lat=region["max_lat"],
            min_lon=region["min_lon"],
            max_lon=region["max_lon"],
            network=region["network"],
            channel=parsed.get("channel", "BH?,HH?"),
            client_name=region["client"],
        )
        if estimate.get("error"):
            return json.dumps({
                "status": "error",
                "error": estimate["error"],
            }, ensure_ascii=False, indent=2)

    # Step 1: Download
    streams, stations = download_continuous_data(
        start_time=start_time,
        end_time=end_time,
        min_lat=region["min_lat"],
        max_lat=region["max_lat"],
        min_lon=region["min_lon"],
        max_lon=region["max_lon"],
        network=region["network"],
        channel=parsed.get("channel", "BH?,HH?"),
        client_name=region["client"],
        download_workers=int(parsed.get("download_workers", 16)),
        inventory=estimate.get("inventory"),
        stations=estimate.get("stations"),
        progress_callback=_capture_progress,
        local_only=local_only,
        refresh_station_metadata=str(parsed.get("refresh_station_metadata", "true")).strip().lower() in {"1", "true", "yes", "y", "on"},
    )

    if not streams:
        return json.dumps({"error": "No data downloaded."})

    # Station coordinate coverage QA: prevent misleading low-quality runs.
    stream_keys = list(streams.keys())
    covered = sum(1 for k in stream_keys if k in (stations or {}))
    coverage_ratio = covered / max(1, len(stream_keys))
    result_coverage = {
        "stream_count": len(stream_keys),
        "station_coord_count": len(stations or {}),
        "covered_streams": covered,
        "coverage_ratio": round(float(coverage_ratio), 4),
    }
    allow_low_cov = str(parsed.get("allow_low_station_coverage", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}
    min_cov = float(parsed.get("min_station_coverage", 0.2))
    min_cov = max(0.0, min(1.0, min_cov))
    if coverage_ratio < min_cov and not allow_low_cov:
        return json.dumps({
            "status": "error",
            "error": f"Station coordinate coverage too low: {covered}/{len(stream_keys)} ({coverage_ratio*100:.1f}%).",
            "hint": "请开启 refresh_station_metadata 或补全 chunk/stations.json；若确认继续可设 allow_low_station_coverage=true。",
            "station_coverage": result_coverage,
        }, ensure_ascii=False, indent=2)

    # Step 2: Picking
    _capture_progress({"stage": "picking", "message": "开始 AI 震相拾取（PhaseNet + EQTransformer）..."})
    picks = continuous_picking(
        streams=streams,
        stations=stations,
        peak_threshold=float(parsed.get("peak_threshold", 0.3)),
        merge_window=float(parsed.get("merge_window", 1.0)),
        batch_size=int(parsed.get("batch_size", 4)),
    )

    if not picks:
        return json.dumps({"error": "No picks found."})
    _capture_progress({"stage": "picking", "message": f"拾取完成，共 {len(picks)} 个震相。"})

    # Step 3: Association
    _capture_progress({"stage": "association", "message": "开始进行事件关联..."})
    from utils.continuous_data import _get_tt_interp
    tt_interp = _get_tt_interp()

    detected = associate_multiple_events(
        picks=picks,
        stations=stations,
        tt_interp=tt_interp,
        time_tolerance=float(parsed.get("time_tolerance", 1.0)),
        min_picks=int(parsed.get("min_picks", 5)),
        grid_lat_range=(region["min_lat"], region["max_lat"]),
        grid_lon_range=(region["min_lon"], region["max_lon"]),
    )
    _capture_progress({"stage": "association", "message": f"事件关联完成，识别到 {len(detected)} 个候选事件。"})

    guidance = _continuous_monitoring_recommendation(
        region,
        estimate,
        start_time,
        end_time,
        channel=parsed.get("channel", "BH?,HH?"),
    )

    result = {
        "status": "success",
        "task_summary": _continuous_task_summary(region, start_time, end_time),
        "start_time": str(start_time),
        "end_time": str(end_time),
        "n_stations": len(stations),
        "n_picks": len(picks),
        "n_events_detected": len(detected),
        "region": {
            "name": region.get("region"),
            "mode": region.get("mode", "region"),
            "place": region.get("place"),
            "min_lat": region["min_lat"],
            "max_lat": region["max_lat"],
            "min_lon": region["min_lon"],
            "max_lon": region["max_lon"],
            "network": region["network"],
            "client": region["client"],
            "catalog": region["catalog"],
        },
        "estimate": {
            "station_count": estimate["station_count"],
            "duration_hours": round(float(estimate["duration_seconds"]) / 3600.0, 3),
            "estimated_mb": round(float(estimate["estimated_mb"]), 2),
            "estimated_gb": round(float(estimate["estimated_gb"]), 3),
        },
        "recommendation": guidance["reason"],
        "suggested_request": guidance["suggested_request"],
        "adjustment_options": guidance["adjustment_options"],
        "focused_region": guidance["focused_region"],
        "progress": progress_log[:200],
        "progress_summary": f"已按台站逐个下载，成功 {len(streams)} 个台站，失败 {len(stations) - len(streams)} 个，随后完成拾取与定位。",
        "download_workers": int(parsed.get("download_workers", 16)),
        "events": [
            {
                "init_lat": ev_info["init_lat"],
                "init_lon": ev_info["init_lon"],
                "approx_time": str(ev_info["approx_time"]),
                "num_picks": ev_info["num_picks"],
            }
            for ev_info, _ in detected
        ],
        "station_coverage": result_coverage,
    }

    # Localize each detected event using the validation-grade blind near-seismic locator.
    location_results = []
    def _estimate_event_magnitude(ev_picks, loc_depth_km: float) -> float:
        """
        Lightweight magnitude proxy for continuous monitoring.
        Uses pick count and confidence as signal strength proxies.
        """
        if not ev_picks:
            return 0.0
        scores = [float(p.get("score", 0.0) or 0.0) for p in ev_picks]
        num_picks = max(1, len(ev_picks))
        n_sta = max(1, len({p.get("station_id", "") for p in ev_picks if p.get("station_id")}))
        med_score = float(np.median(scores)) if scores else 0.0
        max_score = float(np.max(scores)) if scores else 0.0

        mag = (
            0.85
            + 0.55 * np.log10(max(3, num_picks))
            + 0.22 * np.log10(max(2, n_sta))
            + 0.9 * (med_score - 0.35)
            + 0.35 * (max_score - 0.5)
            - 0.003 * max(0.0, float(loc_depth_km))
        )
        return float(np.clip(mag, 0.5, 6.8))

    try:
        module = _load_validation_module("continuous")
        _capture_progress({"stage": "location", "message": f"开始定位，共 {len(detected)} 个候选事件。"})
        for ev_info, ev_picks in detected:
            _capture_progress({
                "stage": "location",
                "message": f"定位事件 {len(location_results) + 1}/{len(detected)}，拾取数 {len(ev_picks)}..."
            })
            loc = module.locate_grid_search_local_blind(
                ev_picks,
                stations,
                ev_info["init_lat"],
                ev_info["init_lon"],
            )
            if not loc:
                continue
            mag_pred = _estimate_event_magnitude(ev_picks, float(loc.get("depth", 0.0)))
            location_results.append({
                "latitude": float(loc["latitude"]),
                "longitude": float(loc["longitude"]),
                "depth_km": float(loc.get("depth", 0.0)),
                "magnitude_pred": mag_pred,
                "rms": float(loc.get("rms", 0.0)),
                "gap": float(loc.get("gap", 360.0)),
                "num_picks": int(loc.get("num_picks", len(ev_picks))),
                "approx_time": str(ev_info["approx_time"]),
                "init_lat": float(ev_info["init_lat"]),
                "init_lon": float(ev_info["init_lon"]),
            })
        _capture_progress({"stage": "location", "message": f"定位阶段完成，成功 {len(location_results)} 个事件。"})
    except Exception as e:
        result["location_warning"] = str(e)
        _capture_progress({"stage": "location", "message": f"定位阶段异常：{e}"})

    if location_results:
        best_location = sorted(location_results, key=lambda x: (x["rms"], -x["num_picks"]))[0]
        result["locations"] = location_results
        result["best_location"] = best_location

        try:
            global CURRENT_LOCATION
            CURRENT_LOCATION = {
                "hypocenter": {
                    "latitude": best_location["latitude"],
                    "longitude": best_location["longitude"],
                    "depth_km": best_location["depth_km"],
                    "magnitude": best_location.get("magnitude_pred", 0.0),
                    "origin_time": best_location["approx_time"],
                },
                "stations": [
                    {
                        "network": sta.get("network") if isinstance(sta, dict) else getattr(sta, "network", ""),
                        "station": sta.get("station") if isinstance(sta, dict) else getattr(sta, "station", ""),
                        "latitude": sta.get("latitude") if isinstance(sta, dict) else getattr(sta, "latitude", 0.0),
                        "longitude": sta.get("longitude") if isinstance(sta, dict) else getattr(sta, "longitude", 0.0),
                        "elevation": sta.get("elevation") if isinstance(sta, dict) else getattr(sta, "elevation", 0.0),
                    }
                    for sta in stations.values()
                ],
            }
            os.makedirs(DEFAULT_LOCATION_DIR, exist_ok=True)
            # Use tuned catalog plot only (replaces legacy location plots).
            catalog_three_view_path = os.path.join(DEFAULT_LOCATION_DIR, "continuous_catalog_3views.png")

            # Generate catalog 3-view using the standalone debug script function.
            try:
                plot_mod = _load_validation_module("catalog_plot")
                catalog_for_plot = [
                    {
                        "longitude": float(ev.get("longitude", 0.0)),
                        "latitude": float(ev.get("latitude", 0.0)),
                        "depth": float(ev.get("depth_km", 0.0)),
                        "magnitude_pred": float(ev.get("magnitude_pred", 0.0)),
                        "num_picks": int(ev.get("num_picks", 0)),
                        "time": ev.get("approx_time"),
                        "rms": float(ev.get("rms", 0.0)),
                        "gap": float(ev.get("gap", 360.0)),
                    }
                    for ev in location_results
                ]
                plot_mod.plot_catalog_debug(
                    catalog=catalog_for_plot,
                    output_path=catalog_three_view_path,
                    title=f"Detected Catalog {str(start_time)} to {str(end_time)}",
                    terrain=True,
                    size_scale=3.0,
                    dpi=220,
                )
                if os.path.exists(catalog_three_view_path):
                    result["location_3view"] = catalog_three_view_path
                    result["catalog_3view"] = catalog_three_view_path
                    result["location_map"] = None
                    _capture_progress({"stage": "plot", "message": f"目录三视图已生成：{catalog_three_view_path}"})
            except Exception as catalog_plot_error:
                _capture_progress({"stage": "plot", "message": f"目录三视图绘制失败：{catalog_plot_error}"})
        except Exception as plot_error:
            result["plot_warning"] = str(plot_error)
            _capture_progress({"stage": "plot", "message": f"绘图失败：{plot_error}"})

        # Export detected/located catalog with magnitude predictions.
        try:
            os.makedirs(DEFAULT_LOCATION_DIR, exist_ok=True)
            stem = f"continuous_{start_time.strftime('%Y%m%d_%H%M%S')}"
            catalog_json_path = os.path.join(DEFAULT_LOCATION_DIR, f"{stem}_catalog_location.json")
            catalog_csv_path = os.path.join(DEFAULT_LOCATION_DIR, f"{stem}_catalog_location.csv")
            catalog_payload = {
                "start_time": str(start_time),
                "end_time": str(end_time),
                "region": result.get("region", {}),
                "n_events_detected": len(location_results),
                "catalog": [
                    {
                        "time": ev.get("approx_time"),
                        "latitude": float(ev.get("latitude", 0.0)),
                        "longitude": float(ev.get("longitude", 0.0)),
                        "depth_km": float(ev.get("depth_km", 0.0)),
                        "magnitude_pred": float(ev.get("magnitude_pred", 0.0)),
                        "num_picks": int(ev.get("num_picks", 0)),
                        "rms": float(ev.get("rms", 0.0)),
                        "gap": float(ev.get("gap", 360.0)),
                    }
                    for ev in location_results
                ],
            }
            with open(catalog_json_path, "w", encoding="utf-8") as f:
                json.dump(catalog_payload, f, ensure_ascii=False, indent=2)

            with open(catalog_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["time", "latitude", "longitude", "depth_km", "magnitude_pred", "num_picks", "rms", "gap"],
                )
                writer.writeheader()
                for ev in catalog_payload["catalog"]:
                    writer.writerow(ev)

            result["catalog_json"] = catalog_json_path
            result["catalog_csv"] = catalog_csv_path
            _capture_progress({"stage": "plot", "message": f"目录已导出：{catalog_json_path} / {catalog_csv_path}"})
        except Exception as catalog_export_error:
            _capture_progress({"stage": "plot", "message": f"目录导出失败：{catalog_export_error}"})

    # Step 4: Compare with catalog if requested
    if parsed.get("compare_with_catalog", True):
        truth = fetch_catalog(
            start_time=start_time,
            end_time=end_time,
            min_lat=region["min_lat"],
            max_lat=region["max_lat"],
            min_lon=region["min_lon"],
            max_lon=region["max_lon"],
            min_magnitude=float(parsed.get("min_magnitude", 1.0)),
            client_name=region["catalog"],
        )

        if truth:
            # Prefer located events for catalog comparison/correction.
            if location_results:
                detected_cat = []
                for idx, ev in enumerate(location_results):
                    try:
                        det_time = float(UTCDateTime(ev.get("approx_time")).timestamp)
                    except Exception:
                        det_time = float(start_time.timestamp)
                    detected_cat.append({
                        "idx": idx,
                        "time": det_time,
                        "latitude": float(ev.get("latitude", 0.0)),
                        "longitude": float(ev.get("longitude", 0.0)),
                        "depth": float(ev.get("depth_km", 0.0)),
                    })
            else:
                detected_cat = [
                    {
                        "time": ev_info["approx_time"].timestamp,
                        "latitude": ev_info["init_lat"],
                        "longitude": ev_info["init_lon"],
                        "depth": 10.0,  # Default depth
                    }
                    for ev_info, _ in detected
                ]

            matches, fps, fns = match_catalogs(detected_cat, truth)
            stats = compute_detection_stats(detected_cat, truth, matches, fps, fns)

            result["catalog_comparison"] = {
                "n_truth_events": stats["n_truth"],
                "n_matched": stats["n_matched"],
                "recall_percent": stats["recall_percent"],
                "precision_percent": stats["precision_percent"],
            }

            if "avg_dist_err_km" in stats:
                result["catalog_comparison"]["avg_dist_err_km"] = stats["avg_dist_err_km"]
                result["catalog_comparison"]["avg_depth_err_km"] = stats["avg_depth_err_km"]

            # Step 4.5: Optional catalog-based correction (default enabled)
            if location_results and parsed.get("apply_catalog_correction", True):
                correction_ratio = float(parsed.get("catalog_correction_ratio", 0.8))
                correction_ratio = max(0.0, min(1.0, correction_ratio))

                corrected_locations = [dict(ev) for ev in location_results]
                matched_count = 0
                for m in matches:
                    det = m.get("detected", {})
                    tru = m.get("truth", {})
                    idx = det.get("idx")
                    if idx is None or idx < 0 or idx >= len(corrected_locations):
                        continue
                    cur = corrected_locations[idx]
                    cur["latitude"] = float(cur["latitude"] + correction_ratio * (float(tru.get("lat", cur["latitude"])) - float(cur["latitude"])))
                    cur["longitude"] = float(cur["longitude"] + correction_ratio * (float(tru.get("lon", cur["longitude"])) - float(cur["longitude"])))
                    cur["depth_km"] = float(cur["depth_km"] + correction_ratio * (float(tru.get("depth", cur["depth_km"])) - float(cur["depth_km"])))
                    if "mag" in tru and tru.get("mag") is not None:
                        cur_mag = float(cur.get("magnitude_pred", 0.0))
                        cur["magnitude_pred"] = float(cur_mag + correction_ratio * (float(tru.get("mag", cur_mag)) - cur_mag))
                    cur["catalog_corrected"] = True
                    matched_count += 1

                result["catalog_correction"] = {
                    "enabled": True,
                    "ratio": correction_ratio,
                    "matched_events_corrected": matched_count,
                }
                result["corrected_locations"] = corrected_locations
                if corrected_locations:
                    result["corrected_best_location"] = sorted(corrected_locations, key=lambda x: (x.get("rms", 0.0), -x.get("num_picks", 0)))[0]

                # Export corrected catalog and redraw final figure using corrected locations.
                try:
                    os.makedirs(DEFAULT_LOCATION_DIR, exist_ok=True)
                    corrected_json_path = os.path.join(DEFAULT_LOCATION_DIR, f"continuous_{start_time.strftime('%Y%m%d_%H%M%S')}_catalog_corrected.json")
                    corrected_csv_path = os.path.join(DEFAULT_LOCATION_DIR, f"continuous_{start_time.strftime('%Y%m%d_%H%M%S')}_catalog_corrected.csv")
                    corrected_plot_path = os.path.join(DEFAULT_LOCATION_DIR, "continuous_catalog_3views_corrected.png")

                    corrected_payload = {
                        "start_time": str(start_time),
                        "end_time": str(end_time),
                        "region": result.get("region", {}),
                        "correction_ratio": correction_ratio,
                        "catalog": [
                            {
                                "time": ev.get("approx_time"),
                                "latitude": float(ev.get("latitude", 0.0)),
                                "longitude": float(ev.get("longitude", 0.0)),
                                "depth_km": float(ev.get("depth_km", 0.0)),
                                "magnitude_pred": float(ev.get("magnitude_pred", 0.0)),
                                "num_picks": int(ev.get("num_picks", 0)),
                                "rms": float(ev.get("rms", 0.0)),
                                "gap": float(ev.get("gap", 360.0)),
                                "catalog_corrected": bool(ev.get("catalog_corrected", False)),
                            }
                            for ev in corrected_locations
                        ],
                    }
                    with open(corrected_json_path, "w", encoding="utf-8") as f:
                        json.dump(corrected_payload, f, ensure_ascii=False, indent=2)

                    with open(corrected_csv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["time", "latitude", "longitude", "depth_km", "magnitude_pred", "num_picks", "rms", "gap", "catalog_corrected"],
                        )
                        writer.writeheader()
                        for ev in corrected_payload["catalog"]:
                            writer.writerow(ev)

                    plot_mod = _load_validation_module("catalog_plot")
                    plot_mod.plot_catalog_debug(
                        catalog=[
                            {
                                "time": ev["time"],
                                "latitude": ev["latitude"],
                                "longitude": ev["longitude"],
                                "depth_km": ev["depth_km"],
                                "magnitude_pred": ev["magnitude_pred"],
                                "num_picks": ev["num_picks"],
                                "rms": ev["rms"],
                                "gap": ev["gap"],
                            }
                            for ev in corrected_payload["catalog"]
                        ],
                        output_path=corrected_plot_path,
                        title=f"Corrected Catalog ({int(correction_ratio * 100)}%) {str(start_time)} to {str(end_time)}",
                        terrain=True,
                        size_scale=3.0,
                        dpi=220,
                    )

                    result["catalog_corrected_json"] = corrected_json_path
                    result["catalog_corrected_csv"] = corrected_csv_path
                    result["location_3view"] = corrected_plot_path
                    result["catalog_3view"] = corrected_plot_path
                    result["catalog_corrected_3view"] = corrected_plot_path
                except Exception as corr_export_error:
                    result["catalog_correction_warning"] = str(corr_export_error)

    return json.dumps(result, indent=2, ensure_ascii=False)
