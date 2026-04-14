from dataclasses import asdict

from langchain.tools import tool
from utils.segy_handler import SegyHandler
from utils.miniseed_handler import MiniSEEDHandler
from utils.hdf5_handler import HDF5Handler
from utils.phase_picker import pick_phases, summarize_pick_results, load_traces, plot_waveform_with_picks
from utils.phase_picker import TraceRecord, PickResult
from utils.sac_handler import SACHandler
import json
from typing import Union
import numpy as np
import os
import h5py

# Global variable to store the current file path being analyzed
# In a multi-user web app, this should be handled via session state or context
CURRENT_SEGY_PATH = None
CURRENT_MINISEED_PATH = None
CURRENT_MINISEED_PATHS = []  # Support multiple MiniSEED files for multi-station location
CURRENT_HDF5_PATH = None
CURRENT_SAC_PATH = None
CURRENT_LANG = "en"  # Current UI language, set from app.py


def set_current_lang(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang


DEFAULT_CONVERT_DIR = "data/convert"


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


def _resolve_output_path(output_path: str | None, *, default_filename: str) -> str:
    """Resolve output file path.

    Rules:
    - If output_path is empty -> data/convert/{default_filename}
    - If output_path is filename only -> data/convert/{output_path}
    - If output_path contains a directory (relative/absolute) -> keep as is
    Also ensures the parent directory exists.
    """

    if not output_path or not str(output_path).strip():
        final_path = os.path.join(DEFAULT_CONVERT_DIR, default_filename)
    else:
        output_path = str(output_path).strip()
        if os.path.dirname(output_path) == "":
            final_path = os.path.join(DEFAULT_CONVERT_DIR, output_path)
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
                output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
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
                output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
                os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
                
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

    return json.dumps(summary, indent=2)

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

    if isinstance(result, dict):
        return json.dumps(result, indent=2)
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

    if isinstance(result, dict):
        return json.dumps(result, indent=2)
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
                output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
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
            output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            
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
            output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)

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
    return json.dumps(result, indent=2)

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
        result["saved_to"] = result.get("output_path", output_path)
        return json.dumps(result, indent=2)
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
    return json.dumps(result, indent=2)

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
    return json.dumps(result, indent=2)

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
    return json.dumps(result, indent=2)

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
                output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
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
            output_path = os.path.join(DEFAULT_CONVERT_DIR, output_filename)
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)

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
    return json.dumps(result, indent=2)

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
    return json.dumps(result, indent=2)

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
    return json.dumps(result, indent=2)

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
    return json.dumps(result, indent=2)


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
        
        # 2. Load traces for plotting
        traces = load_traces(
            source=path,
            file_type=file_type,
            dataset=dataset
        )
        
        # 3. Generate plot
        plot_path = _resolve_output_path(None, default_filename="picks_plot.png")
        plot_waveform_with_picks(traces, picks, plot_path)

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

        lines.append(f"{'初至拾取已完成，结果图表已保存至' if _l == 'zh' else 'Phase picking completed, plot saved to'}：`{plot_path}`")
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
    3. Generate a combined plot showing picks from all stations

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

                # Get summary for this file
                filename = os.path.basename(filepath)
                file_summaries.append({
                    "file": filename,
                    "traces": len(file_traces),
                    "picks": len(file_picks)
                })

            except Exception as e:
                file_summaries.append({
                    "file": os.path.basename(filepath),
                    "error": str(e)
                })

        # Store all picks for location
        CURRENT_PICKS = all_picks

        # Generate combined plot
        plot_path = None
        if all_traces and all_picks:
            plot_path = os.path.join(DEFAULT_CONVERT_DIR, "all_stations_picks.png")
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)
            plot_waveform_with_picks(all_traces, all_picks, plot_path)

        # Build output (language-aware)
        _l = CURRENT_LANG
        lines = []
        if plot_path:
            lines.append(f"![{'所有台站拾取结果图' if _l == 'zh' else 'All Stations Picking Results'}]({plot_path})")
            lines.append("")

        lines.append(f"**{'初至拾取完成' if _l == 'zh' else 'Phase picking completed'}**")
        lines.append(f"- {'处理文件数' if _l == 'zh' else 'Files processed'}: {len(CURRENT_MINISEED_PATHS)}")
        lines.append(f"- {'总拾取数' if _l == 'zh' else 'Total picks'}: {len(all_picks)}")
        lines.append(f"- {'总轨迹数' if _l == 'zh' else 'Total traces'}: {len(all_traces)}")
        lines.append("")

        # Show per-file summary
        lines.append(f"### {'各文件拾取结果' if _l == 'zh' else 'Per-file Results'}")
        for fs in file_summaries:
            if "error" in fs:
                lines.append(f"- **{fs['file']}**: {'错误' if _l == 'zh' else 'error'} - {fs['error']}")
            else:
                if _l == "zh":
                    lines.append(f"- **{fs['file']}**: {fs['picks']} 个拾取, {fs['traces']} 条轨迹")
                else:
                    lines.append(f"- **{fs['file']}**: {fs['picks']} picks, {fs['traces']} traces")

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


@tool
def locate_earthquake(params: Union[str, dict, None] = None):
    """
    Locate earthquake hypocenter using phase picks from multiple stations.

    This tool requires:
    1. Phase picks with arrival times (from pick_first_arrivals)
    2. Station coordinates (latitude, longitude)

    The location algorithm uses:
    - Grid search for initial estimate
    - Geiger's method (least squares) for refinement
    - IASP91 velocity model for travel time calculation

    Args:
        params: Dictionary with optional parameters:
            - method: "auto", "grid_search", or "geiger" (default: "auto")
            - grid_center: [lat, lon] center of search grid
            - grid_size_deg: Search grid half-width in degrees (default: 2.0)
            - depth_range_km: [min, max] depth range in km (default: [0, 50])
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

        # Generate location map using PyGMT
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
            map_path = os.path.join(DEFAULT_CONVERT_DIR, "earthquake_location_map.png")
            os.makedirs(DEFAULT_CONVERT_DIR, exist_ok=True)

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
    Plot earthquake location and station positions on a map using PyGMT.

    Use this tool after locating an earthquake to visualize the result.
    It reads the last location result and station coordinates automatically.

    Args:
        params: Dictionary with optional parameters:
            - region: [west, east, south, north] custom map region in degrees
            - title: Custom map title
            - output: Custom output file path (default: data/convert/earthquake_location_map.png)

    Returns:
        JSON with the path to the saved map image.
    """
    """
    使用 PyGMT 将地震定位结果和台站位置绘制在地图上。

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
    output_path = parsed.get("output", os.path.join(DEFAULT_CONVERT_DIR, "earthquake_location_map.png"))

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
                "error": "Failed to generate map. PyGMT may not be available.",
                "hint": "请确保 PyGMT 已正确安装。"
            }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Map generation failed: {str(e)}",
            "hint": "地图生成失败，请检查 PyGMT 安装。"
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
    from obspy.clients.fdsn import Client as FDSNClient
    from obspy import UTCDateTime
    from obspy.geodetics import locations2degrees
    import glob

    parsed = _parse_param_dict(params)
    _l = CURRENT_LANG

    # Parameters
    lat = _coerce_float(parsed.get("latitude"), allow_none=True, field_name="latitude")
    lon = _coerce_float(parsed.get("longitude"), allow_none=True, field_name="longitude")
    radius_km = _coerce_float(parsed.get("radius_km"), default=200.0, field_name="radius_km")
    minmagnitude = _coerce_float(parsed.get("minmagnitude"), default=4.0, field_name="minmagnitude")
    max_stations = _coerce_int(parsed.get("max_stations"), default=10, field_name="max_stations")
    network = parsed.get("network", None)
    channel = parsed.get("channel", "BH?")
    output_dir = parsed.get("output_dir", "data/fdsn/")

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

    try:
        client = FDSNClient("IRIS")
    except Exception as e:
        return json.dumps({
            "error": f"Cannot connect to IRIS FDSN service: {e}",
            "hint": "无法连接 IRIS FDSN 服务，请检查网络。"
        }, ensure_ascii=False, indent=2)

    lines = []

    # Step 1: Search for events
    lines.append(f"### {'地震事件搜索' if _l == 'zh' else 'Event Search'}")
    lines.append(f"- {'搜索范围' if _l == 'zh' else 'Region'}: ({lat}, {lon}), {'半径' if _l == 'zh' else 'radius'} {radius_km} km")
    lines.append(f"- {'时间范围' if _l == 'zh' else 'Time range'}: {starttime} ~ {endtime}")
    lines.append(f"- {'最小震级' if _l == 'zh' else 'Min magnitude'}: {minmagnitude}")
    lines.append("")

    try:
        catalog = client.get_events(
            starttime=starttime,
            endtime=endtime,
            latitude=lat,
            longitude=lon,
            maxradius=radius_km / 111.0,  # Convert km to degrees (approximate)
            minmagnitude=minmagnitude,
            orderby="magnitude",
        )
    except Exception as e:
        return json.dumps({
            "error": f"Event search failed: {e}",
            "hint": f"地震事件搜索失败: {e}"
        }, ensure_ascii=False, indent=2)

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

    lines.append(f"### {'下载波形数据' if _l == 'zh' else 'Downloading Waveform Data'}")
    lines.append(f"{'目标事件' if _l == 'zh' else 'Target event'}: {evt_time}, ({evt_lat:.2f}, {evt_lon:.2f})")
    lines.append("")

    # Search for stations
    station_kwargs = {
        "starttime": evt_time - 3600,
        "endtime": evt_time + 7200,
        "latitude": evt_lat,
        "longitude": evt_lon,
        "maxradius": radius_km / 111.0,
    }
    if network:
        station_kwargs["network"] = network
    if channel:
        station_kwargs["channel"] = channel

    try:
        inventory = client.get_stations(**station_kwargs, level="station")
    except Exception as e:
        lines.append(f"{'台站搜索失败' if _l == 'zh' else 'Station search failed'}: {e}")
        return "\n".join(lines)

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

    # Step 3: Download waveform data
    downloaded_files = []
    failed_stations = []

    for s in stations_info:
        sta_code = f"{s['network']}.{s['station']}"
        filename = f"{s['network']}.{s['station']}..{channel.replace('?', 'Z')}.mseed"
        filepath = os.path.join(output_dir, filename)

        try:
            waveform_kwargs = {
                "network": s["network"],
                "station": s["station"],
                "location": "*",
                "channel": channel,
                "starttime": evt_time - 60,
                "endtime": evt_time + 1800,
            }
            st = client.get_waveforms(**waveform_kwargs)
            st.write(filepath, format="MSEED")
            downloaded_files.append(filepath)
            add_miniseed_path(filepath)
        except Exception as e:
            failed_stations.append(f"{sta_code}: {e}")

    # Step 4: Save stations.json
    if downloaded_files:
        stations_json_path = os.path.join(output_dir, "stations.json")
        stations_dict = {}
        for s in stations_info:
            if s["network"] in [ds.split("/")[-1].split(".")[0] for ds in downloaded_files if s["station"] in ds]:
                for f in downloaded_files:
                    fname = os.path.basename(f)
                    if s["network"] in fname and s["station"] in fname:
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
        if failed_stations:
            lines.append(f"- {'失败' if _l == 'zh' else 'Failed'}: {len(failed_stations)} {'个台站' if _l == 'zh' else 'stations'}")
            for fs in failed_stations[:5]:
                lines.append(f"  - {fs}")
        lines.append(f"- {'台站坐标已保存至' if _l == 'zh' else 'Station coordinates saved to'}: `{stations_json_path}`")
        lines.append(f"- {'数据保存目录' if _l == 'zh' else 'Data saved to'}: `{output_dir}`")
        lines.append("")
        lines.append(f"{'下一步可以使用 pick_all_miniseed_files 进行震相拾取，然后定位' if _l == 'zh' else 'Next: use pick_all_miniseed_files for phase picking, then locate_earthquake'}.")

    return "\n".join(lines)
