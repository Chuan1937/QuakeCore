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
            trace_data = handler.get_trace_data(trace_index=0)
            if "error" not in trace_data:
                tr = TraceRecord(
                    data=trace_data["data"],
                    sampling_rate=trace_data["sampling_rate"],
                    start_time=trace_data.get("start_time"),
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
            trace_data = handler.get_trace_data(trace_index=0)
            if "error" not in trace_data:
                tr = TraceRecord(
                    data=trace_data["data"],
                    sampling_rate=trace_data["sampling_rate"],
                    start_time=trace_data["start_time"],
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

    Available Methods:
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
    - phasenet: Deep learning P/S picker using PhaseNet (SeisBench).
    - eqtransformer: Deep learning P/S picker using EQTransformer (SeisBench).
    - gpd: Deep learning P/S picker using GPD (SeisBench).

    Args:
      path (optional): File path override.
      file_type (optional): 'hdf5', 'sac', 'segy', 'miniseed'.
      dataset (optional): For HDF5, specific dataset name.
      methods (optional): Comma-separated list of methods (e.g. 'sta_lta,aic').
      method_params (optional): JSON string or dict of params for methods.
    """
    """在当前加载的文件上拾取初至（震相）。

    可用方法：
    - sta_lta: 经典的短长时窗平均比值法 (STA/LTA)。
    - aic: 赤池信息准则 (AIC)，用于精确到时优化。
    - frequency_ratio: 基于频谱变化的检测方法。
    - autocorr: 基于自相关的检测方法。
    - feature_threshold: 简单的振幅/包络阈值法。
    - ar_model: 自回归模型预测误差法。
    - template_correlation: 模板匹配滤波法。
    - pai_k: 基于峰度的 PAI-K 拾取。
    - pai_s: 基于偏度的 PAI-S 拾取。
    - s_phase: 启发式 S 波拾取（P 波后延迟寻找最大 STA/LTA）。
    - phasenet: 深度学习 P/S 波拾取（PhaseNet，基于 SeisBench）。
    - eqtransformer: 深度学习 P/S 波拾取（EQTransformer，基于 SeisBench）。
    - gpd: 深度学习 P/S 波拾取（GPD，基于 SeisBench）。

    参数：
      path（可选）：文件路径覆盖。
      file_type（可选）：'hdf5', 'sac', 'segy', 'miniseed'。
      dataset（可选）：针对 HDF5 的特定数据集名称。
      methods（可选）：逗号分隔的方法列表（例如 'sta_lta,aic,phasenet'）。
      method_params（可选）：方法的参数（JSON 字符串或字典）。
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
        
        # Format as a Markdown output (Chinese)
        lines = []
        lines.append(f"![拾取结果图]({plot_path})")
        lines.append("")

        lines.append(f"初至拾取已完成，结果图表已保存至：`{plot_path}`")
        for trace_index in sorted(best_by_trace.keys()):
            best_p = best_by_trace[trace_index].get("P")
            best_s = best_by_trace[trace_index].get("S")
            parts = []
            if best_p:
                parts.append(
                    f"最佳P波：方法 {best_p.get('method')}，样本点 {best_p.get('sample_index')}，时间 {best_p.get('absolute_time')}，评分 {best_p.get('normalized_score'):.4f}"
                )
            else:
                parts.append("最佳P波：未检出")
            if best_s:
                parts.append(
                    f"最佳S波：方法 {best_s.get('method')}，样本点 {best_s.get('sample_index')}，时间 {best_s.get('absolute_time')}，评分 {best_s.get('normalized_score'):.4f}"
                )
            else:
                parts.append("最佳S波：未检出")
            lines.append(f"摘要（Trace {trace_index}）：" + "；".join(parts))
        lines.append("")
        
        for item in summary:
            lines.append(f"### Trace {item['trace_index']}")
            lines.append("| 相位 | 方法 | 样本点 | 评分 | 绝对时间 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            
            METHOD_TRANSLATIONS = {
                "sta_lta": "STA/LTA (sta_lta)",
                "aic": "AIC (aic)",
                "frequency_ratio": "频率比 (frequency_ratio)",
                "autocorr": "自相关 (autocorr)",
                "feature_threshold": "特征阈值 (feature_threshold)",
                "ar_model": "AR 模型 (ar_model)",
                "template_correlation": "模板相关 (template_correlation)",
                "s_phase": "S 波拾取 (s_phase)",
                "pai_k": "PAI-K (pai_k)",
                "pai_s": "PAI-S (pai_s)"
            }

            # Sort methods by score descending
            sorted_methods = sorted(item['methods'], key=lambda x: x['normalized_score'] or 0, reverse=True)
            for m in sorted_methods:
                score = f"{m['normalized_score']:.4f}" if m['normalized_score'] is not None else "N/A"
                phase = m.get('phase_type', 'P')
                method_key = m['method']
                method_display = METHOD_TRANSLATIONS.get(method_key, method_key)
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

    This tool will:
    1. Pick P and S phases on each loaded MiniSEED file
    2. Store all picks for later use by locate_earthquake
    3. Generate a combined plot showing picks from all stations

    Args:
        params: Dictionary with optional parameters:
            - methods: Comma-separated list of methods (default: uses deep learning + conventional)
            - method_params: JSON string or dict of params for methods
    """
    """
    对所有已加载的 MiniSEED 文件进行初至拾取。

    当你上传了多个 MiniSEED 文件（每个台站一个）并想对所有文件进行震相拾取以进行地震定位时，使用此工具。

    此工具将：
    1. 对每个已加载的 MiniSEED 文件拾取 P 波和 S 波
    2. 存储所有拾取结果供 locate_earthquake 使用
    3. 生成显示所有台站拾取结果的综合图

    参数：
        params: 可选参数字典：
            - methods: 逗号分隔的方法列表（默认：使用深度学习 + 传统方法）
            - method_params: 方法的参数（JSON 字符串或字典）
    """
    global CURRENT_PICKS, CURRENT_MINISEED_PATHS

    parsed = _parse_param_dict(params)

    if not CURRENT_MINISEED_PATHS:
        return json.dumps({
            "error": "No MiniSEED files loaded.",
            "hint": "请先上传 MiniSEED 文件。"
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

        # Build output
        lines = []
        if plot_path:
            lines.append(f"![所有台站拾取结果图]({plot_path})")
            lines.append("")

        lines.append(f"**初至拾取完成**")
        lines.append(f"- 处理文件数: {len(CURRENT_MINISEED_PATHS)}")
        lines.append(f"- 总拾取数: {len(all_picks)}")
        lines.append(f"- 总轨迹数: {len(all_traces)}")
        lines.append("")

        # Show per-file summary
        lines.append("### 各文件拾取结果")
        for fs in file_summaries:
            if "error" in fs:
                lines.append(f"- **{fs['file']}**: 错误 - {fs['error']}")
            else:
                lines.append(f"- **{fs['file']}**: {fs['picks']} 个拾取, {fs['traces']} 条轨迹")

        lines.append("")
        lines.append("拾取结果已保存，可使用 `locate_earthquake` 工具进行定位。")
        lines.append("如需添加台站坐标，请使用 `add_station_coordinates` 工具。")

        return "\n".join(lines)

    except Exception as e:
        return f"批量拾取失败: {str(e)}"


# ==================== Earthquake Location Tools ====================

# Store picks for location
CURRENT_PICKS = None
CURRENT_STATIONS = None


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

        # Generate location map using PyGMT
        try:
            from utils.locator import plot_location_map

            # Get station list for plotting
            station_list = list(stations_dict.values()) if stations_dict else []
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

    if not stations_input:
        return json.dumps({
            "error": "No stations provided.",
            "usage": "Provide a list of stations with network, station, latitude, longitude."
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
