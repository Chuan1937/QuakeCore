from langchain.tools import tool
from utils.segy_handler import SegyHandler
import json
from typing import Union
import numpy as np
import os

# Global variable to store the current file path being analyzed
# In a multi-user web app, this should be handled via session state or context
CURRENT_SEGY_PATH = None


DEFAULT_CONVERT_DIR = "data/convert"


# 设置当前 SEGY 文件路径的辅助函数
def set_current_segy_path(path):
    global CURRENT_SEGY_PATH
    CURRENT_SEGY_PATH = path

# 解析参数字典的辅助函数
def _parse_param_dict(raw_params):
    """Normalize incoming tool arguments to a dict."""
    """归一化传入的工具参数为字典。"""
    if raw_params is None:
        return {}
    if isinstance(raw_params, dict):
        return raw_params
    if isinstance(raw_params, str):
        candidate = raw_params.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            params = {}
            for chunk in candidate.split(","):
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                    params[key.strip()] = value.strip()
            return params
    return {}

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
def get_segy_structure():
    """
    Reads the currently loaded SEGY file and returns a summary of its structure,
    including trace count, sample rate, and sample count.
    Use this tool when the user asks about the 'structure', 'basic info', or 'overview' of the SEGY file.
    """
    """读取当前加载的 SEGY 文件并返回其结构摘要，
    包括道数、采样率和采样点数。
    当用户询问 SEGY 文件的“结构”、“基本信息”或“概览”时使用此工具。
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded. Please ask the user to upload a file first."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    info = handler.get_basic_info()
    return json.dumps(info, indent=2)

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
def read_trace_sample(trace_index: Union[int, str] = 0):
    """
    Reads data from a specific trace index.
    Args:
        trace_index: The 0-based index of the trace to read. Defaults to 0.
    Use this tool to inspect actual seismic data values.
    """
    """读取特定地震道索引的数据。
    参数:
        trace_index: 要读取的地震道的 0 基索引。默认为 0。
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
