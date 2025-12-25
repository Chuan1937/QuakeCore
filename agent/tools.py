from langchain.tools import tool
from utils.segy_handler import SegyHandler
import json
from typing import Union
import numpy as np

# Global variable to store the current file path being analyzed
# In a multi-user web app, this should be handled via session state or context
CURRENT_SEGY_PATH = None

def set_current_segy_path(path):
    global CURRENT_SEGY_PATH
    CURRENT_SEGY_PATH = path

@tool
def get_segy_structure():
    """
    Reads the currently loaded SEGY file and returns a summary of its structure,
    including trace count, sample rate, and sample count.
    Use this tool when the user asks about the 'structure', 'basic info', or 'overview' of the SEGY file.
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded. Please ask the user to upload a file first."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    info = handler.get_basic_info()
    return json.dumps(info, indent=2)

@tool
def get_segy_text_header():
    """
    Reads the EBCDIC text header of the SEGY file.
    Use this tool when the user asks to see the 'text header', 'EBCDIC header', or 'notes' in the file header.
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    header = handler.get_text_header()
    return header

@tool
def get_segy_binary_header():
    """
    Reads the Binary header of the SEGY file.
    Use this tool when the user asks for 'binary header' information like format code, job id, etc.
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    handler = SegyHandler(CURRENT_SEGY_PATH)
    header = handler.get_binary_header()
    return json.dumps(header, indent=2)

@tool
def read_trace_sample(trace_index: Union[int, str] = 0):
    """
    Reads data from a specific trace index.
    Args:
        trace_index: The 0-based index of the trace to read. Defaults to 0.
    Use this tool to inspect actual seismic data values.
    """
    global CURRENT_SEGY_PATH
    if not CURRENT_SEGY_PATH:
        return "No SEGY file is currently loaded."
    
    # Handle potential JSON string input from ReAct agents
    if isinstance(trace_index, str):
        raw_value = trace_index.strip()
        if raw_value.lower().startswith("trace_index"):
            parts = raw_value.split("=", 1)
            if len(parts) == 2:
                raw_value = parts[1].strip()
        try:
            # Try to parse as integer first
            trace_index = int(raw_value)
        except ValueError:
            try:
                # Try to parse as JSON or plain integer string
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
    data = handler.get_trace_data(trace_index=trace_index, count=1)
    
    # Summarize data to avoid context overflow
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

import numpy as np
