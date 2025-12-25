import segyio
import numpy as np
import pandas as pd
import os

class SegyHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self._validate_file()

    def _validate_file(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def get_basic_info(self):
        """获取SEGY文件的基本信息"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                info = {
                    "filename": os.path.basename(self.filepath),
                    "trace_count": int(f.tracecount),
                    "sample_rate": float(segyio.tools.dt(f)) / 1000.0, # ms
                    "samples_per_trace": len(f.samples),
                    "format": int(f.format),
                    "sorting": int(f.sorting) if f.sorting is not None else None
                }
                return info
        except Exception as e:
            return {"error": str(e)}

    def get_text_header(self):
        """读取EBCDIC文本头"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                # segyio automatically handles EBCDIC conversion usually, 
                # but accessing raw text header is useful
                text_header = segyio.tools.wrap(f.text[0])
                return text_header
        except Exception as e:
            return f"Error reading text header: {str(e)}"

    def get_binary_header(self):
        """读取二进制头关键信息"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                bin_header = f.bin
                # Convert dictionary-like object to dict for easier display
                # Note: segyio bin header access might vary, using standard keys often helps
                # Here we just return a summary of common keys if possible, or the raw dict
                common_keys = {
                    segyio.BinField.JobID: "Job ID",
                    segyio.BinField.LineNumber: "Line Number",
                    segyio.BinField.ReelNumber: "Reel Number",
                    segyio.BinField.Traces: "Traces per Ensemble",
                    segyio.BinField.AuxTraces: "Aux Traces",
                    segyio.BinField.Interval: "Sample Interval",
                    segyio.BinField.Samples: "Samples per Trace",
                    segyio.BinField.Format: "Data Format"
                }
                
                result = {}
                for key, name in common_keys.items():
                    if key in bin_header:
                        val = bin_header[key]
                        # Convert numpy types to native python types
                        if hasattr(val, 'item'):
                            val = val.item()
                        result[name] = val
                return result
        except Exception as e:
            return {"error": str(e)}

    def get_trace_data(self, trace_index=0, count=1):
        """获取特定道的数据"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                if trace_index >= f.tracecount:
                    return {"error": "Trace index out of bounds"}
                
                end_index = min(trace_index + count, f.tracecount)
                # f.trace slicing returns a generator, convert to list
                traces = list(f.trace[trace_index:end_index])
                
                # Convert numpy arrays to lists for JSON serialization
                traces_list = [t.tolist() for t in traces]
                
                return {
                    "start_trace": trace_index,
                    "count": len(traces),
                    "data": traces_list
                }
        except Exception as e:
            return {"error": str(e)}
