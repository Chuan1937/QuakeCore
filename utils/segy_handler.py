import segyio
import numpy as np
import pandas as pd
import  h5py
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

    def to_numpy(self, start_trace: int = 0, count: int | None = None, output_path: str | None = None):
        """将SEGY数据转换为 NumPy 数组，可选起始道和数量。"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                total = f.tracecount
                if start_trace >= total:
                    return {"error": "Start trace out of bounds"}

                end = total if count is None else min(start_trace + count, total)
                data = segyio.tools.collect(f.trace[start_trace:end])

                if output_path:
                    np.save(output_path, data)

                return {
                    "array": data,
                    "start_trace": start_trace,
                    "count": end - start_trace
                }
        except Exception as e:
            return {"error": str(e)}

    def to_excel(self, output_path: str, start_trace: int = 0, count: int | None = None):
        """将指定范围的地震道导出为 Excel。count=None 表示从 start_trace 导出到文件末尾。"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                total = f.tracecount
                if start_trace >= total:
                    return {"error": "Start trace out of bounds"}

                end = total if count is None else min(start_trace + count, total)
                data = segyio.tools.collect(f.trace[start_trace:end])

                df = pd.DataFrame(data.T)
                df.columns = [f"Trace_{i}" for i in range(start_trace, end)]
                df.to_excel(output_path, index_label="Sample_Index")

                return f"Saved traces {start_trace}-{end - 1} to {output_path}"
        except Exception as e:
            return {"error": str(e)}

    def to_hdf5(self, output_path: str, start_trace: int = 0, count: int | None = None, compression: str | None = "gzip"):
        """将SEGY数据写入HDF5，支持分块写入与压缩。"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                total = f.tracecount
                if start_trace >= total:
                    return {"error": "Start trace out of bounds"}

                end = total if count is None else min(start_trace + count, total)
                data = segyio.tools.collect(f.trace[start_trace:end])

                with h5py.File(output_path, "w") as h5f:
                    h5f.create_dataset(
                        "traces",
                        data=data,
                        compression=compression,
                        compression_opts=4 if compression == "gzip" else None
                    )
                    h5f.attrs["filepath"] = self.filepath
                    h5f.attrs["start_trace"] = start_trace
                    h5f.attrs["trace_count"] = end - start_trace
                    h5f.attrs["sample_rate_ms"] = float(segyio.tools.dt(f)) / 1000.0

                return f"Saved traces {start_trace}-{end - 1} to {output_path}"
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

    def get_trace_record_data(self, trace_index: int = 0):
        """获取特定道的完整数据用于绘图"""
        try:
            with segyio.open(self.filepath, ignore_geometry=True) as f:
                if trace_index >= f.tracecount:
                    return {"error": "Trace index out of bounds"}
                
                data = f.trace[trace_index]
                dt_micro = segyio.tools.dt(f)
                sampling_rate = 1.0 / (dt_micro / 1_000_000.0) if dt_micro else 100.0
                
                return {
                    "data": data,
                    "sampling_rate": sampling_rate,
                    "start_time": None
                }
        except Exception as e:
            return {"error": str(e)}

