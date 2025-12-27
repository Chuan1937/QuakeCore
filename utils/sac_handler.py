import os
import numpy as np
import pandas as pd
import h5py
from obspy import read as obspy_read


class SACHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._validate_file()

    def _validate_file(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def _load_stream(self):
        return obspy_read(self.filepath)

    def _to_array(self, data):
        if isinstance(data, np.ndarray):
            return data.astype(np.float64)
        return np.array(data, dtype=np.float64)

    def get_basic_info(self):
        try:
            stream = self._load_stream()
            trace_count = len(stream)
            ids = [tr.id for tr in stream]
            sampling_rates = [float(tr.stats.sampling_rate) for tr in stream]
            start_times = [tr.stats.starttime.isoformat() for tr in stream]
            end_times = [tr.stats.endtime.isoformat() for tr in stream]
            npts = [int(tr.stats.npts) for tr in stream]
            same_sr = len(set(sampling_rates)) == 1
            info = {
                "filename": os.path.basename(self.filepath),
                "trace_count": trace_count,
                "sampling_rate": sampling_rates[0] if same_sr else sampling_rates,
                "start_time": start_times[0] if trace_count == 1 else start_times,
                "end_time": end_times[0] if trace_count == 1 else end_times,
                "samples_per_trace": npts[0] if trace_count == 1 else npts,
                "ids": ids,
            }
            sac_headers = []
            for tr in stream:
                sac_hdr = getattr(tr.stats, "sac", None)
                if sac_hdr:
                    sac_headers.append({k: sac_hdr.get(k) for k in ["kstnm", "kcmpnm", "kevnm", "stla", "stlo", "stel", "stdp", "evla", "evlo", "evel", "evdp"] if k in sac_hdr})
                else:
                    sac_headers.append({})
            info["sac_headers"] = sac_headers if trace_count > 1 else sac_headers[0]
            first = self._to_array(stream[0].data)
            info["first_trace_first_10"] = first[:10].tolist()
            return info
        except Exception as exc:
            return {"error": str(exc)}

    def get_trace_data(self, trace_index: int = 0):
        try:
            stream = self._load_stream()
            if trace_index < 0 or trace_index >= len(stream):
                return {"error": "Trace index out of bounds"}
            tr = stream[trace_index]
            data = self._to_array(tr.data)
            summary = {
                "trace_index": trace_index,
                "id": tr.id,
                "start_time": tr.stats.starttime.isoformat(),
                "end_time": tr.stats.endtime.isoformat(),
                "sampling_rate": float(tr.stats.sampling_rate),
                "npts": int(tr.stats.npts),
                "min_value": float(np.min(data)),
                "max_value": float(np.max(data)),
                "mean_value": float(np.mean(data)),
                "first_10_samples": data[:10].tolist(),
            }
            sac_hdr = getattr(tr.stats, "sac", None)
            if sac_hdr:
                summary["sac_meta"] = {k: sac_hdr.get(k) for k in ["kstnm", "kcmpnm", "kevnm", "stla", "stlo", "stel", "stdp", "evla", "evlo", "evel", "evdp"] if k in sac_hdr}
            return summary
        except Exception as exc:
            return {"error": str(exc)}

    def to_numpy(self, output_path: str | None = None):
        try:
            stream = self._load_stream()
            arrays = [self._to_array(tr.data) for tr in stream]
            same_len = len({arr.shape[0] for arr in arrays}) == 1
            if same_len:
                stacked = np.stack(arrays, axis=0)
                result = {
                    "array_shape": list(stacked.shape),
                    "trace_count": stacked.shape[0],
                    "samples_per_trace": stacked.shape[1],
                }
                if output_path:
                    np.save(output_path, stacked)
                    result["output_path"] = output_path
                return result
            final_path = output_path
            if final_path:
                if not final_path.endswith(".npz"):
                    base, _ = os.path.splitext(final_path)
                    final_path = base + ".npz"
                np.savez(final_path, **{f"trace_{i}": arr for i, arr in enumerate(arrays)})
            result = {
                "array_shape": "ragged",
                "trace_count": len(arrays),
                "samples_per_trace": [int(arr.shape[0]) for arr in arrays],
            }
            if final_path:
                result["output_path"] = final_path
            return result
        except Exception as exc:
            return {"error": str(exc)}

    def to_excel(self, output_path: str):
        try:
            stream = self._load_stream()
            arrays = [self._to_array(tr.data) for tr in stream]
            if not arrays:
                return {"error": "No traces available"}
            max_len = max(arr.shape[0] for arr in arrays)
            matrix = np.full((max_len, len(arrays)), np.nan, dtype=np.float64)
            for idx, arr in enumerate(arrays):
                matrix[: arr.shape[0], idx] = arr
            df = pd.DataFrame(matrix, columns=[f"Trace_{i}" for i in range(len(arrays))])
            df.index.name = "Sample_Index"
            df.to_excel(output_path)
            return {
                "saved": True,
                "trace_count": len(arrays),
                "max_samples": int(max_len),
                "output_path": output_path,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def to_hdf5(self, output_path: str, compression: str | None = "gzip"):
        try:
            stream = self._load_stream()
            arrays = [self._to_array(tr.data) for tr in stream]
            ids = [tr.id for tr in stream]
            srs = [float(tr.stats.sampling_rate) for tr in stream]
            starts = [tr.stats.starttime.isoformat() for tr in stream]
            ends = [tr.stats.endtime.isoformat() for tr in stream]
            vlen = h5py.vlen_dtype(np.float64)
            with h5py.File(output_path, "w") as h5f:
                dset = h5f.create_dataset("traces", (len(arrays),), dtype=vlen, compression=compression or None)
                for idx, arr in enumerate(arrays):
                    dset[idx] = arr
                h5f.create_dataset("ids", data=np.array(ids, dtype="S"))
                h5f.create_dataset("sampling_rate", data=np.array(srs, dtype=np.float64))
                h5f.create_dataset("start_time", data=np.array(starts, dtype="S"))
                h5f.create_dataset("end_time", data=np.array(ends, dtype="S"))
                h5f.attrs["filepath"] = self.filepath
                h5f.attrs["trace_count"] = len(arrays)
            return {
                "saved": True,
                "trace_count": len(arrays),
                "output_path": output_path,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def to_miniseed(self, output_path: str):
        try:
            stream = self._load_stream()
            stream.write(output_path, format="MSEED")
            return {
                "saved": True,
                "output_path": output_path,
                "trace_count": len(stream),
            }
        except Exception as exc:
            return {"error": str(exc)}
