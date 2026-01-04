import os
import numpy as np
from obspy import read as obspy_read
import h5py


class MiniSEEDHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._validate_file()

    def _validate_file(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def get_basic_info(self):
        try:
            stream = obspy_read(self.filepath)
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
                "channels": ids,
                "samples_per_trace": npts if trace_count > 1 else npts[0],
            }
            return info
        except Exception as e:
            return {"error": str(e)}

    def get_trace_data(self, trace_index: int = 0):
        try:
            stream = obspy_read(self.filepath)
            if trace_index < 0 or trace_index >= len(stream):
                return {"error": "Trace index out of bounds"}

            tr = stream[trace_index]
            data = tr.data
            if isinstance(data, np.ndarray):
                vals = data.astype(np.float64)
            else:
                vals = np.array(data, dtype=np.float64)

            summary = {
                "trace_index": trace_index,
                "id": tr.id,
                "network": getattr(tr.stats, "network", None),
                "station": getattr(tr.stats, "station", None),
                "location": getattr(tr.stats, "location", None),
                "channel": getattr(tr.stats, "channel", None),
                "start_time": tr.stats.starttime.isoformat(),
                "end_time": tr.stats.endtime.isoformat(),
                "sampling_rate": float(tr.stats.sampling_rate),
                "npts": int(tr.stats.npts),
                "min_value": float(np.min(vals)),
                "max_value": float(np.max(vals)),
                "mean_value": float(np.mean(vals)),
                "first_10_samples": vals[:10].tolist(),
            }
            return summary
        except Exception as e:
            return {"error": str(e)}

    def get_trace_record_data(self, trace_index: int = 0):
        """获取特定道的完整数据用于绘图"""
        try:
            stream = obspy_read(self.filepath)
            if trace_index < 0 or trace_index >= len(stream):
                return {"error": "Trace index out of bounds"}

            tr = stream[trace_index]
            data = tr.data
            if isinstance(data, np.ndarray):
                vals = data.astype(np.float64)
            else:
                vals = np.array(data, dtype=np.float64)

            return {
                "data": vals,
                "sampling_rate": float(tr.stats.sampling_rate),
                "start_time": tr.stats.starttime
            }
        except Exception as e:
            return {"error": str(e)}

    def to_numpy(self, output_path: str | None = None):
        try:
            stream = obspy_read(self.filepath)
            arrays = []
            for tr in stream:
                data = tr.data
                if isinstance(data, np.ndarray):
                    arrays.append(data.astype(np.float64))
                else:
                    arrays.append(np.array(data, dtype=np.float64))
            same_len = len({arr.shape[0] for arr in arrays}) == 1
            if same_len:
                stacked = np.stack(arrays, axis=0)
                if output_path:
                    np.save(output_path, stacked)
                return {
                    "array_shape": list(stacked.shape),
                    "trace_count": stacked.shape[0],
                    "samples_per_trace": stacked.shape[1],
                }
            else:
                if output_path:
                    if not output_path.endswith(".npz"):
                        base, _ = os.path.splitext(output_path)
                        output_path = base + ".npz"
                    np.savez(output_path, **{f"trace_{i}": arr for i, arr in enumerate(arrays)})
                return {
                    "array_shape": "ragged",
                    "trace_count": len(arrays),
                    "samples_per_trace": [int(a.shape[0]) for a in arrays],
                }
        except Exception as e:
            return {"error": str(e)}

    def to_hdf5(self, output_path: str, compression: str | None = "gzip"):
        try:
            stream = obspy_read(self.filepath)
            arrays = []
            ids = []
            srs = []
            starts = []
            ends = []
            for tr in stream:
                data = tr.data
                vals = data.astype(np.float64) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float64)
                arrays.append(vals)
                ids.append(tr.id)
                srs.append(float(tr.stats.sampling_rate))
                starts.append(tr.stats.starttime.isoformat())
                ends.append(tr.stats.endtime.isoformat())
            vlen = h5py.vlen_dtype(np.float64)
            with h5py.File(output_path, "w") as h5f:
                dset = h5f.create_dataset("traces", (len(arrays),), dtype=vlen, compression=compression if compression else None)
                for i, arr in enumerate(arrays):
                    dset[i] = arr
                h5f.create_dataset("ids", data=np.array(ids, dtype="S"))
                h5f.create_dataset("sampling_rate", data=np.array(srs, dtype=np.float64))
                h5f.create_dataset("start_time", data=np.array(starts, dtype="S"))
                h5f.create_dataset("end_time", data=np.array(ends, dtype="S"))
                h5f.attrs["filepath"] = self.filepath
                h5f.attrs["trace_count"] = len(arrays)
            return {
                "saved": True,
                "trace_count": len(arrays),
            }
        except Exception as e:
            return {"error": str(e)}

    def to_sac(self, output_dir: str):
        try:
            stream = obspy_read(self.filepath)
            os.makedirs(output_dir, exist_ok=True)
            count = 0
            for tr in stream:
                fname = f"{tr.id}.{tr.stats.starttime.strftime('%Y%m%dT%H%M%S')}.sac"
                path = os.path.join(output_dir, fname)
                tr.write(path, format="SAC")
                count += 1
            return {
                "saved": True,
                "output_dir": output_dir,
                "file_count": count,
            }
        except Exception as e:
            return {"error": str(e)}
