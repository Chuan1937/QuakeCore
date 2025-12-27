import os
import numpy as np
import pandas as pd
import h5py


class HDF5Handler:
    _PREFERRED_DATASETS = ("traces", "data", "dataset", "waveforms", "waveform", "values")

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._validate_file()

    def _validate_file(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def _is_single_vector(self, dataset: h5py.Dataset) -> bool:
        return dataset.ndim == 1 and dataset.dtype != np.object_ and dataset.shape

    def _coerce_attr_value(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", "ignore")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _collect_datasets(self, h5f):
        datasets = []
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                path = name if name.startswith("/") else f"/{name}"
                datasets.append((node, path))
        h5f.visititems(visitor)
        return datasets

    def _select_dataset(self, h5f, dataset_name: str | None):
        def _candidate_names(name: str):
            norm = str(name).strip()
            if not norm:
                return []
            candidates = [norm]
            if norm.startswith("/"):
                candidates.append(norm.lstrip("/"))
            else:
                candidates.append(f"/{norm}")
            return [c for c in candidates if c]

        if dataset_name:
            for candidate in _candidate_names(dataset_name):
                node = h5f.get(candidate)
                if node is not None and isinstance(node, h5py.Dataset):
                    return node, candidate
            raise ValueError(f"Dataset {dataset_name} not found")

        for preferred in self._PREFERRED_DATASETS:
            node = h5f.get(preferred)
            if node is not None and isinstance(node, h5py.Dataset):
                return node, preferred

        datasets = self._collect_datasets(h5f)
        if datasets:
            return datasets[0]
        raise ValueError("No datasets found in HDF5 file")

    def _normalize_traces(self, data_block):
        arrays = []
        if isinstance(data_block, np.ndarray):
            if data_block.dtype == np.object_:
                for item in data_block:
                    arrays.append(np.asarray(item, dtype=np.float64))
            else:
                arr = np.asarray(data_block, dtype=np.float64)
                if arr.ndim == 1:
                    arrays.append(arr)
                else:
                    for row in arr:
                        arrays.append(row)
        else:
            for item in data_block:
                arrays.append(np.asarray(item, dtype=np.float64))
        return arrays

    def _extract_traces(self, dataset, start_trace: int, count: int | None):
        if dataset.size == 0:
            raise ValueError("Dataset is empty")
        if self._is_single_vector(dataset):
            if start_trace not in (0, None):
                raise ValueError("Single-trace dataset only supports start_trace=0")
            if count not in (None, 1):
                raise ValueError("Single-trace dataset only supports count<=1")
            trace = np.asarray(dataset[:], dtype=np.float64)
            return [trace], 0, 1, 1
        total_traces = int(dataset.shape[0])
        if start_trace < 0 or start_trace >= total_traces:
            raise ValueError("Trace index out of bounds")
        end = total_traces if count is None else min(start_trace + count, total_traces)
        if end <= start_trace:
            raise ValueError("Invalid count/start_trace combination")
        block = dataset[start_trace:end]
        arrays = self._normalize_traces(block)
        return arrays, start_trace, end - start_trace, total_traces

    def get_basic_info(self, dataset: str | None = None):
        try:
            with h5py.File(self.filepath, "r") as h5f:
                dset, dset_name = self._select_dataset(h5f, dataset)
                info = {
                    "filename": os.path.basename(self.filepath),
                    "dataset": dset_name,
                    "shape": list(dset.shape),
                    "dtype": str(dset.dtype),
                    "file_attrs": {k: self._coerce_attr_value(v) for k, v in h5f.attrs.items()},
                    "dataset_attrs": {k: self._coerce_attr_value(v) for k, v in dset.attrs.items()},
                }
                if self._is_single_vector(dset):
                    info["trace_count"] = 1
                    info["samples_per_trace"] = int(dset.shape[0])
                else:
                    trace_count = int(dset.shape[0]) if dset.shape else 0
                    info["trace_count"] = trace_count
                    if dset.ndim >= 2:
                        info["samples_per_trace"] = int(dset.shape[1])
                    elif dset.dtype == np.object_ and trace_count:
                        lengths = []
                        sample = min(trace_count, 5)
                        for idx in range(sample):
                            lengths.append(len(np.asarray(dset[idx])))
                        info["samples_per_trace"] = lengths[0] if len(set(lengths)) == 1 else lengths
                if info.get("trace_count", 0) > 0:
                    first = np.asarray(dset[0]) if not self._is_single_vector(dset) else np.asarray(dset[:])
                    first = np.asarray(first, dtype=np.float64)
                    info["first_trace_first_10"] = first[:10].tolist()
                return info
        except Exception as exc:
            return {"error": str(exc)}

    def get_trace_data(self, trace_index: int = 0, dataset: str | None = None):
        try:
            with h5py.File(self.filepath, "r") as h5f:
                dset, dset_name = self._select_dataset(h5f, dataset)
                if self._is_single_vector(dset):
                    if trace_index not in (0, None):
                        return {"error": "Single-trace dataset only has trace_index 0"}
                    trace = np.asarray(dset[:], dtype=np.float64)
                    total = 1
                else:
                    total = int(dset.shape[0]) if dset.shape else 0
                    if trace_index < 0 or trace_index >= total:
                        return {"error": "Trace index out of bounds"}
                    raw = dset[trace_index]
                    trace = np.asarray(raw, dtype=np.float64)
                summary = {
                    "dataset": dset_name,
                    "trace_index": int(trace_index),
                    "trace_count": total,
                    "min_value": float(np.min(trace)) if trace.size else None,
                    "max_value": float(np.max(trace)) if trace.size else None,
                    "mean_value": float(np.mean(trace)) if trace.size else None,
                    "total_samples": int(trace.shape[0]),
                    "first_10_samples": trace[:10].tolist(),
                }
                return summary
        except Exception as exc:
            return {"error": str(exc)}

    def to_numpy(self, output_path: str, dataset: str | None = None):
        try:
            with h5py.File(self.filepath, "r") as h5f:
                dset, _ = self._select_dataset(h5f, dataset)
                traces, _, count, _ = self._extract_traces(dset, 0, None)
                if not traces:
                    return {"error": "Dataset is empty"}
                same_len = len({tr.shape[0] for tr in traces}) == 1
                if same_len:
                    stacked = np.stack(traces, axis=0)
                    np.save(output_path, stacked)
                    return {
                        "array_shape": list(stacked.shape),
                        "trace_count": stacked.shape[0],
                        "samples_per_trace": stacked.shape[1],
                        "output_path": output_path,
                    }
                if not output_path.endswith(".npz"):
                    base, _ = os.path.splitext(output_path)
                    output_path = base + ".npz"
                np.savez(output_path, **{f"trace_{i}": tr for i, tr in enumerate(traces)})
                return {
                    "array_shape": "ragged",
                    "trace_count": count,
                    "samples_per_trace": [int(tr.shape[0]) for tr in traces],
                    "output_path": output_path,
                }
        except Exception as exc:
            return {"error": str(exc)}

    def to_excel(self, output_path: str, start_trace: int = 0, count: int | None = None, dataset: str | None = None):
        try:
            with h5py.File(self.filepath, "r") as h5f:
                dset, dset_name = self._select_dataset(h5f, dataset)
                traces, actual_start, actual_count, _ = self._extract_traces(dset, start_trace, count)
                if not traces:
                    return {"error": "No data to export"}
                max_len = max(tr.shape[0] for tr in traces)
                matrix = np.full((max_len, len(traces)), np.nan, dtype=np.float64)
                for idx, tr in enumerate(traces):
                    matrix[: tr.shape[0], idx] = tr
                df = pd.DataFrame(matrix, columns=[f"Trace_{actual_start + i}" for i in range(len(traces))])
                df.index.name = "Sample_Index"
                df.to_excel(output_path)
                return {
                    "saved": True,
                    "dataset": dset_name,
                    "start_trace": actual_start,
                    "count": actual_count,
                    "max_samples": int(max_len),
                    "output_path": output_path,
                }
        except Exception as exc:
            return {"error": str(exc)}

    def list_keys(self):
        try:
            entries = []
            groups = []
            def visitor(name, node):
                if isinstance(node, h5py.Dataset):
                    path = name if name.startswith("/") else f"/{name}"
                    entries.append({
                        "path": path,
                        "shape": list(node.shape),
                        "dtype": str(node.dtype),
                    })
                elif isinstance(node, h5py.Group) and name:
                    path = name if name.startswith("/") else f"/{name}"
                    groups.append(path)
            with h5py.File(self.filepath, "r") as h5f:
                h5f.visititems(visitor)
                info = {
                    "filename": os.path.basename(self.filepath),
                    "datasets": entries,
                    "groups": sorted(groups),
                    "file_attrs": {k: self._coerce_attr_value(v) for k, v in h5f.attrs.items()},
                }
                return info
        except Exception as exc:
            return {"error": str(exc)}
