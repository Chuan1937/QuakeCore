import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime, read as obspy_read
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset

try:  # envelope is located in different submodules across ObsPy versions
    from obspy.signal.filter import envelope as _obspy_envelope
except ImportError:  # pragma: no cover - fallback for older releases
    try:
        from obspy.signal.util import envelope as _obspy_envelope
    except ImportError:
        _obspy_envelope = None


def _fallback_envelope(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr
    spectrum = np.fft.fft(arr)
    h = np.zeros(n)
    if n > 0:
        h[0] = 1
        if n % 2 == 0:
            h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    return np.abs(analytic)


def envelope(data: np.ndarray) -> np.ndarray:
    if _obspy_envelope is not None:
        return _obspy_envelope(data)
    return _fallback_envelope(data)

try:  # SEG-Y support is optional
    import segyio  # type: ignore
except Exception:  # pragma: no cover - segyio might be unavailable at runtime
    segyio = None


@dataclass
class TraceRecord:
    """Container holding a single waveform and the metadata required for picking."""

    data: np.ndarray
    sampling_rate: float
    start_time: Optional[UTCDateTime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.data.ndim != 1:
            self.data = np.asarray(self.data).reshape(-1)
        if self.metadata is None:
            self.metadata = {}
@dataclass
class PickResult:
    """Standardized pick output."""

    trace_index: int
    method: str
    sample_index: int
    time_offset_s: Optional[float]
    absolute_time: Optional[str]
    raw_score: Optional[float]
    normalized_score: Optional[float]
    phase_type: str = "P"  # "P" or "S"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _safe_utc(value: Any) -> Optional[UTCDateTime]:
    if value is None:
        return None
    if isinstance(value, UTCDateTime):
        return value
    try:
        return UTCDateTime(value)
    except Exception:
        return None


def _ensure_sampling_rate(provided: Optional[float]) -> float:
    if provided is None or provided <= 0:
        raise ValueError("sampling_rate is required for this data source.")
    return float(provided)


def _standardize_signal(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).copy()
    if data.size == 0:
        return data
    data -= np.mean(data)
    std = np.std(data)
    if std > 0:
        data /= std
    return data


_PREFERRED_HDF5_DATASETS = ("traces", "data", "dataset", "waveforms", "waveform", "values")


def _resolve_hdf5_dataset(h5f: h5py.File, dataset: Optional[str]):
    def _candidate_names(name: str):
        norm = str(name).strip()
        if not norm or norm == "/":
            return []
        candidates = [norm]
        if norm.startswith("/"):
            candidates.append(norm.lstrip("/"))
        else:
            candidates.append(f"/{norm}")
        return [c for c in candidates if c]

    if dataset:
        for candidate in _candidate_names(dataset):
            node = h5f.get(candidate)
            if node is not None and isinstance(node, h5py.Dataset):
                return node, candidate
        raise ValueError(f"Dataset '{dataset}' not found in {h5f.filename}")

    for preferred in _PREFERRED_HDF5_DATASETS:
        node = h5f.get(preferred)
        if node is not None and isinstance(node, h5py.Dataset):
            return node, preferred

    chosen: Optional[Tuple[h5py.Dataset, str]] = None

    def visitor(name, node):
        nonlocal chosen
        if chosen is None and isinstance(node, h5py.Dataset):
            path = name if name.startswith("/") else f"/{name}"
            chosen = (node, path)

    h5f.visititems(visitor)
    if chosen:
        return chosen
    raise ValueError(f"No dataset found in HDF5 file {h5f.filename}")


def _collect_hdf5_traces(
    path: str,
    dataset: Optional[str],
    sampling_rate: Optional[float],
) -> List[TraceRecord]:
    traces: List[TraceRecord] = []
    with h5py.File(path, "r") as h5f:
        dset, resolved_name = _resolve_hdf5_dataset(h5f, dataset)
        dataset_label = dset.name or resolved_name
        sr_attr = dset.attrs.get("sampling_rate") if hasattr(dset, "attrs") else None
        sr_global = h5f.attrs.get("sampling_rate") if hasattr(h5f, "attrs") else None
        sr = sr_attr or sr_global or sampling_rate
        sr = _ensure_sampling_rate(sr)

        start_source = dset.attrs.get("start_time") if hasattr(dset, "attrs") else None
        if start_source is None:
            candidate_paths = ["start_time", "/start_time"]
            parent = os.path.dirname(dataset_label)
            if parent and parent not in ("", "/"):
                stripped = parent.lstrip("/")
                candidate_paths.extend(
                    [
                        f"{parent}/start_time",
                        f"{stripped}/start_time",
                        f"/{stripped}/start_time",
                    ]
                )
            for candidate in candidate_paths:
                node = h5f.get(candidate)
                if isinstance(node, h5py.Dataset):
                    start_source = node[()]
                    break

        def _decode_start_values(values):
            if values is None:
                return None
            if isinstance(values, np.ndarray):
                if values.dtype.kind in {"S", "U", "O"}:
                    return [
                        value.decode("utf-8", "ignore") if isinstance(value, (bytes, bytearray)) else str(value)
                        for value in values
                    ]
                if values.ndim == 0:
                    return _decode_start_values(values.item())
                return None
            if isinstance(values, (list, tuple)):
                return [
                    value.decode("utf-8", "ignore") if isinstance(value, (bytes, bytearray)) else str(value)
                    for value in values
                ]
            if isinstance(values, (bytes, bytearray)):
                return [values.decode("utf-8", "ignore")]
            if isinstance(values, UTCDateTime):
                return [values.isoformat()]
            if isinstance(values, str):
                return [values]
            return None

        decoded = _decode_start_values(start_source)

        def _start_time_for(idx: int):
            if not decoded or idx >= len(decoded):
                return None
            return _safe_utc(decoded[idx])

        ragged = dset.dtype == np.object_
        if ragged:
            for idx in range(len(dset)):
                data = np.asarray(dset[idx], dtype=np.float64)
                traces.append(
                    TraceRecord(
                        data=data,
                        sampling_rate=sr,
                        start_time=_start_time_for(idx),
                        metadata={"dataset": dataset_label, "index": idx},
                    )
                )
        else:
            array = np.asarray(dset, dtype=np.float64)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            for idx, row in enumerate(array):
                traces.append(
                    TraceRecord(
                        data=row,
                        sampling_rate=sr,
                        start_time=_start_time_for(idx),
                        metadata={"dataset": dataset_label, "index": idx},
                    )
                )
    return traces
def _collect_numpy_traces(
    source: Any,
    sampling_rate: Optional[float],
) -> List[TraceRecord]:
    sr = _ensure_sampling_rate(sampling_rate)
    traces: List[TraceRecord] = []
    if isinstance(source, np.ndarray):
        data = source
    else:
        data = np.load(source, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in data.files:
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                traces.append(TraceRecord(data=row, sampling_rate=sr, metadata={"source_key": key}))
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for idx, row in enumerate(arr):
            traces.append(TraceRecord(data=row, sampling_rate=sr, metadata={"row": idx}))
    return traces


def _collect_obspy_traces(path: str) -> List[TraceRecord]:
    stream = obspy_read(path)
    traces: List[TraceRecord] = []
    for tr in stream:
        traces.append(
            TraceRecord(
                data=np.asarray(tr.data, dtype=np.float64),
                sampling_rate=float(tr.stats.sampling_rate),
                start_time=_safe_utc(tr.stats.starttime),
                metadata={
                    "id": tr.id,
                    "network": getattr(tr.stats, "network", None),
                    "station": getattr(tr.stats, "station", None),
                    "channel": getattr(tr.stats, "channel", None),
                },
            )
        )
    return traces


def _collect_segy_traces(path: str) -> List[TraceRecord]:
    if segyio is None:
        raise ImportError("segyio is required for SEG-Y support but is not installed.")
    traces: List[TraceRecord] = []
    with segyio.open(path, ignore_geometry=True) as f:
        data = segyio.tools.collect(f.trace[:])
        dt_micro = segyio.tools.dt(f)
        if not dt_micro:
            raise ValueError("SEG-Y file is missing sampling interval metadata.")
        sampling_interval_s = dt_micro / 1_000_000.0
        sampling_rate = 1.0 / sampling_interval_s
        for idx, row in enumerate(data):
            traces.append(
                TraceRecord(
                    data=np.asarray(row, dtype=np.float64),
                    sampling_rate=sampling_rate,
                    metadata={"trace": idx, "filepath": path},
                )
            )
    return traces


def load_traces(
    source: Any,
    *,
    file_type: Optional[str] = None,
    dataset: Optional[str] = None,
    sampling_rate: Optional[float] = None,
) -> List[TraceRecord]:
    """Normalize arbitrary waveform sources into TraceRecord objects."""

    if isinstance(source, np.ndarray):
        return _collect_numpy_traces(source, sampling_rate)
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], np.ndarray):
        sr = _ensure_sampling_rate(sampling_rate)
        return [TraceRecord(data=np.asarray(arr, dtype=np.float64), sampling_rate=sr) for arr in source]
    if not isinstance(source, str):
        raise ValueError("Unsupported source type. Provide a file path or numpy array.")

    path = source
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext_map = {
        ".mseed": "mseed",
        ".miniseed": "mseed",
        ".sac": "sac",
        ".sgy": "segy",
        ".segy": "segy",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".npy": "npy",
        ".npz": "npz",
    }
    ext = os.path.splitext(path)[1].lower()
    fmt = file_type or ext_map.get(ext)

    if fmt in {"mseed", "miniseed", "sac"}:
        return _collect_obspy_traces(path)
    if fmt == "segy":
        return _collect_segy_traces(path)
    if fmt == "hdf5":
        return _collect_hdf5_traces(path, dataset, sampling_rate)
    if fmt in {"npy", "npz"}:
        return _collect_numpy_traces(path, sampling_rate)

    raise ValueError(f"Unsupported file type '{ext}'. Specify file_type explicitly if needed.")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _burg_prediction_error(signal: np.ndarray, order: int) -> float:
    if order < 1 or len(signal) <= order:
        return float("inf")
    forward = signal.astype(np.float64).copy()
    backward = signal.astype(np.float64).copy()
    error = np.sum(signal ** 2) / len(signal)
    for k in range(order):
        numerator = -2.0 * np.dot(forward[k + 1 :], backward[k:-1])
        denominator = np.sum(forward[k + 1 :] ** 2 + backward[k:-1] ** 2)
        if denominator == 0:
            break
        reflection = numerator / denominator
        new_forward = forward.copy()
        forward[k + 1 :] = forward[k + 1 :] + reflection * backward[k:-1]
        backward[k + 1 :-1] = backward[k + 1 :-1] + reflection * new_forward[k + 1 : -1]
        error *= (1.0 - reflection**2)
    return float(error)


class PhasePickingEngine:
    """Run a suite of classical phase picking algorithms on waveform data."""

    def __init__(self, traces: Sequence[TraceRecord]):
        if not traces:
            raise ValueError("No traces supplied for picking.")
        self.traces = list(traces)
        self._processed_traces = [_standardize_signal(rec.data) for rec in self.traces]

    def run(
        self,
        methods: Optional[Iterable[str]] = None,
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PickResult]:
        available = {
            "sta_lta": self._pick_sta_lta,
            "aic": self._pick_aic,
            "frequency_ratio": self._pick_frequency_ratio,
            "autocorr": self._pick_autocorr_drop,
            "feature_threshold": self._pick_feature_threshold,
            "ar_model": self._pick_ar_model,
            "template_correlation": self._pick_template_correlation,
            "pai_k": self._pick_pai_k,
            "pai_s": self._pick_pai_s,
            "s_phase": self._pick_s_basic,  # New S-wave picker
        }
        selected = list(methods) if methods else list(available.keys())
        params = method_params or {}
        results: List[PickResult] = []
        for idx, record in enumerate(self.traces):
            for name in selected:
                picker = available.get(name)
                if picker is None:
                    continue
                # Some pickers might return a list of picks (e.g. P and S)
                # But current signature is Optional[PickResult].
                # I will modify _pick_s_basic to return a single S-pick.
                # If the user wants both, they select 'sta_lta' (for P) and 's_phase' (for S).
                pick = picker(idx, record, params.get(name, {}))
                if pick:
                    results.append(pick)
        return results

    def _format_result(
        self,
        trace_index: int,
        method: str,
        record: TraceRecord,
        sample_index: int,
        raw_score: Optional[float],
        normalized_score: Optional[float],
        phase_type: str = "P",
        extra: Optional[Dict[str, Any]] = None,
    ) -> PickResult:
        time_offset = sample_index / record.sampling_rate if record.sampling_rate else None
        abs_time = None
        if time_offset is not None and record.start_time is not None:
            abs_time = (record.start_time + time_offset).isoformat()
        payload = {} if extra is None else dict(extra)
        payload.update(record.metadata or {})
        return PickResult(
            trace_index=trace_index,
            method=method,
            sample_index=int(sample_index),
            time_offset_s=time_offset,
            absolute_time=abs_time,
            raw_score=raw_score,
            normalized_score=normalized_score,
            phase_type=phase_type,
            metadata=payload,
        )

    def _signal(self, trace_index: int) -> np.ndarray:
        return self._processed_traces[trace_index]

    def _pick_sta_lta(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        data = self._signal(trace_index)
        short = float(options.get("short_window", 1.0))
        long = float(options.get("long_window", 10.0))
        trig_on = float(options.get("trigger_on", 3.5))
        trig_off = float(options.get("trigger_off", 1.0))
        nsta = max(1, int(short * record.sampling_rate))
        nlta = max(nsta + 1, int(long * record.sampling_rate))
        if nlta >= data.size:
            return None
        cft = classic_sta_lta(data, nsta, nlta)
        on_off = trigger_onset(cft, trig_on, trig_off)
        if on_off.size == 0:
            return None
        start_idx = int(on_off[0][0])
        raw_score = float(cft[start_idx]) if start_idx < cft.size else None
        normalized = None
        if raw_score is not None:
            normalized = max(0.0, min(1.0, (raw_score - trig_on) / max(trig_on * 2.0, 1e-6)))
        return self._format_result(trace_index, "sta_lta", record, start_idx, raw_score, normalized)

    def _pick_aic(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        data = self._signal(trace_index)
        n = data.size
        if n < 4:
            return None
        cumsum = np.cumsum(data)
        cumsum_sq = np.cumsum(data**2)
        idxs = np.arange(1, n - 1)
        var1 = (cumsum_sq[idxs - 1] / idxs) - (cumsum[idxs - 1] / idxs) ** 2
        tail_sum = cumsum_sq[-1] - cumsum_sq[idxs - 1]
        tail_count = n - idxs
        var2 = (tail_sum / tail_count) - ((cumsum[-1] - cumsum[idxs - 1]) / tail_count) ** 2
        var1[var1 <= 0] = np.nan
        var2[var2 <= 0] = np.nan
        aic = idxs * np.log(var1) + (n - idxs - 1) * np.log(var2)
        k = np.nanargmin(aic) if np.isfinite(aic).any() else None
        if k is None:
            return None
        pick = int(idxs[k])
        raw_score = float(aic[k]) if np.isfinite(aic[k]) else None
        normalized = None
        finite = aic[np.isfinite(aic)]
        if raw_score is not None and finite.size > 0:
            span = float(np.nanmax(finite) - np.nanmin(finite))
            if span > 0:
                normalized = max(0.0, min(1.0, (np.nanmax(finite) - raw_score) / span))
        return self._format_result(trace_index, "aic", record, pick, raw_score, normalized)

    def _pick_frequency_ratio(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        data = self._signal(trace_index)
        high_band = options.get("high_band", (5.0, 15.0))
        low_band = options.get("low_band", (0.5, 5.0))
        smooth_window = float(options.get("smooth_window", 0.5))
        df = record.sampling_rate
        nyquist = 0.5 * df
        h_low, h_high = high_band
        l_low, l_high = low_band
        h_high = min(h_high, nyquist - 0.1)
        l_high = min(l_high, nyquist - 0.1)
        if h_low >= h_high or l_low >= l_high:
            return None
        try:
            high = bandpass(data, h_low, h_high, df=df, corners=2, zerophase=True)
            low = bandpass(data, l_low, l_high, df=df, corners=2, zerophase=True)
        except ValueError:
            return None
        window = max(1, int(smooth_window * df))
        hf_energy = _moving_average(high**2, window)
        lf_energy = _moving_average(low**2, window) + 1e-12
        ratio = hf_energy / lf_energy
        gradient = np.gradient(ratio)
        pick = int(np.argmax(gradient))
        raw_score = float(ratio[pick]) if pick < ratio.size else None
        normalized = None
        if raw_score is not None:
            normalized = max(0.0, min(1.0, (raw_score - 1.0) / 4.0))
        return self._format_result(trace_index, "frequency_ratio", record, pick, raw_score, normalized)

    def _pick_autocorr_drop(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        data = self._signal(trace_index)
        window_s = float(options.get("window", 2.0))
        step_s = float(options.get("step", 0.2))
        min_drop = float(options.get("min_drop", 0.2))
        expected_drop = float(options.get("max_drop", 0.8))
        window = max(4, int(window_s * record.sampling_rate))
        step = max(1, int(step_s * record.sampling_rate))
        if data.size <= 2 * window:
            return None
        prev = data[0:window]
        prev_norm = np.linalg.norm(prev) + 1e-12
        prev_corr = 1.0
        best_drop = 0.0
        best_idx: Optional[int] = None
        for start in range(step, data.size - window, step):
            segment = data[start : start + window]
            denom = (np.linalg.norm(segment) * prev_norm) + 1e-12
            corr = float(np.dot(prev, segment) / denom)
            drop = prev_corr - corr
            if drop > min_drop and drop > best_drop:
                best_drop = drop
                best_idx = start
            prev = segment
            prev_norm = np.linalg.norm(prev) + 1e-12
            prev_corr = corr
        if best_idx is None:
            return None
        normalized = max(0.0, min(1.0, best_drop / max(expected_drop, 1e-6)))
        return self._format_result(
            trace_index,
            "autocorr",
            record,
            best_idx,
            best_drop,
            normalized,
            extra={"window": window},
        )

    def _pick_feature_threshold(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        window_s = float(options.get("smooth_window", 0.5))
        sigma = float(options.get("sigma", 3.0))
        data = self._signal(trace_index)
        env = envelope(data)
        smooth = _moving_average(env, max(1, int(window_s * record.sampling_rate)))
        mean = float(np.mean(smooth))
        std = float(np.std(smooth) + 1e-12)
        threshold = mean + sigma * std
        candidates = np.where(smooth >= threshold)[0]
        if candidates.size == 0:
            return None
        pick = int(candidates[0])
        raw_score = float(smooth[pick])
        normalized = max(0.0, min(1.0, (raw_score - threshold) / (abs(threshold) + 1e-9)))
        return self._format_result(
            trace_index,
            "feature_threshold",
            record,
            pick,
            raw_score,
            normalized,
            extra={"threshold": threshold},
        )

    def _pick_ar_model(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        order = int(options.get("order", 4))
        window_s = float(options.get("window", 1.5))
        window = max(order + 1, int(window_s * record.sampling_rate))
        data = self._signal(trace_index)
        if data.size <= 2 * window:
            return None
        scores = []
        idxs = []
        for center in range(window, data.size - window):
            prev = data[center - window : center]
            nxt = data[center : center + window]
            err_prev = _burg_prediction_error(prev, order)
            err_next = _burg_prediction_error(nxt, order)
            if not np.isfinite(err_prev) or not np.isfinite(err_next):
                continue
            score = err_prev - err_next
            scores.append(score)
            idxs.append(center)
        if not scores:
            return None
        best = int(np.argmax(scores))
        pick = int(idxs[best])
        raw_score = float(scores[best])
        scale = float(np.max(np.abs(scores)) + 1e-9)
        normalized = max(0.0, min(1.0, max(0.0, raw_score) / scale))
        return self._format_result(trace_index, "ar_model", record, pick, raw_score, normalized)

    def _pick_template_correlation(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        template: Optional[np.ndarray] = options.get("template")
        if template is None:
            return None
        tpl = np.asarray(template, dtype=np.float64)
        if tpl.ndim != 1 or tpl.size == 0:
            return None
        data = self._signal(trace_index)
        tpl = tpl - np.mean(tpl)
        tpl_norm = np.linalg.norm(tpl) + 1e-12
        best_idx: Optional[int] = None
        best_score = -np.inf
        for start in range(0, data.size - tpl.size):
            window = data[start : start + tpl.size] - np.mean(data[start : start + tpl.size])
            score = float(np.dot(window, tpl) / (np.linalg.norm(window) * tpl_norm + 1e-12))
            if score > best_score:
                best_score = score
                best_idx = start
        if best_idx is None:
            return None
        normalized = (best_score + 1.0) / 2.0 if np.isfinite(best_score) else None
        return self._format_result(trace_index, "template_correlation", record, best_idx, best_score, normalized)

    def _pick_pai_k(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        """PAI-K (kurtosis-based) picker: rolling kurtosis with simple threshold."""
        data = self._signal(trace_index)
        win_sec = float(options.get("window", 0.6))
        threshold = float(options.get("threshold", 3.5))
        min_index = int(options.get("min_index", 0))

        win = max(3, int(win_sec * record.sampling_rate))
        if win >= data.size:
            return None

        x = data
        mean = np.convolve(x, np.ones(win), mode="valid") / win
        x2 = np.convolve(x * x, np.ones(win), mode="valid") / win
        x4 = np.convolve(x**4, np.ones(win), mode="valid") / win
        var = np.maximum(x2 - mean**2, 1e-12)
        std = np.sqrt(var)
        kurtosis = x4 / (var * var) - 3.0

        if kurtosis.size == 0:
            return None

        start_offset = win // 2
        idx = int(np.argmax(kurtosis))
        if idx < 0 or idx >= kurtosis.size:
            return None

        raw_score = float(kurtosis[idx])
        pick_idx = idx + start_offset

        if pick_idx < min_index or raw_score < threshold:
            return None

        normalized = max(0.0, min(1.0, (raw_score - threshold) / max(threshold * 2.0, 1e-6)))
        return self._format_result(trace_index, "pai_k", record, pick_idx, raw_score, normalized)

    def _pick_pai_s(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        """PAI-S (skewness-based) picker: rolling skewness with simple threshold."""
        data = self._signal(trace_index)
        win_sec = float(options.get("window", 0.6))
        threshold = float(options.get("threshold", 2.5))
        min_index = int(options.get("min_index", 0))

        win = max(3, int(win_sec * record.sampling_rate))
        if win >= data.size:
            return None

        x = data
        mean = np.convolve(x, np.ones(win), mode="valid") / win
        x2 = np.convolve(x * x, np.ones(win), mode="valid") / win
        x3 = np.convolve(x**3, np.ones(win), mode="valid") / win
        var = np.maximum(x2 - mean**2, 1e-12)
        std = np.sqrt(var)
        skewness = (x3 - 3 * mean * var - mean**3) / (std**3 + 1e-12)

        if skewness.size == 0:
            return None

        start_offset = win // 2
        idx = int(np.argmax(np.abs(skewness)))
        if idx < 0 or idx >= skewness.size:
            return None

        raw_score = float(skewness[idx])
        pick_idx = idx + start_offset

        if pick_idx < min_index or abs(raw_score) < threshold:
            return None

        normalized = max(0.0, min(1.0, (abs(raw_score) - threshold) / max(threshold * 2.0, 1e-6)))
        return self._format_result(trace_index, "pai_s", record, pick_idx, raw_score, normalized)

    def _pick_s_basic(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[PickResult]:
        """Basic S-wave picker using STA/LTA and time delay."""
        data = self._signal(trace_index)
        short = float(options.get("short_window", 1.0))
        long = float(options.get("long_window", 10.0))
        trig_on = float(options.get("trigger_on", 3.5))
        trig_off = float(options.get("trigger_off", 1.0))
        min_s_delay = float(options.get("min_s_delay", 2.0)) # Minimum delay after P to look for S
        
        nsta = max(1, int(short * record.sampling_rate))
        nlta = max(nsta + 1, int(long * record.sampling_rate))
        
        if nlta >= data.size:
            return None
            
        cft = classic_sta_lta(data, nsta, nlta)
        on_off = trigger_onset(cft, trig_on, trig_off)
        
        if on_off.size == 0:
            return None
            
        # Assume first trigger is P
        p_idx = int(on_off[0][0])
        
        # Look for S after P + delay
        s_search_start = p_idx + int(min_s_delay * record.sampling_rate)
        
        if s_search_start >= cft.size:
            return None
            
        # Find max STA/LTA in the search window
        s_window = cft[s_search_start:]
        if s_window.size == 0:
            return None
            
        s_relative_idx = np.argmax(s_window)
        s_idx = s_search_start + s_relative_idx
        
        raw_score = float(cft[s_idx])
        
        # Check if it's a valid trigger (above threshold)
        if raw_score < trig_on:
            return None
            
        normalized = max(0.0, min(1.0, (raw_score - trig_on) / max(trig_on * 2.0, 1e-6)))
        
        return self._format_result(
            trace_index, 
            "s_phase", 
            record, 
            s_idx, 
            raw_score, 
            normalized, 
            phase_type="S"
        )


def pick_phases(
    source: Any,
    *,
    file_type: Optional[str] = None,
    dataset: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    methods: Optional[Iterable[str]] = None,
    method_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PickResult]:
    """Convenience helper: load traces and run the requested picking methods."""

    traces = load_traces(source, file_type=file_type, dataset=dataset, sampling_rate=sampling_rate)
    engine = PhasePickingEngine(traces)
    return engine.run(methods=methods, method_params=method_params)


def summarize_pick_results(picks: Sequence[PickResult]) -> List[Dict[str, Any]]:
    """Aggregate pick confidences per trace for quick consumption."""

    summary: Dict[int, Dict[str, Any]] = {}
    for pick in picks:
        info = summary.setdefault(
            pick.trace_index,
            {
                "count": 0,
                "scores": [],
                "best": None,
                "methods": [],
            },
        )
        info["count"] += 1
        info["methods"].append(
            {
                "method": pick.method,
                "sample_index": pick.sample_index,
                "absolute_time": pick.absolute_time,
                "normalized_score": pick.normalized_score,
                "phase_type": pick.phase_type,
            }
        )
        if pick.normalized_score is not None:
            info["scores"].append(pick.normalized_score)
            best = info.get("best")
            if best is None or (best["score"] < pick.normalized_score):
                info["best"] = {
                    "score": pick.normalized_score,
                    "method": pick.method,
                    "sample_index": pick.sample_index,
                    "absolute_time": pick.absolute_time,
                    "phase_type": pick.phase_type,
                }

    summaries: List[Dict[str, Any]] = []
    for trace_index, info in summary.items():
        scores = info["scores"]
        avg = float(sum(scores) / len(scores)) if scores else None
        best = info.get("best") or {}
        summaries.append(
            {
                "trace_index": trace_index,
                "method_count": info["count"],
                "average_score": avg,
                "best_method": best.get("method"),
                "best_sample_index": best.get("sample_index"),
                "best_absolute_time": best.get("absolute_time"),
                "best_phase_type": best.get("phase_type"),
                "methods": info["methods"],
            }
        )
    return summaries


def plot_waveform_with_picks(
    traces: List[TraceRecord],
    picks: List[PickResult],
    output_path: str,
    max_traces: int = 5
) -> str:
    """Plot waveforms with picks and save to file."""

    def _method_to_color(method: str):
        # Stable method->color mapping using matplotlib categorical palettes.
        # Falls back gracefully if method list exceeds palette length.
        palette = plt.get_cmap('tab20')
        if not method:
            return palette(0)
        idx = abs(hash(method)) % palette.N
        return palette(idx)
    
    # Group picks by trace index
    picks_by_trace = {}
    for p in picks:
        if p.trace_index not in picks_by_trace:
            picks_by_trace[p.trace_index] = []
        picks_by_trace[p.trace_index].append(p)
        
    # Select traces to plot (first N)
    traces_to_plot = traces[:max_traces]
    
    n_traces = len(traces_to_plot)
    if n_traces == 0:
        return "No traces to plot."
        
    fig, axes = plt.subplots(n_traces, 1, figsize=(10, 3 * n_traces), sharex=True, squeeze=False)
    
    for i, trace in enumerate(traces_to_plot):
        ax = axes[i, 0]
        data = trace.data
        sr = trace.sampling_rate
        times = np.arange(len(data)) / sr
        
        waveform_line, = ax.plot(times, data, 'k-', linewidth=0.6, label="Waveform")
        ax.set_ylabel("Amplitude")
        
        # Plot picks
        trace_picks = picks_by_trace.get(i, [])

        legend_items = {"Waveform": waveform_line}

        for p in trace_picks:
            if p.sample_index < 0 or p.sample_index >= len(data):
                continue
                
            t_pick = p.sample_index / sr

            color = _method_to_color(p.method)
            linestyle = '-.' if p.phase_type == "S" else '--'
            
            # Vertical line

            legend_label = f"{p.phase_type}-{p.method}" if p.method else f"{p.phase_type}"
            line = ax.axvline(
                x=t_pick,
                color=color,
                linestyle=linestyle,
                alpha=0.9,
                linewidth=1.2,
            )

            # Keep one legend item per (phase, method) to avoid a huge legend.
            legend_items.setdefault(legend_label, line)

        if len(legend_items) > 1:
            ax.legend(list(legend_items.values()), list(legend_items.keys()), loc='upper right', fontsize=9)
        else:
            # No picks: avoid showing a redundant legend.
            ax.legend().remove()
            
    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    return output_path
