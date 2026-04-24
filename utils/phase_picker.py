import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import Stream, Trace, UTCDateTime, read as obspy_read
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset

try:  # envelope is located in different submodules across ObsPy versions
    from obspy.signal.filter import envelope as _obspy_envelope
except ImportError:  # pragma: no cover - fallback for older releases
    try:
        from obspy.signal.util import envelope as _obspy_envelope
    except ImportError:
        _obspy_envelope = None

# SeisBench lazy import for deep learning models
try:
    import seisbench
    import seisbench.models as sbm
    import torch
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    seisbench = None
    sbm = None
    torch = None


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


def _bandpass_filter(data: np.ndarray, sampling_rate: float,
                     freqmin: float = 1.0, freqmax: float = 45.0) -> np.ndarray:
    """Apply Butterworth bandpass filter using ObsPy."""
    nyquist = sampling_rate / 2.0
    # Clamp frequencies to valid range
    freqmin = max(freqmin, 0.001)
    freqmax = min(freqmax, nyquist * 0.9)
    if freqmin >= freqmax:
        return data
    try:
        return bandpass(data, freqmin, freqmax, df=sampling_rate, corners=4, zerophase=True)
    except Exception:
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

    def _attach_trace_index(traces: List[TraceRecord]) -> List[TraceRecord]:
        for idx, tr in enumerate(traces):
            if tr.metadata is None:
                tr.metadata = {}
            tr.metadata.setdefault("trace_index", idx)
        return traces

    if isinstance(source, np.ndarray):
        return _attach_trace_index(_collect_numpy_traces(source, sampling_rate))
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], np.ndarray):
        sr = _ensure_sampling_rate(sampling_rate)
        return _attach_trace_index([TraceRecord(data=np.asarray(arr, dtype=np.float64), sampling_rate=sr) for arr in source])
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
        return _attach_trace_index(_collect_obspy_traces(path))
    if fmt == "segy":
        return _attach_trace_index(_collect_segy_traces(path))
    if fmt == "hdf5":
        return _attach_trace_index(_collect_hdf5_traces(path, dataset, sampling_rate))
    if fmt in {"npy", "npz"}:
        return _attach_trace_index(_collect_numpy_traces(path, sampling_rate))

    raise ValueError(f"Unsupported file type '{ext}'. Specify file_type explicitly if needed.")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _robust_normalize(data: np.ndarray) -> np.ndarray:
    """
    Robust normalization for picking:
    1) sanitize NaN/Inf
    2) clip extreme amplitudes
    3) normalize by robust scale (MAD) with std fallback
    """
    if data.size == 0:
        return data

    arr = np.nan_to_num(data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # Clip rare spikes that can destabilize classic and ML pickers
    p1, p99 = np.percentile(arr, [1.0, 99.0])
    if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
        arr = np.clip(arr, p1, p99)

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    robust_std = 1.4826 * mad

    if robust_std > 1e-12:
        return (arr - median) / robust_std

    std = np.std(arr)
    if std > 1e-12:
        return (arr - np.mean(arr)) / std

    return arr


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
    """Run a suite of classical and deep learning phase picking algorithms on waveform data."""

    def __init__(self, traces: Sequence[TraceRecord]):
        if not traces:
            raise ValueError("No traces supplied for picking.")
        self.traces = list(traces)
        self._processed_traces = [self._preprocess(rec) for rec in self.traces]
        self._seisbench_models = {}  # Cache for loaded models

    @staticmethod
    def _preprocess(rec: TraceRecord) -> np.ndarray:
        """Full preprocessing pipeline: detrend, taper, bandpass, normalize."""
        from scipy.signal import detrend

        data = np.asarray(rec.data, dtype=np.float64).copy()
        sr = rec.sampling_rate
        if data.size == 0:
            return data

        # 0. Sanitize invalid values early
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Remove linear trend
        data = detrend(data, type='linear')

        # 2. Demean
        data -= np.mean(data)

        # 3. Taper edges (5% each side)
        n = len(data)
        taper_len = max(1, int(n * 0.05))
        taper = np.ones(n)
        taper[:taper_len] = np.linspace(0, 1, taper_len)
        taper[-taper_len:] = np.linspace(1, 0, taper_len)
        data *= taper

        # 4. Bandpass filter (1-45 Hz for local/regional events)
        data = _bandpass_filter(data, sr, freqmin=1.0, freqmax=min(45.0, sr * 0.45))

        # 5. Robust normalization
        return _robust_normalize(data)

    def _get_seisbench_model(self, model_name: str, pretrained: bool = True):
        """Load and cache a SeisBench model."""
        if not SEISBENCH_AVAILABLE:
            return None

        cache_key = f"{model_name}_{pretrained}"
        if cache_key in self._seisbench_models:
            return self._seisbench_models[cache_key]

        try:
            pretrained_name = "scedc" if pretrained else "original"
            fallback_names = (pretrained_name, "original", "instance")
            if model_name.lower() == "phasenet":
                for name in fallback_names:
                    try:
                        model = sbm.PhaseNet.from_pretrained(name)
                        break
                    except Exception:
                        model = None
                if model is None:
                    return None
            elif model_name.lower() == "eqtransformer":
                for name in fallback_names:
                    try:
                        model = sbm.EQTransformer.from_pretrained(name)
                        break
                    except Exception:
                        model = None
                if model is None:
                    return None
            elif model_name.lower() == "gpd":
                model = sbm.GPD.from_pretrained("instance" if pretrained else "original")
            else:
                return None

            # Set to evaluation mode
            if torch is not None:
                model.eval()

            self._seisbench_models[cache_key] = model
            return model
        except Exception as e:
            warnings.warn(f"Failed to load SeisBench model {model_name}: {e}")
            return None

    def run(
        self,
        methods: Optional[Iterable[str]] = None,
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PickResult]:
        # Default methods: deep learning only (EQTransformer + PhaseNet)
        # Traditional methods (sta_lta, aic, etc.) are only used when explicitly requested
        DEFAULT_METHODS = [
            "eqtransformer",
            "phasenet",
        ]

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
            "s_phase": self._pick_s_basic,
            # SeisBench deep learning methods
            "phasenet": self._pick_phasenet,
            "eqtransformer": self._pick_eqtransformer,
            "gpd": self._pick_gpd,
        }
        selected = list(methods) if methods else DEFAULT_METHODS
        params = method_params or {}
        results: List[PickResult] = []
        for idx, record in enumerate(self.traces):
            for name in selected:
                picker = available.get(name)
                if picker is None:
                    continue
                pick = picker(idx, record, params.get(name, {}))
                if pick:
                    # Handle both single pick and list of picks
                    if isinstance(pick, list):
                        results.extend(pick)
                    else:
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
        # Keep sample index within current trace bounds so picks are always plottable.
        npts = int(len(record.data))
        if npts > 0:
            sample_index = int(max(0, min(int(sample_index), npts - 1)))
        else:
            sample_index = int(sample_index)

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

    @staticmethod
    def _to_1d_prob(arr: Any) -> Optional[np.ndarray]:
        """Convert a model output fragment to a 1D probability array."""
        if arr is None:
            return None
        try:
            if torch is not None and hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
        except Exception:
            return None
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            return None
        if arr.ndim > 1:
            return None
        return arr

    @staticmethod
    def _resample_prob(prob: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
        """Resample probability curve to a target length for index alignment."""
        if prob is None or target_len <= 0:
            return None
        p = np.asarray(prob, dtype=np.float64).reshape(-1)
        if p.size == 0:
            return None
        if p.size == target_len:
            return p
        if p.size == 1:
            return np.full(target_len, float(p[0]), dtype=np.float64)
        x_old = np.linspace(0.0, 1.0, p.size)
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, p)

    def _extract_phase_probs(self, output: Any, expected_len: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse model output robustly and return aligned (P, S) probability curves.
        Supports dict/tuple/tensor outputs from different SeisBench model versions.
        """
        p_pred = None
        s_pred = None

        if isinstance(output, dict):
            p_pred = output.get("p_pick", output.get("p", output.get("P", None)))
            s_pred = output.get("s_pick", output.get("s", output.get("S", None)))
        elif isinstance(output, (list, tuple)):
            # Common order for EQTransformer: (det, p, s)
            if len(output) >= 3:
                p_pred = output[1]
                s_pred = output[2]
            elif len(output) == 2:
                p_pred = output[0]
                s_pred = output[1]
            elif len(output) == 1:
                output = output[0]

        if p_pred is None and s_pred is None:
            # Fallback tensor/array path: preserve channel axis if present.
            try:
                raw = output
                if torch is not None and hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                raw = np.asarray(raw, dtype=np.float64)
                raw = np.squeeze(raw)
                if raw.ndim == 3 and raw.shape[0] == 1:
                    raw = raw[0]
                if raw.ndim == 2:
                    if raw.shape[0] >= 3:
                        p_pred = raw[1]
                        s_pred = raw[2]
                    elif raw.shape[1] >= 3:
                        p_pred = raw[:, 1]
                        s_pred = raw[:, 2]
                    elif raw.shape[0] == 2:
                        p_pred = raw[0]
                        s_pred = raw[1]
                    else:
                        p_pred = raw[0]
                elif raw.ndim == 1:
                    p_pred = raw
            except Exception:
                pass

        p = self._resample_prob(self._to_1d_prob(p_pred), expected_len)
        s = self._resample_prob(self._to_1d_prob(s_pred), expected_len)
        return p, s

    def _first_peak(self, prob: Optional[np.ndarray], threshold: float) -> Optional[Tuple[int, float]]:
        """Return the earliest peak index/score above threshold."""
        if prob is None:
            return None
        peaks = self._find_peaks_simple(prob, threshold)
        if len(peaks) > 0:
            best_idx = int(peaks[0])
            return best_idx, float(prob[best_idx])
        # Fallback: allow global max if above threshold (for plateau outputs)
        idx = int(np.argmax(prob))
        score = float(prob[idx])
        if score > threshold:
            return idx, score
        return None

    @staticmethod
    def _annotation_stream(record: TraceRecord) -> Stream:
        """Build a 3-component stream from a single record for SeisBench annotate()."""
        data = np.asarray(record.data, dtype=np.float32)
        start_time = record.start_time or UTCDateTime(0)
        metadata = record.metadata or {}
        network = str(metadata.get("network") or "XX")
        station = str(metadata.get("station") or metadata.get("id") or "STA")
        channel = str(metadata.get("channel") or "BHZ").upper()
        prefix = channel[:-1] if len(channel) >= 3 else "BH"
        if len(prefix) < 2:
            prefix = "BH"
        location = str(metadata.get("location") or "")

        traces = []
        for comp in ("Z", "N", "E"):
            tr = Trace(data=data.copy())
            tr.stats.network = network
            tr.stats.station = station
            tr.stats.location = location
            tr.stats.channel = f"{prefix}{comp}"
            tr.stats.starttime = start_time
            tr.stats.sampling_rate = record.sampling_rate
            traces.append(tr)
        return Stream(traces)

    def _pick_from_annotation(
        self,
        annotated: Stream,
        trace_index: int,
        record: TraceRecord,
        method: str,
        model_label: str,
        threshold: float,
    ) -> Optional[List[PickResult]]:
        """Convert annotated probability traces into PickResult objects."""
        results: List[PickResult] = []
        for phase in ("P", "S"):
            phase_trace = next((tr for tr in annotated if str(getattr(tr.stats, "channel", "")).upper().endswith(phase)), None)
            if phase_trace is None:
                continue
            prob = np.asarray(phase_trace.data, dtype=np.float64).reshape(-1)
            peak = self._first_peak(prob, threshold)
            if peak is None:
                continue
            sample_index, raw_score = peak
            normalized = min(1.0, raw_score)
            results.append(
                self._format_result(
                    trace_index,
                    method,
                    record,
                    sample_index,
                    raw_score,
                    normalized,
                    phase_type=phase,
                    extra={"model": model_label},
                )
            )
        return results if results else None

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
        if len(on_off) == 0:
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

        if len(on_off) == 0:
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

    def _pick_phasenet(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[List[PickResult]]:
        """
        PhaseNet deep learning picker from SeisBench.
        Uses a U-Net architecture for P and S phase picking.

        Options:
            - threshold: Detection threshold (default: 0.3)
            - pretrained: Use pretrained weights (default: True)
        """
        if not SEISBENCH_AVAILABLE:
            return None

        threshold = float(options.get("threshold", 0.3))
        pretrained = options.get("pretrained", True)

        model = self._get_seisbench_model("phasenet", pretrained)
        if model is None:
            return None

        try:
            data = self._signal(trace_index)
            if hasattr(model, "annotate"):
                try:
                    annotated = model.annotate(self._annotation_stream(record))
                    picks = self._pick_from_annotation(
                        annotated,
                        trace_index,
                        record,
                        "phasenet",
                        "PhaseNet",
                        threshold,
                    )
                    if picks:
                        return picks
                except Exception:
                    pass

            # Fallback for model versions that do not support annotate() cleanly.
            model_samples = model.in_samples if hasattr(model, "in_samples") else 3001
            best_p_score = -1.0
            best_p_idx = None
            best_s_score = -1.0
            best_s_idx = None

            if len(data) >= model_samples:
                step = max(1, model_samples // 2)
                starts = list(range(0, len(data) - model_samples + 1, step))
                tail_start = len(data) - model_samples
                if starts and starts[-1] != tail_start:
                    starts.append(tail_start)
                for start in starts:
                    window = data[start:start + model_samples]
                    window_data = np.stack([window, window, window]).astype(np.float32)
                    for i in range(3):
                        std = np.std(window_data[i]) + 1e-10
                        window_data[i] = (window_data[i] - np.mean(window_data[i])) / std
                    tensor_data = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = model(tensor_data)
                    p_pred, s_pred = self._extract_phase_probs(output, expected_len=model_samples)
                    p_peak = self._first_peak(p_pred, threshold)
                    if p_peak is not None:
                        local_idx, p_score = p_peak
                        global_idx = start + local_idx
                        if best_p_idx is None or global_idx < best_p_idx or (global_idx == best_p_idx and p_score > best_p_score):
                            best_p_idx = global_idx
                            best_p_score = p_score
                    s_peak = self._first_peak(s_pred, threshold)
                    if s_peak is not None:
                        local_idx, s_score = s_peak
                        global_idx = start + local_idx
                        if best_s_idx is None or global_idx < best_s_idx or (global_idx == best_s_idx and s_score > best_s_score):
                            best_s_idx = global_idx
                            best_s_score = s_score
            else:
                padded_data = np.zeros(model_samples, dtype=np.float32)
                padded_data[:len(data)] = data
                window_data = np.stack([padded_data, padded_data, padded_data]).astype(np.float32)
                for i in range(3):
                    std = np.std(window_data[i]) + 1e-10
                    window_data[i] = (window_data[i] - np.mean(window_data[i])) / std
                tensor_data = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor_data)
                p_pred, s_pred = self._extract_phase_probs(output, expected_len=model_samples)
                p_peak = self._first_peak(p_pred, threshold)
                if p_peak is not None:
                    best_p_idx, best_p_score = p_peak
                s_peak = self._first_peak(s_pred, threshold)
                if s_peak is not None:
                    best_s_idx, best_s_score = s_peak

            results = []
            if best_p_idx is not None and best_p_score > threshold:
                results.append(self._format_result(
                    trace_index, "phasenet", record, int(best_p_idx), float(best_p_score), min(1.0, float(best_p_score)),
                    phase_type="P", extra={"model": "PhaseNet"}
                ))
            if best_s_idx is not None and best_s_score > threshold:
                results.append(self._format_result(
                    trace_index, "phasenet", record, int(best_s_idx), float(best_s_score), min(1.0, float(best_s_score)),
                    phase_type="S", extra={"model": "PhaseNet"}
                ))
            return results if results else None

        except Exception as e:
            warnings.warn(f"PhaseNet picking failed: {e}")
            return None

    def _pick_eqtransformer(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[List[PickResult]]:
        """
        EQTransformer deep learning picker from SeisBench.
        Uses a transformer architecture for earthquake detection and picking.

        Options:
            - threshold: Detection threshold (default: 0.3)
            - pretrained: Use pretrained weights (default: True)
        """
        if not SEISBENCH_AVAILABLE:
            return None

        threshold = float(options.get("threshold", 0.3))
        pretrained = options.get("pretrained", True)

        model = self._get_seisbench_model("eqtransformer", pretrained)
        if model is None:
            return None

        try:
            data = self._signal(trace_index)
            if hasattr(model, "annotate"):
                try:
                    annotated = model.annotate(self._annotation_stream(record))
                    picks = self._pick_from_annotation(
                        annotated,
                        trace_index,
                        record,
                        "eqtransformer",
                        "EQTransformer",
                        threshold,
                    )
                    if picks:
                        return picks
                except Exception:
                    pass

            # Fallback for model versions that do not support annotate() cleanly.
            model_samples = model.in_samples if hasattr(model, "in_samples") else 6000
            best_p_score = -1.0
            best_p_idx = None
            best_s_score = -1.0
            best_s_idx = None

            if len(data) >= model_samples:
                step = max(1, model_samples // 2)
                starts = list(range(0, len(data) - model_samples + 1, step))
                tail_start = len(data) - model_samples
                if starts and starts[-1] != tail_start:
                    starts.append(tail_start)
                for start in starts:
                    window = data[start:start + model_samples]
                    window_data = np.stack([window, window, window]).astype(np.float32)
                    for i in range(3):
                        std = np.std(window_data[i]) + 1e-10
                        window_data[i] = (window_data[i] - np.mean(window_data[i])) / std
                    tensor_data = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = model(tensor_data)
                    p_pred, s_pred = self._extract_phase_probs(output, expected_len=model_samples)
                    p_peak = self._first_peak(p_pred, threshold)
                    if p_peak is not None:
                        local_idx, p_score = p_peak
                        global_idx = start + local_idx
                        if best_p_idx is None or global_idx < best_p_idx or (global_idx == best_p_idx and p_score > best_p_score):
                            best_p_idx = global_idx
                            best_p_score = p_score
                    s_peak = self._first_peak(s_pred, threshold)
                    if s_peak is not None:
                        local_idx, s_score = s_peak
                        global_idx = start + local_idx
                        if best_s_idx is None or global_idx < best_s_idx or (global_idx == best_s_idx and s_score > best_s_score):
                            best_s_idx = global_idx
                            best_s_score = s_score
            else:
                padded_data = np.zeros(model_samples, dtype=np.float32)
                padded_data[:len(data)] = data
                window_data = np.stack([padded_data, padded_data, padded_data]).astype(np.float32)
                for i in range(3):
                    std = np.std(window_data[i]) + 1e-10
                    window_data[i] = (window_data[i] - np.mean(window_data[i])) / std
                tensor_data = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor_data)
                p_pred, s_pred = self._extract_phase_probs(output, expected_len=model_samples)
                p_peak = self._first_peak(p_pred, threshold)
                if p_peak is not None:
                    best_p_idx, best_p_score = p_peak
                s_peak = self._first_peak(s_pred, threshold)
                if s_peak is not None:
                    best_s_idx, best_s_score = s_peak

            results = []
            if best_p_idx is not None and best_p_score > threshold:
                results.append(self._format_result(
                    trace_index, "eqtransformer", record, int(best_p_idx), float(best_p_score), min(1.0, float(best_p_score)),
                    phase_type="P", extra={"model": "EQTransformer"}
                ))
            if best_s_idx is not None and best_s_score > threshold:
                results.append(self._format_result(
                    trace_index, "eqtransformer", record, int(best_s_idx), float(best_s_score), min(1.0, float(best_s_score)),
                    phase_type="S", extra={"model": "EQTransformer"}
                ))

            return results if results else None

        except Exception as e:
            warnings.warn(f"EQTransformer picking failed: {e}")
            return None

    def _pick_gpd(
        self,
        trace_index: int,
        record: TraceRecord,
        options: Dict[str, Any],
    ) -> Optional[List[PickResult]]:
        """
        GPD (Generalized Phase Detection) deep learning picker from SeisBench.
        Uses a CNN architecture with sliding windows for phase picking.

        Options:
            - threshold: Detection threshold (default: 0.5)
            - pretrained: Use pretrained weights (default: True)
        """
        if not SEISBENCH_AVAILABLE:
            return None

        threshold = float(options.get("threshold", 0.5))
        pretrained = options.get("pretrained", True)

        model = self._get_seisbench_model("gpd", pretrained)
        if model is None:
            return None

        try:
            data = self._signal(trace_index)
            sr = record.sampling_rate

            # GPD uses fixed window size (typically 400 samples)
            window_size = model.in_samples if hasattr(model, 'in_samples') else 400
            step = window_size // 4  # 75% overlap

            # Prepare 3-component data
            if data.ndim == 1:
                window_data = np.zeros((3, len(data)), dtype=np.float32)
                window_data[0] = data
                window_data[1] = data
                window_data[2] = data
            else:
                window_data = np.asarray(data, dtype=np.float32)
                if window_data.shape[0] != 3 and window_data.shape[1] == 3:
                    window_data = window_data.T

            # Normalize per component
            for i in range(3):
                std = np.std(window_data[i]) + 1e-10
                window_data[i] = (window_data[i] - np.mean(window_data[i])) / std

            # Sliding window inference
            p_probs = np.zeros(len(data))
            s_probs = np.zeros(len(data))
            n_windows = 0

            for start in range(0, len(data) - window_size + 1, step):
                end = start + window_size
                win = window_data[:, start:end]

                # Convert to tensor
                tensor_win = torch.tensor(win, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    output = model(tensor_win)

                # Get probabilities
                output_np = output[0].cpu().numpy()
                exp_output = np.exp(output_np - np.max(output_np))
                probs = exp_output / np.sum(exp_output)

                center = start + window_size // 2
                if center < len(data):
                    p_probs[center] = max(p_probs[center], probs[1] if len(probs) > 1 else 0)
                    s_probs[center] = max(s_probs[center], probs[2] if len(probs) > 2 else 0)
                n_windows += 1

            results = []

            # Find P peaks
            p_peaks = self._find_peaks_simple(p_probs, threshold)
            if len(p_peaks) > 0:
                p_idx = int(p_peaks[0])
                p_score = float(p_probs[p_idx])
                results.append(self._format_result(
                    trace_index, "gpd", record, p_idx, p_score, min(1.0, p_score),
                    phase_type="P", extra={"model": "GPD"}
                ))

            # Find S peaks
            s_peaks = self._find_peaks_simple(s_probs, threshold)
            if len(s_peaks) > 0:
                s_idx = int(s_peaks[0])
                s_score = float(s_probs[s_idx])
                results.append(self._format_result(
                    trace_index, "gpd", record, s_idx, s_score, min(1.0, s_score),
                    phase_type="S", extra={"model": "GPD"}
                ))

            return results if results else None

        except Exception as e:
            warnings.warn(f"GPD picking failed: {e}")
            return None

    def _find_peaks_simple(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Simple peak finding above threshold."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return np.array(peaks)


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
    """
    Aggregate pick confidences per trace for quick consumption.
    Includes scoring that prioritizes deep learning methods and consensus.
    """

    # Method priority weights (deep learning methods get higher weight)
    METHOD_WEIGHTS = {
        "phasenet": 1.2,
        "eqtransformer": 1.2,
        "gpd": 1.1,
        "sta_lta": 1.0,
        "aic": 0.9,
        "pai_k": 0.9,
        "pai_s": 0.9,
        "s_phase": 0.8,
        "frequency_ratio": 0.8,
        "autocorr": 0.7,
        "feature_threshold": 0.7,
        "ar_model": 0.7,
        "template_correlation": 0.8,
    }

    # Group picks by trace and phase type
    summary: Dict[int, Dict[str, Any]] = {}
    for pick in picks:
        trace_key = pick.trace_index
        phase_key = pick.phase_type or "P"

        if trace_key not in summary:
            summary[trace_key] = {
                "count": 0,
                "scores": [],
                "best_p": None,
                "best_s": None,
                "methods": [],
                "p_picks": [],
                "s_picks": [],
            }

        info = summary[trace_key]
        info["count"] += 1
        info["methods"].append(
            {
                "method": pick.method,
                "sample_index": pick.sample_index,
                "absolute_time": pick.absolute_time,
                "normalized_score": pick.normalized_score,
                "phase_type": phase_key,
            }
        )

        # Store picks by phase type
        if phase_key == "P":
            info["p_picks"].append(pick)
        else:
            info["s_picks"].append(pick)

        if pick.normalized_score is not None:
            info["scores"].append(pick.normalized_score)

    # Find best picks with scoring
    def _score_pick(pick: PickResult) -> float:
        """Score a pick considering confidence and method weight."""
        base_score = pick.normalized_score or 0.5
        method_weight = METHOD_WEIGHTS.get(pick.method, 0.8)
        return base_score * method_weight

    def _find_best_pick(picks_list: List[PickResult]):
        """Find the best pick from a list based on scoring."""
        if not picks_list:
            return None
        best = max(picks_list, key=_score_pick)
        return {
            "score": best.normalized_score,
            "weighted_score": _score_pick(best),
            "method": best.method,
            "sample_index": best.sample_index,
            "absolute_time": best.absolute_time,
            "phase_type": best.phase_type,
        }

    summaries: List[Dict[str, Any]] = []
    for trace_index, info in summary.items():
        scores = info["scores"]
        avg = float(sum(scores) / len(scores)) if scores else None

        # Find best P and S picks
        best_p = _find_best_pick(info["p_picks"])
        best_s = _find_best_pick(info["s_picks"])

        summaries.append(
            {
                "trace_index": trace_index,
                "method_count": info["count"],
                "average_score": avg,
                "best_p": best_p,
                "best_s": best_s,
                "best_method": (best_p or best_s or {}).get("method"),
                "best_sample_index": (best_p or best_s or {}).get("sample_index"),
                "best_absolute_time": (best_p or best_s or {}).get("absolute_time"),
                "best_phase_type": (best_p or best_s or {}).get("phase_type"),
                "methods": info["methods"],
            }
        )
    return summaries


def get_best_picks_for_plotting(
    picks: List[PickResult],
    max_per_phase: int = 1
) -> List[PickResult]:
    """
    Select the best picks for plotting.
    Returns top N picks per trace per phase type, prioritizing:
    1. High confidence score
    2. Deep learning method priority as tie-breaker
    """
    # Group picks by trace and phase
    grouped: Dict[Tuple[int, str], List[PickResult]] = {}
    for pick in picks:
        key = (pick.trace_index, pick.phase_type or "P")
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pick)

    # Method priority for plotting
    PLOT_PRIORITY = {
        "phasenet": 0,
        "eqtransformer": 1,
        "gpd": 2,
        "sta_lta": 3,
        "aic": 4,
        "pai_k": 5,
    }

    def _sort_key(pick: PickResult) -> Tuple[float, int]:
        """Sort by score first, then by method priority."""
        priority = PLOT_PRIORITY.get(pick.method, 10)
        score = pick.normalized_score or 0
        return (-score, priority)

    best_picks = []
    for (trace_idx, phase), pick_list in grouped.items():
        # Sort by priority and score
        sorted_picks = sorted(pick_list, key=_sort_key)
        # Take top N
        best_picks.extend(sorted_picks[:max_per_phase])

    return best_picks


def plot_waveform_with_picks(
    traces: List[TraceRecord],
    picks: List[PickResult],
    output_path: str,
    max_traces: int = 5,
    max_picks_per_phase: int = 1
) -> str:
    """Plot waveforms with best picks and save to file."""

    # Filter to best picks for cleaner visualization
    plot_picks = get_best_picks_for_plotting(picks, max_per_phase=max_picks_per_phase)

    def _pick_to_color(method: str, phase: str):
        """Scientific palette by phase: P=green, S=magenta-purple."""
        phase_key = (phase or "P").upper()
        if phase_key == "S":
            return "#C2185B"  # Magenta-purple
        return "#1B9E77"      # Scientific green

    # Group filtered picks by original trace index
    picks_by_trace = {}
    for p in plot_picks:
        if p.trace_index not in picks_by_trace:
            picks_by_trace[p.trace_index] = []
        picks_by_trace[p.trace_index].append(p)

    # Select traces to plot (first N)
    traces_to_plot = traces[:max_traces]

    n_traces = len(traces_to_plot)
    if n_traces == 0:
        return "No traces to plot."

    fig, axes = plt.subplots(n_traces, 1, figsize=(12, 3 * n_traces), sharex=True, squeeze=False)

    for i, trace in enumerate(traces_to_plot):
        ax = axes[i, 0]
        data = trace.data
        sr = trace.sampling_rate
        times = np.arange(len(data)) / sr

        # Plot waveform
        ax.plot(times, data, 'k-', linewidth=0.6, label="Waveform")
        ax.set_ylabel("Amplitude")

        # Match picks using the original trace index if available
        trace_idx = trace.metadata.get("trace_index")
        if trace_idx is None:
            trace_idx = i
        trace_picks = picks_by_trace.get(int(trace_idx), [])
        legend_items = {}

        for p in trace_picks:
            if p.sample_index < 0 or p.sample_index >= len(data):
                continue

            t_pick = p.sample_index / sr
            color = _pick_to_color(p.method, p.phase_type)
            linestyle = '-.' if p.phase_type == "S" else '--'

            # Create legend label with score
            score_str = f" ({p.normalized_score:.2f})" if p.normalized_score else ""
            legend_label = f"{p.phase_type}: {p.method}{score_str}"

            ax.axvline(
                x=t_pick,
                color=color,
                linestyle=linestyle,
                alpha=0.95,
                linewidth=1.1,
                label=legend_label
            )

            # Track legend items
            if legend_label not in legend_items:
                legend_items[legend_label] = True

        if legend_items:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, 'No picks detected', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')

        # Add trace info
        if trace.metadata:
            info_str = trace.metadata.get('id', '') or f"Trace {i}"
            ax.set_title(info_str, fontsize=10, loc='left')

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
