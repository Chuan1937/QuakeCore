"""
Continuous Seismic Picking Module for QuakeCore
===============================================
AI-powered continuous phase picking on multi-station waveform data.
Uses ensemble of PhaseNet and EQTransformer for robust detection.
"""

import os
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from obspy import read as obspy_read
from scipy.signal import find_peaks
from tqdm import tqdm

try:
    import seisbench.models as sbm
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    sbm = None


# =================== Device Setup ===================

def get_device() -> torch.device:
    """Get the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            torch.mps.set_per_process_memory_fraction(0.33)
        except Exception:
            pass
        torch.set_num_threads(4)
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =================== Model Caching ===================

_pht_model = None
_eqt_model = None
DEVICE = None


def _get_pht():
    """Get cached PhaseNet model."""
    global _pht_model
    if _pht_model is None:
        for name in ("scedc", "original"):
            try:
                _pht_model = sbm.PhaseNet.from_pretrained(name)
                _pht_model.to(DEVICE).eval()
                print(f"  [INFO] PhaseNet ({name}) loaded on {DEVICE}")
                break
            except Exception:
                continue
    return _pht_model


def _get_eqt():
    """Get cached EQTransformer model."""
    global _eqt_model
    if _eqt_model is None:
        for name in ("scedc", "original"):
            try:
                _eqt_model = sbm.EQTransformer.from_pretrained(name)
                _eqt_model.to(DEVICE).eval()
                print(f"  [INFO] EQTransformer ({name}) loaded on {DEVICE}")
                break
            except Exception:
                continue
    return _eqt_model


# =================== Continuous Picking ===================

def continuous_picking(
    streams: Dict[str, str],
    stations: Dict[str, Dict[str, float]],
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    peak_threshold: float = 0.3,
    merge_window: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Run continuous AI picking on multi-station waveform data.

    Uses ensemble of PhaseNet and EQTransformer, with peak detection
    and model consensus merging.

    Args:
        streams: Dict mapping station_id to mseed file path
        stations: Dict mapping station_id to station info
        device: PyTorch device (auto-detected if None)
        batch_size: Number of stations to process at once
        peak_threshold: Minimum peak height for detection
        merge_window: Time window (s) for merging picks from different models

    Returns:
        List of pick dicts with keys: station_id, phase, time, time_epoch,
        time_str, score, model
    """
    global DEVICE

    if device is None:
        DEVICE = get_device()
    else:
        DEVICE = device

    if not SEISBENCH_AVAILABLE:
        raise ImportError("seisbench is required for continuous picking")

    pht = _get_pht()
    eqt = _get_eqt()

    # Pre-load all station data
    station_data = []
    for net_sta, mseed_path in streams.items():
        if net_sta not in stations:
            continue
        try:
            st = obspy_read(mseed_path)
            _best = {}
            for tr in st:
                comp = tr.stats.channel[-1].upper()
                comp = {"1": "N", "2": "E"}.get(comp, comp)
                if comp in ("Z", "N", "E") and (comp not in _best or len(tr.data) > len(_best[comp].data)):
                    _best[comp] = tr
            if len(_best) >= 3:
                station_data.append((net_sta, _best))
        except Exception:
            continue

    all_picks = []
    total_batches = (len(station_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="  [Picking] Batches", unit="batch", leave=False):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(station_data))

        for net_sta, _best in station_data[batch_start:batch_end]:
            try:
                st_3c = st.__class__([_best[c] for c in ("Z", "N", "E")])

                t0 = max(tr.stats.starttime for tr in st_3c)
                t1 = min(tr.stats.endtime for tr in st_3c)
                for tr in st_3c:
                    tr.trim(starttime=t0, endtime=t1, pad=True, fill_value=0)
                    tr.detrend("demean").detrend("linear").taper(max_percentage=0.01)
                    tr.interpolate(sampling_rate=100.0)

                sta_candidates = []

                # PhaseNet
                try:
                    with torch.inference_mode():
                        anno_pht = pht.annotate(st_3c.copy())
                    for phase in ["P", "S"]:
                        tr_prob = next((tr for tr in anno_pht if tr.stats.channel.endswith(phase)), None)
                        if tr_prob:
                            dt = tr_prob.stats.delta
                            peaks, props = find_peaks(tr_prob.data, height=peak_threshold, distance=int(1.0 / dt))
                            for pk, conf in zip(peaks, props['peak_heights']):
                                sta_candidates.append({
                                    "station_id": net_sta, "phase": phase, "score": float(conf),
                                    "time": tr_prob.stats.starttime + (pk * dt), "model": "pht"
                                })
                except Exception:
                    pass

                # EQT
                try:
                    with torch.inference_mode():
                        anno_eqt = eqt.annotate(st_3c.copy())
                    for phase in ["P", "S"]:
                        tr_prob = next((tr for tr in anno_eqt if tr.stats.channel.endswith(phase)), None)
                        if tr_prob:
                            dt = tr_prob.stats.delta
                            peaks, props = find_peaks(tr_prob.data, height=peak_threshold, distance=int(1.0 / dt))
                            for pk, conf in zip(peaks, props['peak_heights']):
                                sta_candidates.append({
                                    "station_id": net_sta, "phase": phase, "score": float(conf),
                                    "time": tr_prob.stats.starttime + (pk * dt), "model": "eqt"
                                })
                except Exception:
                    pass

                # Ensemble merge: keep best pick per phase per time window
                sta_candidates.sort(key=lambda x: x["time"])
                merged = []
                for pick in sta_candidates:
                    if not merged:
                        merged.append(pick)
                        continue
                    if pick["phase"] == merged[-1]["phase"] and abs(pick["time"] - merged[-1]["time"]) < merge_window:
                        if pick["model"] == "pht" or pick["score"] > merged[-1]["score"]:
                            merged[-1] = pick
                    else:
                        merged.append(pick)

                for m in merged:
                    m["time_str"] = str(m["time"])
                    try:
                        m["time_epoch"] = float(m["time"].timestamp)
                    except AttributeError:
                        m["time_epoch"] = float(m["time"])
                all_picks.extend(merged)

            except Exception:
                continue

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print(f"  [Picking] Found {len(all_picks)} continuous peaks across network.")
    return all_picks


def clear_model_cache():
    """Clear cached models to free memory."""
    global _pht_model, _eqt_model
    _pht_model = None
    _eqt_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
