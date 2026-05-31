"""
Continuous Seismic Monitoring & Cataloging Pipeline (Southern California)
===========================================================================
Downloads continuous data for a specific time window, runs blind picking +
greedy REAL-Lite association to detect MULTIPLE events, locates them, and
matches the detected catalog against the SCEDC ground-truth catalog.

Usage:
    python continuous_monitoring_validation.py --start "2019-07-04T17:00:00" --end "2019-07-04T18:00:00"
"""

"""
Continuous Seismic Monitoring & Cataloging Pipeline (Southern California)
===========================================================================
Downloads continuous data for a specific time window, runs blind picking +
greedy REAL-Lite association to detect MULTIPLE events, locates them, and
matches the detected catalog against the SCEDC ground-truth catalog.

Usage:
    python continuous_monitoring_validation.py --start "2019-07-04T17:00:00" --end "2019-07-04T18:00:00"
"""

import os
import sys
import json
import argparse
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# QuakeCore imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from obspy import UTCDateTime, read_inventory, read as obspy_read
from obspy.clients.fdsn import Client as FDSNClient
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import torch
import seisbench.models as sbm

warnings.filterwarnings("ignore")

# =================== Device Setup ===================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cpu")
if DEVICE.type == "mps":
    try:
        torch.mps.set_per_process_memory_fraction(0.33)
    except Exception:
        pass
    # Use more threads for MPS operations
    torch.set_num_threads(4)
elif DEVICE.type == "cuda":
    torch.set_num_threads(4)

print(f"  [INFO] PyTorch device: {DEVICE}, threads: {torch.get_num_threads()}")

# =================== Paths ===================
DATA_DIR = os.path.join(PROJECT_ROOT, "Validation", "ContinuousData")
TAUP_DIR = os.path.join(os.path.dirname(__file__), "taup")
os.makedirs(DATA_DIR, exist_ok=True)


# =================== Helper Functions ===================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _dist_deg(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.minimum(np.sqrt(a), 1.0)))


# =================== Velocity Model & TT Cache ===================

_taup_model = None
_tt_interp = None


def _get_taup():
    global _taup_model
    if _taup_model is None:
        socal_tvel_path = os.path.join(TAUP_DIR, "socal.tvel")
        socal_npz_path = os.path.join(TAUP_DIR, "socal.npz")
        if not os.path.exists(socal_npz_path):
            os.makedirs(TAUP_DIR, exist_ok=True)
            import obspy.taup
            iasp91_path = os.path.join(os.path.dirname(obspy.taup.__file__), "data", "iasp91.tvel")
            with open(iasp91_path) as f:
                iasp91_lines = f.readlines()
            mantle_start = next(i + 2 for i, line in enumerate(iasp91_lines[2:]) if float(line.split()[0]) > 35.0)
            socal_layers = [
                (0.0, 5.50, 3.18, 2.60), (5.5, 5.50, 3.18, 2.60),
                (5.5, 6.30, 3.64, 2.67), (16.0, 6.30, 3.64, 2.67),
                (16.0, 6.70, 3.87, 2.80), (32.0, 6.70, 3.87, 2.80),
                (32.0, 7.80, 4.50, 3.30), (60.0, 7.80, 4.50, 3.30)
            ]
            with open(socal_tvel_path, "w") as f:
                f.write("SOUTHERN CALIFORNIA 1D VELOCITY MODEL\n")
                f.write(iasp91_lines[1])
                for dep, vp, vs, rho in socal_layers:
                    f.write(f"{dep:9.3f}{vp:7.4f}{vs:7.4f}{rho:7.4f}\n")
                f.writelines(iasp91_lines[mantle_start:])
            build_taup_model(socal_tvel_path, output_folder=TAUP_DIR)
        _taup_model = TauPyModel(model=socal_npz_path)
    return _taup_model


def _get_tt_interp():
    global _tt_interp
    if _tt_interp is not None:
        return _tt_interp
    cache_file = os.path.join(TAUP_DIR, "tt_interp_cache_v2.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        dist_grid = data["dist_grid"]
        depth_grid = data["depth_grid"]
        _tt_interp = {
            ph: RegularGridInterpolator(
                (dist_grid, depth_grid), data[f"tt_{ph}"],
                method='linear', bounds_error=False, fill_value=np.inf
            ) for ph in ["P", "S"]
        }
        return _tt_interp
    taup = _get_taup()
    dist_grid = np.linspace(0, 4.5, 450)
    depth_grid = np.linspace(0, 40, 41)
    _tt_interp = {}
    save_dict = {"dist_grid": dist_grid, "depth_grid": depth_grid}
    phase_map = {"P": ["p", "P", "Pg", "Pn", "Pdiff"], "S": ["s", "S", "Sg", "Sn", "Sdiff"]}
    for ph in ["P", "S"]:
        mat = np.full((len(dist_grid), len(depth_grid)), np.inf)
        for i, d in enumerate(dist_grid):
            for j, z in enumerate(depth_grid):
                try:
                    arrs = taup.get_travel_times(max(0.001, z), max(0.001, d), phase_list=phase_map[ph])
                    if arrs:
                        mat[i, j] = min(a.time for a in arrs)
                except Exception:
                    pass
        mask = np.isfinite(mat)
        if mask.any():
            mat[~mask] = np.max(mat[mask]) * 2
        save_dict[f"tt_{ph}"] = mat
        _tt_interp[ph] = RegularGridInterpolator(
            (dist_grid, depth_grid), mat,
            method='linear', bounds_error=False, fill_value=np.inf
        )
    np.savez(cache_file, **save_dict)
    return _tt_interp


# =================== 1. 数据下载模块 ===================

def download_continuous_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon):
    """下载指定区域和时间的连续波形"""
    chunk_dir = os.path.join(DATA_DIR, f"chunk_{start_time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(chunk_dir, exist_ok=True)

    # 检查是否已有下载好的数据
    existing_mseed = glob.glob(os.path.join(chunk_dir, "*.mseed"))
    if existing_mseed:
        print(f"  [INFO] Found {len(existing_mseed)} existing mseed files in {chunk_dir}")
        print(f"  [INFO] Loading existing data directly...")

        client = FDSNClient("SCEDC", timeout=120)
        try:
            inv = client.get_stations(
                network="CI", station="*", channel="BH?,HH?",
                minlatitude=min_lat, maxlatitude=max_lat,
                minlongitude=min_lon, maxlongitude=max_lon,
                starttime=start_time, endtime=end_time, level="response")
        except Exception as e:
            print(f"  [ERROR] Failed to fetch inventory: {e}")
            return None, None

        stations = {}
        for net in inv:
            for sta in net:
                stations[f"{net.code}.{sta.code}"] = {
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation": sta.elevation or 0.0
                }

        streams = {}
        for net_sta_file in existing_mseed:
            net_sta = os.path.splitext(os.path.basename(net_sta_file))[0]
            streams[net_sta] = net_sta_file

        print(f"  Loaded {len(streams)} stations from existing data.")
        return streams, stations

    # 如果没有已下载数据，则下载新数据
    client = FDSNClient("SCEDC", timeout=120)

    print(f"--- Fetching Metadata for CI network ---")
    try:
        inv = client.get_stations(
            network="CI", station="*", channel="BH?,HH?",
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon,
            starttime=start_time, endtime=end_time, level="response")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch inventory: {e}")
        return None, None

    stations = {}
    for net in inv:
        for sta in net:
            stations[f"{net.code}.{sta.code}"] = {
                "latitude": sta.latitude,
                "longitude": sta.longitude,
                "elevation": sta.elevation or 0.0
            }

    print(f"  Found {len(stations)} stations. Downloading waveforms...")

    streams = {}
    for net in inv:
        for sta in net:
            net_sta = f"{net.code}.{sta.code}"
            mseed_file = os.path.join(chunk_dir, f"{net_sta}.mseed")
            if os.path.exists(mseed_file):
                streams[net_sta] = mseed_file
                continue
            try:
                st = client.get_waveforms(net.code, sta.code, "*", "BH?,HH?", start_time, end_time)
                st.write(mseed_file, format="MSEED")
                streams[net_sta] = mseed_file
            except Exception:
                pass

    print(f"  Successfully downloaded data for {len(streams)} stations.")
    return streams, stations


# =================== 2. 连续双模拾取 ===================

_pht_model = None


def _get_pht():
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


_eqt_model = None


def _get_eqt():
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


def continuous_picking(streams, stations):
    pht = _get_pht()
    eqt = _get_eqt()
    all_picks = []
    batch_size = 4  # Process 4 stations at a time for memory efficiency

    # Pre-load all station data first
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
                            peaks, props = find_peaks(tr_prob.data, height=0.3, distance=int(1.0 / dt))
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
                            peaks, props = find_peaks(tr_prob.data, height=0.3, distance=int(1.0 / dt))
                            for pk, conf in zip(peaks, props['peak_heights']):
                                sta_candidates.append({
                                    "station_id": net_sta, "phase": phase, "score": float(conf),
                                    "time": tr_prob.stats.starttime + (pk * dt), "model": "eqt"
                                })
                except Exception:
                    pass

                # Ensemble merge
                sta_candidates.sort(key=lambda x: x["time"])
                merged = []
                for pick in sta_candidates:
                    if not merged:
                        merged.append(pick)
                        continue
                    if pick["phase"] == merged[-1]["phase"] and abs(pick["time"] - merged[-1]["time"]) < 1.0:
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


# =================== 3. 贪心多事件关联 (Multi-Event REAL-Lite) ===================

def associate_multiple_events(picks, stations, tt_interp, time_tolerance=2.0, min_picks=6):
    """
    贪心算法：不断寻找最强事件，记录后，将其 Picks 移出池子，直到没有足够 Picks 组网。
    """
    unassociated_picks = picks.copy()
    detected_events = []

    # 南加州粗网格
    lats = np.arange(32.0, 36.5, 0.1)
    lons = np.arange(-120.0, -115.0, 0.1)
    trial_depth = 10.0

    iteration = 1
    while len(unassociated_picks) >= min_picks:
        best_cluster_count = 0
        best_event_info = None
        best_event_picks = []

        # Pre-collect valid picks once per association iteration
        valid_picks = []
        for p in unassociated_picks:
            sid = p.get("station_id")
            sta = stations.get(sid)
            if sta is None:
                continue
            ph = p.get("phase")
            if ph not in tt_interp:
                continue
            t_epoch = p.get("time_epoch")
            if t_epoch is None:
                try:
                    t_epoch = float(p["time"].timestamp)
                except AttributeError:
                    t_epoch = float(p["time"])
            valid_picks.append((p, sta["latitude"], sta["longitude"], ph, float(t_epoch)))

        if len(valid_picks) < min_picks:
            break

        pick_refs = [v[0] for v in valid_picks]
        pick_lats = np.asarray([v[1] for v in valid_picks], dtype=np.float64)
        pick_lons = np.asarray([v[2] for v in valid_picks], dtype=np.float64)
        pick_phases = np.asarray([v[3] for v in valid_picks], dtype=object)
        pick_times = np.asarray([v[4] for v in valid_picks], dtype=np.float64)

        phase_masks = {ph: (pick_phases == ph) for ph in tt_interp.keys()}

        import itertools
        grid_points = list(itertools.product(lats, lons))
        for trial_lat, trial_lon in tqdm(grid_points, desc="  [Association] Grid scan", unit="grid", leave=False):
                dists = _dist_deg(pick_lats, pick_lons, trial_lat, trial_lon)
                theo = np.full_like(pick_times, np.inf, dtype=np.float64)
                for ph, mask in phase_masks.items():
                    if np.any(mask):
                        theo[mask] = tt_interp[ph](np.column_stack([dists[mask], np.full(np.sum(mask), trial_depth)]))

                valid = np.isfinite(theo)
                if int(np.sum(valid)) < min_picks:
                    continue

                ots = pick_times[valid] - theo[valid]
                if ots.size < min_picks:
                    continue

                ots_sorted = np.sort(ots)
                right = np.searchsorted(ots_sorted, ots_sorted + time_tolerance, side="right")
                counts = right - np.arange(ots_sorted.size)
                max_idx = int(np.argmax(counts))
                count = int(counts[max_idx])

                if count > best_cluster_count:
                    best_cluster_count = count
                    cluster_start = float(ots_sorted[max_idx])
                    cluster_end = cluster_start + time_tolerance
                    valid_idx = np.flatnonzero(valid)
                    win_mask = (ots >= cluster_start) & (ots <= cluster_end)
                    selected_idx = valid_idx[win_mask]
                    sel_picks = [pick_refs[i] for i in selected_idx.tolist()]

                    unique = {(p["station_id"], p["phase"]): p for p in sel_picks}
                    final_picks = list(unique.values())

                    best_event_info = {
                        "init_lat": float(trial_lat),
                        "init_lon": float(trial_lon),
                        "approx_time": UTCDateTime(cluster_start + time_tolerance / 2.0),
                        "num_picks": len(final_picks)
                    }
                    best_event_picks = final_picks

        if best_event_info and best_event_info["num_picks"] >= min_picks:
            print(f"  [Association Iter {iteration}] Found event at {best_event_info['approx_time']} "
                  f"with {len(best_event_picks)} picks.")
            detected_events.append((best_event_info, best_event_picks))
            # 核心：从池子中剥离已经被关联的 picks
            picked_ids = {id(p) for p in best_event_picks}
            unassociated_picks = [p for p in unassociated_picks if id(p) not in picked_ids]
            iteration += 1
        else:
            break

    print(f"  [Association Done] Total {len(detected_events)} events isolated.")
    return detected_events


# =================== 4. 定位函数 (EDT Grid Search) ===================

def locate_grid_search_local_blind(picks, stations, init_lat, init_lon, init_depth=10.0):
    """4-stage EDT grid search with S-P depth refinement."""
    sta_coords = {
        p["station_id"]: (stations[p["station_id"]]["latitude"], stations[p["station_id"]]["longitude"])
        for p in picks if p["station_id"] in stations
    }
    if len(sta_coords) < 3:
        return None

    tt_interp = _get_tt_interp()
    def _pick_epoch(p):
        t_epoch = p.get("time_epoch")
        if t_epoch is not None:
            return float(t_epoch)
        return float(UTCDateTime(p["time_str"]))

    t0 = min(_pick_epoch(p) for p in picks)
    obs_rel = [
        (p["station_id"], p["phase"], _pick_epoch(p) - t0)
        for p in picks
    ]

    def _vec_dist2deg(lat1, lon1, lat2, lon2):
        la1, lo1, la2, lo2 = map(
            lambda x: np.radians(np.asarray(x, dtype=np.float64)), [lat1, lon1, lat2, lon2]
        )
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return np.degrees(2 * np.arcsin(np.minimum(np.sqrt(a), 1.0)))

    def _edt_grid(obs_rel, sta_coords, lat_range, lon_range, depth_range):
        N = len(obs_rel)
        sta_lats = np.array([sta_coords[s][0] for s, _, _ in obs_rel])
        sta_lons = np.array([sta_coords[s][1] for s, _, _ in obs_rel])
        obs_times = np.array([t for _, _, t in obs_rel])
        phases = [ph for _, ph, _ in obs_rel]

        n_grid = len(lat_range) * len(lon_range) * len(depth_range)
        theo = np.full((n_grid, N), np.inf, dtype=np.float32)
        lats_f = np.empty(n_grid, dtype=np.float32)
        lons_f = np.empty(n_grid, dtype=np.float32)
        deps_f = np.empty(n_grid, dtype=np.float32)
        idx = 0
        for lat in lat_range:
            for lon in lon_range:
                for dep in depth_range:
                    lats_f[idx], lons_f[idx], deps_f[idx] = lat, lon, dep
                    idx += 1

        for k in range(N):
            if phases[k] in tt_interp:
                dists = _vec_dist2deg(sta_lats[k], sta_lons[k], lats_f, lons_f)
                theo[:, k] = tt_interp[phases[k]](np.column_stack([dists, deps_f])).astype(np.float32)

        valid = np.isfinite(theo)
        n_val = valid.sum(axis=1).astype(np.float64)
        n_pairs = n_val * (n_val - 1) / 2
        c = np.where(valid, obs_times[None, :] - theo.astype(np.float64), np.nan)
        c.sort(axis=1)
        np.nan_to_num(c, nan=0.0, copy=False)
        S = c.sum(axis=1)
        SJ = (c * np.arange(N, dtype=np.float64)).sum(axis=1)
        misfit = np.where(n_pairs > 0, (2 * SJ - (n_val - 1) * S) / n_pairs, np.inf)
        best_idx = int(np.argmin(misfit))
        best_rms = float(misfit[best_idx])
        return float(lats_f[best_idx]), float(lons_f[best_idx]), float(deps_f[best_idx]), best_rms

    # Stage 1: Coarse grid
    lat_c = np.arange(init_lat - 0.4, init_lat + 0.401, 0.05)
    lon_c = np.arange(init_lon - 0.4, init_lon + 0.401, 0.05)
    dep_c = np.arange(0, 41, 4)
    best_lat, best_lon, best_dep, _ = _edt_grid(obs_rel, sta_coords, lat_c, lon_c, dep_c)
    best_rms = None

    # Stage 2: Fine grid
    lat_f = np.arange(best_lat - 0.1, best_lat + 0.101, 0.01)
    lon_f = np.arange(best_lon - 0.1, best_lon + 0.101, 0.01)
    dep_f = np.arange(max(0, best_dep - 6), best_dep + 6.1, 1.0)
    best_lat, best_lon, best_dep, best_rms = _edt_grid(obs_rel, sta_coords, lat_f, lon_f, dep_f)

    # Stage 3: Global S-P depth search
    sta_ps = {}
    for sta, ph, rel_t in obs_rel:
        if sta in sta_coords:
            sta_ps.setdefault(sta, {})[ph] = rel_t
    ps_valid = {s: v for s, v in sta_ps.items() if "P" in v and "S" in v}

    if len(ps_valid) >= 2:
        dep_fine = np.arange(0.0, 35.5, 0.5)
        ps_stas = list(ps_valid.keys())
        obs_sp = np.array([ps_valid[s]["S"] - ps_valid[s]["P"] for s in ps_stas])
        ps_dists = _vec_dist2deg(
            [sta_coords[s][0] for s in ps_stas],
            [sta_coords[s][1] for s in ps_stas], best_lat, best_lon
        )
        misfit_sp = np.full(len(dep_fine), np.inf)
        for di, d in enumerate(dep_fine):
            residuals = []
            for si, dist in enumerate(ps_dists):
                tp = float(tt_interp["P"]((dist, float(d))))
                ts = float(tt_interp["S"]((dist, float(d))))
                if not (np.isinf(tp) or np.isinf(ts)):
                    residuals.append(abs(obs_sp[si] - (ts - tp)))
            if residuals:
                misfit_sp[di] = np.median(residuals)
        best_dep = float(dep_fine[np.argmin(misfit_sp)])

    # Stage 4: Micro-polish
    lat_m = np.arange(best_lat - 0.03, best_lat + 0.031, 0.005)
    lon_m = np.arange(best_lon - 0.03, best_lon + 0.031, 0.005)
    dep_m = np.arange(max(0, best_dep - 2.0), best_dep + 2.1, 0.5)
    best_lat, best_lon, best_dep, best_rms = _edt_grid(obs_rel, sta_coords, lat_m, lon_m, dep_m)

    # 计算方位角 GAP
    if len(sta_coords) >= 3:
        event_lat, event_lon = best_lat, best_lon
        azims = []
        for sta, (slat, slon) in sta_coords.items():
            az = np.degrees(np.arctan2(np.sin(np.radians(slon - event_lon)) * np.cos(np.radians(slat)),
                                        np.cos(np.radians(event_lat)) * np.sin(np.radians(slat)) - np.sin(np.radians(event_lat)) * np.cos(np.radians(slat)) * np.cos(np.radians(slon - event_lon))))
            azims.append((az + 360) % 360)
        azims = sorted(azims)
        gaps = [(azims[(i + 1) % len(azims)] - azims[i]) for i in range(len(azims))]
        gap = max(gaps) if gaps else 360.0
    else:
        gap = 360.0

    return {"latitude": best_lat, "longitude": best_lon, "depth": best_dep, "rms": best_rms, "gap": gap, "num_picks": len(obs_rel)}


# =================== 5. 目录匹配与评估模块 ===================

def fetch_scedc_catalog(start_time, end_time, min_lat, max_lat, min_lon, max_lon):
    """通过 FDSN 下载真实的 SCEDC 目录作为 Ground Truth"""
    client = FDSNClient("SCEDC")
    try:
        cat = client.get_events(
            starttime=start_time, endtime=end_time,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon
        )
        truth = []
        for ev in cat:
            origin = ev.preferred_origin() or ev.origins[0]
            mag = ev.preferred_magnitude() or ev.magnitudes[0]
            truth.append({
                "time": origin.time,
                "lat": origin.latitude,
                "lon": origin.longitude,
                "depth": origin.depth / 1000.0 if origin.depth else 0.0,
                "mag": mag.mag if mag else 0.0,
                "id": str(ev.resource_id)
            })
        return truth
    except Exception as e:
        print(f"  [WARN] Failed to fetch truth catalog: {e}")
        return []


def match_catalogs(detected_cat, truth_cat):
    """将盲检目录与真实目录进行双向匹配"""
    matched_truth_indices = set()
    matches = []
    false_positives = []

    for det in detected_cat:
        best_match = None
        min_time_diff = 999
        best_truth_idx = -1

        for i, tru in enumerate(truth_cat):
            t_diff = abs(det["time"] - tru["time"])
            dist = haversine_km(det["latitude"], det["longitude"], tru["lat"], tru["lon"])

            # 判定标准：时间误差 < 5s，空间误差 < 30km
            if t_diff <= 5.0 and dist <= 30.0:
                if t_diff < min_time_diff:
                    min_time_diff = t_diff
                    best_match = tru
                    best_truth_idx = i

        if best_match:
            matched_truth_indices.add(best_truth_idx)
            matches.append({
                "detected": det,
                "truth": best_match,
                "dist_err": haversine_km(det["latitude"], det["longitude"], best_match["lat"], best_match["lon"]),
                "depth_err": abs(det["depth"] - best_match["depth"])
            })
        else:
            false_positives.append(det)

    false_negatives = [tru for i, tru in enumerate(truth_cat) if i not in matched_truth_indices]

    return matches, false_positives, false_negatives


def plot_detected_catalog_three_views(detected_catalog, output_png, title=None):
    """
    参考目录绘图风格：
    一张大图包含：
    - 平面图（带地形底图）
    - 纬度-深度剖面
    - 经度-深度剖面
    - 震级(可视化尺寸)图例
    """
    if not detected_catalog:
        return None

    ev_lon = np.array([float(ev["longitude"]) for ev in detected_catalog], dtype=float)
    ev_lat = np.array([float(ev["latitude"]) for ev in detected_catalog], dtype=float)
    ev_dep = np.array([max(0.0, float(ev.get("depth", 0.0))) for ev in detected_catalog], dtype=float)
    # 目录里没有真实震级，用拾取数构造一个稳定的可视化等级（仅用于点大小）
    ev_mag_vis = np.array([1.8 + 0.18 * np.sqrt(max(1, int(ev.get("num_picks", 1)))) for ev in detected_catalog], dtype=float)

    min_lat, max_lat = float(np.min(ev_lat)), float(np.max(ev_lat))
    min_lon, max_lon = float(np.min(ev_lon)), float(np.max(ev_lon))
    lat_pad = max(0.08, (max_lat - min_lat) * 0.08 + 0.02)
    lon_pad = max(0.08, (max_lon - min_lon) * 0.08 + 0.02)

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    gs0 = fig.add_gridspec(8, 12)

    # 布局与参考代码一致
    ax1 = fig.add_subplot(gs0[0:5, 7:12])  # depth-lat
    ax2 = fig.add_subplot(gs0[5:9, 0:7])   # lon-depth
    ax3 = fig.add_subplot(gs0[5:9, 7:12])  # legend

    # 地图面板优先用 Cartopy，并加入 terrain/imagery 背景
    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        use_cartopy = True
    except Exception:
        use_cartopy = False

    if use_cartopy:
        data_crs = ccrs.PlateCarree()
        ax0 = fig.add_subplot(gs0[0:5, 0:7], projection=data_crs)
        ax0.set_extent(
            [min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad],
            crs=data_crs,
        )
        # terrain-like 底图（Cartopy 内置，离线可用）
        try:
            ax0.stock_img()
        except Exception:
            ax0.set_facecolor("#e9eef2")
        gl = ax0.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.7,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax0 = fig.add_subplot(gs0[0:5, 0:7])
        ax0.set_facecolor("#e9eef2")
        ax0.grid(linestyle="--", alpha=0.6)
        ax0.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax0.set_ylim(min_lat - lat_pad, max_lat + lat_pad)

    dep_max = max(45.0, float(np.max(ev_dep)) + 5.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=dep_max)
    cmap = cm.jet_r
    colors = cmap(norm(ev_dep))
    ms = np.clip(ev_mag_vis * 3.0, 4.5, 18.0)

    # xoy
    if use_cartopy:
        ax0.scatter(ev_lon, ev_lat, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, transform=data_crs, alpha=0.9)
    else:
        ax0.scatter(ev_lon, ev_lat, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9)
    ax0.set_title("Plan View")

    # yoz: depth-lat
    ax1.scatter(ev_dep, ev_lat, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9)
    ax1.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    ax1.set_xlim(0.0, dep_max)
    ax1.set_facecolor("#bfc0c2")
    ax1.set_xlabel("Depth (km)", fontsize=12)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_yticklabels([])
    ax1.set_title("Lat-Depth")
    ax1.tick_params(axis="both", which="major", labelsize=11)

    # xoz: lon-depth
    ax2.scatter(ev_lon, ev_dep, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9)
    ax2.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax2.set_ylim(dep_max, 0.0)
    ax2.set_facecolor("#bfc0c2")
    ax2.set_xlabel("Lon. (°)", fontsize=12)
    ax2.set_ylabel("Depth (km)", fontsize=12)
    ax2.set_title("Lon-Depth")
    ax2.tick_params(axis="both", which="major", labelsize=11)

    # 右下角图例
    ax3.plot(0.10, 0.90, "o", mec="k", mfc="none", mew=1, ms=3 * 3)
    ax3.plot(0.10, 0.78, "o", mec="k", mfc="none", mew=1, ms=4 * 3)
    ax3.plot(0.10, 0.65, "o", mec="k", mfc="none", mew=1, ms=5 * 3)
    ax3.plot(0.10, 0.50, "o", mec="k", mfc="none", mew=1, ms=6 * 3)
    ax3.text(0.20, 0.88, "M 3.0", fontsize=12)
    ax3.text(0.20, 0.76, "M 4.0", fontsize=12)
    ax3.text(0.20, 0.62, "M 5.0", fontsize=12)
    ax3.text(0.20, 0.47, "M 6.0", fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical")
    cbar.set_label(label="Depth (km)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(title or "Detected Event Catalog", fontsize=14)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png


# =================== 主程序流程 ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2019-07-04T17:00:00", help="Start time")
    parser.add_argument("--end", type=str, default="2019-07-04T18:00:00", help="End time")
    args = parser.parse_args()

    start_time = UTCDateTime(args.start)
    end_time = UTCDateTime(args.end)

    # 南加州区域
    min_lat, max_lat = 32.0, 36.5
    min_lon, max_lon = -120.0, -115.0

    print("=" * 70)
    print(f"  Continuous Blind Monitoring: {start_time} to {end_time}")
    print("=" * 70)

    _get_taup()
    tt_interp = _get_tt_interp()

    # Step 1: 获取数据
    print("\n--- Step 1: Downloading continuous data ---")
    streams, stations = download_continuous_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    if not streams:
        return

    # Step 2: 连续拾取
    print("\n--- Step 2: AI continuous picking (PhaseNet + EQT) ---")
    all_picks = continuous_picking(streams, stations)

    # Step 3: 多事件关联剥离
    print("\n--- Step 3: Greedy multi-event association ---")
    detected_clusters = associate_multiple_events(all_picks, stations, tt_interp, time_tolerance=1.0, min_picks=5)

    # Step 4: 定位所有捕获到的事件
    detected_catalog = []
    print("\n--- Step 4: Locating detected events with STRICT QC ---")
    for idx, (ev_info, ev_picks) in enumerate(tqdm(detected_clusters, desc="  [Locating] Events", unit="ev", leave=False)):
        loc = locate_grid_search_local_blind(
            ev_picks, stations, ev_info["init_lat"], ev_info["init_lon"]
        )
        if loc:
            # ==========================================
            # 【终极防过拟合质控 (Anti-Overfitting QC)】
            # ==========================================
            rms = loc.get('rms', 0)
            gap = loc.get('gap', 360)
            depth = loc.get('depth', 10)
            num_picks = loc.get('num_picks', len(ev_picks))

            # 统计台站与评分状态
            sta_phases = {}
            for p in ev_picks:
                sta_phases.setdefault(p["station_id"], set()).add(p["phase"])

            ps_pairs = sum(1 for phases in sta_phases.values() if "P" in phases and "S" in phases)
            max_score = max(p.get("score", 0) for p in ev_picks)

            dists_km = []
            for p in ev_picks:
                if p["station_id"] in stations:
                    s = stations[p["station_id"]]
                    d = haversine_km(loc["latitude"], loc["longitude"], s["latitude"], s["longitude"])
                    dists_km.append(d)
            min_dist = min(dists_km) if dists_km else 999.0

            # ---------------------------------------------------------
            # 1. 绝对底线 (Physical Bottom Line)
            # ---------------------------------------------------------
            if rms > 1.5: continue
            if gap > 300.0: continue
            if min_dist > 80.0: continue
            if ps_pairs < 1: continue
            if (depth <= 0.5 or depth >= 39.5) and rms > 1.2: continue

            # ---------------------------------------------------------
            # 2. 自由度动态审查 (Degrees of Freedom Check)
            # ---------------------------------------------------------
            if num_picks <= 6:
                if max_score < 0.75 or ps_pairs < 2:
                    continue
            elif num_picks <= 8:
                if max_score < 0.60:
                    continue
            else:
                if max_score < 0.45:
                    continue

            # ==========================================
            # 恭喜，确认为真实地震！
            # ==========================================
            t0 = min(float(p.get("time_epoch", p["time"])) for p in ev_picks)
            ot_abs = UTCDateTime(t0)

            loc["time"] = ot_abs
            loc["num_picks"] = len(ev_picks)
            detected_catalog.append(loc)

    # 保存检测结果到文件
    continuous_results_dir = os.path.join(os.path.dirname(__file__), "continuous_results")
    os.makedirs(continuous_results_dir, exist_ok=True)
    result_file = os.path.join(continuous_results_dir, f"continuous_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w") as f:
        json.dump({
            "start_time": str(start_time),
            "end_time": str(end_time),
            "detected_catalog": [
                {
                    "time": str(ev["time"]),
                    "latitude": ev["latitude"],
                    "longitude": ev["longitude"],
                    "depth": ev["depth"],
                    "num_picks": ev["num_picks"],
                    "rms": ev.get("rms", 0),
                    "gap": ev.get("gap", 360),
                }
                for ev in detected_catalog
            ]
        }, f, indent=2)
    print(f"\n  Saved detection results to: {result_file}")

    # Step 4.5: 目录三视图绘图
    if detected_catalog:
        fig_file = os.path.join(
            continuous_results_dir,
            f"continuous_{start_time.strftime('%Y%m%d_%H%M%S')}_catalog_3views.png",
        )
        try:
            plot_detected_catalog_three_views(
                detected_catalog=detected_catalog,
                output_png=fig_file,
                title=f"Detected Catalog {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')} UTC",
            )
            print(f"  Saved catalog 3-view figure to: {fig_file}")
        except Exception as e:
            print(f"  [WARN] Failed to draw 3-view catalog figure: {e}")

    # Step 5: 下载真实目录并对比
    print("\n--- Evaluating vs SCEDC Ground Truth ---")
    truth_catalog = fetch_scedc_catalog(start_time, end_time, min_lat, max_lat, min_lon, max_lon)

    # 过滤 Mw < 1.0
    truth_filtered = [t for t in truth_catalog if t["mag"] >= 1.0]

    matches, fps, fns = match_catalogs(detected_catalog, truth_filtered)

    n_truth = len(truth_filtered)
    n_detected = len(detected_catalog)
    n_matched = len(matches)

    recall = (n_matched / n_truth * 100) if n_truth > 0 else 0
    precision = (n_matched / n_detected * 100) if n_detected > 0 else 0

    print(f"\n{'=' * 40}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'=' * 40}")
    print(f"  Ground Truth Events (M>=1.0) : {n_truth}")
    print(f"  Total Detected Events        : {n_detected}")
    print(f"  True Positives (Matched)     : {n_matched}")
    print(f"  False Negatives (Missed)     : {len(fns)}")
    print(f"  False Positives (Ghosts)     : {len(fps)}")
    print(f"")
    print(f"  Recall    : {recall:.1f}%")
    print(f"  Precision : {precision:.1f}%")

    if matches:
        avg_dist = np.mean([m["dist_err"] for m in matches])
        avg_dep = np.mean([m["depth_err"] for m in matches])
        print(f"  Avg Dist Error : {avg_dist:.2f} km")
        print(f"  Avg Dep Error  : {avg_dep:.2f} km")


if __name__ == "__main__":
    main()
