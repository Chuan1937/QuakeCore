"""
Near-Seismic Blind Detection & Location Pipeline (SCEDC Southern California)
============================================================================
Blind continuous picking (PhaseNet annotate + find_peaks) + REAL-Lite OT
Consensus Association + Local EDT grid search + comparison with SCEDC catalog.

Usage:
    python nearseismic_location_blind_validation.py --event EVENT_ID
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from obspy import UTCDateTime, read_inventory, read as obspy_read
from obspy.geodetics import locations2degrees
from scipy.signal import find_peaks
import pandas as pd
import torch
import seisbench.models as sbm
from tqdm import tqdm

# =================== Device Setup ===================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cpu")
if DEVICE.type == "mps":
    try:
        torch.mps.set_per_process_memory_fraction(0.33)
    except Exception:
        pass
print(f"  [INFO] PyTorch device: {DEVICE}")

# =================== Paths ===================
WAVEFORMS_DIR = os.path.join(PROJECT_ROOT, "Validation", "Waveforms-near-seismic")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Validation", "near-seismic-location", "results_blind")
CATALOG_TXT = os.path.join(PROJECT_ROOT, "Validation", "near-seismic-catalog", "SCEDC_selected.txt")
TAUP_DIR = os.path.join(os.path.dirname(__file__), "taup")

os.makedirs(WAVEFORMS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =================== Helper Functions ===================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def load_scedc_catalog():
    rows = []
    with open(CATALOG_TXT) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "": continue
            parts = line.split()
            if len(parts) < 11: continue
            try:
                rows.append({
                    "date": parts[0] + " " + parts[1], "mw": float(parts[4]),
                    "lat": float(parts[6]), "lon": float(parts[7]),
                    "depth": float(parts[8]), "eventid": parts[10],
                })
            except: continue
    return pd.DataFrame(rows)

def load_stations_from_xml(event_dir, event_id):
    xml_path = os.path.join(event_dir, f"stations_{event_id}.xml")
    inv = read_inventory(xml_path)
    stations = {}
    for network in inv:
        for station in network:
            key = f"{network.code}.{station.code}"
            stations[key] = {
                "latitude": station.latitude, "longitude": station.longitude, "elevation": station.elevation or 0.0,
            }
    return stations

# =================== Velocity Model & TT Cache (Reused directly) ===================

_taup_model = None
_tt_interp = None

def _get_taup():
    global _taup_model
    if _taup_model is None:
        from obspy.taup import TauPyModel
        from obspy.taup.taup_create import build_taup_model
        socal_tvel_path = os.path.join(TAUP_DIR, "socal.tvel")
        socal_npz_path = os.path.join(TAUP_DIR, "socal.npz")
        if not os.path.exists(socal_npz_path):
            os.makedirs(TAUP_DIR, exist_ok=True)
            import obspy.taup
            iasp91_path = os.path.join(os.path.dirname(obspy.taup.__file__), "data", "iasp91.tvel")
            with open(iasp91_path) as f: iasp91_lines = f.readlines()
            mantle_start = next(i+2 for i, line in enumerate(iasp91_lines[2:]) if float(line.split()[0]) > 35.0)
            socal_layers = [(0.0, 5.50, 3.18, 2.60), (5.5, 5.50, 3.18, 2.60), (5.5, 6.30, 3.64, 2.67),
                            (16.0, 6.30, 3.64, 2.67), (16.0, 6.70, 3.87, 2.80), (32.0, 6.70, 3.87, 2.80),
                            (32.0, 7.80, 4.50, 3.30), (60.0, 7.80, 4.50, 3.30)]
            with open(socal_tvel_path, "w") as f:
                f.write("SOUTHERN CALIFORNIA 1D VELOCITY MODEL\n")
                f.write(iasp91_lines[1])
                for dep, vp, vs, rho in socal_layers: f.write(f"{dep:9.3f}{vp:7.4f}{vs:7.4f}{rho:7.4f}\n")
                f.writelines(iasp91_lines[mantle_start:])
            build_taup_model(socal_tvel_path, output_folder=TAUP_DIR)
        _taup_model = TauPyModel(model=socal_npz_path)
    return _taup_model

def _get_tt_interp():
    global _tt_interp
    if _tt_interp is not None: return _tt_interp
    from scipy.interpolate import RegularGridInterpolator
    cache_file = os.path.join(TAUP_DIR, "tt_interp_cache_v2.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        dist_grid = data["dist_grid"]
        depth_grid = data["depth_grid"]
        _tt_interp = {ph: RegularGridInterpolator((dist_grid, depth_grid), data[f"tt_{ph}"],
                    method='linear', bounds_error=False, fill_value=np.inf) for ph in ["P", "S"]}
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
                    if arrs: mat[i, j] = min(a.time for a in arrs)
                except: pass
        mask = np.isfinite(mat)
        if mask.any(): mat[~mask] = np.max(mat[mask]) * 2
        save_dict[f"tt_{ph}"] = mat
        _tt_interp[ph] = RegularGridInterpolator((dist_grid, depth_grid), mat,
                   method='linear', bounds_error=False, fill_value=np.inf)
    np.savez(cache_file, **save_dict)
    return _tt_interp


# =================== REAL-Lite OT Consensus Associator (Pure Physics, No GaMMA) ===================

def associate_by_origin_time(picks, stations, tt_interp, time_tolerance=1.5, min_picks=4):
    """
    Pure physics-based near-seismic phase associator (similar to REAL algorithm).
    Core idea: If a grid point is the true source, all real phases back-extrapolated
    to OT will perfectly overlap within time_tolerance seconds.
    False picks will scatter randomly and be filtered out naturally.

    :param picks: list of pick dicts (from PhaseNet annotate + find_peaks)
    :param stations: dict of station info
    :param tt_interp: pre-computed travel time interpolators
    :param time_tolerance: max OT scatter for a valid cluster (seconds)
    :param min_picks: minimum picks required to form an event
    :return: (associated_picks, init_estimate_dict)
    """
    if len(picks) < min_picks:
        return [], None

    # Determine search grid from station distribution
    sta_lats = [stations[p["station_id"]]["latitude"] for p in picks if p["station_id"] in stations]
    sta_lons = [stations[p["station_id"]]["longitude"] for p in picks if p["station_id"] in stations]
    if not sta_lats:
        return [], None

    min_lat, max_lat = min(sta_lats) - 0.5, max(sta_lats) + 0.5
    min_lon, max_lon = min(sta_lons) - 0.5, max(sta_lons) + 0.5

    lats = np.arange(min_lat, max_lat, 0.08)
    lons = np.arange(min_lon, max_lon, 0.08)
    trial_depth = 10.0  # km, fixed for initial association

    def _vec_dist2deg(lat1, lon1, lat2, lon2):
        la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
        return np.degrees(2 * np.arcsin(np.minimum(np.sqrt(a), 1.0)))

    best_event = None
    max_associated_picks = 0
    best_associated_pick_list = []

    print(f"  [REAL-Lite] Scanning {len(lats)*len(lons)} grid points for OT consensus...")

    for trial_lat in lats:
        for trial_lon in lons:
            origin_times = []
            valid_picks_for_grid = []

            for p in picks:
                sid = p["station_id"]
                if sid not in stations:
                    continue
                s_lat = stations[sid]["latitude"]
                s_lon = stations[sid]["longitude"]
                dist_deg = _vec_dist2deg(s_lat, s_lon, trial_lat, trial_lon)
                phase = p["phase"]

                if phase in tt_interp:
                    t_theo = float(tt_interp[phase]((dist_deg, trial_depth)))
                    if np.isinf(t_theo):
                        continue
                    try:
                        pick_ts = float(p["time"].timestamp)
                    except AttributeError:
                        pick_ts = float(p["time"])
                    ot = pick_ts - t_theo
                    origin_times.append(ot)
                    valid_picks_for_grid.append((ot, p))

            if len(origin_times) < min_picks:
                continue

            origin_times_arr = np.array(sorted(origin_times))

            # Slide a time_tolerance window to find the densest OT cluster
            for i in range(len(origin_times_arr)):
                cluster_mask = (origin_times_arr >= origin_times_arr[i]) & \
                               (origin_times_arr <= origin_times_arr[i] + time_tolerance)
                cluster_count = int(np.sum(cluster_mask))

                if cluster_count > max_associated_picks:
                    max_associated_picks = cluster_count
                    cluster_ot_start = origin_times_arr[i]
                    cluster_ot_end = cluster_ot_start + time_tolerance

                    # Extract real picks belonging to this cluster
                    selected_picks = []
                    for ot, p in valid_picks_for_grid:
                        if cluster_ot_start <= ot <= cluster_ot_end:
                            selected_picks.append(p)

                    # Deduplicate: same station + same phase -> keep highest score
                    unique = {}
                    for p in selected_picks:
                        key = (p["station_id"], p["phase"])
                        if key not in unique or p["score"] > unique[key]["score"]:
                            unique[key] = p
                    final_picks = list(unique.values())

                    best_event = {
                        "latitude": float(trial_lat),
                        "longitude": float(trial_lon),
                        "depth": trial_depth,
                        "time": cluster_ot_start + (time_tolerance / 2.0),
                        "num_picks": len(final_picks),
                    }

    if best_event is None or max_associated_picks < min_picks:
        print("  [WARN] Associator: no reliable event cluster found.")
        return [], None

    print(f"  [REAL-Lite] Associated {best_event['num_picks']} picks (from {len(picks)} raw peaks).")
    print(f"  [REAL-Lite] Init estimate: lat={best_event['latitude']:.3f} lon={best_event['longitude']:.3f}")
    return final_picks, best_event


# =================== Continuous Blind Picking (Annotate + find_peaks) ===================

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

def pick_event_blind_local(event_dir, stations):
    """Dual-model (PhaseNet + EQT) annotate() on continuous traces + ensemble merge."""
    mseed_files = sorted(glob.glob(os.path.join(event_dir, "*.mseed")))
    pht = _get_pht()
    eqt = _get_eqt()
    all_picks = []

    sta_files = {}
    for f in mseed_files:
        net_sta = os.path.basename(f).replace(".mseed", "")
        sta_files[net_sta] = f

    for net_sta, mseed_path in sta_files.items():
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
            if len(_best) < 3:
                continue

            st_3c = st.__class__([_best[c] for c in ("Z", "N", "E")])
            t0 = max(tr.stats.starttime for tr in st_3c)
            t1 = min(tr.stats.endtime for tr in st_3c)
            for tr in st_3c:
                tr.trim(starttime=t0, endtime=t1, pad=True, fill_value=0)
                tr.detrend("demean").detrend("linear").taper(max_percentage=0.05)
                tr.interpolate(sampling_rate=100.0)

            sta_candidates = []
            dt = 0.01
            dist_samples = max(1, int(1.0 / dt))

            # ---- PhaseNet ----
            try:
                anno_pht = pht.annotate(st_3c.copy())
                tr_p_pht = next((tr for tr in anno_pht if tr.stats.channel.endswith('P')), None)
                tr_s_pht = next((tr for tr in anno_pht if tr.stats.channel.endswith('S')), None)
                if tr_p_pht:
                    dt = tr_p_pht.stats.delta
                    dist_samples = max(1, int(1.0 / dt))
                for phase, tr_prob in [("P", tr_p_pht), ("S", tr_s_pht)]:
                    if tr_prob is not None:
                        peaks, props = find_peaks(tr_prob.data, height=0.3, distance=dist_samples)
                        for peak_idx, conf in zip(peaks, props['peak_heights']):
                            pick_time = tr_prob.stats.starttime + (peak_idx * dt)
                            sta_candidates.append({
                                "station_id": net_sta, "phase": phase,
                                "time": pick_time, "time_str": str(pick_time),
                                "score": float(conf), "model": "pht",
                            })
            except Exception:
                pass

            # ---- EQTransformer ----
            try:
                anno_eqt = eqt.annotate(st_3c.copy())
                tr_p_eqt = next((tr for tr in anno_eqt if tr.stats.channel.endswith('P')), None)
                tr_s_eqt = next((tr for tr in anno_eqt if tr.stats.channel.endswith('S')), None)
                for phase, tr_prob in [("P", tr_p_eqt), ("S", tr_s_eqt)]:
                    if tr_prob is not None:
                        peaks, props = find_peaks(tr_prob.data, height=0.3, distance=dist_samples)
                        for peak_idx, conf in zip(peaks, props['peak_heights']):
                            pick_time = tr_prob.stats.starttime + (peak_idx * dt)
                            sta_candidates.append({
                                "station_id": net_sta, "phase": phase,
                                "time": pick_time, "time_str": str(pick_time),
                                "score": float(conf), "model": "eqt",
                            })
            except Exception:
                pass

            # ---- Ensemble merge: deduplicate same-phase picks within 1.0s ----
            sta_candidates.sort(key=lambda x: x["time"])
            merged_sta_picks = []
            for pick in sta_candidates:
                if not merged_sta_picks:
                    merged_sta_picks.append(pick)
                    continue
                last_pick = merged_sta_picks[-1]
                if pick["phase"] == last_pick["phase"] and abs(pick["time"] - last_pick["time"]) < 1.0:
                    # PhaseNet preferred for near-seismic onsets; same model -> keep higher score
                    if pick["model"] == "pht" and last_pick["model"] == "eqt":
                        merged_sta_picks[-1] = pick
                    elif pick["model"] == last_pick["model"] and pick["score"] > last_pick["score"]:
                        merged_sta_picks[-1] = pick
                else:
                    merged_sta_picks.append(pick)

            all_picks.extend(merged_sta_picks)

        except Exception:
            continue

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print(f"  [Blind Picking] Ensemble extracted {len(all_picks)} picks (PhaseNet + EQT).")

    # Run REAL-Lite OT association
    tt_interp = _get_tt_interp()
    return associate_by_origin_time(all_picks, stations, tt_interp, time_tolerance=1.5, min_picks=4)


# =================== Local EDT Grid Search (Blind) ===================

def locate_grid_search_local_blind(picks, stations, init_lat, init_lon, init_depth=10.0):
    """Optimized Local EDT with Global S-P Depth Search and Stage 4 Micro-Polish."""
    sta_coords = {p["station_id"]: (stations[p["station_id"]]["latitude"],
                  stations[p["station_id"]]["longitude"])
                  for p in picks if p["station_id"] in stations}
    if len(sta_coords) < 3:
        return None

    tt_interp = _get_tt_interp()
    t0 = min(float(UTCDateTime(p["time_str"])) for p in picks)
    obs_rel = [(p["station_id"], p["phase"], float(UTCDateTime(p["time_str"])) - t0) for p in picks]

    def _vec_dist2deg(lat1, lon1, lat2, lon2):
        la1, lo1, la2, lo2 = map(lambda x: np.radians(np.asarray(x, dtype=np.float64)), [lat1, lon1, lat2, lon2])
        a = np.sin((la2 - la1)/2)**2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1)/2)**2
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
        return float(lats_f[best_idx]), float(lons_f[best_idx]), float(deps_f[best_idx]), float(misfit[best_idx])

    # Stage 1: Coarse grid
    lat_c = np.arange(init_lat - 0.4, init_lat + 0.401, 0.05)
    lon_c = np.arange(init_lon - 0.4, init_lon + 0.401, 0.05)
    dep_c = np.arange(0, 41, 4)
    best_lat, best_lon, best_dep, mis1 = _edt_grid(obs_rel, sta_coords, lat_c, lon_c, dep_c)

    # Stage 2: Fine grid
    lat_f = np.arange(best_lat - 0.1, best_lat + 0.101, 0.01)
    lon_f = np.arange(best_lon - 0.1, best_lon + 0.101, 0.01)
    dep_f = np.arange(max(0, best_dep - 6), best_dep + 6.1, 1.0)
    best_lat, best_lon, best_dep, mis2 = _edt_grid(obs_rel, sta_coords, lat_f, lon_f, dep_f)

    # Stage 3: Global S-P depth search (unbound, median-based)
    sta_ps = {}
    for sta, ph, rel_t in obs_rel:
        if sta in sta_coords:
            sta_ps.setdefault(sta, {})[ph] = rel_t
    ps_valid = {s: v for s, v in sta_ps.items() if "P" in v and "S" in v}

    if len(ps_valid) >= 2:
        dep_fine = np.arange(0.0, 35.5, 0.5)
        ps_stas = list(ps_valid.keys())
        obs_sp = np.array([ps_valid[s]["S"] - ps_valid[s]["P"] for s in ps_stas])
        ps_dists = _vec_dist2deg([sta_coords[s][0] for s in ps_stas],
                                 [sta_coords[s][1] for s in ps_stas], best_lat, best_lon)

        misfit_sp = np.full(len(dep_fine), np.inf)
        for di, d in enumerate(dep_fine):
            residuals = []
            for si, dist in enumerate(ps_dists):
                tp = float(tt_interp["P"]((dist, float(d))))
                ts = float(tt_interp["S"]((dist, float(d))))
                if not (np.isinf(tp) or np.isinf(ts)):
                    residuals.append(abs(obs_sp[si] - (ts - tp)))
            if residuals:
                misfit_sp[di] = np.median(residuals)  # Robust median vs mean
        best_dep = float(dep_fine[np.argmin(misfit_sp)])

    # Stage 4: Micro-polish
    lat_micro = np.arange(best_lat - 0.03, best_lat + 0.031, 0.005)
    lon_micro = np.arange(best_lon - 0.03, best_lon + 0.031, 0.005)
    dep_micro = np.arange(max(0, best_dep - 2.0), best_dep + 2.1, 0.5)
    best_lat, best_lon, best_dep, mis_final = _edt_grid(obs_rel, sta_coords, lat_micro, lon_micro, dep_micro)

    return {"latitude": best_lat, "longitude": best_lon, "depth": best_dep, "num_picks": len(obs_rel)}


# =================== Main Event Process ===================

def process_local_event_blind(event_id, catalog_row):
    event_dir = os.path.join(WAVEFORMS_DIR, f"Event_{event_id}")
    if not os.path.isdir(event_dir):
        print(f"  [ERROR] Event directory not found: {event_dir}")
        return None

    cat_lat, cat_lon, cat_depth = catalog_row["lat"], catalog_row["lon"], catalog_row["depth"]
    print(f"\n{'#' * 70}")
    print(f"# Event {event_id} [BLIND LOCAL]")
    print(f"# Truth (hidden): lat={cat_lat:.4f} lon={cat_lon:.4f} dep={cat_depth:.1f} km")
    print(f"{'#' * 70}")

    stations = load_stations_from_xml(event_dir, event_id)
    print(f"  {len(stations)} stations loaded")
    work_dir = os.path.join(RESULTS_DIR, f"Event_{event_id}")

    # Step 1: Blind picking & association
    print("\n--- Step 1: Blind continuous picking + REAL-Lite association ---")
    result = pick_event_blind_local(event_dir, stations)
    if result is None or result == ([], None):
        print("  [ERROR] Blind detection failed to form an event.")
        return None
    associated_picks, init_est = result

    if not associated_picks or init_est is None:
        print("  [ERROR] No associated picks or init estimate.")
        return None

    # Step 2: Local EDT location (portable, no external binary dependency)
    print(f"\n--- Step 2: Local EDT grid search (init: {init_est['latitude']:.3f}, {init_est['longitude']:.3f}) ---")
    loc = locate_grid_search_local_blind(associated_picks, stations,
                                         init_est["latitude"], init_est["longitude"])
    if not loc:
        return None

    dist_error = haversine_km(cat_lat, cat_lon, loc["latitude"], loc["longitude"])
    depth_error = abs(loc["depth"] - cat_depth)

    print(f"\n  {'=' * 40}\n  RESULTS COMPARISON\n  {'=' * 40}")
    print(f"  Catalog : lat={cat_lat:.4f} lon={cat_lon:.4f} Z={cat_depth:.1f} km")
    print(f"  Located : lat={loc['latitude']:.4f} lon={loc['longitude']:.4f} Z={loc['depth']:.1f} km")
    print(f"  Dist Err: {dist_error:.2f} km")
    print(f"  Dep Err : {depth_error:.2f} km")

    return {
        "event_id": str(event_id), "catalog_mw": float(catalog_row["mw"]),
        "catalog": {"latitude": cat_lat, "longitude": cat_lon, "depth_km": cat_depth},
        "located": {"latitude": round(loc["latitude"], 4), "longitude": round(loc["longitude"], 4),
                    "depth_km": round(loc["depth"], 2)},
        "errors": {"distance_km": round(dist_error, 2), "depth_km": round(depth_error, 2)},
        "quality": {"num_picks": loc.get("num_picks", len(associated_picks))},
        "method": "blind_real_lite_edt",
    }


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, default=None, help="Single event ID (if not specified, process all)")
    args = parser.parse_args()

    locator_name = "Python EDT + S-P"
    print("=" * 70)
    print("  Near-Seismic Blind Detection & Location Pipeline")
    print("  Velocity model: SoCal (Hadley-Kanamori 1977)")
    print("  Picking: PhaseNet + EQT ensemble annotate (continuous)")
    print("  Association: REAL-Lite OT consensus (pure physics, no ML)")
    print(f"  Locator: {locator_name}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    _get_taup()
    _get_tt_interp()

    catalog = load_scedc_catalog()

    if args.event:
        # 单个事件模式
        row = catalog[catalog["eventid"].astype(str) == args.event]
        if row.empty:
            print(f"Event {args.event} not found in catalog.")
            return
        row = row.iloc[0]
        result = process_local_event_blind(args.event, row)
        if result:
            work_dir = os.path.join(RESULTS_DIR, f"Event_{args.event}")
            os.makedirs(work_dir, exist_ok=True)
            out_path = os.path.join(work_dir, "comparison.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n  Saved: {out_path}")
    else:
        # 批量处理所有事件模式
        event_dirs = sorted([d for d in os.listdir(WAVEFORMS_DIR)
                           if os.path.isdir(os.path.join(WAVEFORMS_DIR, d)) and d.startswith("Event_")])

        # 过滤出需要处理的事件
        to_process = []
        for event_dir in event_dirs:
            event_id = event_dir.replace("Event_", "")
            result_path = os.path.join(RESULTS_DIR, f"Event_{event_id}", "comparison.json")
            if os.path.exists(result_path):
                continue
            download_done = os.path.join(WAVEFORMS_DIR, event_dir, "download_done.txt")
            if not os.path.exists(download_done):
                continue
            row = catalog[catalog["eventid"].astype(str) == event_id]
            if row.empty:
                continue
            to_process.append((event_id, row.iloc[0]))

        processed = 0
        errors = 0

        print(f"\n--- Batch Processing ({len(to_process)} events) ---")
        for i, (event_id, row) in enumerate(tqdm(to_process, desc="  Processing", unit="ev")):
            try:
                result = process_local_event_blind(event_id, row)
                if result:
                    work_dir = os.path.join(RESULTS_DIR, f"Event_{event_id}")
                    os.makedirs(work_dir, exist_ok=True)
                    out_path = os.path.join(work_dir, "comparison.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1

        print(f"\n{'=' * 50}")
        print(f"  Batch Complete!")
        print(f"  Processed: {processed}")
        print(f"  Errors:    {errors}")

if __name__ == "__main__":
    main()