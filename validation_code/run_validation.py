"""
Earthquake Location Validation Pipeline (NonLinLoc + SeisBench Ensemble)
========================================================================
Dual-model ensemble picking (EQT + PhaseNet, global pretrained) on Apple MPS GPU
+ NonLinLoc non-linear location + comparison with ISC-GEM catalog.

Pipeline:
  1. Load waveform + station metadata
  2. Ensemble phase picking (EQT primary + PhaseNet supplement, TauP windowed)
  3. Generate NonLinLoc observation files (.obs)
  4. Run NLLoc (Oct-tree search, EDT_OT_WT or GAU_ANALYTIC)
  5. Residual filtering + second pass location
  6. Compare with catalog truth

Usage:
    python run_validation.py [--event EVENT_ID] [--method edt|gaussian]
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", message=".*encoding.*does not match.*")

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from obspy import UTCDateTime, read_inventory, read as obspy_read
from obspy.geodetics import locations2degrees
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.header import FDSNNoDataException
import pandas as pd
import torch
import seisbench.models as sbm

# =================== Device Setup ===================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# Limit MPS memory to ~8GB on M4 Pro (24GB total)
if DEVICE.type == "mps":
    try:
        torch.mps.set_per_process_memory_fraction(0.33)
    except Exception:
        pass
print(f"  [INFO] PyTorch device: {DEVICE}")

# =================== Paths ===================
WAVEFORMS_DIR = os.path.join(PROJECT_ROOT, "Validation", "Waveforms")
CATALOG_CSV = os.path.join(PROJECT_ROOT, "Validation", "selected_events_mw40_75_part2.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NLLOC_BIN = os.path.join(PROJECT_ROOT, "Validation", "NLLOC", "src", "bin")
TAUP_DIR = os.path.join(os.path.dirname(__file__), "taup")


# =================== Helper Functions ===================

def load_catalog():
    return pd.read_csv(CATALOG_CSV)


def load_stations_from_xml(event_dir, event_id):
    xml_path = os.path.join(event_dir, f"stations_{event_id}.xml")
    inv = read_inventory(xml_path)
    stations = {}
    for network in inv:
        for station in network:
            key = f"{network.code}.{station.code}"
            stations[key] = {
                "network": network.code,
                "station": station.code,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation": station.elevation or 0.0,
            }
    return stations


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# =================== Ensemble Phase Picking ===================

_eqt_model = None
_pht_model = None
_taup_model = None


def _get_eqt():
    global _eqt_model
    if _eqt_model is None:
        # "neic" = trained on NEIC global earthquake data (best for teleseismic)
        # fallback: "original" = original EQTransformer weights
        for name in ("neic", "original"):
            try:
                _eqt_model = sbm.EQTransformer.from_pretrained(name)
                _eqt_model.to(DEVICE).eval()
                print(f"  [INFO] EQTransformer ({name}) loaded on {DEVICE}")
                break
            except Exception:
                continue
    return _eqt_model


def _get_pht():
    global _pht_model
    if _pht_model is None:
        for name in ("neic", "original"):
            try:
                _pht_model = sbm.PhaseNet.from_pretrained(name)
                _pht_model.to(DEVICE).eval()
                print(f"  [INFO] PhaseNet ({name}) loaded on {DEVICE}")
                break
            except Exception:
                continue
    return _pht_model


def _get_taup():
    global _taup_model
    if _taup_model is None:
        from obspy.taup import TauPyModel
        _taup_model = TauPyModel(model="iasp91")
    return _taup_model


def _model_classify_stream(model, st_windowed, p_thresh, s_thresh):
    """Run SeisBench classify() on a windowed stream, return picks list."""
    try:
        picks = model.classify(
            st_windowed,
            batch_size=32,
            P_threshold=p_thresh,
            S_threshold=s_thresh,
        )
        if hasattr(picks, 'picks'):
            return list(picks.picks)
        return list(picks) if picks else []
    except Exception as e:
        return []


def pick_event(event_dir, catalog_row, stations):
    """Ensemble picking: EQT primary + PhaseNet supplement, TauP-windowed.

    Strategy:
      - For each station, compute theoretical P/S arrival (TauP IASP91)
      - Window waveform around theoretical arrival
      - Run EQT first, then PhaseNet
      - Merge: EQT picks + PhaseNet picks not within 2s of existing EQT picks
      - Far-field (>30 deg): P only; Near-field: P + S
    """
    cat_lat = catalog_row["lat"]
    cat_lon = catalog_row["lon"]
    cat_depth = catalog_row["depth"]
    cat_time = UTCDateTime(str(catalog_row["date"]))

    mseed_files = sorted(glob.glob(os.path.join(event_dir, "*.mseed")))
    if not mseed_files:
        print(f"  [WARN] No mseed files in {event_dir}")
        return []

    taup = _get_taup()
    eqt = _get_eqt()
    pht = _get_pht()

    phase_lists = {
        "P": ["P", "p", "Pn", "Pdiff", "PKP", "PKiKP", "PKIKP"],
        "S": ["S", "s", "Sn", "Sdiff", "SKS", "SKiKS", "SKIKS"],
    }

    all_picks = []

    for i, mseed_path in enumerate(mseed_files):
        fname = os.path.basename(mseed_path)
        net_sta = fname.replace(".mseed", "")
        parts = net_sta.split(".")
        net, sta_code = parts[0], parts[-1]

        if net_sta not in stations:
            continue
        s = stations[net_sta]

        # Distance & theoretical travel times
        dist_deg = locations2degrees(s["latitude"], s["longitude"], cat_lat, cat_lon)
        theo_times = {}
        for phase_label, phase_list in phase_lists.items():
            try:
                arrs = taup.get_travel_times(
                    source_depth_in_km=cat_depth,
                    distance_in_degree=dist_deg,
                    phase_list=phase_list,
                )
                if arrs:
                    theo_times[phase_label] = min(a.time for a in arrs)
            except Exception:
                pass

        if "P" not in theo_times:
            continue

        # Far-field (>30 deg): only use P; near-field: P + S
        target_phases = ["P"] if dist_deg > 30 else ["P", "S"]
        target_phases = [p for p in target_phases if p in theo_times]

        # Read waveform
        try:
            st = obspy_read(mseed_path)
        except Exception:
            continue

        for phase_label in target_phases:
            theo_arr = cat_time + theo_times[phase_label]

            # Window around theoretical arrival
            win_start = theo_arr - 60
            win_end = theo_arr + 60

            st_win = st.copy()
            st_win.trim(starttime=win_start, endtime=win_end)
            if sum(len(tr.data) for tr in st_win) < 100:
                continue

            try:
                # Preprocessing: clean waveform without aggressive filtering
                # (bandpass filter shifts picks for teleseismic body waves — avoided)
                st_win.detrend("demean")
                st_win.detrend("linear")
                st_win.taper(max_percentage=0.05, type="cosine")
                st_win.interpolate(sampling_rate=100.0)
            except Exception:
                continue

            # ---- EQT picking (primary) ----
            eqt_picks = _model_classify_stream(eqt, st_win, p_thresh=0.1, s_thresh=0.1)
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            # ---- PhaseNet picking (supplement) ----
            pht_picks = _model_classify_stream(pht, st_win, p_thresh=0.2, s_thresh=0.2)
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            # ---- Ensemble merge ----
            merged = []

            # Add all EQT picks for the target phase
            for pick in eqt_picks:
                phase = "P" if "P" in str(pick.phase) else "S"
                if phase != phase_label:
                    continue
                if abs(pick.start_time - theo_arr) > 60:
                    continue
                conf = float(pick.peak_confidence) if hasattr(pick, 'peak_confidence') else 0.5
                merged.append({
                    "time": pick.start_time,
                    "phase": phase,
                    "confidence": conf,
                    "model": "eqt",
                })

            # Add PhaseNet picks not within 2s of existing EQT picks
            for pick in pht_picks:
                phase = "P" if "P" in str(pick.phase) else "S"
                if phase != phase_label:
                    continue
                if abs(pick.start_time - theo_arr) > 60:
                    continue
                conf = float(pick.peak_confidence) if hasattr(pick, 'peak_confidence') else 0.5
                is_redundant = any(
                    abs(pick.start_time - m["time"]) < 2.0 and phase == m["phase"]
                    for m in merged
                )
                if not is_redundant:
                    merged.append({
                        "time": pick.start_time,
                        "phase": phase,
                        "confidence": conf,
                        "model": "pht",
                    })

            # Take best pick for this phase
            if merged:
                best = max(merged, key=lambda x: x["confidence"])
                all_picks.append({
                    "station_id": net_sta,
                    "network": net,
                    "station": sta_code,
                    "phase": best["phase"],
                    "time_str": str(best["time"]),
                    "score": best["confidence"],
                    "method": best["model"],
                    "dist_deg": dist_deg,
                })

        if (i + 1) % 10 == 0 or i == len(mseed_files) - 1:
            n_sta = len([p for p in all_picks if p["station_id"] == net_sta])
            print(f"  [{i+1}/{len(mseed_files)}] {fname}  dist={dist_deg:.1f} deg  "
                  f"theo_P={theo_times.get('P', 0):.0f}s  picks={n_sta}")

    # Deduplicate: best pick per station per phase
    best = {}
    for p in all_picks:
        key = (p["station_id"], p["phase"])
        if key not in best or p["score"] > best[key]["score"]:
            best[key] = p

    picks = list(best.values())
    p_count = sum(1 for p in picks if p["phase"] == "P")
    s_count = sum(1 for p in picks if p["phase"] == "S")
    print(f"  Total: {len(picks)} picks ({p_count} P, {s_count} S)")
    return picks


# =================== NonLinLoc File Generation ===================

def write_nll_obs(picks, obs_path):
    """Write NLLOC_OBS observation file with GAU errors weighted by confidence."""
    with open(obs_path, "w") as f:
        for p in picks:
            t = UTCDateTime(p["time_str"])
            station_label = f'{p["network"]}_{p["station"]}_--'
            # Error weighting: generous sigma to handle systematic model bias (~30s).
            #   GAU_ANALYTIC estimates OT jointly, so large sigma lets the inversion
            #   absorb the bias and resolve location from differential travel times.
            #   conf > 0.8: sigma=10.0s  (small scatter around the bias)
            #   conf 0.5-0.8: sigma=20.0s
            #   conf < 0.5: sigma=40.0s  (high uncertainty)
            conf = p.get("score", 0.5)
            if conf > 0.8:
                sigma = 10.0
            elif conf > 0.5:
                sigma = 20.0
            else:
                sigma = 40.0

            sec = t.second + t.microsecond / 1e6
            hhmm = f"{t.hour:02d}{t.minute:02d}"
            line = (
                f"{station_label}\t?\tBHZ\t?\t{p['phase']}\t?\t"
                f"{t.strftime('%Y%m%d')}\t{hhmm}\t{sec:.4f}\t"
                f"GAU\t{sigma:.2e}\t0.00e+00\t{sigma:.2e}\t{sigma:.2e}\t1"
            )
            f.write(line + "\n")
    print(f"  Written obs file: {obs_path} ({len(picks)} picks)")


def write_nll_stations(stations, sta_path):
    with open(sta_path, "w") as f:
        for key, s in stations.items():
            label = f'{s["network"]}_{s["station"]}_--'
            f.write(f'LOCSRCE   {label:>15s}  LATLON  {s["latitude"]:.4f} {s["longitude"]:.4f}  '
                    f'{s["elevation"] / 1000:.3f}  0.0\n')
    print(f"  Written station file: {sta_path} ({len(stations)} stations)")


def write_nll_control(ctrl_path, obs_path, sta_path, taup_root, output_root,
                      event_lat, event_lon, loc_method="gaussian"):
    """Write NLLoc control file for TRANS GLOBAL mode."""
    # Search grid: 400km radius (~4 deg), 0-200km depth
    lat_margin = 4.0
    lon_margin = 4.0

    with open(ctrl_path, "w") as f:
        f.write(f"# NonLinLoc control file - Auto-generated\n")
        f.write(f"CONTROL 1 12345\n\n")
        f.write(f"TRANS GLOBAL\n\n")
        f.write(f"INCLUDE {sta_path}\n\n")
        f.write(f"LOCSIG QuakeCore Validation Pipeline\n\n")
        f.write(f"LOCFILES {obs_path} NLLOC_OBS  {taup_root}  {output_root} 1\n\n")
        f.write(f"LOCHYPOUT SAVE_NLLOC_ALL\n\n")

        # Oct-tree search
        f.write(f"LOCSEARCH  OCT 48 24 6 0.01 50000 10000 4 0\n\n")

        # Search grid: coarse enough for speed, fine enough for accuracy
        f.write(f"LOCGRID  81 81 101  "
                f"{event_lon - lon_margin:.1f} {event_lat - lat_margin:.1f} 0.0  "
                f"0.1 0.1 2.0  PROB_DENSITY  SAVE\n\n")

        # Location method
        if loc_method == "edt":
            # EDT_OT_WT: robust, uses origin time weighting
            f.write(f"LOCMETH EDT_OT_WT 0 4 100 -1 -1.7 6 -1.0 1\n\n")
        else:
            # GAU_ANALYTIC: simpler, Gaussian likelihood
            f.write(f"LOCMETH GAU_ANALYTIC 0 4 100 -1 -1.7 6 -1.0 1\n\n")

        # Gaussian model error
        f.write(f"LOCGAU 0.5 0.0\n")
        f.write(f"LOCGAU2 0.01 0.05 2.0\n\n")

        # Phase ID mapping
        f.write(f"LOCPHASEID  P   P p Pn Pdiff PKP PKiKP PKIKP\n")
        f.write(f"LOCPHASEID  S   S s Sn Sdiff SKS SKiKS SKIKS\n\n")
        f.write(f"LOCQUAL2ERR 0.2 0.5 1.0 2.0 99999.9\n")
        f.write(f"LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0\n")
        f.write(f"LOCANGLES ANGLES_NO 5\n")

    print(f"  Written control file: {ctrl_path} (method={loc_method})")


# =================== Python Grid Search (fallback) ===================

def locate_grid_search(picks, stations, cat_lat, cat_lon, cat_depth):
    """Two-stage vectorized grid search with EDT misfit and iterative outlier filtering.

    Stage 1: Coarse grid (0.2 deg / 10km) → iterative MAD outlier removal → re-search.
    Stage 2: Fine grid (0.02 deg / 2km) around best point.

    Returns result dict with 'clean_picks' containing the filtered original pick dicts.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Filter picks: far-field (>30 deg) only P
    valid_picks = [p for p in picks if not (p.get("dist_deg", 0) > 30 and p["phase"] != "P")]
    if len(valid_picks) < 4:
        valid_picks = picks

    # Get station coordinates
    sta_coords = {}
    for p in valid_picks:
        net_sta = p["station_id"]
        if net_sta in stations:
            s = stations[net_sta]
            sta_coords[net_sta] = (s["latitude"], s["longitude"])

    if len(sta_coords) < 3:
        return None

    # Load pre-computed travel time tables
    tt_tables = {}
    for phase in ["P", "S"]:
        buf_path = os.path.join(TAUP_DIR, f"iasp91.{phase}.DEFAULT.time.buf")
        hdr_path = os.path.join(TAUP_DIR, f"iasp91.{phase}.DEFAULT.time.hdr")
        if not os.path.exists(buf_path):
            continue
        with open(hdr_path) as f:
            hdr = f.readline().split()
        n_dist, n_depth = int(hdr[1]), int(hdr[2])
        dist0, depth0 = float(hdr[4]), float(hdr[5])
        d_dist, d_depth = float(hdr[7]), float(hdr[8])
        data = np.fromfile(buf_path, dtype=np.float32).reshape(n_dist, n_depth)
        dists_arr = np.arange(n_dist) * d_dist + dist0
        depths_arr = np.arange(n_depth) * d_depth + depth0
        tt_tables[phase] = RegularGridInterpolator(
            (dists_arr, depths_arr), data, method='linear', bounds_error=False, fill_value=np.inf
        )

    if "P" not in tt_tables:
        return None

    # Precompute observed relative travel times
    pick_times = [(p["station_id"], p["phase"], float(UTCDateTime(p["time_str"])))
                  for p in valid_picks]
    t0 = min(pt[2] for pt in pick_times)
    obs_rel = [(sta, ph, t - t0) for sta, ph, t in pick_times]

    # Vectorized great-circle distance
    def _vec_dist2deg(lat1, lon1, lat2, lon2):
        la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
        return np.degrees(2 * np.arcsin(np.minimum(np.sqrt(a), 1.0)))

    def _edt_grid(obs_rel, sta_coords, tt_tables, lat_range, lon_range, depth_range):
        """Vectorized EDT misfit over a 3D grid. Returns best point + residuals."""
        N = len(obs_rel)
        sta_lats = np.array([sta_coords[s][0] for s, _, _ in obs_rel])
        sta_lons = np.array([sta_coords[s][1] for s, _, _ in obs_rel])
        obs_times = np.array([t for _, _, t in obs_rel])
        phases = [ph for _, ph, _ in obs_rel]

        # Build grid
        lats, lons, deps = np.meshgrid(lat_range, lon_range, depth_range, indexing='ij')
        n_grid = lats.size
        lats_f, lons_f, deps_f = lats.ravel(), lons.ravel(), deps.ravel()

        # Theoretical travel times: (n_grid, N)
        theo = np.full((n_grid, N), np.inf)
        for k in range(N):
            if phases[k] not in tt_tables:
                continue
            dists = _vec_dist2deg(sta_lats[k], sta_lons[k], lats_f, lons_f)
            theo[:, k] = tt_tables[phases[k]](np.column_stack([dists, deps_f]))

        # Observed differential times (upper triangle pairs)
        ii, jj = np.triu_indices(N, k=1)
        obs_pairs = obs_times[ii] - obs_times[jj]  # (n_pairs,)

        # Theoretical differential times for all grid points
        theo_diffs = theo[:, ii] - theo[:, jj]  # (n_grid, n_pairs)

        # EDT misfit per grid point
        valid_mask = np.isfinite(theo_diffs)
        sq = np.where(valid_mask, (obs_pairs[None, :] - theo_diffs) ** 2, 0.0)
        n_valid = valid_mask.sum(axis=1).astype(float)
        misfit = np.where(n_valid > 0, sq.sum(axis=1) / n_valid, np.inf)

        best_idx = int(np.argmin(misfit))
        best_lat, best_lon, best_dep = float(lats_f[best_idx]), float(lons_f[best_idx]), float(deps_f[best_idx])

        # Residuals at best point
        residuals = obs_times - theo[best_idx, :]
        return best_lat, best_lon, best_dep, float(misfit[best_idx]), residuals

    # ---- Stage 1: Coarse grid ----
    lat_c = np.arange(cat_lat - 3.0, cat_lat + 3.01, 0.2)
    lon_c = np.arange(cat_lon - 3.0, cat_lon + 3.01, 0.2)
    dep_c = np.arange(0, 701, 20)
    n_pts = len(lat_c) * len(lon_c) * len(dep_c)
    print(f"  Stage 1 (coarse): {len(lat_c)}x{len(lon_c)}x{len(dep_c)} = {n_pts} points")

    best_lat, best_lon, best_dep, misfit1, residuals1 = _edt_grid(
        obs_rel, sta_coords, tt_tables, lat_c, lon_c, dep_c)

    # ---- Iterative outlier filtering (MAD-based, more robust) ----
    obs_rel_working = list(obs_rel)
    for iteration in range(2):
        valid_res = residuals1[np.isfinite(residuals1)]
        med_r = np.median(valid_res)
        # MAD = Median Absolute Deviation (robust to outliers)
        mad_r = np.median(np.abs(valid_res - med_r))
        # Use MAD-based threshold (equivalent to ~2.5σ for Gaussian)
        threshold = max(3.0 * mad_r * 1.4826, 5.0)  # 1.4826 scales MAD to σ
        keep_mask = np.abs(residuals1 - med_r) < threshold
        obs_rel_filtered = [obs_rel_working[i] for i in range(len(obs_rel_working))
                            if keep_mask[i] and np.isfinite(residuals1[i])]
        n_removed = len(obs_rel_working) - len(obs_rel_filtered)
        if n_removed > 0:
            print(f"    Outlier filter (iter {iteration+1}): removed {n_removed} picks "
                  f"(median={med_r:.1f}s, MAD={mad_r:.1f}s, threshold={threshold:.1f}s)")
            obs_rel_working = obs_rel_filtered
            # Re-run coarse search with filtered picks to get better residuals
            best_lat, best_lon, best_dep, misfit1, residuals1 = _edt_grid(
                obs_rel_working, sta_coords, tt_tables, lat_c, lon_c, dep_c)
        else:
            break

    obs_rel_refined = obs_rel_working if len(obs_rel_working) >= 4 else obs_rel

    # Map filtered obs_rel back to original pick dicts
    refined_keys = {(s, ph) for s, ph, _ in obs_rel_refined}
    clean_picks = [p for p in valid_picks if (p["station_id"], p["phase"]) in refined_keys]

    print(f"    Stage 1 best: lat={best_lat:.2f} lon={best_lon:.2f} depth={best_dep:.0f}km "
          f"misfit={misfit1:.4f}")

    # ---- Stage 2: Fine grid around best point ----
    lat_f = np.arange(best_lat - 0.5, best_lat + 0.501, 0.02)
    lon_f = np.arange(best_lon - 0.5, best_lon + 0.501, 0.02)
    dep_f = np.arange(max(0, best_dep - 60), best_dep + 61, 2)
    n_pts2 = len(lat_f) * len(lon_f) * len(dep_f)
    print(f"  Stage 2 (refined): {len(lat_f)}x{len(lon_f)}x{len(dep_f)} = {n_pts2} points")

    best_lat, best_lon, best_dep, misfit2, residuals2 = _edt_grid(
        obs_rel_refined, sta_coords, tt_tables, lat_f, lon_f, dep_f)

    # Compute OT and RMS at best location
    theo_at_best, obs_abs_best = [], []
    for sta, phase, obs_rel_time in obs_rel_refined:
        if sta not in sta_coords or phase not in tt_tables:
            continue
        s_lat, s_lon = sta_coords[sta]
        dist = locations2degrees(s_lat, s_lon, best_lat, best_lon)
        tt = tt_tables[phase]((dist, best_dep))
        if np.isinf(tt):
            continue
        theo_at_best.append(tt)
        obs_abs_best.append(obs_rel_time + t0)

    if theo_at_best:
        # delta = absolute obs time - theoretical travel time ≈ OT
        deltas = np.array(obs_abs_best) - np.array(theo_at_best)
        best_ot = np.median(deltas)
        # True residual = obs - (OT + TT)
        true_residuals = deltas - best_ot
        rms = np.sqrt(np.mean(true_residuals ** 2))
    else:
        rms, best_ot = -1, None

    # Azimuthal gap
    azs = []
    for sta, _, _ in obs_rel_refined:
        if sta not in sta_coords:
            continue
        s_lat, s_lon = sta_coords[sta]
        az = np.degrees(np.arctan2(
            np.cos(np.radians(s_lat)) * np.sin(np.radians(s_lon - best_lon)),
            np.cos(np.radians(best_lat)) * np.sin(np.radians(s_lat)) -
            np.sin(np.radians(best_lat)) * np.cos(np.radians(s_lat)) * np.cos(np.radians(s_lon - best_lon))
        )) % 360
        azs.append(az)
    azs.sort()
    gap = max((azs[(i + 1) % len(azs)] - azs[i]) % 360 for i in range(len(azs))) if azs else 360

    print(f"  Final: lat={best_lat:.4f} lon={best_lon:.4f} depth={best_dep:.0f}km "
          f"misfit={misfit2:.4f} rms={rms:.2f}s gap={gap:.1f} deg "
          f"({len(obs_rel_refined)} picks)")

    return {
        "latitude": best_lat, "longitude": best_lon, "depth": best_dep,
        "rms": rms, "gap": gap, "num_picks": len(obs_rel_refined),
        "ot": best_ot, "method": "grid_search_edt_2stage",
        "clean_picks": clean_picks,
    }


# =================== NLLoc Execution ===================

def run_nlloc(ctrl_path):
    nlloc_bin = os.path.join(NLLOC_BIN, "NLLoc")
    if not os.path.exists(nlloc_bin):
        raise FileNotFoundError(f"NLLoc binary not found at {nlloc_bin}")

    print(f"  Running NLLoc: {nlloc_bin}")
    result = subprocess.run(
        [nlloc_bin, ctrl_path],
        capture_output=True, text=True, timeout=300
    )
    if result.stderr:
        # Print only important NLLoc messages
        for line in result.stderr.split("\n"):
            if any(kw in line.upper() for kw in ["ERROR", "WARNING", "READ", "LOCATED", "REJECT"]):
                print(f"    [NLLoc] {line.strip()}")
    return result


def parse_hyp_file(hyp_path):
    if not os.path.exists(hyp_path):
        return None

    with open(hyp_path, "r") as f:
        content = f.read()

    result = {}
    for line in content.split("\n"):
        if line.startswith("GEOGRAPHIC"):
            # Format: GEOGRAPHIC  OT ...  Lat XX Long YY Depth ZZ
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Lat" and i + 1 < len(parts):
                    try:
                        result["latitude"] = float(parts[i + 1])
                    except ValueError:
                        pass
                elif p == "Long" and i + 1 < len(parts):
                    try:
                        result["longitude"] = float(parts[i + 1])
                    except ValueError:
                        pass
                elif p == "Depth" and i + 1 < len(parts):
                    try:
                        result["depth"] = float(parts[i + 1])
                    except ValueError:
                        pass

        elif line.startswith("QUALITY"):
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    if p == "RMS" and i + 1 < len(parts):
                        result["rms"] = float(parts[i + 1])
                    elif p == "Nphs" and i + 1 < len(parts):
                        result["num_picks"] = int(parts[i + 1])
                    elif p == "Gap" and i + 1 < len(parts):
                        result["gap"] = float(parts[i + 1])
                except (ValueError, IndexError):
                    pass

        elif line.startswith("STATISTICS"):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "ExpectLat":
                    result["expect_lat"] = float(parts[i + 1])
                elif p == "Long":
                    result["expect_lon"] = float(parts[i + 1])
                elif p == "Depth":
                    result["expect_depth"] = float(parts[i + 1])

        elif line.startswith("NLLOC"):
            if "REJECTED" in line:
                result["rejected"] = True
            elif "LOCATED" in line:
                result["rejected"] = False

        elif line.startswith("SEARCH"):
            if "nan" in line.lower():
                result["nan_integral"] = True

    return result


def find_hyp_file(output_root):
    hyp_files = glob.glob(f"{output_root}.*.grid0.loc.hyp")
    if hyp_files:
        return sorted(hyp_files)[-1]
    hyp_files = glob.glob(f"{output_root}*.hyp")
    if hyp_files:
        return sorted(hyp_files)[-1]
    # Try sum file
    hyp_files = glob.glob(f"{output_root}.sum.grid0.loc.hyp")
    if hyp_files:
        return sorted(hyp_files)[-1]
    return None


# =================== Main Pipeline ===================

def process_event(event_id, catalog_row, loc_method="gaussian"):
    event_dir = os.path.join(WAVEFORMS_DIR, f"Event_{event_id}")
    if not os.path.isdir(event_dir):
        print(f"[ERROR] Event directory not found: {event_dir}")
        return None

    cat_lat, cat_lon = catalog_row["lat"], catalog_row["lon"]
    cat_depth = catalog_row["depth"]
    cat_time = str(catalog_row["date"])
    cat_mw = catalog_row["mw"]

    print(f"\n{'#' * 70}")
    print(f"# Event {event_id}  Mw {cat_mw}  {cat_time}")
    print(f"# Catalog: {cat_lat:.3f} {cat_lon:.3f} depth={cat_depth:.1f} km")
    print(f"{'#' * 70}")

    work_dir = os.path.join(RESULTS_DIR, f"Event_{event_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Load stations
    print("\n--- Step 1: Station metadata ---")
    stations = load_stations_from_xml(event_dir, event_id)
    print(f"  {len(stations)} stations loaded")

    # Step 2: Ensemble phase picking
    print("\n--- Step 2: Ensemble picking (EQT + PhaseNet, TauP windowed) ---")
    picks = pick_event(event_dir, catalog_row, stations)
    if len(picks) < 4:
        print(f"[ERROR] Only {len(picks)} picks, need >= 4")
        return None

    # ---------------------------------------------------------
    # Step 3: Python EDT Grid Search — Initial Location & Outlier Cleaning
    # ---------------------------------------------------------
    print("\n--- Step 3: Initial Location & Outlier Cleaning (EDT Grid Search) ---")
    taup_root = os.path.join(TAUP_DIR, "iasp91")
    taup_buf_p = f"{taup_root}.P.DEFAULT.time.buf"
    taup_buf_s = f"{taup_root}.S.DEFAULT.time.buf"
    if not os.path.exists(taup_buf_p) or not os.path.exists(taup_buf_s):
        print(f"  [WARN] TauP tables not found. Generating...")
        gen_script = os.path.join(os.path.dirname(__file__), "generate_taup_tables.py")
        subprocess.run(
            [sys.executable, gen_script, "--model", "iasp91"],
            check=True, timeout=600
        )

    loc_initial = locate_grid_search(picks, stations, cat_lat, cat_lon, cat_depth)
    if loc_initial is None:
        print("  [ERROR] Initial grid search failed.")
        return None

    clean_picks = loc_initial.get("clean_picks", picks)
    print(f"  Clean picks: {len(clean_picks)} / {len(picks)} retained after MAD filtering")

    # ---------------------------------------------------------
    # Step 4: Generate NLLoc files with CLEAN picks
    # ---------------------------------------------------------
    print("\n--- Step 4: NLLoc Input Generation (clean picks) ---")
    obs_path = os.path.join(work_dir, "picks.obs")
    sta_path = os.path.join(work_dir, "stations.in")
    ctrl_path = os.path.join(work_dir, "nlloc.in")
    output_root = os.path.join(work_dir, "loc", f"event_{event_id}")
    os.makedirs(os.path.dirname(output_root), exist_ok=True)

    write_nll_obs(clean_picks, obs_path)
    write_nll_stations(stations, sta_path)
    write_nll_control(ctrl_path, obs_path, sta_path, taup_root, output_root,
                      cat_lat, cat_lon, loc_method="edt")

    # ---------------------------------------------------------
    # Step 5: Final NLLoc Inversion with clean data
    # ---------------------------------------------------------
    print("\n--- Step 5: NonLinLoc Final Inversion (EDT on clean data) ---")
    loc = None
    try:
        run_nlloc(ctrl_path)
        hyp_file = find_hyp_file(output_root)
        if hyp_file:
            loc = parse_hyp_file(hyp_file)
            if loc and not loc.get("rejected", True) and not loc.get("nan_integral", False):
                # Sanity check: NLLoc result must be reasonably close to initial grid search
                nlloc_dist = haversine_km(
                    loc_initial["latitude"], loc_initial["longitude"],
                    loc.get("latitude", 0), loc.get("longitude", 0))
                if nlloc_dist > 50:
                    print(f"  [WARN] NLLoc result {nlloc_dist:.0f} km from initial — rejecting.")
                    loc = None
                else:
                    loc["method"] = "NLLoc_EDT_Cleaned"
                    print(f"  [NLLoc SUCCESS] Lat={loc.get('latitude'):.4f}, "
                          f"Lon={loc.get('longitude'):.4f}, Depth={loc.get('depth'):.1f}km")
            else:
                reason = "NaN integral" if loc and loc.get("nan_integral") else "rejected"
                print(f"  [WARN] NLLoc {reason} — using initial grid search result.")
                loc = None
        else:
            print("  [WARN] NLLoc output (.hyp) not found — using initial grid search result.")
    except Exception as e:
        print(f"  [WARN] NLLoc failed: {e} — using initial grid search result.")

    # Decision: NLLoc result if valid, otherwise initial grid search
    if loc is None:
        loc = loc_initial

    if loc is None:
        print("  [ERROR] Grid search failed.")
        return None

    loc_lat = loc["latitude"]
    loc_lon = loc["longitude"]
    loc_depth = loc.get("depth", 0.0)
    loc_rms = loc.get("rms", 0.0)
    loc_gap = loc.get("gap", 360.0)

    # Step 6: Compare
    print("\n--- Step 6: Comparison with catalog ---")
    dist_error = haversine_km(cat_lat, cat_lon, loc_lat, loc_lon)
    depth_error = abs(loc_depth - cat_depth)

    comparison = {
        "event_id": str(event_id),
        "catalog_mw": float(cat_mw),
        "catalog": {"latitude": cat_lat, "longitude": cat_lon, "depth_km": cat_depth, "time": cat_time},
        "located": {"latitude": round(loc_lat, 4), "longitude": round(loc_lon, 4),
                     "depth_km": round(loc_depth, 2)},
        "errors": {"distance_km": round(dist_error, 2), "depth_km": round(depth_error, 2)},
        "quality": {"rms_residual": round(loc_rms, 3), "azimuthal_gap": round(loc_gap, 1),
                     "num_picks": loc.get("num_picks", len(picks))},
        "method": loc.get("method", "unknown"),
    }

    print(f"\n  {'=' * 60}")
    print(f"  {'':18s} {'Catalog':>20s}  {'Located':>20s}")
    print(f"  {'-' * 18} {'-' * 20}  {'-' * 20}")
    print(f"  {'Latitude':18s} {cat_lat:>20.4f}  {loc_lat:>20.4f}")
    print(f"  {'Longitude':18s} {cat_lon:>20.4f}  {loc_lon:>20.4f}")
    print(f"  {'Depth (km)':18s} {cat_depth:>20.1f}  {loc_depth:>20.2f}")
    print(f"  {'=' * 60}")
    print(f"  Distance error : {dist_error:.2f} km")
    print(f"  Depth error    : {depth_error:.2f} km")
    print(f"  RMS residual   : {loc_rms:.3f} s")
    print(f"  Azimuthal gap  : {loc_gap:.1f} deg")
    print(f"  Num picks used : {loc.get('num_picks', '?')}")

    result_path = os.path.join(work_dir, "comparison.json")
    with open(result_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Saved: {result_path}")

    return comparison


# =================== Data Download ===================

def download_event(event_id, catalog_row):
    """Download waveform + station metadata for a single event from IRIS FDSN.

    Returns True if download succeeded, False otherwise.
    """
    event_dir = os.path.join(WAVEFORMS_DIR, f"Event_{event_id}")
    os.makedirs(event_dir, exist_ok=True)

    done_file = os.path.join(event_dir, "download_done.txt")
    if os.path.exists(done_file):
        print(f"  [SKIP] Already downloaded.")
        return True

    event_time = UTCDateTime(str(catalog_row["date"]))
    event_lat, event_lon = catalog_row["lat"], catalog_row["lon"]
    target_networks = "II,IU,G,GE,IC"

    try:
        client = FDSNClient("EARTHSCOPE", timeout=120)

        print(f"  Downloading station metadata...")
        inv_near = client.get_stations(
            network=target_networks, station="*", channel="BH?",
            latitude=event_lat, longitude=event_lon,
            minradius=0, maxradius=15,
            starttime=event_time, endtime=event_time + 3600, level="response")
        inv_far = client.get_stations(
            network=target_networks, station="*", channel="BH?",
            latitude=event_lat, longitude=event_lon,
            minradius=30, maxradius=90,
            starttime=event_time, endtime=event_time + 3600, level="response")

        inv_total = inv_near + inv_far
        n_sta = sum(len(s) for n in inv_total for s in n)
        inv_total.write(os.path.join(event_dir, f"stations_{event_id}.xml"), format="STATIONXML")
        print(f"  Station metadata: {n_sta} stations")

        t_start = event_time - 120
        t_end = event_time + 2400

        # Build station list, skip already downloaded
        tasks = []
        for network in inv_total:
            for station in network:
                net, sta = network.code, station.code
                mseed_file = os.path.join(event_dir, f"{net}.{sta}.mseed")
                if os.path.exists(mseed_file):
                    continue
                tasks.append((net, sta, mseed_file))

        existing = sum(len(s) for n in inv_total for s in n) - len(tasks)
        print(f"  Downloading {len(tasks)} waveforms ({existing} already cached)...")

        def _download_one(args):
            net, sta, mseed_file = args
            try:
                c = FDSNClient("EARTHSCOPE", timeout=60)
                st = c.get_waveforms(
                    network=net, station=sta, location="*", channel="BH?",
                    starttime=t_start, endtime=t_end)
                inv_single = c.get_stations(
                    network=net, station=sta, channel="BH?",
                    starttime=t_start, endtime=t_end, level="response")
                st.remove_response(inventory=inv_single, output="VEL",
                                   pre_filt=(0.01, 0.05, 10.0, 20.0))
                st.write(mseed_file, format="MSEED")
                return True
            except Exception:
                return False

        download_count = existing
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_download_one, t): t for t in tasks}
            for future in as_completed(futures):
                if future.result():
                    download_count += 1

        with open(done_file, "w") as f:
            f.write(f"Completed {download_count} stations.")
        print(f"  Downloaded {download_count} waveforms")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


# =================== Main ===================

def main():
    parser = argparse.ArgumentParser(description="Earthquake Location Validation (NonLinLoc + SeisBench)")
    parser.add_argument("--event", type=str, default=None, help="Single event ID")
    parser.add_argument("--download", action="store_true", help="Download data + locate all events")
    parser.add_argument("--locate-only", action="store_true", help="Locate already-downloaded events")
    args = parser.parse_args()

    print("=" * 70)
    print("  Earthquake Location Validation Pipeline")
    print(f"  Picking: EQT (global) + PhaseNet (global) ensemble")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    catalog = load_catalog()
    total = len(catalog)

    # Determine which events to process
    if args.event:
        target_rows = catalog[catalog["eventid"].astype(str) == args.event]
        if target_rows.empty:
            print(f"Event {args.event} not in catalog.")
            return
    elif args.download:
        target_rows = catalog
    else:
        # Default: locate already-downloaded events
        target_rows = catalog[
            catalog["eventid"].astype(str).map(
                lambda eid: os.path.exists(
                    os.path.join(WAVEFORMS_DIR, f"Event_{eid}", "download_done.txt")))
        ]

    print(f"Events to process: {len(target_rows)} / {total} total\n")
    all_results = []
    failed = []

    for idx, (_, row) in enumerate(target_rows.iterrows()):
        eid = str(row["eventid"])
        mw = row["mw"]
        result_path = os.path.join(RESULTS_DIR, f"Event_{eid}", "comparison.json")

        # Skip if already located (resume capability)
        if os.path.exists(result_path) and not args.event:
            print(f"[{idx+1}/{len(target_rows)}] Event {eid} Mw{mw:.2f} — already located, loading result.")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n{'=' * 70}")
        print(f"[{idx+1}/{len(target_rows)}] Event {eid}  Mw {mw:.2f}  {row['date']}")

        # Download if needed
        if args.download or args.event:
            event_dir = os.path.join(WAVEFORMS_DIR, f"Event_{eid}")
            if not os.path.exists(os.path.join(event_dir, "download_done.txt")):
                print(f"--- Downloading Event {eid} ---")
                if not download_event(eid, row):
                    failed.append((eid, "download_failed"))
                    continue
            else:
                print(f"  Data already downloaded.")

        # Locate
        result = process_event(eid, row)
        if result:
            all_results.append(result)
        else:
            failed.append((eid, "location_failed"))

    # ---- Final Summary ----
    if all_results:
        print(f"\n\n{'=' * 70}")
        print(f"  SUMMARY ({len(all_results)} located, {len(failed)} failed)")
        print(f"{'=' * 70}")
        print(f"  {'Event':>12s} {'Mw':>5s} {'Dist Err':>10s} {'Depth Err':>10s} "
              f"{'RMS':>8s} {'Gap':>6s} {'Method':>20s}")
        print(f"  {'-' * 12} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 6} {'-' * 20}")
        for r in all_results:
            print(f"  {r['event_id']:>12s} {r['catalog_mw']:>5.2f} "
                  f"{r['errors']['distance_km']:>8.2f} km "
                  f"{r['errors']['depth_km']:>8.2f} km "
                  f"{r['quality']['rms_residual']:>6.3f}s "
                  f"{r['quality']['azimuthal_gap']:>5.1f} deg "
                  f"{r.get('method', '?'):>20s}")

    if failed:
        print(f"\n  Failed events:")
        for eid, reason in failed:
            print(f"    {eid}: {reason}")

    print("\nDone.")


if __name__ == "__main__":
    main()
