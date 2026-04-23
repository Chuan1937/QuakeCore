"""
Continuous Seismic Data Download Module for QuakeCore
====================================================
Downloads continuous waveform data from FDSN data centers (SCEDC, etc.)
for specified time windows and geographic regions.
"""

import os
import glob
from typing import Dict, Optional, Tuple, Any

import numpy as np
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client as FDSNClient


# =================== Paths ===================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "continuous")


def get_default_data_dir() -> str:
    """Get the default directory for continuous data storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


# =================== Distance Functions ===================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# =================== Data Download ===================

def download_continuous_data(
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    network: str = "CI",
    channel: str = "BH?,HH?",
    data_dir: Optional[str] = None,
    client_name: str = "SCEDC",
    timeout: int = 120,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Download continuous waveform data for specified region and time window.

    Args:
        start_time: Start time of data window
        end_time: End time of data window
        min_lat: Minimum latitude of region
        max_lat: Maximum latitude of region
        min_lon: Minimum longitude of region
        max_lon: Maximum longitude of region
        network: Seismic network code (default: "CI" for SCEDC)
        channel: Channel codes to download (default: "BH?,HH?")
        data_dir: Directory to store downloaded data
        client_name: FDSN client name (default: "SCEDC")
        timeout: Request timeout in seconds

    Returns:
        Tuple of (streams dict mapping station_id to mseed file path,
                  stations dict mapping station_id to station info)
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    chunk_dir = os.path.join(data_dir, f"chunk_{start_time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Check for existing data
    existing_mseed = glob.glob(os.path.join(chunk_dir, "*.mseed"))
    if existing_mseed:
        print(f"  [INFO] Found {len(existing_mseed)} existing mseed files in {chunk_dir}")

        client = FDSNClient(client_name, timeout=timeout)
        try:
            inv = client.get_stations(
                network=network, station="*", channel=channel,
                minlatitude=min_lat, maxlatitude=max_lat,
                minlongitude=min_lon, maxlongitude=max_lon,
                starttime=start_time, endtime=end_time, level="response"
            )
        except Exception as e:
            print(f"  [ERROR] Failed to fetch inventory: {e}")
            return {}, {}

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

    # Download new data
    client = FDSNClient(client_name, timeout=timeout)

    print(f"--- Fetching Metadata for {network} network ---")
    try:
        inv = client.get_stations(
            network=network, station="*", channel=channel,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon,
            starttime=start_time, endtime=end_time, level="response"
        )
    except Exception as e:
        print(f"  [ERROR] Failed to fetch inventory: {e}")
        return {}, {}

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
                st = client.get_waveforms(net.code, sta.code, "*", channel, start_time, end_time)
                st.write(mseed_file, format="MSEED")
                streams[net_sta] = mseed_file
            except Exception:
                pass

    print(f"  Successfully downloaded data for {len(streams)} stations.")
    return streams, stations


def load_station_data(
    streams: Dict[str, str],
    stations: Dict[str, Dict[str, float]]
) -> list:
    """
    Load and preprocess station data from mseed files.

    Args:
        streams: Dict mapping station_id to mseed file path
        stations: Dict mapping station_id to station info (lat, lon, elevation)

    Returns:
        List of tuples (net_sta, best_traces_dict) where best_traces_dict
        maps component (Z, N, E) to best trace
    """
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
    return station_data


def fetch_catalog(
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    min_magnitude: float = 1.0,
    client_name: str = "SCEDC",
) -> list:
    """
    Fetch earthquake catalog from FDSN client as ground truth reference.

    Args:
        start_time: Start time of catalog window
        end_time: End time of catalog window
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        min_lon: Minimum longitude
        max_lon: Maximum longitude
        min_magnitude: Minimum magnitude to include (default: 1.0)
        client_name: FDSN client name

    Returns:
        List of event dictionaries with time, lat, lon, depth, mag, id
    """
    client = FDSNClient(client_name)
    try:
        cat = client.get_events(
            starttime=start_time, endtime=end_time,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon
        )
        truth = []
        for ev in cat:
            origin = ev.preferred_origin() or ev.origins[0]
            mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if mag and mag.mag < min_magnitude:
                continue
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
        print(f"  [WARN] Failed to fetch catalog: {e}")
        return []


# =================== Southern California Velocity Model & TT Cache ===================

_taup_model = None
_tt_interp = None

TAUP_DIR = os.path.join(PROJECT_ROOT, "resources", "taup")


def _get_taup():
    """
    Get or build Southern California TauPy model with custom velocity model.

    Creates a custom socal.tvel file based on IASP91 with Southern California
    velocity layers, then builds the TauP model.
    """
    global _taup_model
    if _taup_model is not None:
        return _taup_model

    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model

    socal_tvel_path = os.path.join(TAUP_DIR, "socal.tvel")
    socal_npz_path = os.path.join(TAUP_DIR, "socal.npz")

    if not os.path.exists(socal_npz_path):
        os.makedirs(TAUP_DIR, exist_ok=True)
        import obspy.taup
        iasp91_path = os.path.join(os.path.dirname(obspy.taup.__file__), "data", "iasp91.tvel")
        with open(iasp91_path) as f:
            iasp91_lines = f.readlines()
        mantle_start = next(i + 2 for i, line in enumerate(iasp91_lines[2:]) if float(line.split()[0]) > 35.0)

        # Southern California velocity layers
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
    """
    Get or build travel time interpolation cache for Southern California.

    Returns a dict mapping phase ("P", "S") to RegularGridInterpolator.
    The cache is built the first time and reused for subsequent calls.
    """
    global _tt_interp
    if _tt_interp is not None:
        return _tt_interp

    from scipy.interpolate import RegularGridInterpolator

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
