"""
Continuous Seismic Data Download Module for QuakeCore
====================================================
Downloads continuous waveform data from FDSN data centers (SCEDC, etc.)
for specified time windows and geographic regions.
"""

import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, Any, List, Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client as FDSNClient

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =================== Predefined Seismic Regions ===================

REGIONS = {
    # Southern California
    "南加州": {
        "min_lat": 32.0, "max_lat": 36.5, "min_lon": -120.0, "max_lon": -115.0,
        "network": "CI", "client": "SCEDC", "catalog": "SCEDC",
        "description": "Southern California (CI network, SCEDC)"
    },
    # Northern California
    "北加州": {
        "min_lat": 36.0, "max_lat": 42.0, "min_lon": -124.0, "max_lon": -120.0,
        "network": "BK", "client": "NCEDC", "catalog": "NCEDC",
        "description": "Northern California (BK network, NCEDC)"
    },
    # Central California
    "中加州": {
        "min_lat": 34.0, "max_lat": 38.0, "min_lon": -122.0, "max_lon": -118.0,
        "network": "CI,BK", "client": "SCEDC", "catalog": "SCEDC",
        "description": "Central California (CI+BK networks)"
    },
    # All California
    "加州": {
        "min_lat": 32.0, "max_lat": 42.0, "min_lon": -124.0, "max_lon": -114.0,
        "network": "CI,BK", "client": "SCEDC", "catalog": "SCEDC",
        "description": "All California (CI+BK networks)"
    },
    # USGS National/Regional
    "美国西部": {
        "min_lat": 30.0, "max_lat": 50.0, "min_lon": -130.0, "max_lon": -100.0,
        "network": "*", "client": "USGS", "catalog": "USGS",
        "description": "US Western US (USGS)"
    },
    # Southern California with USGS catalog comparison
    "南加州_对比USGS": {
        "min_lat": 32.0, "max_lat": 36.5, "min_lon": -120.0, "max_lon": -115.0,
        "network": "CI", "client": "SCEDC", "catalog": "USGS",
        "description": "Southern California with USGS ground truth"
    },
    # Pacific Northwest
    "太平洋西北": {
        "min_lat": 40.0, "max_lat": 49.0, "min_lon": -125.0, "max_lon": -116.0,
        "network": "UW,CC,CN", "client": "IRIS", "catalog": "USGS",
        "description": "Pacific Northwest (UW/CC/CN networks)"
    },
    # Japan
    "日本": {
        "min_lat": 30.0, "max_lat": 46.0, "min_lon": 125.0, "max_lon": 150.0,
        "network": "*", "client": "NIED", "catalog": "NIED",
        "description": "Japan (NIED)"
    },
    # New Zealand
    "新西兰": {
        "min_lat": -48.0, "max_lat": -34.0, "min_lon": 165.0, "max_lon": 180.0,
        "network": "NZ", "client": "GEONET", "catalog": "GEONET",
        "description": "New Zealand (NZ network)"
    },
    # Europe (example: Italy)
    "欧洲": {
        "min_lat": 36.0, "max_lat": 48.0, "min_lon": 6.0, "max_lon": 19.0,
        "network": "IV,MI", "client": "INGV", "catalog": "INGV",
        "description": "Europe (Italy, IV/MI networks)"
    },
}

# Region aliases
REGION_ALIASES = {
    "socal": "南加州",
    "southern_ca": "南加州",
    "socalifornia": "南加州",
    "nocal": "北加州",
    "northern_ca": "北加州",
    "northern_california": "北加州",
    "central_ca": "中加州",
    "central_california": "中加州",
    "california": "加州",
    "california_all": "加州",
    "ca": "加州",
    "us_west": "美国西部",
    "western_us": "美国西部",
    "pacific_nw": "太平洋西北",
    "pnw": "太平洋西北",
    "japan": "日本",
    "nz": "新西兰",
    "new_zealand": "新西兰",
    "europe": "欧洲",
    "italy": "欧洲",
}

# Catalog to client mapping for ground truth fetching
CATALOG_CLIENTS = {
    "SCEDC": "SCEDC",
    "NCEDC": "NCEDC",
    "USGS": "USGS",
    "NIED": "NIED",
    "GEONET": "GEONET",
    "INGV": "INGV",
}

CATALOG_ALIASES = {
    "usgc": "USGS",
    "usgs": "USGS",
    "scedc": "SCEDC",
    "ncedc": "NCEDC",
    "nied": "NIED",
    "geonet": "GEONET",
    "ingv": "INGV",
}

PLACE_ALIASES = {
    "usc": {
        "name": "南加州大学",
        "en_name": "University of Southern California",
        "latitude": 34.0224,
        "longitude": -118.2851,
        "radius_km": 40.0,
        "network": "CI",
        "client": "SCEDC",
        "catalog": "SCEDC",
    },
    "university of southern california": {
        "name": "南加州大学",
        "en_name": "University of Southern California",
        "latitude": 34.0224,
        "longitude": -118.2851,
        "radius_km": 40.0,
        "network": "CI",
        "client": "SCEDC",
        "catalog": "SCEDC",
    },
    "南加州大学": {
        "name": "南加州大学",
        "en_name": "University of Southern California",
        "latitude": 34.0224,
        "longitude": -118.2851,
        "radius_km": 40.0,
        "network": "CI",
        "client": "SCEDC",
        "catalog": "SCEDC",
    },
}

_PLACE_RESOLUTION_CACHE: Dict[str, Dict[str, Any]] = {}


def get_region(region_name: str) -> Optional[Dict[str, Any]]:
    """Get region config by name, supporting aliases."""
    # Direct match
    if region_name in REGIONS:
        return REGIONS[region_name].copy()
    # Alias match
    alias = region_name.lower().strip()
    if alias in REGION_ALIASES:
        return REGIONS[REGION_ALIASES[alias]].copy()
    return None


def list_regions() -> List[str]:
    """List all available region names."""
    return list(REGIONS.keys())


def _normalize_client_name(client_name: str) -> str:
    """Normalize official client names and common aliases."""
    if not client_name:
        return "SCEDC"
    raw = str(client_name).strip()
    upper = raw.upper()
    if upper in CATALOG_CLIENTS:
        return CATALOG_CLIENTS[upper]
    lower = raw.lower()
    if lower in CATALOG_ALIASES:
        return CATALOG_ALIASES[lower]
    return upper


def _client_candidates(client_name: str) -> List[str]:
    """Candidate FDSN clients for metadata fallback."""
    primary = _normalize_client_name(client_name)
    ordered = [primary]
    for alt in ["IRIS", "USGS"]:
        if alt not in ordered:
            ordered.append(alt)
    return ordered


def _place_cache_path() -> str:
    return os.path.join(PROJECT_ROOT, "data", "continuous", "place_cache.json")


def _load_place_cache() -> Dict[str, Dict[str, Any]]:
    if _PLACE_RESOLUTION_CACHE:
        return _PLACE_RESOLUTION_CACHE

    cache_path = _place_cache_path()
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _PLACE_RESOLUTION_CACHE.update(data)
    except Exception:
        pass
    return _PLACE_RESOLUTION_CACHE


def _save_place_cache() -> None:
    try:
        cache_path = _place_cache_path()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(_PLACE_RESOLUTION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _infer_provider_from_coordinates(lat: float, lon: float, country_code: str = "") -> Dict[str, str]:
    country_code = (country_code or "").lower()
    if country_code == "us" and 32.0 <= lat <= 42.5 and -125.0 <= lon <= -114.0:
        if lat < 36.5:
            return {"network": "CI", "client": "SCEDC", "catalog": "SCEDC"}
        return {"network": "BK", "client": "NCEDC", "catalog": "NCEDC"}
    if country_code == "us":
        return {"network": "*", "client": "IRIS", "catalog": "USGS"}
    return {"network": "*", "client": "IRIS", "catalog": "USGS"}


def _geocode_place(query: str) -> Optional[Dict[str, Any]]:
    query = str(query).strip()
    if not query:
        return None

    cache = _load_place_cache()
    cached = cache.get(query.lower())
    if cached:
        return cached.copy()

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?"
            f"q={quote_plus(query)}&format=jsonv2&addressdetails=1&limit=1"
        )
        request = Request(url, headers={"User-Agent": "QuakeCore/1.0"})
        with urlopen(request, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload:
            return None

        top = payload[0]
        lat = float(top.get("lat"))
        lon = float(top.get("lon"))
        address = top.get("address") or {}
        provider = _infer_provider_from_coordinates(lat, lon, address.get("country_code", ""))

        lower_query = query.lower()
        radius_km = 20.0 if any(term in lower_query for term in ("university", "campus", "college", "school", "大学", "学院")) else 30.0
        display_name = top.get("display_name") or query
        result = {
            "name": query,
            "en_name": display_name,
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius_km,
            "network": provider["network"],
            "client": provider["client"],
            "catalog": provider["catalog"],
            "source": "nominatim",
        }
        cache[query.lower()] = result
        _save_place_cache()
        return result.copy()
    except Exception:
        return None


def resolve_named_place(name: str) -> Optional[Dict[str, Any]]:
    """Resolve a named place to a center point and recommended monitoring defaults."""
    if not name:
        return None
    key = str(name).strip().lower()
    if key in PLACE_ALIASES:
        return PLACE_ALIASES[key].copy()
    geocoded = _geocode_place(name)
    if geocoded:
        return geocoded
    # Try a looser substring match for common punctuation/casing differences.
    for alias, data in PLACE_ALIASES.items():
        if alias in key or key in alias:
            return data.copy()
    return None


def _inventory_to_stations(inv) -> Dict[str, Dict[str, float]]:
    stations = {}
    for net in inv:
        for sta in net:
            stations[f"{net.code}.{sta.code}"] = {
                "latitude": sta.latitude,
                "longitude": sta.longitude,
                "elevation": sta.elevation or 0.0
            }
    return stations


def estimate_continuous_download(
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    network: str = "CI",
    channel: str = "BH?,HH?",
    client_name: str = "SCEDC",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Estimate the size and station count for a continuous download."""
    client = FDSNClient(_normalize_client_name(client_name), timeout=timeout)
    try:
        inv = client.get_stations(
            network=network, station="*", channel=channel,
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon,
            starttime=start_time, endtime=end_time, level="response"
        )
    except Exception as e:
        return {"error": f"Failed to fetch inventory: {e}"}

    stations = _inventory_to_stations(inv)
    duration_seconds = max(0.0, float(end_time - start_time))
    station_count = len(stations)
    channel_groups = max(1, len([part for part in str(channel).split(",") if part.strip()]))
    estimated_bytes = station_count * duration_seconds * 100.0 * 3.0 * channel_groups * 4.0

    return {
        "station_count": station_count,
        "duration_seconds": duration_seconds,
        "estimated_bytes": float(estimated_bytes),
        "estimated_mb": float(estimated_bytes / (1024.0 ** 2)),
        "estimated_gb": float(estimated_bytes / (1024.0 ** 3)),
        "stations": stations,
        "inventory": inv,
    }


# =================== Paths ===================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "continuous")


def get_default_data_dir() -> str:
    """Get the default directory for continuous data storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def get_chunk_dir(start_time: UTCDateTime, data_dir: Optional[str] = None) -> str:
    """Return chunk directory path for a given start time."""
    if data_dir is None:
        data_dir = get_default_data_dir()
    return os.path.join(data_dir, f"chunk_{start_time.strftime('%Y%m%d_%H%M%S')}")


def _chunk_stations_json_path(chunk_dir: str) -> str:
    return os.path.join(chunk_dir, "stations.json")


def _save_chunk_stations_cache(chunk_dir: str, stations: Dict[str, Dict[str, float]]) -> None:
    """Persist station coordinates for local/offline re-runs."""
    if not stations:
        return
    try:
        os.makedirs(chunk_dir, exist_ok=True)
        with open(_chunk_stations_json_path(chunk_dir), "w", encoding="utf-8") as f:
            json.dump(stations, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_chunk_stations_cache(chunk_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load station cache for a chunk.
    Fallbacks:
    1) chunk_dir/stations.json
    2) data/fdsn/stations.json (legacy/global cache)
    """
    candidates = [
        _chunk_stations_json_path(chunk_dir),
        os.path.join(PROJECT_ROOT, "data", "fdsn", "stations.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            normalized = {}
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                if "latitude" in val and "longitude" in val:
                    sta_key = os.path.splitext(str(key))[0]
                    normalized[sta_key] = {
                        "latitude": float(val.get("latitude", 0.0)),
                        "longitude": float(val.get("longitude", 0.0)),
                        "elevation": float(val.get("elevation", 0.0) or 0.0),
                    }
            if normalized:
                return normalized
        except Exception:
            continue
    return {}


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
    download_workers: Optional[int] = None,
    inventory=None,
    stations: Optional[Dict[str, Dict[str, float]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    local_only: bool = False,
    refresh_station_metadata: bool = False,
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
        download_workers: Number of parallel station-download workers

    Returns:
        Tuple of (streams dict mapping station_id to mseed file path,
                  stations dict mapping station_id to station info)
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    chunk_dir = get_chunk_dir(start_time=start_time, data_dir=data_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    def _emit(stage: str, **payload):
        message = {"stage": stage, **payload}
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                pass
        else:
            text = payload.get("message")
            if text:
                print(f"  [{stage}] {text}")

    # Check for existing data
    existing_mseed = glob.glob(os.path.join(chunk_dir, "*.mseed"))
    if existing_mseed:
        _emit("download", message=f"Found {len(existing_mseed)} existing mseed files in {chunk_dir}")
        existing_keys = {os.path.splitext(os.path.basename(p))[0] for p in existing_mseed}
        if stations is None:
            stations = load_chunk_stations_cache(chunk_dir)
            if stations:
                _emit("metadata", message=f"Loaded {len(stations)} stations from local cache.")
        if local_only and (stations is None or not stations):
            _emit("error", message="local_only=true but no local station cache found for this chunk.")
            return {}, {}

        coverage = 0.0
        if stations:
            covered = sum(1 for k in existing_keys if k in stations)
            coverage = covered / max(1, len(existing_keys))
            _emit("metadata", message=f"Local station coverage: {covered}/{len(existing_keys)} ({coverage*100:.1f}%).")

        if stations is None or not stations or coverage < 0.6:
            if not refresh_station_metadata:
                _emit("metadata", message="Use local waveform/cache only; skip remote station metadata refresh.")
            elif local_only:
                _emit("metadata", message="local_only=true, skip remote station metadata补全。")
            else:
                # Fallback to remote metadata when local cache missing or coverage too low.
                if inventory is None:
                    last_err = None
                    for cname in _client_candidates(client_name):
                        try:
                            client = FDSNClient(cname, timeout=timeout)
                            inventory = client.get_stations(
                                network=network, station="*", channel=channel,
                                minlatitude=min_lat, maxlatitude=max_lat,
                                minlongitude=min_lon, maxlongitude=max_lon,
                                starttime=start_time, endtime=end_time, level="response"
                            )
                            _emit("metadata", message=f"Station metadata refreshed from {cname}.")
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    if inventory is None:
                        if stations:
                            _emit("metadata", message=f"Remote metadata unavailable, continue with local cache: {last_err}")
                        else:
                            _emit("error", message=f"Failed to fetch inventory: {last_err}")
                            return {}, {}
                if inventory is not None:
                    stations_remote = _inventory_to_stations(inventory)
                    if stations_remote:
                        stations = stations_remote
                        _save_chunk_stations_cache(chunk_dir, stations)
                        _emit("metadata", message=f"Updated local station cache: {len(stations)} stations.")

        streams = {}
        for net_sta_file in existing_mseed:
            net_sta = os.path.splitext(os.path.basename(net_sta_file))[0]
            streams[net_sta] = net_sta_file

        _emit("download", message=f"Loaded {len(streams)} stations from existing data.")
        return streams, stations

    # Download new data
    client = FDSNClient(_normalize_client_name(client_name), timeout=timeout)

    if inventory is None or stations is None:
        _emit("metadata", message=f"Fetching metadata for {network} network")
        try:
            inventory = client.get_stations(
                network=network, station="*", channel=channel,
                minlatitude=min_lat, maxlatitude=max_lat,
                minlongitude=min_lon, maxlongitude=max_lon,
                starttime=start_time, endtime=end_time, level="response"
            )
        except Exception as e:
            _emit("error", message=f"Failed to fetch inventory: {e}")
            return {}, {}

    if stations is None:
        stations = _inventory_to_stations(inventory)
    _save_chunk_stations_cache(chunk_dir, stations)

    inv_list = list(inventory)
    _emit("metadata", message=f"Found {len(stations)} stations. Downloading waveforms...")

    streams = {}
    downloaded = 0
    failed = 0
    total = sum(len(net) for net in inv_list)
    workers = int(download_workers or 16)
    workers = max(1, min(16, workers))
    progress = None
    if TQDM_AVAILABLE:
        progress = tqdm(total=total, desc=f"  [Download] {network}", unit="sta", leave=False)

    tasks = []
    existing = 0
    for net in inv_list:
        for sta in net:
            net_sta = f"{net.code}.{sta.code}"
            mseed_file = os.path.join(chunk_dir, f"{net_sta}.mseed")
            if os.path.exists(mseed_file):
                streams[net_sta] = mseed_file
                existing += 1
                continue
            tasks.append((net.code, sta.code, net_sta, mseed_file))

    downloaded += existing
    if existing > 0:
        _emit("download", message=f"{downloaded}/{total} stations ready", downloaded=downloaded, total=total)
    if progress is not None and existing > 0:
        progress.update(existing)

    def _download_one(net_code: str, sta_code: str, net_sta: str, mseed_file: str):
        local_client = FDSNClient(_normalize_client_name(client_name), timeout=timeout)
        st = local_client.get_waveforms(net_code, sta_code, "*", channel, start_time, end_time)
        st.write(mseed_file, format="MSEED")
        return net_sta, mseed_file

    try:
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(_download_one, net_code, sta_code, net_sta, mseed_file): (net_sta,)
                    for net_code, sta_code, net_sta, mseed_file in tasks
                }
                for fut in as_completed(future_to_task):
                    net_sta = future_to_task[fut][0]
                    try:
                        sta_id, out_file = fut.result()
                        streams[sta_id] = out_file
                        downloaded += 1
                        _emit("download", message=f"{downloaded}/{total} stations ready", downloaded=downloaded, total=total)
                    except Exception as exc:
                        failed += 1
                        _emit("download", message=f"{net_sta} failed: {exc}", downloaded=downloaded, total=total, failed=failed)
                    finally:
                        if progress is not None:
                            progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    _emit("download", message=f"Successfully downloaded data for {len(streams)} stations.", downloaded=downloaded, failed=failed, total=total)
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
    client = FDSNClient(_normalize_client_name(client_name))
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
