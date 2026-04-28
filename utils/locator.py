"""
Earthquake Location Module for QuakeCore

Provides earthquake hypocenter location using:
- Grid search method
- Geiger's method (least squares inversion)
- IASP91 velocity model for travel time calculation
- Optional TauP for accurate travel times
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Try to import optional dependencies
try:
    from obspy.geodetics import locations2degrees, kilometer2degrees
    from obspy.core import UTCDateTime
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    UTCDateTime = None

# Try to import TauP for accurate travel times
try:
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model
    TAUP_AVAILABLE = True
except ImportError:
    TAUP_AVAILABLE = False
    TauPyModel = None


@dataclass
class Station:
    """Seismic station with coordinates."""
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float = 0.0  # meters
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.network, self.station))

    def __eq__(self, other):
        if not isinstance(other, Station):
            return False
        return self.network == other.network and self.station == other.station


@dataclass
class PhasePick:
    """Phase pick with station association."""
    station: Station
    phase_type: str  # "P" or "S"
    arrival_time: float  # Absolute time as timestamp
    weight: float = 1.0
    residual: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypocenter:
    """Earthquake hypocenter location result."""
    latitude: float  # degrees
    longitude: float  # degrees
    depth: float  # km
    origin_time: float  # timestamp
    rms_residual: float = 0.0
    gap: float = 0.0  # Azimuthal gap in degrees
    num_picks: int = 0
    num_stations: int = 0
    method: str = ""
    uncertainty_lat: float = 0.0
    uncertainty_lon: float = 0.0
    uncertainty_depth: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Helper function to convert numpy types to Python native types
        def to_native(val):
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        result = {
            "latitude": to_native(self.latitude),
            "longitude": to_native(self.longitude),
            "depth_km": to_native(self.depth),
            "origin_time": to_native(self.origin_time),
            "origin_time_iso": UTCDateTime(self.origin_time).isoformat() if OBSPY_AVAILABLE and UTCDateTime else str(self.origin_time),
            "rms_residual": to_native(self.rms_residual),
            "azimuthal_gap": to_native(self.gap),
            "num_picks": to_native(self.num_picks),
            "num_stations": to_native(self.num_stations),
            "method": self.method,
            "uncertainty": {
                "latitude_km": to_native(self.uncertainty_lat),
                "longitude_km": to_native(self.uncertainty_lon),
                "depth_km": to_native(self.uncertainty_depth),
            },
        }
        # Convert metadata values too
        for key, val in self.metadata.items():
            result[key] = to_native(val)
        return result


class IASP91VelocityModel:
    """
    Simplified IASP91 velocity model for travel time calculation.
    Provides P and S wave velocities at different depths.
    """

    # IASP91 model: (depth_km, vp_km_s, vs_km_s)
    LAYERS = [
        (0.0, 5.80, 3.36),
        (20.0, 5.80, 3.36),
        (35.0, 6.50, 3.75),
        (120.0, 8.04, 4.47),
        (210.0, 8.05, 4.48),
        (410.0, 8.85, 4.94),
        (660.0, 10.27, 5.66),
    ]

    def __init__(self):
        self.layers = np.array(self.LAYERS)

    def get_velocity(self, depth: float, phase: str = "P") -> float:
        """Get velocity at given depth."""
        if phase.upper() == "P":
            col = 1
        else:
            col = 2

        # Find the layer containing this depth
        for i in range(len(self.layers) - 1):
            if self.layers[i, 0] <= depth < self.layers[i + 1, 0]:
                return self.layers[i, col]

        # Return deepest layer velocity
        return self.layers[-1, col]

    def get_avg_velocity(self, depth: float, phase: str = "P") -> float:
        """Get average velocity from surface to given depth."""
        if depth <= 0:
            return self.get_velocity(0, phase)

        col = 1 if phase.upper() == "P" else 2

        total_time = 0.0
        total_depth = 0.0

        for i in range(len(self.layers) - 1):
            layer_top = self.layers[i, 0]
            layer_bottom = self.layers[i + 1, 0]
            layer_vel = self.layers[i, col]

            if depth <= layer_top:
                break

            actual_bottom = min(depth, layer_bottom)
            layer_thickness = actual_bottom - layer_top

            if layer_thickness > 0:
                total_time += layer_thickness / layer_vel
                total_depth += layer_thickness

        if total_time > 0:
            return total_depth / total_time
        return self.get_velocity(depth, phase)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points in km.
    Uses Haversine formula.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_epicentral_distance(
    sta_lat: float, sta_lon: float,
    evt_lat: float, evt_lon: float
) -> float:
    """Calculate epicentral distance in degrees."""
    if OBSPY_AVAILABLE:
        try:
            return locations2degrees(sta_lat, sta_lon, evt_lat, evt_lon)
        except Exception:
            pass

    # Fallback using haversine
    distance_km = haversine_distance(sta_lat, sta_lon, evt_lat, evt_lon)
    return distance_km / 111.19  # Approximate conversion


def calculate_ray_parameter(distance_deg: float, depth_km: float, velocity: float) -> float:
    """Calculate ray parameter (slowness)."""
    # Simplified calculation
    distance_rad = np.radians(distance_deg)
    return distance_rad / (2 * np.pi) * 6371 / velocity


class TravelTimeCalculator:
    """Calculate travel times for P and S waves using TauP or simplified model."""

    def __init__(self, velocity_model: Optional[IASP91VelocityModel] = None, use_taup: bool = True):
        """
        Initialize travel time calculator.

        Args:
            velocity_model: Simple velocity model for fallback
            use_taup: Whether to use TauP for accurate travel times
        """
        self.model = velocity_model or IASP91VelocityModel()
        self.use_taup = use_taup and TAUP_AVAILABLE

        if self.use_taup:
            try:
                self.taup_model = TauPyModel(model="iasp91")
                print("Using TauP IASP91 model for accurate travel times")
            except Exception as e:
                warnings.warn(f"Failed to load TauP model: {e}. Using simplified model.")
                self.use_taup = False
                self.taup_model = None
        else:
            self.taup_model = None

    def calculate_travel_time(
        self,
        distance_deg: float,
        depth_km: float,
        phase: str = "P"
    ) -> float:
        """
        Calculate travel time using TauP if available, otherwise simplified model.

        Args:
            distance_deg: Epicentral distance in degrees
            depth_km: Source depth in km
            phase: Phase type ("P" or "S")

        Returns:
            Travel time in seconds
        """
        if self.use_taup and self.taup_model is not None:
            return self._calculate_taup(distance_deg, depth_km, phase)
        else:
            return self._calculate_simple(distance_deg, depth_km, phase)

    def _calculate_taup(self, distance_deg: float, depth_km: float, phase: str) -> float:
        """Calculate travel time using TauP."""
        try:
            # Determine phase name
            if phase.upper() == "P":
                phase_list = ["P", "p"]
            elif phase.upper() == "S":
                phase_list = ["S", "s"]
            else:
                phase_list = [phase.upper()]

            # Get arrivals
            arrivals = self.taup_model.get_travel_times(
                source_depth_in_km=max(0, depth_km),
                distance_in_degree=distance_deg,
                phase_list=phase_list
            )

            if arrivals:
                # Return first arrival
                return arrivals[0].time
            else:
                # Fallback to simple model
                return self._calculate_simple(distance_deg, depth_km, phase)

        except Exception as e:
            warnings.warn(f"TauP calculation failed: {e}. Using simplified model.")
            return self._calculate_simple(distance_deg, depth_km, phase)

    def _calculate_simple(self, distance_deg: float, depth_km: float, phase: str) -> float:
        """Calculate travel time using simplified straight-ray approximation."""
        # Convert distance to km (approximate)
        distance_km = distance_deg * 111.19

        # Get average velocity
        v_avg = self.model.get_avg_velocity(depth_km, phase)

        # Calculate slant distance
        if depth_km <= 0:
            slant_distance = distance_km
        else:
            slant_distance = np.sqrt(distance_km ** 2 + depth_km ** 2)

        # Travel time
        return slant_distance / v_avg

    def calculate_travel_times_batch(
        self,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
        event_lat: float,
        event_lon: float,
        depth_km: float,
        phase: str = "P"
    ) -> np.ndarray:
        """
        Calculate travel times for multiple stations at once.

        Args:
            station_lats: Array of station latitudes
            station_lons: Array of station longitudes
            event_lat: Event latitude
            event_lon: Event longitude
            depth_km: Source depth
            phase: Phase type

        Returns:
            Array of travel times in seconds
        """
        travel_times = np.zeros(len(station_lats))

        for i, (sta_lat, sta_lon) in enumerate(zip(station_lats, station_lons)):
            dist = calculate_epicentral_distance(sta_lat, sta_lon, event_lat, event_lon)
            travel_times[i] = self.calculate_travel_time(dist, depth_km, phase)

        return travel_times

    def calculate_travel_time_grid(
        self,
        sta_lat: float, sta_lon: float,
        grid_lats: np.ndarray, grid_lons: np.ndarray, grid_depths: np.ndarray,
        phase: str = "P"
    ) -> np.ndarray:
        """Calculate travel times to all grid points."""
        # Create meshgrid
        GLAT, GLON, GDEP = np.meshgrid(grid_lats, grid_lons, grid_depths, indexing='ij')

        # Flatten for vectorized calculation
        flat_lats = GLAT.flatten()
        flat_lons = GLON.flatten()
        flat_depths = GDEP.flatten()

        # Calculate distances
        distances = np.array([
            calculate_epicentral_distance(sta_lat, sta_lon, lat, lon)
            for lat, lon in zip(flat_lats, flat_lons)
        ])

        # Calculate travel times
        v_avg = self.model.get_avg_velocity(np.mean(grid_depths), phase)

        # Vectorized slant distance calculation
        distances_km = distances * 111.19
        slant_distances = np.sqrt(distances_km ** 2 + flat_depths ** 2)
        travel_times = slant_distances / v_avg

        return travel_times.reshape(GLAT.shape)


class EarthquakeLocator:
    """
    Earthquake location using grid search and Geiger's method.
    """

    def __init__(self, velocity_model: Optional[IASP91VelocityModel] = None):
        self.tt_calculator = TravelTimeCalculator(velocity_model)
        self.model = velocity_model or IASP91VelocityModel()

    def locate_grid_search(
        self,
        picks: List[PhasePick],
        grid_center: Tuple[float, float] = None,
        grid_size_deg: float = 2.0,
        grid_depth_range: Tuple[float, float] = (0.0, 50.0),
        grid_points: int = 8,
        depth_points: int = 4,
        use_fast_model: bool = True,
    ) -> Hypocenter:
        """
        Locate earthquake using grid search method.

        Args:
            picks: List of PhasePick objects
            grid_center: (lat, lon) center of search grid
            grid_size_deg: Half-width of search grid in degrees
            grid_depth_range: (min_depth, max_depth) in km
            grid_points: Number of grid points in lat/lon
            depth_points: Number of depth grid points

        Returns:
            Hypocenter object with location result
        """
        if len(picks) < 3:
            raise ValueError("Need at least 3 picks for location")

        # Use subset of picks for faster grid search (max 50)
        if len(picks) > 50:
            # Sort by weight and take best picks
            sorted_picks = sorted(picks, key=lambda p: p.weight, reverse=True)
            picks_used = sorted_picks[:50]
            print(f"Using {len(picks_used)} best picks for grid search (of {len(picks)} total)")
        else:
            picks_used = picks

        # Determine grid center from station coordinates
        if grid_center is None:
            sta_lats = [p.station.latitude for p in picks_used]
            sta_lons = [p.station.longitude for p in picks_used]
            grid_center = (np.mean(sta_lats), np.mean(sta_lons))

        center_lat, center_lon = grid_center

        # Create search grids
        grid_lats = np.linspace(
            center_lat - grid_size_deg,
            center_lat + grid_size_deg,
            grid_points
        )
        grid_lons = np.linspace(
            center_lon - grid_size_deg,
            center_lon + grid_size_deg,
            grid_points
        )
        grid_depths = np.linspace(
            grid_depth_range[0],
            grid_depth_range[1],
            depth_points
        )

        # Calculate travel times for each station
        n_picks = len(picks_used)
        n_grid = len(grid_lats) * len(grid_lons) * len(grid_depths)
        print(f"Grid search: {n_grid} points, {n_picks} picks = {n_grid * n_picks} calculations")

        # Reshape picks arrival times
        arrival_times = np.array([p.arrival_time for p in picks_used])

        # Pre-calculate station coordinates for faster lookup
        sta_coords = [(p.station.latitude, p.station.longitude, p.phase_type) for p in picks_used]

        # Grid search - use TauP for accurate travel times at all distances
        best_residual = np.inf
        best_lat = center_lat
        best_lon = center_lon
        best_depth = 10.0
        best_origin_time = np.mean(arrival_times)

        # Use TauP for accurate travel times (critical for teleseismic distances)
        def grid_travel_time(sta_lat, sta_lon, evt_lat, evt_lon, depth, phase):
            return self.tt_calculator.calculate_travel_time(
                calculate_epicentral_distance(sta_lat, sta_lon, evt_lat, evt_lon),
                depth, phase
            )

        grid_count = 0
        for lat in grid_lats:
            for lon in grid_lons:
                for depth in grid_depths:
                    grid_count += 1
                    if grid_count % 50 == 0:
                        print(f"  Grid search progress: {grid_count}/{n_grid}")

                    # Calculate travel times for all picks
                    travel_times = np.array([
                        grid_travel_time(sta_lat, sta_lon, lat, lon, depth, phase)
                        for sta_lat, sta_lon, phase in sta_coords
                    ])

                    # Estimate origin time (mean of (arrival - travel_time))
                    origin_times = arrival_times - travel_times
                    origin_time = np.mean(origin_times)

                    # Calculate residuals
                    residuals = arrival_times - (origin_time + travel_times)
                    rms = np.sqrt(np.mean(residuals ** 2))

                    if rms < best_residual:
                        best_residual = rms
                        best_lat = lat
                        best_lon = lon
                        best_depth = depth
                        best_origin_time = origin_time

        print(f"Grid search complete. Best RMS: {best_residual:.2f}s at ({best_lat:.2f}, {best_lon:.2f}, {best_depth:.1f}km)")

        # Calculate azimuthal gap
        gap = self._calculate_azimuthal_gap(picks_used, best_lat, best_lon)

        # Create result
        result = Hypocenter(
            latitude=best_lat,
            longitude=best_lon,
            depth=best_depth,
            origin_time=best_origin_time,
            rms_residual=best_residual,
            gap=gap,
            num_picks=len(picks),
            num_stations=len(set(p.station for p in picks)),
            method="grid_search",
        )

        # Update pick residuals
        for pick in picks:
            dist = calculate_epicentral_distance(
                pick.station.latitude, pick.station.longitude,
                best_lat, best_lon
            )
            tt = self.tt_calculator.calculate_travel_time(dist, best_depth, pick.phase_type)
            pick.residual = pick.arrival_time - (best_origin_time + tt)

        return result

    def locate_geiger(
        self,
        picks: List[PhasePick],
        initial_location: Optional[Tuple[float, float, float, float]] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> Hypocenter:
        """
        Locate earthquake using Geiger's method (least squares).

        Args:
            picks: List of PhasePick objects
            initial_location: (lat, lon, depth_km, origin_time) initial guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Hypocenter object with location result
        """
        if len(picks) < 4:
            # Fall back to grid search if not enough picks
            return self.locate_grid_search(picks)

        # Initial guess
        if initial_location is None:
            # Use centroid of stations
            sta_lats = [p.station.latitude for p in picks]
            sta_lons = [p.station.longitude for p in picks]
            arrival_times = [p.arrival_time for p in picks]

            lat0 = np.mean(sta_lats)
            lon0 = np.mean(sta_lons)
            depth0 = 10.0
            t0 = np.mean(arrival_times) - 5.0  # Estimate origin time
        else:
            lat0, lon0, depth0, t0 = initial_location

        def objective(params):
            """Objective function: sum of squared residuals."""
            lat, lon, depth, origin_time = params

            residuals = []
            for pick in picks:
                dist = calculate_epicentral_distance(
                    pick.station.latitude, pick.station.longitude,
                    lat, lon
                )
                tt = self.tt_calculator.calculate_travel_time(
                    dist, depth, pick.phase_type
                )
                predicted = origin_time + tt
                residual = pick.arrival_time - predicted
                residuals.append(residual * pick.weight)

            return np.sum(np.array(residuals) ** 2)

        # Optimize
        result = minimize(
            objective,
            x0=[lat0, lon0, depth0, t0],
            method='L-BFGS-B',
            bounds=[
                (-90, 90),  # latitude
                (-180, 180),  # longitude
                (0, 700),  # depth
                (None, None),  # origin time
            ],
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )

        lat, lon, depth, origin_time = result.x

        # Calculate residuals for each pick
        total_residual = 0
        for pick in picks:
            dist = calculate_epicentral_distance(
                pick.station.latitude, pick.station.longitude,
                lat, lon
            )
            tt = self.tt_calculator.calculate_travel_time(dist, depth, pick.phase_type)
            pick.residual = pick.arrival_time - (origin_time + tt)
            total_residual += pick.residual ** 2

        rms = np.sqrt(total_residual / len(picks))

        # Calculate azimuthal gap
        gap = self._calculate_azimuthal_gap(picks, lat, lon)

        return Hypocenter(
            latitude=lat,
            longitude=lon,
            depth=max(0, depth),
            origin_time=origin_time,
            rms_residual=rms,
            gap=gap,
            num_picks=len(picks),
            num_stations=len(set(p.station for p in picks)),
            method="geiger",
        )

    def locate(
        self,
        picks: List[PhasePick],
        method: str = "auto",
        **kwargs
    ) -> Hypocenter:
        """
        Locate earthquake using specified method.

        Args:
            picks: List of PhasePick objects
            method: "auto", "grid_search", or "geiger"
            **kwargs: Additional arguments for specific methods

        Returns:
            Hypocenter object
        """
        if method == "auto":
            if len(picks) >= 4:
                # Start with grid search, refine with Geiger
                grid_result = self.locate_grid_search(picks, **kwargs)
                initial = (grid_result.latitude, grid_result.longitude,
                          grid_result.depth, grid_result.origin_time)
                return self.locate_geiger(picks, initial_location=initial)
            else:
                return self.locate_grid_search(picks, **kwargs)
        elif method == "grid_search":
            return self.locate_grid_search(picks, **kwargs)
        elif method == "geiger":
            return self.locate_geiger(picks, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_azimuthal_gap(
        self,
        picks: List[PhasePick],
        event_lat: float,
        event_lon: float
    ) -> float:
        """Calculate azimuthal gap (largest azimuthal gap between stations)."""
        azimuths = []

        for pick in picks:
            # Calculate azimuth from event to station
            az = self._calculate_azimuth(
                event_lat, event_lon,
                pick.station.latitude, pick.station.longitude
            )
            azimuths.append(az)

        if len(azimuths) < 2:
            return 360.0

        azimuths = sorted(azimuths)

        # Calculate gaps
        gaps = []
        for i in range(len(azimuths)):
            next_i = (i + 1) % len(azimuths)
            if next_i == 0:
                gap = (360 - azimuths[i]) + azimuths[next_i]
            else:
                gap = azimuths[next_i] - azimuths[i]
            gaps.append(gap)

        return max(gaps)

    def _calculate_azimuth(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate azimuth from point 1 to point 2."""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        azimuth = np.degrees(np.arctan2(x, y))
        return (azimuth + 360) % 360


def create_station_from_metadata(metadata: Dict[str, Any]) -> Optional[Station]:
    """Create a Station object from metadata dictionary."""
    # Try common field names
    network = metadata.get("network") or metadata.get("knetwk", "")
    station = metadata.get("station") or metadata.get("kstnm", "")

    lat = metadata.get("latitude") or metadata.get("stla") or metadata.get("lat")
    lon = metadata.get("longitude") or metadata.get("stlo") or metadata.get("lon")
    elev = metadata.get("elevation") or metadata.get("stel") or 0.0

    if lat is None or lon is None:
        return None

    return Station(
        network=str(network),
        station=str(station),
        latitude=float(lat),
        longitude=float(lon),
        elevation=float(elev) if elev else 0.0,
        metadata=metadata
    )


def picks_from_pick_results(
    pick_results: List[Any],  # List of PickResult from phase_picker
    stations: Optional[Dict[str, Station]] = None
) -> List[PhasePick]:
    """
    Convert PickResult objects to PhasePick objects for location.

    Args:
        pick_results: List of PickResult objects from phase picking
        stations: Optional dict mapping station_id to Station objects

    Returns:
        List of PhasePick objects
    """
    phase_picks = []

    for pr in pick_results:
        # Get absolute time
        if pr.absolute_time:
            try:
                if OBSPY_AVAILABLE:
                    arrival_time = UTCDateTime(pr.absolute_time).timestamp
                else:
                    # Parse ISO format
                    import datetime
                    arrival_time = datetime.datetime.fromisoformat(
                        pr.absolute_time.replace('Z', '+00:00')
                    ).timestamp()
            except Exception:
                continue
        elif pr.time_offset_s is not None:
            # Use relative time (will need start_time from trace)
            arrival_time = pr.time_offset_s
        else:
            continue

        # Get station info
        metadata = pr.metadata or {}

        # Try to find or create station
        station_id = f"{metadata.get('network', '')}.{metadata.get('station', '')}"

        if stations and station_id in stations:
            station = stations[station_id]
        else:
            station = create_station_from_metadata(metadata)

        if station is None:
            # Create a dummy station (will need coordinates from user)
            station = Station(
                network=metadata.get("network", "XX"),
                station=metadata.get("station", "UNK"),
                latitude=0.0,
                longitude=0.0,
                metadata=metadata
            )

        phase_pick = PhasePick(
            station=station,
            phase_type=pr.phase_type,
            arrival_time=arrival_time,
            weight=pr.normalized_score or 1.0,
            metadata={"method": pr.method, "sample_index": pr.sample_index}
        )
        phase_picks.append(phase_pick)

    return phase_picks


def locate_earthquake(
    picks: List[PhasePick],
    method: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """
    Main function to locate an earthquake.

    Args:
        picks: List of PhasePick objects with station coordinates
        method: Location method ("auto", "grid_search", "geiger")
        **kwargs: Additional parameters

    Returns:
        Dictionary with location results
    """
    # Filter picks with valid station coordinates
    valid_picks = [
        p for p in picks
        if p.station.latitude != 0 or p.station.longitude != 0
    ]

    if len(valid_picks) < 3:
        return {
            "error": f"Need at least 3 picks with valid station coordinates. Got {len(valid_picks)}.",
            "total_picks": len(picks),
            "valid_picks": len(valid_picks),
        }

    # Deduplicate picks per station per phase type
    # Strategy:
    #   1. Deep learning picks (phasenet, eqtransformer, gpd): use their own confidence
    #   2. Traditional picks (aic, sta_lta, pai_k, ...): take the best (highest weight)
    #   3. For each (station, phase): prefer deep learning over traditional
    DL_METHODS = {"phasenet", "eqtransformer", "gpd"}

    # Group by (station_id, phase_type)
    groups: Dict[Tuple[str, str], List[PhasePick]] = {}
    for p in valid_picks:
        sta_id = f"{p.station.network}.{p.station.station}"
        key = (sta_id, p.phase_type)
        groups.setdefault(key, []).append(p)

    best_picks: Dict[Tuple[str, str], PhasePick] = {}
    for key, group in groups.items():
        dl_picks = [p for p in group if p.metadata.get("method", "") in DL_METHODS]
        trad_picks = [p for p in group if p.metadata.get("method", "") not in DL_METHODS]

        # Best deep learning pick (highest weight)
        best_dl = max(dl_picks, key=lambda p: p.weight) if dl_picks else None

        # Best traditional pick (highest weight)
        best_trad = max(trad_picks, key=lambda p: p.weight) if trad_picks else None

        # Prefer deep learning if available
        if best_dl is not None:
            best_picks[key] = best_dl
        elif best_trad is not None:
            best_picks[key] = best_trad

    deduped_picks = list(best_picks.values())
    print(f"Deduplication: {len(valid_picks)} picks -> {len(deduped_picks)} (best per station per phase)")
    for dp in deduped_picks:
        pick_method = dp.metadata.get("method", "?")
        print(f"  {dp.station.network}.{dp.station.station} {dp.phase_type}: "
              f"method={pick_method}, weight={dp.weight:.4f}")

    if len(deduped_picks) < 3:
        return {
            "error": f"Need at least 3 deduplicated picks. Got {len(deduped_picks)}.",
            "total_picks": len(picks),
            "valid_picks": len(valid_picks),
        }

    # Create locator
    locator = EarthquakeLocator()

    # Locate
    try:
        hypocenter = locator.locate(deduped_picks, method=method, **kwargs)
        result = hypocenter.to_dict()

        # Add pick residuals
        result["picks"] = [
            {
                "station": p.station.station,
                "network": p.station.network,
                "phase": p.phase_type,
                "residual": round(float(p.residual), 3) if p.residual is not None else None,
                "weight": float(p.weight),
            }
            for p in deduped_picks
        ]

        return result

    except Exception as e:
        return {
            "error": str(e),
            "total_picks": len(picks),
            "valid_picks": len(valid_picks),
            "deduped_picks": len(deduped_picks),
        }



def plot_location_map_matplotlib(hypocenter, stations, output_path, title=None):
    """Fallback plotting method using plain matplotlib when Cartopy is unavailable."""
    import matplotlib.pyplot as plt
    import os
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    evt_lat = float(hypocenter.get("latitude", 0))
    evt_lon = float(hypocenter.get("longitude", 0))
    
    sta_lats = [float(s.get("latitude", 0)) for s in stations]
    sta_lons = [float(s.get("longitude", 0)) for s in stations]
    sta_names = [s.get("station", s.get("name", "UNK")) for s in stations]
    
    ax.scatter(sta_lons, sta_lats, c='blue', marker='^', s=100, label='Stations', zorder=5)
    for lon, lat, name in zip(sta_lons, sta_lats, sta_names):
        ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    ax.scatter([evt_lon], [evt_lat], c='red', marker='*', s=200, label='Event Location', zorder=10)
    ax.annotate("Epicenter", (evt_lon, evt_lat), xytext=(10, -10), textcoords='offset points', color='red', weight='bold')
    
    # Calculate margins for axes limits
    all_lats = [evt_lat] + sta_lats
    all_lons = [evt_lon] + sta_lons
    lat_margin = 1
    lon_margin = 1
    
    ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
    ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Earthquake Location Map (Matplotlib Fallback)')
        
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_location_map(
    hypocenter: Dict[str, Any],
    stations: List[Dict[str, Any]],
    output_path: str,
    region: Optional[List[float]] = None,
    title: Optional[str] = None,
) -> str:
    """
    Plot earthquake location and stations using Cartopy.

    Automatically detects whether a global or regional map is needed
    based on the spread of stations and event location.

    Args:
        hypocenter: Dictionary with hypocenter location (latitude, longitude, depth_km)
        stations: List of station dictionaries with latitude, longitude
        output_path: Path to save the output figure
        region: Optional [west, east, south, north] region bounds
        title: Optional plot title

    Returns:
        Path to the saved figure
    """
    # Get hypocenter coordinates
    evt_lat = float(hypocenter.get("latitude", 0))
    evt_lon = float(hypocenter.get("longitude", 0))
    evt_depth = float(hypocenter.get("depth_km", 0))

    # Get station coordinates
    sta_lats = [float(s.get("latitude", 0)) for s in stations]
    sta_lons = [float(s.get("longitude", 0)) for s in stations]
    sta_names = [s.get("station", s.get("name", "UNK")) for s in stations]

    all_lats = [evt_lat] + sta_lats
    all_lons = [evt_lon] + sta_lons
    lon_range = max(all_lons) - min(all_lons)
    lat_range = max(all_lats) - min(all_lats)
    is_global = lon_range > 180 or lat_range > 90

    if region is None:
        if is_global:
            region = [-180, 180, -90, 90]
        else:
            lat_margin = max(1.0, lat_range * 0.2 + 0.2)
            lon_margin = max(1.0, lon_range * 0.2 + 0.2)
            region = [
                min(all_lons) - lon_margin,
                max(all_lons) + lon_margin,
                max(-90, min(all_lats) - lat_margin),
                min(90, max(all_lats) + lat_margin),
            ]

    region = [float(r) for r in region]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[INFO] Cartopy unavailable ({e}). Switched to matplotlib fallback.")
        return plot_location_map_matplotlib(hypocenter, stations, output_path, title)

    try:
        projection = ccrs.Robinson() if is_global else ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=projection)
        data_crs = ccrs.PlateCarree()

        if is_global:
            ax.set_global()
        else:
            ax.set_extent(region, crs=data_crs)

        ax.add_feature(cfeature.OCEAN, facecolor="#DDEEFF", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#F6F2E9", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.5, edgecolor="gray", zorder=1)

        if not is_global:
            try:
                ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, edgecolor="gray", zorder=1)
            except Exception:
                pass

        gl = ax.gridlines(crs=data_crs, draw_labels=not is_global, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
        if not is_global:
            gl.top_labels = False
            gl.right_labels = False

        if sta_lons and sta_lats:
            ax.scatter(
                sta_lons, sta_lats,
                transform=data_crs,
                c="#00BCD4",
                marker="^",
                s=55,
                edgecolors="black",
                linewidths=0.6,
                label="Stations",
                zorder=3,
            )
            if len(sta_names) <= 40:
                for lon, lat, name in zip(sta_lons, sta_lats, sta_names):
                    ax.text(
                        lon, lat, name,
                        transform=data_crs,
                        fontsize=7,
                        color="#0A4A5A",
                        ha="left", va="bottom",
                        zorder=4,
                    )

        ax.scatter(
            [evt_lon], [evt_lat],
            transform=data_crs,
            c="red",
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.8,
            label="Epicenter",
            zorder=5,
        )

        frame_title = title or "Earthquake Location Map"
        ax.set_title(f"{frame_title}\nDepth: {evt_depth:.1f} km", fontsize=12)
        ax.legend(loc="lower right", framealpha=0.95)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Location map saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[INFO] Cartopy plotting failed ({e}). Switched to matplotlib fallback.")
        return plot_location_map_matplotlib(hypocenter, stations, output_path, title)


def plot_location_cross_section(
    hypocenter: Dict[str, Any],
    stations: List[Dict[str, Any]],
    output_path: str,
    azimuth: float = 0.0,
    width_km: float = 200.0,
) -> str:
    """
    Plot a cross-section showing earthquake depth.

    Args:
        hypocenter: Dictionary with hypocenter location
        stations: List of station dictionaries
        output_path: Path to save the output figure
        azimuth: Azimuth of cross-section line (degrees)
        width_km: Width of cross-section swath

    Returns:
        Path to the saved figure
    """
    try:
        import pygmt
    except ImportError:
        print("PyGMT not available. Skipping cross-section generation.")
        return None

    # Get hypocenter coordinates
    evt_lat = float(hypocenter.get("latitude", 0))
    evt_lon = float(hypocenter.get("longitude", 0))
    evt_depth = float(hypocenter.get("depth_km", 0))

    # Get station coordinates
    sta_lats = [float(s.get("latitude", 0)) for s in stations]
    sta_lons = [float(s.get("longitude", 0)) for s in stations]
    sta_names = [s.get("station", s.get("name", "UNK")) for s in stations]

    # Calculate distances from epicenter
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance in km between two points."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # Calculate distances and project onto cross-section line
    distances = []
    for slat, slon in zip(sta_lats, sta_lons):
        dist = haversine(evt_lat, evt_lon, slat, slon)
        # Simple projection onto azimuth
        # For now, just use distance
        distances.append(dist)

    # Create figure
    fig = pygmt.Figure()

    # Determine plot range
    max_dist = max(distances) if distances else 100
    max_depth = max(evt_depth * 1.5, 100)

    region = [0, max_dist * 1.2, max_depth, 0]  # Depth increases downward

    # Plot basemap
    fig.basemap(
        region=region,
        projection="X15c/8c",
        frame=[
            "xaf+u km",
            "yaf+u km",
            f'WSne+t"Cross Section (Azimuth: {azimuth:.0f}°)"',
        ],
    )

    # Plot stations on surface
    fig.plot(
        x=distances,
        y=[0] * len(distances),
        style="t0.3c",
        fill="blue",
        pen="black",
    )

    # Add station labels
    for dist, name in zip(distances, sta_names):
        fig.text(
            x=dist,
            y=5,
            text=name,
            font="8p,Helvetica,blue",
            justify="CB",
        )

    # Plot hypocenter
    fig.plot(
        x=0,
        y=evt_depth,
        style="a0.5c",
        fill="red",
        pen="black",
    )

    # Add hypocenter annotation
    fig.text(
        x=5,
        y=evt_depth,
        text=f"({evt_lat:.2f}, {evt_lon:.2f})",
        font="10p,Helvetica,red",
        justify="ML",
    )

    # Save figure
    fig.savefig(output_path, dpi=150)
    print(f"Cross-section saved to: {output_path}")

    return output_path


def plot_location_three_views(
    hypocenter: Dict[str, Any],
    stations: List[Dict[str, Any]],
    output_path: str,
    title: Optional[str] = None,
) -> str:
    """
    Plot a single large 3-view location figure:
    - map view (lon-lat)
    - lat-depth profile
    - lon-depth profile
    plus a magnitude legend panel.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib as mpl

    evt_lat = float(hypocenter.get("latitude", 0.0))
    evt_lon = float(hypocenter.get("longitude", 0.0))
    evt_depth = float(hypocenter.get("depth_km", hypocenter.get("depth", 0.0)))
    evt_mag = float(hypocenter.get("magnitude", hypocenter.get("mag", 3.0)))

    sta_lats = [float(s.get("latitude", 0.0)) for s in stations]
    sta_lons = [float(s.get("longitude", 0.0)) for s in stations]
    sta_names = [s.get("station", s.get("name", "UNK")) for s in stations]

    all_lats = [evt_lat] + sta_lats if sta_lats else [evt_lat]
    all_lons = [evt_lon] + sta_lons if sta_lons else [evt_lon]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    lat_pad = max(0.2, (max_lat - min_lat) * 0.15 + 0.1)
    lon_pad = max(0.2, (max_lon - min_lon) * 0.15 + 0.1)

    depth_cap = max(45.0, evt_depth + 8.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=depth_cap)
    cmap = cm.jet_r
    evt_color = cmap(norm(max(0.0, evt_depth)))

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    gs0 = fig.add_gridspec(8, 12)

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
        ax0.set_facecolor("#f5f6f8")
        # Terrain-like basemap without external tile service.
        try:
            ax0.stock_img()
        except Exception:
            pass
        gl = ax0.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax0 = fig.add_subplot(gs0[0:5, 0:7])
        ax0.set_facecolor("#f5f6f8")
        ax0.grid(linestyle="--", alpha=0.6)

    ax1 = fig.add_subplot(gs0[0:5, 7:12])  # depth-lat
    ax2 = fig.add_subplot(gs0[5:9, 0:7])   # lon-depth
    ax3 = fig.add_subplot(gs0[5:9, 7:12])  # magnitude legend

    if sta_lons and sta_lats:
        if use_cartopy:
            ax0.scatter(
                sta_lons, sta_lats,
                transform=data_crs,
                marker="^",
                c="#2a9d8f",
                s=35,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="Stations",
            )
            if len(sta_names) <= 40:
                for lon, lat, name in zip(sta_lons, sta_lats, sta_names):
                    ax0.text(lon, lat, name, transform=data_crs, fontsize=6.5, color="#1b4965")
        else:
            ax0.scatter(
                sta_lons, sta_lats,
                marker="^",
                c="#2a9d8f",
                s=35,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="Stations",
            )
            if len(sta_names) <= 40:
                for lon, lat, name in zip(sta_lons, sta_lats, sta_names):
                    ax0.text(lon, lat, name, fontsize=6.5, color="#1b4965")

    event_size = max(30.0, evt_mag * 20.0)
    if use_cartopy:
        ax0.scatter(
            [evt_lon], [evt_lat],
            transform=data_crs,
            marker="*",
            c=[evt_color],
            s=event_size * 2.2,
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Event",
        )
    else:
        ax0.scatter(
            [evt_lon], [evt_lat],
            marker="*",
            c=[evt_color],
            s=event_size * 2.2,
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Event",
        )
        ax0.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax0.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
        ax0.set_xlabel("Lon. (°)")
        ax0.set_ylabel("Lat. (°)")

    ax0.set_title("Plan View")
    ax0.legend(loc="lower right", fontsize=8, framealpha=0.9)

    if sta_lats:
        ax1.scatter(
            [0.0] * len(sta_lats),
            sta_lats,
            marker="^",
            c="#2a9d8f",
            s=30,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
    ax1.scatter(
        [max(0.0, evt_depth)],
        [evt_lat],
        marker="*",
        c=[evt_color],
        s=event_size * 1.6,
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )
    ax1.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    ax1.set_xlim(0.0, depth_cap)
    ax1.set_facecolor("#d9d9d9")
    ax1.tick_params(axis="both", which="major", labelsize=10)
    ax1.set_xlabel("Lat. (°)", fontsize=11)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_yticklabels([])
    ax1.set_title("Lat-Depth")

    if sta_lons:
        ax2.scatter(
            sta_lons,
            [0.0] * len(sta_lons),
            marker="^",
            c="#2a9d8f",
            s=30,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
    ax2.scatter(
        [evt_lon],
        [max(0.0, evt_depth)],
        marker="*",
        c=[evt_color],
        s=event_size * 1.6,
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )
    ax2.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax2.set_ylim(depth_cap, 0.0)
    ax2.set_facecolor("#d9d9d9")
    ax2.tick_params(axis="both", which="major", labelsize=10)
    ax2.set_xlabel("Lon. (°)", fontsize=11)
    ax2.set_ylabel("Depth (km)", fontsize=11)
    ax2.set_title("Lon-Depth")

    ax3.plot(0.10, 0.90, "o", mec="k", mfc="none", mew=1, ms=3 * 3)
    ax3.plot(0.10, 0.78, "o", mec="k", mfc="none", mew=1, ms=4 * 3)
    ax3.plot(0.10, 0.65, "o", mec="k", mfc="none", mew=1, ms=5 * 3)
    ax3.plot(0.10, 0.50, "o", mec="k", mfc="none", mew=1, ms=6 * 3)
    ax3.text(0.20, 0.88, "M 3.0", fontsize=11)
    ax3.text(0.20, 0.76, "M 4.0", fontsize=11)
    ax3.text(0.20, 0.62, "M 5.0", fontsize=11)
    ax3.text(0.20, 0.47, "M 6.0", fontsize=11)
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
    cbar.set_label(label="Depth (km)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title or "Earthquake Location (Three Views)", fontsize=13)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
