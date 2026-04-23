"""
Multi-Event Association Module for QuakeCore
============================================
Greedy REAL-Lite algorithm for associating seismic picks into multiple events.
Designed for continuous monitoring scenarios where multiple events may occur
in a time window.
"""

import itertools
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from obspy import UTCDateTime
from scipy.interpolate import RegularGridInterpolator


# =================== Distance Functions ===================

def _dist_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate angular distance in degrees between two points."""
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.minimum(np.sqrt(a), 1.0)))


# =================== Greedy Multi-Event Association ===================

def associate_multiple_events(
    picks: List[Dict[str, Any]],
    stations: Dict[str, Dict[str, float]],
    tt_interp: Dict[str, RegularGridInterpolator],
    time_tolerance: float = 2.0,
    min_picks: int = 6,
    grid_lat_range: Tuple[float, float] = (32.0, 36.5),
    grid_lon_range: Tuple[float, float] = (-120.0, -115.0),
    trial_depth: float = 10.0,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Greedy algorithm to associate picks into multiple events.

    Iteratively finds the strongest event cluster, records it, and removes
    associated picks from the pool until no sufficient picks remain.

    Args:
        picks: List of pick dicts with keys: station_id, phase, time/time_epoch, score, model
        stations: Dict mapping station_id to station info (latitude, longitude, elevation)
        tt_interp: Dict mapping phase ("P", "S") to RegularGridInterpolator for travel times
        time_tolerance: Time window for associating picks (seconds)
        min_picks: Minimum picks required to form an event
        grid_lat_range: (min, max) latitude for grid search
        grid_lon_range: (min, max) longitude for grid search
        trial_depth: Trial depth for initial grid search (km)

    Returns:
        List of (event_info, event_picks) tuples for all detected events
    """
    unassociated_picks = picks.copy()
    detected_events = []

    # Southern California coarse grid
    lats = np.arange(grid_lat_range[0], grid_lat_range[1], 0.1)
    lons = np.arange(grid_lon_range[0], grid_lon_range[1], 0.1)

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

        grid_points = list(itertools.product(lats, lons))
        for trial_lat, trial_lon in grid_points:
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

                # Deduplicate picks (best per station per phase)
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
            # Remove associated picks from pool
            picked_ids = {id(p) for p in best_event_picks}
            unassociated_picks = [p for p in unassociated_picks if id(p) not in picked_ids]
            iteration += 1
        else:
            break

    print(f"  [Association Done] Total {len(detected_events)} events isolated.")
    return detected_events
