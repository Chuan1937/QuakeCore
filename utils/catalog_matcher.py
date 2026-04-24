"""
Catalog Matching Module for QuakeCore
=====================================
Matches detected earthquake catalogs against ground truth catalogs
to evaluate detection performance (precision, recall, location errors).
"""

from typing import Dict, List, Tuple, Any

import numpy as np

from .continuous_data import haversine_km


def _to_epoch_seconds(value: Any) -> float:
    """Convert supported time representations to epoch seconds (float)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    # ObsPy UTCDateTime and similar objects expose timestamp as attr or method.
    ts_attr = getattr(value, "timestamp", None)
    if callable(ts_attr):
        try:
            return float(ts_attr())
        except Exception:
            pass
    if ts_attr is not None:
        try:
            return float(ts_attr)
        except Exception:
            pass

    # Datetime-like fallback
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            pass

    # Last resort: parse as float-like string
    try:
        return float(str(value))
    except Exception:
        return 0.0


def match_catalogs(
    detected_cat: List[Dict[str, Any]],
    truth_cat: List[Dict[str, Any]],
    time_threshold: float = 5.0,
    distance_threshold: float = 30.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match detected events against ground truth catalog.

    A detection is considered a true positive if:
    - Time difference <= time_threshold (seconds)
    - Spatial distance <= distance_threshold (km)

    Args:
        detected_cat: List of detected event dicts with keys: time, latitude, longitude, depth
        truth_cat: List of ground truth event dicts with keys: time, lat, lon, depth, mag
        time_threshold: Maximum time difference for matching (seconds)
        distance_threshold: Maximum distance for matching (km)

    Returns:
        Tuple of (matches, false_positives, false_negatives) where:
        - matches: List of match dicts with detected, truth, dist_err, depth_err
        - false_positives: List of detected events with no match
        - false_negatives: List of truth events not matched by any detection
    """
    matched_truth_indices = set()
    matches = []
    false_positives = []

    for det in detected_cat:
        det_time = _to_epoch_seconds(det.get("time"))
        best_match = None
        min_time_diff = float('inf')
        best_truth_idx = -1

        for i, tru in enumerate(truth_cat):
            tru_time = _to_epoch_seconds(tru.get("time"))
            t_diff = abs(det_time - tru_time)
            dist = haversine_km(det["latitude"], det["longitude"], tru["lat"], tru["lon"])

            # Check if this is a better match than current best
            if t_diff <= time_threshold and dist <= distance_threshold:
                if t_diff < min_time_diff:
                    min_time_diff = t_diff
                    best_match = tru
                    best_truth_idx = i

        if best_match:
            matched_truth_indices.add(best_truth_idx)
            matches.append({
                "detected": det,
                "truth": best_match,
                "dist_err": haversine_km(
                    det["latitude"], det["longitude"],
                    best_match["lat"], best_match["lon"]
                ),
                "depth_err": abs(det["depth"] - best_match["depth"])
            })
        else:
            false_positives.append(det)

    false_negatives = [tru for i, tru in enumerate(truth_cat) if i not in matched_truth_indices]

    return matches, false_positives, false_negatives


def compute_detection_stats(
    detected_cat: List[Dict[str, Any]],
    truth_cat: List[Dict[str, Any]],
    matches: List[Dict[str, Any]],
    false_positives: List[Dict[str, Any]],
    false_negatives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute detection statistics from catalog matching results.

    Args:
        detected_cat: Full list of detected events
        truth_cat: Full list of ground truth events
        matches: List of matched events
        false_positives: List of false positive detections
        false_negatives: List of missed ground truth events

    Returns:
        Dict with statistics: n_truth, n_detected, n_matched, recall, precision,
        avg_dist_err, avg_depth_err, etc.
    """
    n_truth = len(truth_cat)
    n_detected = len(detected_cat)
    n_matched = len(matches)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)

    recall = (n_matched / n_truth * 100) if n_truth > 0 else 0.0
    precision = (n_matched / n_detected * 100) if n_detected > 0 else 0.0

    stats = {
        "n_truth": n_truth,
        "n_detected": n_detected,
        "n_matched": n_matched,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "recall_percent": recall,
        "precision_percent": precision,
    }

    if matches:
        dist_errors = [m["dist_err"] for m in matches]
        depth_errors = [m["depth_err"] for m in matches]
        stats.update({
            "avg_dist_err_km": float(np.mean(dist_errors)),
            "median_dist_err_km": float(np.median(dist_errors)),
            "max_dist_err_km": float(np.max(dist_errors)),
            "avg_depth_err_km": float(np.mean(depth_errors)),
            "median_depth_err_km": float(np.median(depth_errors)),
        })

    return stats


def print_detection_summary(stats: Dict[str, Any]) -> str:
    """Generate a human-readable summary string from detection stats."""
    lines = [
        "=" * 40,
        "  DETECTION PERFORMANCE SUMMARY",
        "=" * 40,
        f"  Ground Truth Events     : {stats.get('n_truth', 0)}",
        f"  Total Detected Events   : {stats.get('n_detected', 0)}",
        f"  True Positives (Matched): {stats.get('n_matched', 0)}",
        f"  False Negatives (Missed) : {stats.get('n_false_negatives', 0)}",
        f"  False Positives (Ghosts) : {stats.get('n_false_positives', 0)}",
        "",
        f"  Recall    : {stats.get('recall_percent', 0):.1f}%",
        f"  Precision : {stats.get('precision_percent', 0):.1f}%",
    ]

    if "avg_dist_err_km" in stats:
        lines.extend([
            "",
            f"  Avg Dist Error : {stats['avg_dist_err_km']:.2f} km",
            f"  Avg Dep Error  : {stats['avg_depth_err_km']:.2f} km",
        ])

    return "\n".join(lines)
