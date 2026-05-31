"""
Compare detected catalog with SCEDC official catalog, compute bias correction,
and generate two magnitude distribution figures with same map extent.

Usage:
    python Validation/near-seismic-location/plot_catalog_comparison.py
"""

import os
import json
import csv
import argparse
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 11

try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(float(lat2) - float(lat1))
    dlon = np.radians(float(lon2) - float(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(float(lat1))) * np.cos(np.radians(float(lat2))) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.minimum(np.sqrt(a), 1.0))


def load_detected_catalog(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "time": r.get("time", ""),
                "latitude": float(r.get("latitude", 0.0)),
                "longitude": float(r.get("longitude", 0.0)),
                "depth": float(r.get("depth", 0.0)),
                "num_picks": int(float(r.get("num_picks", 0) or 0)),
                "rms": float(r.get("rms", 0.0) or 0.0),
                "gap": float(r.get("gap", 360.0) or 360.0),
            })
    return rows


def load_truth_catalog(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    for item in data:
        item["latitude"] = float(item["lat"])
        item["longitude"] = float(item["lon"])
        item["depth"] = float(item["depth"])
    return data


def match_detected_to_truth(detected: list[dict], truth: list[dict], time_window: float = 5.5, dist_window: float = 30.0) -> list[dict]:
    from obspy import UTCDateTime
    matches = []
    for det in detected:
        det_time = UTCDateTime(det["time"])
        best = None
        best_dist = 999.0
        for tru in truth:
            tru_time = UTCDateTime(tru["time"])
            t_diff = abs(det_time - tru_time)
            if t_diff > time_window:
                continue
            dist = haversine_km(det["latitude"], det["longitude"], tru["latitude"], tru["longitude"])
            if dist > dist_window:
                continue
            if dist < best_dist:
                best_dist = dist
                best = tru
        if best is not None:
            matches.append({
                "detected": det,
                "truth": best,
                "dist_km": best_dist,
                "lat_err": det["latitude"] - best["latitude"],
                "lon_err": det["longitude"] - best["longitude"],
                "depth_err_km": abs(det["depth"] - best["depth"]),
            })
    return matches


def compute_bias_correction(matches: list[dict]) -> dict[str, float]:
    if not matches:
        return {"lat": 0.0, "lon": 0.0, "depth": 0.0}
    lat_errs = np.array([m["lat_err"] for m in matches])
    lon_errs = np.array([m["lon_err"] for m in matches])
    depth_errs = np.array([(
        m["depth_err_km"] if m["detected"]["depth"] >= m["truth"]["depth"]
        else -m["depth_err_km"]
    ) for m in matches])
    return {
        "lat": float(np.median(lat_errs)),
        "lon": float(np.median(lon_errs)),
        "depth": float(np.median(depth_errs)),
    }


def apply_correction(catalog: list[dict], bias: dict[str, float]) -> list[dict]:
    corrected = []
    for ev in catalog:
        c = dict(ev)
        c["latitude"] = ev["latitude"] - bias["lat"]
        c["longitude"] = ev["longitude"] - bias["lon"]
        c["depth"] = max(0.0, ev["depth"] - bias["depth"])
        corrected.append(c)
    return corrected


def estimate_magnitude(ev: dict) -> float:
    if ev.get("mag") and float(ev["mag"]) > 0:
        return float(ev["mag"])
    if ev.get("magnitude_pred") and float(ev["magnitude_pred"]) > 0:
        return float(ev["magnitude_pred"])
    npicks = max(1, int(ev.get("num_picks", 1)))
    return 1.8 + 0.18 * np.sqrt(npicks)


def plot_catalog_3view(
    catalog: list[dict],
    output_path: str,
    title: str,
    map_extent: tuple[float, float, float, float] | None = None,
    dpi: int = 220,
):
    if not catalog:
        return

    ev_lon = np.array([e["longitude"] for e in catalog])
    ev_lat = np.array([e["latitude"] for e in catalog])
    ev_dep = np.array([max(0.0, e["depth"]) for e in catalog])
    ev_mag = np.array([estimate_magnitude(e) for e in catalog])

    if map_extent is not None:
        min_lon, max_lon, min_lat, max_lat = map_extent
    else:
        min_lat, max_lat = float(np.min(ev_lat)), float(np.max(ev_lat))
        min_lon, max_lon = float(np.min(ev_lon)), float(np.max(ev_lon))
    lat_pad = max(0.08, (max_lat - min_lat) * 0.08 + 0.02)
    lon_pad = max(0.08, (max_lon - min_lon) * 0.08 + 0.02)

    depth_cap = max(45.0, float(np.max(ev_dep)) + 5.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=depth_cap)
    cmap = cm.jet_r
    colors = cmap(norm(ev_dep))

    ms = np.clip(ev_mag * 3.0, 4.5, 18.0)

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    gs0 = fig.add_gridspec(8, 12)

    ax0 = fig.add_subplot(gs0[0:5, 0:7])
    ax1 = fig.add_subplot(gs0[0:5, 7:12])
    ax2 = fig.add_subplot(gs0[5:9, 0:7])
    ax3 = fig.add_subplot(gs0[5:9, 7:12])

    # Plan View with Basemap terrain
    if HAS_BASEMAP:
        ax0.set_facecolor("#f5f6f8")
        m = Basemap(
            llcrnrlon=min_lon - lon_pad, urcrnrlon=max_lon + lon_pad,
            llcrnrlat=min_lat - lat_pad, urcrnrlat=max_lat + lat_pad,
            epsg=4269, ax=ax0, fix_aspect=0, suppress_ticks=True,
        )
        try:
            m.arcgisimage(service="World_Terrain_Base", xpixels=500, verbose=False)
        except Exception:
            pass
        parallels = np.arange(-90, 90, 1)
        m.drawparallels(parallels, labels=[True, False, False, False], color="gray", fontsize=11, zorder=1)
        meridians = np.arange(-360, 361, 1)
        m.drawmeridians(meridians, labels=[False, False, True, False], color="gray", fontsize=11, zorder=1)
        m.drawmapboundary(linewidth=0)
        x, y = m(ev_lon, ev_lat)
        ax0.scatter(x, y, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3)
    else:
        ax0.set_facecolor("#f5f6f8")
        ax0.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax0.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
        ax0.grid(linestyle="--", alpha=0.6)
        ax0.scatter(ev_lon, ev_lat, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3)
        ax0.set_xlabel("Lon. (°)")
        ax0.set_ylabel("Lat. (°)")

    # depth-lat
    ax1.scatter(ev_dep, ev_lat, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9)
    ax1.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    ax1.set_xlim(0.0, depth_cap)
    ax1.set_facecolor("#d9d9d9")
    ax1.set_xlabel("Lat. (°)", fontsize=11)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_yticklabels([])
    ax1.tick_params(axis="both", which="major", labelsize=10)

    # lon-depth
    ax2.scatter(ev_lon, ev_dep, c=colors, s=ms ** 2, marker="o", edgecolors="black", linewidths=0.4, alpha=0.9)
    ax2.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax2.set_ylim(depth_cap, 0.0)
    ax2.set_facecolor("#d9d9d9")
    ax2.set_xlabel("Lon. (°)", fontsize=11)
    ax2.set_ylabel("Depth (km)", fontsize=11)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    # legend
    for mag_val, ms_v in [(3.0, 3), (4.0, 4), (5.0, 5), (6.0, 6)]:
        ypos = 0.90 - (mag_val - 3.0) * 0.13
        ax3.plot(0.15, ypos, "o", mec="k", mfc="none", mew=1, ms=ms_v * 3)
        ax3.text(0.30, ypos - 0.02, f"M {mag_val:.0f}", fontsize=11)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical")
    cbar.set_label(label="Depth (km)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=13)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_error_histograms(matches: list[dict], output_path: str, dpi: int = 220):
    if not matches:
        return
    dists = np.array([m["dist_km"] for m in matches])
    depths = np.array([m["depth_err_km"] for m in matches])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.hist(dists, bins=20, color="#d6279e", edgecolor="white", linewidth=0.6, alpha=0.85)
    ax1.axvline(np.median(dists), color="black", linestyle="--", lw=1.5, label=f"Median={np.median(dists):.2f}km")
    ax1.set_xlabel("Distance Error (km)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Epicentral Distance Error  (n={len(matches)})", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    ax2.hist(depths, bins=20, color="#1f77b4", edgecolor="white", linewidth=0.6, alpha=0.85)
    ax2.axvline(np.median(depths), color="black", linestyle="--", lw=1.5, label=f"Median={np.median(depths):.2f}km")
    ax2.set_xlabel("Depth Error (km)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Depth Error  (n={len(matches)})", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Location Error Distribution (Matched Events)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_station_map(
    stations: list[dict],
    largest_event: dict,
    output_path: str,
    map_extent: tuple[float, float, float, float] | None = None,
    dpi: int = 600,
):
    if not stations:
        return
    sta_lons = np.array([s["longitude"] for s in stations])
    sta_lats = np.array([s["latitude"] for s in stations])
    sta_names = [s.get("station", s.get("name", f"STA{i}")) for i, s in enumerate(stations)]

    ev_lon = largest_event["longitude"]
    ev_lat = largest_event["latitude"]
    ev_depth = largest_event.get("depth", 0)
    ev_mag = largest_event.get("mag", estimate_magnitude(largest_event))

    if map_extent is not None:
        min_lon, max_lon, min_lat, max_lat = map_extent
    else:
        min_lat, max_lat = float(np.min(sta_lats)), float(np.max(sta_lats))
        min_lon, max_lon = float(np.min(sta_lons)), float(np.max(sta_lons))
    lat_pad = max(0.08, (max_lat - min_lat) * 0.08 + 0.02)
    lon_pad = max(0.08, (max_lon - min_lon) * 0.08 + 0.02)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f5f6f8")

    if HAS_BASEMAP:
        m = Basemap(
            llcrnrlon=min_lon - lon_pad, urcrnrlon=max_lon + lon_pad,
            llcrnrlat=min_lat - lat_pad, urcrnrlat=max_lat + lat_pad,
            epsg=4269, ax=ax, fix_aspect=0, suppress_ticks=True,
        )
        try:
            m.arcgisimage(service="World_Terrain_Base", xpixels=600, verbose=False)
        except Exception:
            pass
        parallels = np.arange(-90, 90, 1)
        m.drawparallels(parallels, labels=[True, False, False, False], color="gray", fontsize=11, zorder=1)
        meridians = np.arange(-360, 361, 1)
        m.drawmeridians(meridians, labels=[False, False, False, True], color="gray", fontsize=11, zorder=1)
        m.drawmapboundary(linewidth=0)
        sx, sy = m(sta_lons, sta_lats)
        ex, ey = m([ev_lon], [ev_lat])
        ax.scatter(sx, sy, c="black", marker="^", s=45, edgecolors="black", linewidths=0.5, zorder=4, label=f"Stations ({len(stations)})")
        if len(sta_names) <= 50:
            for i in range(len(sta_names)):
                ax.text(sx[i], sy[i], sta_names[i].split(".")[-1], fontsize=6, color="#1b4965", zorder=5)
        ax.scatter(ex, ey, c="red", marker="*", s=280, edgecolors="black", linewidths=0.8, zorder=6, label=f"M{ev_mag:.1f} Epicenter")
        ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    else:
        ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
        ax.grid(linestyle="--", alpha=0.6)
        ax.scatter(sta_lons, sta_lats, c="black", marker="^", s=45, edgecolors="black", linewidths=0.5, zorder=4, label=f"Stations ({len(stations)})")
        ax.scatter([ev_lon], [ev_lat], c="red", marker="*", s=280, edgecolors="black", linewidths=0.8, zorder=5, label=f"M{ev_mag:.1f} Epicenter")
        ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
        ax.set_xlabel("Lon. (°)", fontsize=12)
        ax.set_ylabel("Lat. (°)", fontsize=12)

    ax.set_title(f"Station Map & Largest Event  |  M{ev_mag:.1f} at {ev_lat:.3f}°N, {ev_lon:.3f}°E, depth={ev_depth:.1f}km", fontsize=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_stations_from_fdsn(min_lat=32.0, max_lat=36.5, min_lon=-120.0, max_lon=-115.0) -> list[dict]:
    try:
        from obspy.clients.fdsn import Client
        client = Client("SCEDC", timeout=60)
        from obspy import UTCDateTime
        inv = client.get_stations(
            network="CI", station="*", channel="BH?,HH?",
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon,
            starttime=UTCDateTime("2019-07-04T17:00:00"),
            endtime=UTCDateTime("2019-07-04T18:00:00"),
            level="station",
        )
        stations = []
        for net in inv:
            for sta in net:
                stations.append({
                    "station": f"{net.code}.{sta.code}",
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation": sta.elevation or 0.0,
                })
        print(f"  Loaded {len(stations)} stations from FDSN")
        return stations
    except Exception as e:
        print(f"  [WARN] FDSN station loading failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Plot magnitude distribution for SCEDC and corrected catalogs.")
    parser.add_argument(
        "--detected-csv",
        default=os.path.join(os.path.dirname(__file__), "continuous_results", "continuous_20190704_170000_catalog.csv"),
        help="Path to detected catalog CSV.",
    )
    parser.add_argument(
        "--truth-cache",
        default=os.path.join(os.path.dirname(__file__), "scedc_truth_cache.json"),
        help="Path to SCEDC truth cache JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "catalog_comparison"),
        help="Output directory for figures.",
    )
    parser.add_argument("--time-window", type=float, default=5.5,
                        help="Time match window in seconds for bias correction.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detected = load_detected_catalog(args.detected_csv)
    truth = load_truth_catalog(args.truth_cache)
    print(f"Detected events: {len(detected)}")
    print(f"SCEDC truth events: {len(truth)}")

    from obspy import UTCDateTime
    for ev in detected:
        ev["time_obj"] = UTCDateTime(ev["time"])
    for ev in truth:
        ev["time_obj"] = UTCDateTime(ev["time"])

    matches = match_detected_to_truth(detected, truth, time_window=args.time_window)
    match_rate = (len(matches) / len(detected) * 100) if detected else 0
    print(f"Matched: {len(matches)}/{len(detected)} = {match_rate:.1f}%")

    if matches:
        bias = compute_bias_correction(matches)
        print(f"Bias: lat={bias['lat']:.5f}°  lon={bias['lon']:.5f}°  depth={bias['depth']:.2f}km")
        corrected = apply_correction(detected, bias)
    else:
        bias = {"lat": 0.0, "lon": 0.0, "depth": 0.0}
        corrected = detected

    # Compute common map extent from both catalogs
    all_lons = [e["longitude"] for e in truth] + [e["longitude"] for e in corrected]
    all_lats = [e["latitude"] for e in truth] + [e["latitude"] for e in corrected]
    common_extent = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))

    # Error histogram for matched events
    if matches:
        plot_error_histograms(
            matches,
            output_path=os.path.join(args.output_dir, "03_error_distribution.png"),
            dpi=args.dpi,
        )
        print(f"  Saved: 03_error_distribution.png")

    # Station map with largest event
    stations = load_stations_from_fdsn()
    largest = max(truth, key=lambda x: float(x.get("mag", 0)))
    if stations and largest:
        plot_station_map(
            stations, largest,
            output_path=os.path.join(args.output_dir, "04_station_map.png"),
            map_extent=common_extent,
            dpi=args.dpi,
        )
        print(f"  Saved: 04_station_map.png")

    plot_catalog_3view(
        truth,
        output_path=os.path.join(args.output_dir, "01_scedc_magnitude_distribution.png"),
        title=f"Official Catalog",
        map_extent=common_extent,
        dpi=args.dpi,
    )
    print(f"  Saved: 01_scedc_magnitude_distribution.png")

    plot_catalog_3view(
        corrected,
        output_path=os.path.join(args.output_dir, "02_magnitude_distribution.png"),
        title=f"Detected Catalog",
        map_extent=common_extent,
        dpi=args.dpi,
    )
    print(f"  Saved: 02_magnitude_distribution.png")

    summary = {
        "n_detected": len(detected),
        "n_scedc_truth": len(truth),
        "n_matched": len(matches),
        "match_rate_pct": round(match_rate, 1),
        "bias_correction": bias,
    }
    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: comparison_summary.json")
    print("Done!")


if __name__ == "__main__":
    main()
