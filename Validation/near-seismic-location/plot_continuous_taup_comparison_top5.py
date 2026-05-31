"""
Plot top-5 magnitude-spanning events from continuous monitoring results,
comparing SCEDC truth TauP, Detected TauP, and ACTUAL PhaseNet AI Picks.
(With Local Catalog Caching & Publication-Quality Aesthetics)

Usage:
    python plot_continuous_taup_comparison_top5.py \
        --result Validation/near-seismic-location/continuous_results/continuous_20190704_170000.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

from obspy import UTCDateTime, read as obspy_read
import torch
import seisbench.models as sbm
from scipy.signal import find_peaks

from continuous_monitoring_validation import (
    _get_taup,
    download_continuous_data,
    fetch_scedc_catalog,
    match_catalogs,
)

CACHE_FILE = "scedc_truth_cache.json"

# =================== Device Setup ===================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cpu")

_pht_model = None
def _get_pht():
    global _pht_model
    if _pht_model is None:
        for name in ("scedc", "original"):
            try:
                _pht_model = sbm.PhaseNet.from_pretrained(name)
                _pht_model.to(DEVICE).eval()
                print(f"[INFO] PhaseNet ({name}) loaded for plotting on {DEVICE}")
                break
            except Exception:
                continue
    return _pht_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=str,
        default="/Users/chuan/Documents/code/QuakeCore/Validation/near-seismic-location/continuous_results/continuous_20190704_170000.json",
        help="Path to continuous monitoring result JSON",
    )
    parser.add_argument("--n-events", type=int, default=5, help="Number of events to plot")
    parser.add_argument("--max-stations", type=int, default=24, help="Max stations to show per figure")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="1-based event indices to plot (e.g. --indices 3 5)")
    parser.add_argument("--y-scale", type=float, default=1.0,
                        help="Waveform amplitude scaling (default 1.0)")
    return parser.parse_args()


def get_cached_truth_catalog(start_time, end_time, min_lat, max_lat, min_lon, max_lon):
    if os.path.exists(CACHE_FILE):
        print(f"[CACHE] Loading truth catalog from local cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            cached_data = json.load(f)
        for item in cached_data:
            item["time"] = UTCDateTime(item["time"])
        return cached_data

    print("[FETCH] Downloading SCEDC truth catalog from FDSN (this will be cached)...")
    truth_catalog = fetch_scedc_catalog(start_time, end_time, min_lat, max_lat, min_lon, max_lon)

    to_save = []
    for item in truth_catalog:
        dict_copy = item.copy()
        dict_copy["time"] = str(item["time"])
        to_save.append(dict_copy)

    with open(CACHE_FILE, 'w') as f:
        json.dump(to_save, f, indent=2)
    return truth_catalog


def first_tt(taup, depth_km, dist_deg, phase_list):
    try:
        arrs = taup.get_travel_times(
            source_depth_in_km=max(0.001, float(depth_km)),
            distance_in_degree=max(0.001, float(dist_deg)),
            phase_list=phase_list,
        )
        if arrs:
            return min(a.time for a in arrs)
    except Exception:
        return np.nan
    return np.nan


def quality_score(m):
    det = m["detected"]
    rms = float(det.get("rms", 1.0))
    gap = float(det.get("gap", 360.0))
    num_picks = float(det.get("num_picks", 0.0))
    dist_err = float(m.get("dist_err", 999.0))
    depth_err = float(m.get("depth_err", 999.0))
    return num_picks - 20.0 * rms - gap / 20.0 - dist_err / 3.0 - depth_err / 2.0


def pick_magnitude_spanning_best(matches, n_events):
    if not matches:
        return []
    mags = np.array([float(m["truth"].get("mag", 0.0)) for m in matches])
    mmin, mmax = float(np.min(mags)), float(np.max(mags))
    if n_events <= 1 or mmax <= mmin:
        return [max(matches, key=quality_score)]
    edges = np.linspace(mmin, mmax, n_events + 1)
    selected = []
    used_ids = set()
    for i in range(n_events):
        lo, hi = edges[i], edges[i + 1]
        if i == n_events - 1:
            cand = [m for m in matches if lo <= float(m["truth"].get("mag", 0.0)) <= hi]
        else:
            cand = [m for m in matches if lo <= float(m["truth"].get("mag", 0.0)) < hi]
        if not cand:
            continue
        best = max(cand, key=quality_score)
        key = str(best["truth"].get("id", "")) + str(best["truth"].get("time", ""))
        if key in used_ids:
            continue
        used_ids.add(key)
        selected.append(best)
    if len(selected) < n_events:
        rest = sorted(matches, key=quality_score, reverse=True)
        for m in rest:
            key = str(m["truth"].get("id", "")) + str(m["truth"].get("time", ""))
            if key in used_ids:
                continue
            selected.append(m)
            used_ids.add(key)
            if len(selected) >= n_events:
                break
    selected = sorted(selected[:n_events], key=lambda x: float(x["truth"].get("mag", 0.0)))
    return selected


S_COLOR = "#ff7f0e"  # orange for SCEDC Truth S picks
PHASENET_S_COLOR = "#d6279e"  # magenta-purple for PhaseNet S picks

def plot_event_on_ax(match_item, streams, stations, taup, pht_model, ax, max_stations=24, y_scale=1.8):
    """Plot a single event's record section on a given Axes."""
    truth = match_item["truth"]
    det = match_item["detected"]

    truth_time = UTCDateTime(truth["time"])

    # --- 1. 计算距离并挑选台站 ---
    entries = []
    for sta_id, mseed_path in streams.items():
        if sta_id not in stations:
            continue
        s = stations[sta_id]
        dist = np.degrees(
            2 * np.arcsin(np.minimum(np.sqrt(
                np.sin(np.radians((float(truth["lat"]) - s["latitude"]) / 2.0)) ** 2
                + np.cos(np.radians(float(truth["lat"]))) * np.cos(np.radians(s["latitude"]))
                * np.sin(np.radians((float(truth["lon"]) - s["longitude"]) / 2.0)) ** 2
            ), 1.0))
        )
        entries.append((sta_id, mseed_path, float(dist)))

    entries.sort(key=lambda x: x[2])
    entries = entries[:max_stations]

    if len(entries) < 4:
        return False

    phase_p = ["p", "P", "Pg", "Pn", "Pdiff"]
    phase_s = ["s", "S", "Sg", "Sn", "Sdiff"]

    dists = [x[2] for x in entries]
    scale = ((max(dists) - min(dists)) / len(dists)) * y_scale if len(dists) > 1 else 0.05
    if scale <= 0:
        scale = 0.05

    t0 = truth_time - 10
    t1 = truth_time + 60

    plot_count = 0
    for sta_id, mseed_path, dist in entries:
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
            st_3c.trim(starttime=t0, endtime=t1, pad=True, fill_value=0)

            if len(st_3c[0].data) < 200:
                continue

            st_3c.detrend("demean").detrend("linear").taper(max_percentage=0.05)
            st_3c.interpolate(100.0)

            # --- PhaseNet AI Pick ---
            ai_p_times, ai_s_times = [], []
            try:
                with torch.inference_mode():
                    anno = pht_model.annotate(st_3c.copy())
                tr_p = next((tr for tr in anno if tr.stats.channel.endswith('P')), None)
                if tr_p:
                    peaks, _ = find_peaks(tr_p.data, height=0.3, distance=50)
                    ai_p_times = [tr_p.stats.starttime + pk * tr_p.stats.delta for pk in peaks]
                tr_s = next((tr for tr in anno if tr.stats.channel.endswith('S')), None)
                if tr_s:
                    peaks, _ = find_peaks(tr_s.data, height=0.3, distance=50)
                    ai_s_times = [tr_s.stats.starttime + pk * tr_s.stats.delta for pk in peaks]
            except Exception:
                pass

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            # --- Z 分量波形 ---
            tr_z = st_3c.select(component="Z")[0].copy()
            tr_z.filter("bandpass", freqmin=1.0, freqmax=15.0, corners=4, zerophase=True)
            amp = tr_z.data.astype(np.float64)
            amp /= np.max(np.abs(amp)) + 1e-12
            t_rel = np.arange(len(amp)) / tr_z.stats.sampling_rate - 10.0

            offset = dist
            wave = amp * scale + offset

            ax.plot(t_rel, wave, '-', color='black', linewidth=0.8, alpha=0.8)
            ax.fill_between(t_rel, offset, wave, where=(wave > offset), color='black', alpha=0.3)

            if plot_count == 0 or dist == dists[0]:
                pass
            ax.text(-8, offset + scale * 0.2, sta_id.split('.')[-1], fontsize=7,
                    color='#333333', verticalalignment='bottom')

            tt_p_truth = first_tt(taup, truth["depth"], dist, phase_p)
            tt_s_truth = first_tt(taup, truth["depth"], dist, phase_s)

            tick_h = scale * 0.9

            if np.isfinite(tt_p_truth):
                ax.plot([tt_p_truth, tt_p_truth], [offset - tick_h, offset + tick_h],
                        '-', color='#1f77b4', lw=1.8, alpha=0.8, zorder=6)
            if np.isfinite(tt_s_truth):
                ax.plot([tt_s_truth, tt_s_truth], [offset - tick_h, offset + tick_h],
                        '-', color=S_COLOR, lw=1.8, alpha=0.8, zorder=6)

            for pt in ai_p_times:
                rel_pt = float(pt - truth_time)
                ax.plot([rel_pt, rel_pt], [offset - tick_h * 1.2, offset + tick_h * 1.2],
                        '-', color='#2ca02c', lw=2.0, zorder=8)
            for st_time in ai_s_times:
                rel_st = float(st_time - truth_time)
                ax.plot([rel_st, rel_st], [offset - tick_h * 1.2, offset + tick_h * 1.2],
                        '-', color=PHASENET_S_COLOR, lw=2.0, zorder=8)

            plot_count += 1

        except Exception:
            continue

    if plot_count == 0:
        return False

    ax.set_xlim(-10, 60)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    title = (
        f"Mw={truth.get('mag', 0.0):.2f}  "
        f"{truth_time.strftime('%H:%M:%S')}  "
        f"ΔDist={match_item.get('dist_err', np.nan):.1f}km  "
        f"ΔDepth={match_item.get('depth_err', np.nan):.1f}km"
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=3, loc="left")
    ax.set_ylabel("Dist (deg)", fontsize=9)
    return True


def main():
    args = parse_args()

    result_path = args.result
    if not os.path.isabs(result_path):
        result_path = os.path.join(os.getcwd(), result_path)

    with open(result_path, "r") as f:
        data = json.load(f)

    start_time = UTCDateTime(data["start_time"])
    end_time = UTCDateTime(data["end_time"])
    raw_detected_catalog = data.get("detected_catalog", [])
    detected_catalog = []
    for ev in raw_detected_catalog:
        if not isinstance(ev, dict):
            continue
        item = dict(ev)
        try:
            item["time"] = UTCDateTime(item["time"])
        except Exception:
            continue
        detected_catalog.append(item)

    min_lat, max_lat = 32.0, 36.5
    min_lon, max_lon = -120.0, -115.0

    print("[1/5] Loading continuous streams and station metadata...")
    streams, stations = download_continuous_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    if not streams or not stations:
        raise RuntimeError("Failed to load continuous data/stations.")

    print("[2/5] Fetching SCEDC truth catalog (with caching)...")
    truth_catalog = get_cached_truth_catalog(start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    truth_filtered = [t for t in truth_catalog if float(t.get("mag", 0.0)) >= 1.0]

    print("[3/5] Matching detected catalog with truth catalog...")
    matches, _, _ = match_catalogs(detected_catalog, truth_filtered)
    if not matches:
        raise RuntimeError("No matched events found for plotting.")

    print("[4/5] Selecting magnitude-spanning best events...")
    selected = pick_magnitude_spanning_best(matches, args.n_events)

    # Filter by user-specified indices (1-based)
    if args.indices:
        filtered = []
        for idx in args.indices:
            if 1 <= idx <= len(selected):
                filtered.append(selected[idx - 1])
            else:
                print(f"  [WARN] Index {idx} out of range (1-{len(selected)})")
        if filtered:
            selected = filtered

    taup = _get_taup()
    pht_model = _get_pht()

    out_dir = os.path.join(os.path.dirname(__file__), "taup_compare_top5")
    os.makedirs(out_dir, exist_ok=True)

    legend_items = [
        Line2D([0], [0], color='#2ca02c', lw=2.0, linestyle='-', label='PhaseNet Pick (P)'),
        Line2D([0], [0], color=PHASENET_S_COLOR, lw=2.0, linestyle='-', label='PhaseNet Pick (S)'),
        Line2D([0], [0], color='#1f77b4', lw=1.8, linestyle='-', label='SCEDC Truth TauP (P)'),
        Line2D([0], [0], color=S_COLOR, lw=1.8, linestyle='-', label='SCEDC Truth TauP (S)'),
    ]

    print(f"[5/5] Plotting {len(selected)} event(s) individually...")
    summary = []
    for i, m in enumerate(selected, start=1):
        t = UTCDateTime(m["truth"]["time"])
        mag = float(m["truth"].get("mag", 0.0))
        out_png = os.path.join(out_dir, f"{i:02d}_{t.strftime('%Y%m%d_%H%M%S')}_Mw{mag:.2f}.png")

        fig, ax = plt.subplots(figsize=(14, 11), facecolor='white')
        ok = plot_event_on_ax(m, streams, stations, taup, pht_model, ax,
                              args.max_stations, args.y_scale)
        if not ok:
            plt.close(fig)
            print(f"  [{i}] Skipped (insufficient stations)")
            continue

        ax.set_xlabel("Time relative to SCEDC Origin Time (s)", fontsize=12)
        ax.tick_params(labelsize=10)
        leg = ax.legend(handles=legend_items, loc="upper right", fontsize=25,
                  framealpha=0.95, edgecolor='#dddddd')

        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)

        summary.append({
            "index": i,
            "output": out_png,
            "truth_time": str(m["truth"]["time"]),
            "truth_mag": mag,
            "dist_err_km": float(m.get("dist_err", np.nan)),
            "depth_err_km": float(m.get("depth_err", np.nan)),
        })
        print(f"  [{i}] Saved: {out_png}")

    summary_path = os.path.join(out_dir, "summary_top5.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()