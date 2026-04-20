"""
Generate NonLinLoc 2D travel-time tables for TRANS GLOBAL mode.
Uses ObsPy TauP with progress output.

Generates files in the format expected by NLLoc GLOBAL mode:
  {root}.{PHASE}.DEFAULT.time.buf
  {root}.{PHASE}.DEFAULT.time.hdr

Header format (from TauP_Table_NLL.java writeGridHeader):
  Line 1: 1 {numDist} {numDepth} 0.0 {dist0} {depth0} 0.0 {deltaDist} {deltaDepth} {TYPE} FLOAT
  Line 2: {stationName} {stationLon} {stationLat} {stationDepth}

Buffer format: float32, written as (for each distance point, write all depth values)
  -> dist-major order: for dist in distances: for depth in depths: write float

Default grid: 0-180 deg distance (181 points), 0-700 km depth (141 points)
"""

import os
import sys
import time
import struct
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from obspy.taup import TauPyModel


def generate_tables(model_name="iasp91", out_dir=None,
                    n_dist=181, dist_max=180.0,
                    n_depth=141, depth_max=700.0):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taup")
    os.makedirs(out_dir, exist_ok=True)

    model = TauPyModel(model=model_name)
    distances = np.linspace(0.0, dist_max, n_dist)
    depths = np.linspace(0.0, depth_max, n_depth)

    dist0 = distances[0]
    depth0 = depths[0]
    delta_dist = distances[1] - distances[0] if n_dist > 1 else 1.0
    delta_depth = depths[1] - depths[0] if n_depth > 1 else 1.0

    phase_sets = {
        "P": ["P", "p", "Pn", "Pdiff", "PKP", "PKiKP", "PKIKP"],
        "S": ["S", "s", "Sn", "Sdiff", "SKS", "SKiKS", "SKIKS"],
    }

    total_queries = n_dist * n_depth * len(phase_sets)
    done = 0
    t0 = time.time()

    for phase_label, phase_list in phase_sets.items():
        print(f"\n--- Generating {phase_label} ({n_dist}x{n_depth} = {n_dist*n_depth} queries) ---")

        # Write header (TauP_Table_NLL format)
        hdr_path = os.path.join(out_dir, f"{model_name}.{phase_label}.DEFAULT.time.hdr")
        with open(hdr_path, "w") as f:
            f.write(f"1 {n_dist} {n_depth} 0.0 {dist0:.4f} {depth0:.4f} 0.0 "
                    f"{delta_dist:.4f} {delta_depth:.4f} TIME2D FLOAT\n")
            f.write(f"DEFAULT 0.0 0.0 0.0\n")

        # Open binary file
        buf_path = os.path.join(out_dir, f"{model_name}.{phase_label}.DEFAULT.time.buf")
        buf_file = open(buf_path, "wb")

        for i, dist in enumerate(distances):
            depth_times = np.full(n_depth, -1.0, dtype=np.float32)

            for j, depth in enumerate(depths):
                try:
                    arrivals = model.get_travel_times(
                        source_depth_in_km=depth,
                        distance_in_degree=dist,
                        phase_list=phase_list,
                    )
                    if arrivals:
                        depth_times[j] = arrivals[0].time
                except Exception:
                    pass
                done += 1

            # Write all depth values for this distance point (dist-major)
            depth_times.tofile(buf_file)

            if (i + 1) % 20 == 0 or i == n_dist - 1:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_queries - done) / rate if rate > 0 else 0
                print(f"  {phase_label}: dist {i+1}/{n_dist}  "
                      f"done {done}/{total_queries}  "
                      f"rate {rate:.0f}/s  ETA {eta:.0f}s")

        buf_file.close()

        size_mb = os.path.getsize(buf_path) / 1024 / 1024
        print(f"  -> {buf_path} ({size_mb:.1f} MB)")
        print(f"  -> {hdr_path}")

    print(f"\nDone in {time.time()-t0:.0f}s. Tables in {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="iasp91")
    parser.add_argument("--n-dist", type=int, default=181)
    parser.add_argument("--n-depth", type=int, default=141)
    args = parser.parse_args()
    generate_tables(args.model, n_dist=args.n_dist, n_depth=args.n_depth)
