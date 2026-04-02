#!/usr/bin/env python
"""
Download test seismic data for QuakeCore demonstration.

This script downloads waveform data from IRIS for a well-recorded earthquake.
"""

import os
import sys
import json

def download_test_data():
    """Download test data using ObsPy FDSN client."""

    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime
        from obspy.geodetics import locations2degrees
        OBSPY_AVAILABLE = True
    except ImportError:
        print("ObsPy not installed. Installing required packages...")
        os.system("pip install obspy")
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime
        from obspy.geodetics import locations2degrees

    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("QuakeCore Test Data Downloader")
    print("=" * 60)

    # Event: 2020-10-19 M7.6 South of Alaska
    # This is a well-recorded teleseismic event with good global coverage
    event_info = {
        "name": "2020-10-19 M7.6 South of Alaska",
        "time": UTCDateTime("2020-10-19T20:54:39"),
        "latitude": 54.65,
        "longitude": -159.67,
        "depth_km": 28.0,
        "magnitude": 7.6,
    }

    print(f"\nEvent: {event_info['name']}")
    print(f"  Time: {event_info['time']}")
    print(f"  Location: {event_info['latitude']:.2f}°N, {abs(event_info['longitude']):.2f}°W")
    print(f"  Depth: {event_info['depth_km']} km")
    print(f"  Magnitude: M{event_info['magnitude']}")

    # Select stations with good global distribution
    # Format: (network, station, description, known_lat, known_lon, known_elev)
    stations = [
        ("IU", "ANMO", "Albuquerque, New Mexico, USA", 34.9459, -106.4572, 1820),
        ("IU", "COLA", "College Outpost, Alaska, USA", 64.8734, -147.8613, 157),
        ("IU", "HRV", "Harvard, Massachusetts, USA", 42.5064, -71.5557, 282),
        ("IU", "MAJO", "Matsushiro, Japan", 36.5459, 138.2056, 406),
        ("IU", "TATO", "Taipei, Taiwan", 24.9754, 121.4878, 93),
        ("II", "KDAK", "Kodiak Island, Alaska, USA", 57.7819, -152.5844, 114),
    ]

    print(f"\nDownloading data from {len(stations)} stations...")

    # Connect to IRIS
    client = Client("IRIS")

    # Download parameters
    time_before = 60  # seconds before origin
    time_after = 1800  # seconds after origin (30 min)

    downloaded_stations = []
    event_file_path = os.path.join(data_dir, "event_info.txt")

    for network, station, description, sta_lat, sta_lon, sta_elev in stations:
        print(f"\n  {network}.{station} - {description}")

        try:
            # Get waveforms
            st = client.get_waveforms(
                network, station, "*", "BH?",
                event_info["time"] - time_before,
                event_info["time"] + time_after
            )

            if len(st) == 0:
                print(f"    No data available")
                continue

            # Merge and detrend
            st.merge(method=1, fill_value='interpolate')
            st.detrend('demean')

            # Save as MiniSEED
            filename = f"{network}.{station}..BH.mseed"
            filepath = os.path.join(data_dir, filename)
            st.write(filepath, format="MSEED")

            # Calculate distance using known coordinates
            dist_deg = locations2degrees(
                sta_lat, sta_lon,
                event_info["latitude"], event_info["longitude"]
            )

            print(f"    ✓ Downloaded: {filename}")
            print(f"      Distance: {dist_deg:.1f}° ({dist_deg * 111:.0f} km)")
            print(f"      Channels: {len(st)}")

            downloaded_stations.append({
                "network": network,
                "station": station,
                "latitude": sta_lat,
                "longitude": sta_lon,
                "elevation": sta_elev,
                "distance_deg": dist_deg,
                "description": description,
                "filename": filename,
            })

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    # Save event and station info
    print(f"\n{'=' * 60}")
    print("Saving metadata...")

    with open(event_file_path, 'w') as f:
        f.write("# QuakeCore Test Event Information\n")
        f.write(f"# Event: {event_info['name']}\n")
        f.write(f"event_name = {event_info['name']}\n")
        f.write(f"event_time = {event_info['time'].isoformat()}\n")
        f.write(f"event_latitude = {event_info['latitude']}\n")
        f.write(f"event_longitude = {event_info['longitude']}\n")
        f.write(f"event_depth_km = {event_info['depth_km']}\n")
        f.write(f"event_magnitude = {event_info['magnitude']}\n")
        f.write("\n# Station Information\n")
        f.write("# Format: network,station,latitude,longitude,elevation,distance_deg\n")

        for sta in downloaded_stations:
            f.write(f"station = {sta['network']},{sta['station']},"
                   f"{sta['latitude']:.4f},{sta['longitude']:.4f},"
                   f"{sta['elevation']:.1f},{sta['distance_deg']:.2f}\n")

    print(f"  Event info saved to: {event_file_path}")

    # Create station coordinates JSON for easy use
    stations_json_path = os.path.join(data_dir, "stations.json")

    stations_list = []
    for sta in downloaded_stations:
        stations_list.append({
            "network": sta["network"],
            "station": sta["station"],
            "latitude": sta["latitude"],
            "longitude": sta["longitude"],
            "elevation": sta["elevation"],
            "description": sta["description"],
        })

    with open(stations_json_path, 'w') as f:
        json.dump({
            "event": {
                "name": event_info["name"],
                "time": str(event_info["time"]),
                "latitude": event_info["latitude"],
                "longitude": event_info["longitude"],
                "depth_km": event_info["depth_km"],
                "magnitude": event_info["magnitude"],
            },
            "stations": stations_list
        }, f, indent=2)

    print(f"  Station metadata saved to: {stations_json_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"Total stations downloaded: {len(downloaded_stations)}")
    print(f"Data directory: {data_dir}")
    print(f"\nFiles downloaded:")

    for sta in downloaded_stations:
        print(f"  - {sta['filename']}")

    print(f"\n{'=' * 60}")
    print("Usage Instructions")
    print(f"{'=' * 60}")
    print("""
1. Start QuakeCore:
   streamlit run app.py

2. Upload the MiniSEED files (*.mseed) from the data/ directory

3. Use the agent to:
   - "读取这些文件的基本信息"
   - "进行初至拾取"
   - "添加台站坐标（使用 stations.json）"
   - "定位地震"

4. Compare the location result with the true event location:
   - True location: 54.65°N, 159.67°W, Depth: 28 km
""")

    return downloaded_stations


if __name__ == "__main__":
    download_test_data()
