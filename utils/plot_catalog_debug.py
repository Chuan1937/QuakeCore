"""
Standalone catalog 3-view plotter for continuous monitoring results.

Generates plan view (map), depth-latitude cross-section, and longitude-depth
cross-section in a single figure. Used by the run_continuous_monitoring tool
to visualize detected event catalogs.

Based on continuous_monitoring_validation.py::plot_detected_catalog_three_views
"""

import os
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


def _estimate_magnitude(ev: dict) -> float:
    """Estimate magnitude from event dict, preferring explicit values."""
    mag = ev.get("magnitude_pred") or ev.get("mag") or 0.0
    if mag and float(mag) > 0:
        return float(mag)
    npicks = max(1, int(ev.get("num_picks", 1)))
    return 1.8 + 0.18 * np.sqrt(npicks)


def plot_catalog_debug(
    catalog: list[dict],
    output_path: str,
    title: str = "Detected Catalog",
    terrain: bool = True,
    size_scale: float = 3.0,
    dpi: int = 220,
) -> str:
    """
    Plot a 3-view catalog figure: map plan view, depth-lat section, lon-depth section.

    Args:
        catalog: List of event dicts with keys: longitude, latitude, depth (or depth_km),
                 magnitude_pred (optional), num_picks (optional).
        output_path: Path to save the output PNG.
        title: Figure title.
        terrain: Whether to render terrain basemap (requires Cartopy or Basemap).
        size_scale: Marker size scaling factor.
        dpi: Output resolution.

    Returns:
        Absolute path to the saved figure.
    """
    if not catalog:
        return ""

    ev_lon = np.array([float(e.get("longitude", 0.0)) for e in catalog], dtype=float)
    ev_lat = np.array([float(e.get("latitude", 0.0)) for e in catalog], dtype=float)
    ev_dep = np.array([max(0.0, float(e.get("depth_km", e.get("depth", 0.0)))) for e in catalog], dtype=float)
    ev_mag = np.array([_estimate_magnitude(e) for e in catalog], dtype=float)

    min_lat, max_lat = float(np.min(ev_lat)), float(np.max(ev_lat))
    min_lon, max_lon = float(np.min(ev_lon)), float(np.max(ev_lon))
    lat_pad = max(0.08, (max_lat - min_lat) * 0.08 + 0.02)
    lon_pad = max(0.08, (max_lon - min_lon) * 0.08 + 0.02)

    dep_max = max(45.0, float(np.max(ev_dep)) + 5.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=dep_max)
    cmap = cm.jet_r
    colors = cmap(norm(ev_dep))
    ms = np.clip(ev_mag * size_scale, 4.5, 18.0)

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    gs0 = fig.add_gridspec(8, 12)

    ax1 = fig.add_subplot(gs0[0:5, 7:12])  # depth-lat
    ax2 = fig.add_subplot(gs0[5:9, 0:7])   # lon-depth
    ax3 = fig.add_subplot(gs0[5:9, 7:12])  # legend

    # Plan view - try Cartopy first, then Basemap, then plain matplotlib
    use_cartopy = False
    if terrain:
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
        try:
            ax0.stock_img()
        except Exception:
            ax0.set_facecolor("#e9eef2")
        gl = ax0.gridlines(
            crs=data_crs, draw_labels=True,
            linewidth=0.4, color="gray", alpha=0.7, linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        ax0.scatter(ev_lon, ev_lat, c=colors, s=ms ** 2, marker="o",
                    edgecolors="black", linewidths=0.4, transform=data_crs, alpha=0.9)
    elif terrain and HAS_BASEMAP:
        ax0 = fig.add_subplot(gs0[0:5, 0:7])
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
        ax0.scatter(x, y, c=colors, s=ms ** 2, marker="o",
                    edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3)
    else:
        ax0 = fig.add_subplot(gs0[0:5, 0:7])
        ax0.set_facecolor("#e9eef2")
        ax0.grid(linestyle="--", alpha=0.6)
        ax0.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
        ax0.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
        ax0.scatter(ev_lon, ev_lat, c=colors, s=ms ** 2, marker="o",
                    edgecolors="black", linewidths=0.4, alpha=0.9)
        ax0.set_xlabel("Lon. (°)")
        ax0.set_ylabel("Lat. (°)")

    ax0.set_title("Plan View")

    # depth-lat cross-section
    ax1.scatter(ev_dep, ev_lat, c=colors, s=ms ** 2, marker="o",
                edgecolors="black", linewidths=0.4, alpha=0.9)
    ax1.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    ax1.set_xlim(0.0, dep_max)
    ax1.set_facecolor("#bfc0c2")
    ax1.set_xlabel("Depth (km)", fontsize=12)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_yticklabels([])
    ax1.set_title("Lat-Depth")
    ax1.tick_params(axis="both", which="major", labelsize=11)

    # lon-depth cross-section
    ax2.scatter(ev_lon, ev_dep, c=colors, s=ms ** 2, marker="o",
                edgecolors="black", linewidths=0.4, alpha=0.9)
    ax2.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax2.set_ylim(dep_max, 0.0)
    ax2.set_facecolor("#bfc0c2")
    ax2.set_xlabel("Lon. (°)", fontsize=12)
    ax2.set_ylabel("Depth (km)", fontsize=12)
    ax2.set_title("Lon-Depth")
    ax2.tick_params(axis="both", which="major", labelsize=11)

    # Legend panel
    ax3.plot(0.10, 0.90, "o", mec="k", mfc="none", mew=1, ms=3 * size_scale)
    ax3.plot(0.10, 0.78, "o", mec="k", mfc="none", mew=1, ms=4 * size_scale)
    ax3.plot(0.10, 0.65, "o", mec="k", mfc="none", mew=1, ms=5 * size_scale)
    ax3.plot(0.10, 0.50, "o", mec="k", mfc="none", mew=1, ms=6 * size_scale)
    ax3.text(0.20, 0.88, "M 3.0", fontsize=12)
    ax3.text(0.20, 0.76, "M 4.0", fontsize=12)
    ax3.text(0.20, 0.62, "M 5.0", fontsize=12)
    ax3.text(0.20, 0.47, "M 6.0", fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical")
    cbar.set_label(label="Depth (km)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(title, fontsize=14)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)
