"""Quick stats tool — demonstrates the @register_tool pattern.

This tool provides quick statistics about loaded seismic data.
To add a similar tool, just:
1. Create a file in quakecore_tools/
2. Use @register_tool
3. Done — no other files need modification.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.tools import tool

from quakecore_tools.helpers import parse_param_dict, tool_error, tool_success
from quakecore_tools.registry import register_tool


@register_tool(
    name="get_quick_stats",
    category="analysis",
    description="Get amplitude statistics (min/max/mean/std) and histogram of waveform. NOT for phase picking — use pick_first_arrivals for picking.",
    triggers=["统计", "statistics", "quick stats", "波形统计", "amplitude stats"],
    needs_file=True,
)
@tool
def get_quick_stats(params: str | dict | None = None) -> str:
    """
    Get amplitude statistics (min, max, mean, std) and histogram of the loaded waveform.
    Use ONLY when the user asks for 'statistics', 'stats', '数据统计', '波形统计', or 'amplitude stats'.
    Do NOT use for phase picking — use pick_first_arrivals instead.
    """
    Get quick statistics (min, max, mean, std) of the currently loaded waveform data.
    Use this when the user asks for 'statistics', 'stats', '数据统计', or '波形统计'.
    Returns: min, max, mean, std values and a histogram plot if possible.
    """
    from quakecore_tools.context import get_context

    parsed = parse_param_dict(params)
    ctx = get_context()

    # Determine which file to read
    path = parsed.get("path")
    if not path:
        path = ctx.active_path
    if not path or not os.path.exists(path):
        return tool_error("未找到可处理的波形文件。请先上传数据。")

    try:
        from obspy import read as obspy_read

        st = obspy_read(path)
        if not st or len(st) == 0:
            return tool_error("文件中未找到波形数据。")

        import numpy as np

        all_data = []
        trace_stats = []
        for tr in st:
            d = tr.data.astype(float)
            all_data.append(d)
            trace_stats.append({
                "trace": f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}",
                "samples": len(d),
                "sampling_rate": float(tr.stats.sampling_rate),
                "min": float(np.min(d)),
                "max": float(np.max(d)),
                "mean": float(np.mean(d)),
                "std": float(np.std(d)),
            })

        combined = np.concatenate(all_data)
        summary = {
            "num_traces": len(st),
            "total_samples": int(len(combined)),
            "global_min": float(np.min(combined)),
            "global_max": float(np.max(combined)),
            "global_mean": float(np.mean(combined)),
            "global_std": float(np.std(combined)),
            "traces": trace_stats[:10],  # Limit to first 10 traces
        }

        # Try to generate a histogram
        artifacts = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(combined, bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Count")
            ax.set_title("Waveform Amplitude Distribution")
            fig.tight_layout()

            from quakecore_tools.helpers import DEFAULT_STRUCTURE_DIR
            os.makedirs(DEFAULT_STRUCTURE_DIR, exist_ok=True)
            plot_path = os.path.join(DEFAULT_STRUCTURE_DIR, "quick_stats_hist.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            from quakecore_tools.helpers import build_artifact_entry
            entry = build_artifact_entry(plot_path, "image")
            if entry:
                artifacts.append(entry)
        except Exception:
            pass  # Histogram is optional

        return tool_success(
            f"统计完成：{len(st)} 条道，{len(combined)} 个采样点",
            data=summary,
            artifacts=artifacts,
        )

    except ImportError:
        return tool_error("需要安装 ObsPy 才能读取波形数据。")
    except Exception as e:
        return tool_error(f"统计失败: {e}")
