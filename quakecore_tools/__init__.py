"""QuakeCore tools package — auto-discovers and registers all tools.

When this package is imported, all modules under ``quakecore_tools/`` are
imported, triggering ``@register_tool`` decorators.  Legacy tools that still
live in ``agent/tools.py`` are registered via :func:`_register_legacy_tools`.
"""

from quakecore_tools.registry import auto_discover, get_registry, register_tool  # noqa: F401


def _register_legacy_tools() -> None:
    """Bridge: register tools that haven't been migrated out of agent/tools.py yet.

    Each tool is registered into the global registry so that
    :func:`quakecore_tools.registry.build_tool_list` returns a complete set.
    """
    try:
        from agent.tools_facade import (
            run_analysis_sandbox,
            add_station_coordinates,
            associate_continuous_events,
            compress_hdf5_to_zfp,
            convert_hdf5_to_excel,
            convert_hdf5_to_numpy,
            convert_miniseed_to_hdf5,
            convert_miniseed_to_numpy,
            convert_miniseed_to_sac,
            convert_sac_to_excel,
            convert_sac_to_hdf5,
            convert_sac_to_miniseed,
            convert_sac_to_numpy,
            convert_segy_to_excel,
            convert_segy_to_hdf5,
            convert_segy_to_numpy,
            download_continuous_waveforms,
            download_seismic_data,
            get_file_structure,
            get_hdf5_keys,
            get_hdf5_structure,
            get_loaded_context,
            get_miniseed_structure,
            get_sac_structure,
            get_segy_binary_header,
            get_segy_structure,
            get_segy_text_header,
            load_local_data,
            locate_earthquake,
            locate_place_data_nearseismic,
            locate_uploaded_data_nearseismic,
            pick_all_miniseed_files,
            pick_first_arrivals,
            plot_location_map,
            prepare_nearseismic_taup_cache,
            read_file_trace,
            read_hdf5_trace,
            read_miniseed_trace,
            read_sac_trace,
            read_trace_sample,
            run_continuous_monitoring,
            run_continuous_picking,
            run_dsa_depth_scanning,
            list_dsa_examples_tool,
            run_telehypo_location,
            run_telehypo_plots_tool,
            predict_polarity_tool,
            list_polarity_models_tool,
            get_demo_progress,
        )
    except ImportError:
        return

    _LEGACY: list[tuple[str, str, str, list[str], bool]] = [
        # (name, category, description, triggers, needs_file)
        ("get_loaded_context", "file", "Return currently loaded file type and paths", [], False),
        ("load_local_data", "file", "Load a local directory or file into the current workspace", ["加载本地", "load local"], False),
        ("download_seismic_data", "file", "Download seismic waveform data from FDSN web services", ["下载数据", "download data"], False),
        ("get_file_structure", "file", "Read structure of currently loaded file", ["文件结构", "file structure"], True),
        ("get_segy_structure", "file", "Read SEGY file structure info", ["segy 结构", "segy structure"], True),
        ("get_segy_text_header", "file", "Read SEGY EBCDIC text header", ["text header", "文本头"], True),
        ("get_segy_binary_header", "file", "Read SEGY binary header", ["binary header", "二进制头"], True),
        ("get_miniseed_structure", "file", "Read MiniSEED file structure info", ["miniseed 结构"], True),
        ("get_hdf5_structure", "file", "Read HDF5 file structure info", ["hdf5 结构"], True),
        ("get_hdf5_keys", "file", "List HDF5 groups and datasets", ["hdf5 keys"], True),
        ("get_sac_structure", "file", "Read SAC file structure info", ["sac 结构"], True),
        ("read_file_trace", "waveform", "Read one trace from currently loaded file", ["读取轨迹", "read trace"], True),
        ("read_trace_sample", "waveform", "Read data from a specific SEGY trace index", ["trace sample"], True),
        ("read_miniseed_trace", "waveform", "Read one MiniSEED trace by index", ["miniseed trace"], True),
        ("read_hdf5_trace", "waveform", "Read one trace from HDF5 file", ["hdf5 trace"], True),
        ("read_sac_trace", "waveform", "Read SAC trace by index", ["sac trace"], True),
        ("convert_segy_to_numpy", "conversion", "Convert SEGY to NumPy", ["segy to numpy"], True),
        ("convert_segy_to_excel", "conversion", "Convert SEGY to Excel", ["segy to excel"], True),
        ("convert_segy_to_hdf5", "conversion", "Convert SEGY to HDF5", ["segy to hdf5"], True),
        ("convert_miniseed_to_numpy", "conversion", "Convert MiniSEED to NumPy", ["miniseed to numpy"], True),
        ("convert_miniseed_to_hdf5", "conversion", "Convert MiniSEED to HDF5", ["miniseed to hdf5"], True),
        ("convert_miniseed_to_sac", "conversion", "Convert MiniSEED to SAC", ["miniseed to sac"], True),
        ("convert_hdf5_to_numpy", "conversion", "Convert HDF5 to NumPy", ["hdf5 to numpy"], True),
        ("convert_hdf5_to_excel", "conversion", "Convert HDF5 to Excel", ["hdf5 to excel"], True),
        ("compress_hdf5_to_zfp", "conversion", "Compress HDF5 dataset using ZFP", ["zfp 压缩"], True),
        ("convert_sac_to_numpy", "conversion", "Convert SAC to NumPy", ["sac to numpy"], True),
        ("convert_sac_to_hdf5", "conversion", "Convert SAC to HDF5", ["sac to hdf5"], True),
        ("convert_sac_to_miniseed", "conversion", "Convert SAC to MiniSEED", ["sac to miniseed"], True),
        ("convert_sac_to_excel", "conversion", "Convert SAC to Excel", ["sac to excel"], True),
        ("pick_first_arrivals", "picking", "Pick P/S first arrivals (初至拾取) from loaded waveform — use this for phase picking", ["震相拾取", "初至拾取", "phase picking", "pick arrivals", "first arrival"], True),
        ("pick_all_miniseed_files", "picking", "Batch pick phases from all loaded MiniSEED files", ["批量拾取", "batch pick"], True),
        ("locate_earthquake", "location", "Locate earthquake using picked phases", ["地震定位", "locate earthquake"], False),
        ("locate_uploaded_data_nearseismic", "location", "Locate earthquake from uploaded data", ["上传数据定位"], False),
        ("locate_place_data_nearseismic", "location", "Download and locate earthquake near a place", ["按地点定位"], False),
        ("add_station_coordinates", "location", "Add station coordinates for location", ["台站坐标", "station coordinates"], False),
        ("prepare_nearseismic_taup_cache", "location", "Prepare TauP travel-time cache", ["taup cache"], False),
        ("plot_location_map", "location", "Plot earthquake location and stations on map", ["绘制地图", "plot map"], False),
        ("download_continuous_waveforms", "monitoring", "Download continuous waveform data", ["下载连续波形"], False),
        ("run_continuous_picking", "monitoring", "Run phase picking on continuous data", ["连续拾取"], False),
        ("associate_continuous_events", "monitoring", "Associate picks into events", ["事件关联"], False),
        ("run_continuous_monitoring", "monitoring", "Run full continuous monitoring workflow", ["连续监测", "continuous monitoring"], False),
        ("run_analysis_sandbox", "analysis", "Run lightweight analysis templates on session artifacts", ["分析沙箱", "analysis sandbox"], False),
        ("run_dsa_depth_scanning", "professional", "Run DSA depth-scanning algorithm for focal depth", ["dsa", "深度扫描", "depth scanning"], False),
        ("list_dsa_examples_tool", "professional", "List available DSA examples", ["列出dsa"], False),
        ("run_telehypo_location", "professional", "Run TeleHypo teleseismic hypocenter location", ["telehypo", "远震定位"], False),
        ("run_telehypo_plots_tool", "professional", "Generate TeleHypo result plots", ["telehypo plots"], False),
        ("predict_polarity_tool", "professional", "Predict P-wave first-motion polarity", ["极性预测", "polarity prediction"], True),
        ("list_polarity_models_tool", "professional", "List available polarity models", ["极性模型", "polarity models"], False),
        ("get_demo_progress", "demo", "Get demo workflow progress", ["演示进度", "demo progress"], False),
    ]

    _TOOL_MAP = {
        "get_loaded_context": get_loaded_context,
        "load_local_data": load_local_data,
        "download_seismic_data": download_seismic_data,
        "get_file_structure": get_file_structure,
        "get_segy_structure": get_segy_structure,
        "get_segy_text_header": get_segy_text_header,
        "get_segy_binary_header": get_segy_binary_header,
        "get_miniseed_structure": get_miniseed_structure,
        "get_hdf5_structure": get_hdf5_structure,
        "get_hdf5_keys": get_hdf5_keys,
        "get_sac_structure": get_sac_structure,
        "read_file_trace": read_file_trace,
        "read_trace_sample": read_trace_sample,
        "read_miniseed_trace": read_miniseed_trace,
        "read_hdf5_trace": read_hdf5_trace,
        "read_sac_trace": read_sac_trace,
        "convert_segy_to_numpy": convert_segy_to_numpy,
        "convert_segy_to_excel": convert_segy_to_excel,
        "convert_segy_to_hdf5": convert_segy_to_hdf5,
        "convert_miniseed_to_numpy": convert_miniseed_to_numpy,
        "convert_miniseed_to_hdf5": convert_miniseed_to_hdf5,
        "convert_miniseed_to_sac": convert_miniseed_to_sac,
        "convert_hdf5_to_numpy": convert_hdf5_to_numpy,
        "convert_hdf5_to_excel": convert_hdf5_to_excel,
        "compress_hdf5_to_zfp": compress_hdf5_to_zfp,
        "convert_sac_to_numpy": convert_sac_to_numpy,
        "convert_sac_to_hdf5": convert_sac_to_hdf5,
        "convert_sac_to_miniseed": convert_sac_to_miniseed,
        "convert_sac_to_excel": convert_sac_to_excel,
        "pick_first_arrivals": pick_first_arrivals,
        "pick_all_miniseed_files": pick_all_miniseed_files,
        "locate_earthquake": locate_earthquake,
        "locate_uploaded_data_nearseismic": locate_uploaded_data_nearseismic,
        "locate_place_data_nearseismic": locate_place_data_nearseismic,
        "add_station_coordinates": add_station_coordinates,
        "prepare_nearseismic_taup_cache": prepare_nearseismic_taup_cache,
        "plot_location_map": plot_location_map,
        "download_continuous_waveforms": download_continuous_waveforms,
        "run_continuous_picking": run_continuous_picking,
        "associate_continuous_events": associate_continuous_events,
        "run_continuous_monitoring": run_continuous_monitoring,
        "run_analysis_sandbox": run_analysis_sandbox,
        "run_dsa_depth_scanning": run_dsa_depth_scanning,
        "list_dsa_examples_tool": list_dsa_examples_tool,
        "run_telehypo_location": run_telehypo_location,
        "run_telehypo_plots_tool": run_telehypo_plots_tool,
        "predict_polarity_tool": predict_polarity_tool,
        "list_polarity_models_tool": list_polarity_models_tool,
        "get_demo_progress": get_demo_progress,
    }

    from quakecore_tools.registry import ToolMeta, _REGISTRY

    for name, category, desc, triggers, needs_file in _LEGACY:
        if name in _REGISTRY:
            continue
        func = _TOOL_MAP.get(name)
        if func is None:
            continue
        _REGISTRY[name] = ToolMeta(
            name=name,
            func=func,
            description=desc,
            category=category,
            triggers=triggers,
            needs_file=needs_file,
        )


# Auto-discover new-style @register_tool tools from all modules in this package.
# Legacy tools are registered lazily on first access to avoid circular imports.
auto_discover()

_legacy_registered = False


def ensure_legacy_tools() -> None:
    """Register legacy tools from agent/tools.py. Called lazily."""
    global _legacy_registered
    if _legacy_registered:
        return
    _legacy_registered = True
    _register_legacy_tools()
