import importlib


EXPECTED_TOOLS = [
    "run_analysis_sandbox",
    "add_station_coordinates",
    "associate_continuous_events",
    "compress_hdf5_to_zfp",
    "convert_hdf5_to_excel",
    "convert_hdf5_to_numpy",
    "convert_miniseed_to_hdf5",
    "convert_miniseed_to_numpy",
    "convert_miniseed_to_sac",
    "convert_sac_to_excel",
    "convert_sac_to_hdf5",
    "convert_sac_to_miniseed",
    "convert_sac_to_numpy",
    "convert_segy_to_excel",
    "convert_segy_to_hdf5",
    "convert_segy_to_numpy",
    "download_continuous_waveforms",
    "download_seismic_data",
    "get_file_structure",
    "get_hdf5_keys",
    "get_hdf5_structure",
    "get_loaded_context",
    "get_miniseed_structure",
    "get_sac_structure",
    "get_segy_binary_header",
    "get_segy_structure",
    "get_segy_text_header",
    "load_local_data",
    "locate_earthquake",
    "locate_place_data_nearseismic",
    "locate_uploaded_data_nearseismic",
    "pick_all_miniseed_files",
    "pick_first_arrivals",
    "plot_location_map",
    "prepare_nearseismic_taup_cache",
    "read_file_trace",
    "read_hdf5_trace",
    "read_miniseed_trace",
    "read_sac_trace",
    "read_trace_sample",
    "run_continuous_monitoring",
    "run_continuous_picking",
]


def test_tools_facade_import_and_expected_exports():
    facade = importlib.import_module("agent.tools_facade")

    for tool_name in EXPECTED_TOOLS:
        assert hasattr(facade, tool_name), f"Missing tool export: {tool_name}"
        tool_obj = getattr(facade, tool_name)
        assert callable(tool_obj) or hasattr(tool_obj, "name"), f"Invalid tool object: {tool_name}"


def test_agent_core_imports_successfully():
    core = importlib.import_module("agent.core")
    assert hasattr(core, "get_agent_executor")
