from backend.services.router_service import RouterService


def test_route_intent_covers_expected_routes():
    service = RouterService()

    cases = {
        "请帮我定位这个事件": "earthquake_location",
        "Help me locate the earthquake": "earthquake_location",
        "对当前波形做 phase picking": "phase_picking",
        "请对当前波形做初至拾取": "phase_picking",
        "读取当前文件结构": "file_structure",
        "Read the waveform trace at channel 0": "waveform_reading",
        "Please convert this file format": "format_conversion",
        "最近 10 小时连续监测": "continuous_monitoring",
        "帮我做连续地震监测": "continuous_monitoring",
        "Plot the map for this event": "map_plotting",
        "I have a question, what is a P-wave?": "seismo_qa",
        "Settings and config for LLM": "settings",
        "Just chat with me": "general_chat",
    }

    for message, expected in cases.items():
        assert service.route_intent(message) == expected


def test_extract_artifacts_includes_metadata():
    service = RouterService()

    artifacts = service.extract_artifacts("![demo](data/results/plot.png)")

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "image"
    assert artifact.name == "plot.png"
    assert artifact.path == "results/plot.png"
    assert artifact.url == "/api/artifacts/results/plot.png"
