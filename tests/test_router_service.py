from backend.services.router_service import RouterService


def test_route_intent_covers_expected_routes():
    service = RouterService()

    cases = {
        "请帮我定位这个事件": "earthquake_location",
        "Help me locate the earthquake": "earthquake_location",
        "解释一下定位结果": "result_explanation",
        "看看第3个事件": "result_analysis",
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
    from pathlib import Path

    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/plot.png").write_bytes(b"stub")

    service = RouterService()

    artifacts = service.extract_artifacts("![demo](data/results/plot.png)")

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "image"
    assert artifact.name == "plot.png"
    assert artifact.path == "results/plot.png"
    assert artifact.url == "/api/artifacts/results/plot.png"


def test_extract_artifacts_from_plain_text_paths():
    from pathlib import Path

    Path("data/picks").mkdir(parents=True, exist_ok=True)
    Path("data/picks/demo_picks.png").write_bytes(b"stub")
    Path("data/picks/demo_picks.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    service = RouterService()
    artifacts = service.extract_artifacts(
        "拾取结果图表已保存至 data/picks/demo_picks.png，CSV 文件已保存至 data/picks/demo_picks.csv。"
    )

    assert len(artifacts) == 2
    by_name = {item.name: item for item in artifacts}
    assert by_name["demo_picks.png"].type == "image"
    assert by_name["demo_picks.csv"].type == "file"
