from backend.services.tool_result import normalize_tool_output


def test_normalize_plain_string_output():
    result = normalize_tool_output("hello world")

    assert result.success is True
    assert result.message == "hello world"
    assert result.error is None
    assert result.data == {}


def test_normalize_json_string_output():
    payload = '{"success": true, "message": "ok", "data": {"count": 2}}'
    result = normalize_tool_output(payload)

    assert result.success is True
    assert result.message == "ok"
    assert result.data == {"count": 2}
    assert result.raw == payload


def test_normalize_markdown_image_output_extracts_artifacts():
    result = normalize_tool_output("done\n![plot](data/results/plot.png)")

    assert result.success is True
    assert len(result.artifacts) == 1
    assert result.artifacts[0]["type"] == "image"
    assert result.artifacts[0]["name"] == "plot.png"
    assert result.artifacts[0]["path"] == "results/plot.png"
    assert result.artifacts[0]["url"] == "/api/artifacts/results/plot.png"


def test_normalize_exception_output():
    result = normalize_tool_output(RuntimeError("boom"))

    assert result.success is False
    assert result.error == "boom"


def test_normalize_dict_preserves_data_and_artifacts():
    payload = {
        "success": True,
        "message": "ready",
        "data": {"value": 42},
        "artifacts": [{"type": "image", "path": "data/demo.png"}],
    }

    result = normalize_tool_output(payload)

    assert result.success is True
    assert result.message == "ready"
    assert result.data == {"value": 42}
    assert result.artifacts == [
        {
            "type": "image",
            "name": "demo.png",
            "path": "demo.png",
            "url": "/api/artifacts/demo.png",
        }
    ]
