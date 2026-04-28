import json


def test_run_opencode_admin_workspace_basic():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "save_text('hello.txt', 'hello opencode')\nset_message('done')",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["message"] == "done"
    assert payload["artifacts"]


def test_run_opencode_admin_workspace_shell():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "res = shell('python --version')\nset_data('returncode', res['returncode'])\nset_message('shell ok')",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["data"]["returncode"] == 0


def test_run_opencode_admin_workspace_import():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "import numpy as np\narr = np.array([1, 2, 3])\nset_data('sum', int(np.sum(arr)))\nset_message('import ok')",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["data"]["sum"] == 6


def test_run_opencode_admin_workspace_save_plot():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.plot([0, 1], [0, 1])\nsave_plot('test.png')\nset_message('plot ok')",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert any(a["name"] == "test.png" for a in payload["artifacts"])


def test_run_opencode_admin_workspace_timeout():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "import time\ntime.sleep(5)\nset_message('should not reach')",
                "timeout_seconds": 2,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is False


def test_run_opencode_admin_workspace_empty_code():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is False


def test_run_opencode_admin_workspace_gmt_check():
    from quakecore_tools.opencode_admin_tool import run_opencode_admin_workspace

    raw = run_opencode_admin_workspace.invoke(
        {
            "params": {
                "session_id": "test",
                "input_key": "",
                "runtime_results": {},
                "code": "res = shell('gmt --version')\nset_data('returncode', res['returncode'])\nset_message('gmt checked')",
                "timeout_seconds": 30,
            }
        }
    )

    payload = json.loads(raw)
    assert payload["success"] is True
