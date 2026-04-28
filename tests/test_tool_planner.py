from backend.services.tool_planner import ToolPlanner


def test_tool_planner_rule_trace_plot_zh():
    planner = ToolPlanner()
    plan = planner.plan(
        message="看看第二道的拾取结果图",
        route="result_analysis",
        runtime_results={"last_picks_csv": "picks/a.csv"},
        uploaded_files=[],
        current_file=None,
        lang="zh",
    )
    assert plan.route == "result_analysis"
    assert plan.tool == "picks_trace_plot"
    assert plan.params.get("trace_index") == 1


def test_tool_planner_rule_trace_detail_en_zero_based():
    planner = ToolPlanner()
    plan = planner.plan(
        message="show trace 3 picks",
        route="result_analysis",
        runtime_results={"last_picks_csv": "picks/a.csv"},
        uploaded_files=[],
        current_file=None,
        lang="en",
    )
    assert plan.tool in {"picks_trace_detail", "picks_trace_plot"}
    assert plan.params.get("trace_index") == 3


def test_tool_planner_continuous_monitoring_parses_region_and_time():
    planner = ToolPlanner()
    plan = planner.plan(
        message="对加州2019年7月4日的17到18点进行地震监测",
        route="continuous_monitoring",
        runtime_results={},
        uploaded_files=[],
        current_file=None,
        lang="zh",
    )
    assert plan.route == "continuous_monitoring"
    assert plan.tool == "run_continuous_monitoring"
    assert plan.params.get("region") == "加州"
    assert plan.params.get("start") == "2019-07-04T17:00:00"
    assert plan.params.get("end") == "2019-07-04T18:00:00"


def test_tool_planner_continuous_monitoring_parses_time_without_region():
    planner = ToolPlanner()
    plan = planner.plan(
        message="2019年7月4日的17到18点进行地震监测",
        route="continuous_monitoring",
        runtime_results={},
        uploaded_files=[],
        current_file=None,
        lang="zh",
    )
    assert plan.tool == "run_continuous_monitoring"
    assert plan.params.get("start") == "2019-07-04T17:00:00"
    assert plan.params.get("end") == "2019-07-04T18:00:00"
