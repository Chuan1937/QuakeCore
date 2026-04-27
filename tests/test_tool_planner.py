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

