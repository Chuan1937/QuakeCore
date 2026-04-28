# Task 16: Phase 2 Workflow 稳定化集中任务

推荐模型：`gpt-5.3-codex`

## 背景

当前 `refactor` 分支已经完成：

- session file context
- ToolResult 标准化
- `quakecore_tools/*` facade 分层
- `backend/workflows/location_workflow.py`
- `AgentService` 中对 `earthquake_location` route 的 workflow 优先调用

现在不要继续分散做很多小功能。这个任务集中解决 Phase 2 当前最关键的问题：**让地震定位 workflow 稳定、可解释、可测试，并减少无意义 fallback 到 LLM Agent**。

## 总目标

把 `run_location_workflow()` 从“能跑”升级为“可稳定演示”：

1. workflow 返回 `status`，区分 `success` / `partial_success` / `failed`。
2. 每个 step 包含 `name`、`status`、`message`、`error`、`data`、`artifacts`、`duration_ms`。
3. 区分 fatal step 与 optional step。
4. `plot_location_map` 失败不能导致定位 workflow 整体失败。
5. workflow 部分失败时，不要直接丢给 LLM，让用户能看到步骤结果。
6. `/api/chat` 对 `earthquake_location` 的处理要返回 workflow summary，而不是只有 success 时才返回。
7. 添加一个专用 API：`POST /api/workflows/location/run`，用于不经过 LLM 直接运行 workflow。
8. 增加 smoke 脚本验证 workflow API。

## 允许修改

- `backend/workflows/location_workflow.py`
- `backend/services/agent_service.py`
- `backend/schemas.py`
- `backend/main.py`
- `backend/routes/*`
- `tests/*`
- `scripts/*`
- `docs/*`

## 禁止修改

- 不要重写 `agent/tools.py`
- 不要删除 `app.py`
- 不要默认启用 LangGraph
- 不要引入数据库、Redis、Docker、Qdrant
- 不要让测试依赖真实 DeepSeek API key
- 不要让测试依赖外网模型下载

## 详细实现要求

### 1. Workflow result schema

`run_location_workflow(session_id: str)` 返回字典至少包含：

```python
{
    "success": bool,
    "status": "success" | "partial_success" | "failed",
    "message": str,
    "summary": str,
    "steps": list[dict],
    "location": dict,
    "artifacts": list[dict],
    "error": str | None,
}
```

### 2. Step schema

每个 step 至少包含：

```python
{
    "name": "pick_all_miniseed_files",
    "status": "ok" | "warning" | "error" | "skipped",
    "required": true,
    "message": "...",
    "error": None,
    "data": {},
    "artifacts": [],
    "duration_ms": 123,
}
```

### 3. required / optional step 规则

建议：

Required steps:

- `get_loaded_context`
- `pick_all_miniseed_files` 或已有 picks 检测成功
- `add_station_coordinates`
- `locate_uploaded_data_nearseismic`

Optional steps:

- `load_local_data`
- `prepare_nearseismic_taup_cache`
- `plot_location_map`

注意：`plot_location_map` 失败不能让整体 status 变成 failed，最多 `partial_success`。

### 4. 状态判定

建议：

- 所有 required steps ok，optional 可失败：`success`
- 定位成功但部分 required/optional 有 warning：`partial_success`
- 定位 step 失败且没有 location data：`failed`

不要只用 `normalized.success` 简单判断。

### 5. AgentService 行为调整

当前逻辑可能只有 workflow success 时才返回 workflow result，失败则 fallback 到 LLM。

请改为：

- `status == success`：直接返回 workflow result
- `status == partial_success`：直接返回 workflow result，并把 warning 写入 answer
- `status == failed`：只有在 workflow 发生 fatal exception 或没有任何可用 steps 时，才 fallback 到 LLM

这能让用户看到 workflow 为什么失败，而不是被 LLM 模糊回答覆盖。

### 6. Workflow API

新增：

```text
POST /api/workflows/location/run
```

请求：

```json
{
  "session_id": "optional-session-id"
}
```

返回 workflow result。

建议新增文件：

- `backend/routes/workflows.py`

并在 `backend/main.py` 注册。

### 7. Smoke script

新增：

```text
scripts/smoke_location_workflow.py
```

行为：

1. 检查 `/health`
2. 调用 `/api/workflows/location/run`
3. 检查 JSON 包含 `status`、`steps`、`artifacts`
4. 允许 status 为 `partial_success` 或 `failed`
5. 不允许 HTTP 500 或非 JSON

### 8. Tests

修改或新增：

- `tests/test_location_workflow.py`
- `tests/test_location_workflow_route.py`

必须测试：

1. workflow result 包含 `status`。
2. 每个 step 包含 `duration_ms`。
3. optional step 失败不会导致 fatal failure。
4. `/api/workflows/location/run` 返回 200。
5. `/api/chat` 对 earthquake_location 能返回 workflow summary 或 structured fallback。

## 验证命令

```bash
pytest tests/test_location_workflow.py -q
pytest tests/test_location_workflow_route.py -q
pytest tests -q
python scripts/smoke_location_workflow.py
```

如果没有运行后端，smoke script 应给出清晰提示。

## 常见失败点

1. 工具返回字符串，不能直接当 dict 解析。
2. plotting 失败导致 workflow 整体失败。
3. workflow failed 后 fallback 到 LLM，导致用户看不到 steps。
4. route 没有注册到 FastAPI main。
5. 测试调用 workflow 时真实工具太重，必要时可以 monkeypatch 工具调用。

## 完成后报告

请输出：

1. 修改了哪些文件
2. 新增了哪个 API
3. workflow status 如何判定
4. partial_success 如何返回给用户
5. 测试命令与结果
