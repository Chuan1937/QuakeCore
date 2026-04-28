# Task 18: 下一阶段集中计划：工作流可观测性 + 前端 Workspace + 合并前稳定化

推荐模型：`gpt-5.3-codex` 起步；如果只修前端 build，可切换 `gpt-5.4-mini`。

## 当前代码状态

`refactor` 分支已经完成以下核心能力：

1. FastAPI 后端。
2. Next.js 前端。
3. DeepSeek `deepseek-v4-flash` 真实验证脚本。
4. Session file context。
5. ToolResult 标准化。
6. `quakecore_tools/*` facade 分层。
7. 地震定位 deterministic workflow。
8. `/api/workflows/location/run` 路由。
9. `scripts/smoke_location_workflow.py`。
10. `tests/test_location_workflow.py` 和相关后端测试。

当前不建议继续拆很多小任务。本任务集中处理下一阶段最重要的事情：

- 让 workflow 结果对用户可见。
- 让前端能展示 workflow steps。
- 让真实 DeepSeek + workflow + upload 的闭环稳定。
- 做合并 main 前的工程加固。

## 总目标

完成后，QuakeCore 应达到：

```text
上传地震文件
→ session 记录当前文件
→ chat 或 workflow API 触发定位 workflow
→ 后端返回 status / steps / artifacts
→ 前端显示步骤、警告、错误、地图/文件
→ DeepSeek v4 flash 可以真实解释结果
→ pytest 和 smoke scripts 可验证
```

## 禁止事项

- 不要重写 `agent/tools.py`。
- 不要删除 `app.py`。
- 不要默认启用 LangGraph。
- 不要引入 Docker、Redis、Qdrant、数据库。
- 不要引入复杂权限系统。
- 不要让普通 `pytest tests -q` 依赖真实 DeepSeek API。
- 不要把真实 API key 写入仓库。
- 不要大规模重写前端 UI。

---

# Part A: 后端 Workflow 返回结构完善

## A1. ChatResponse 增加 workflow 字段

当前 `/api/chat` 只返回：

```text
session_id, answer, route, artifacts, error
```

建议新增可选字段：

```python
workflow: dict | None = None
```

用于传递：

```json
{
  "status": "partial_success",
  "summary": "...",
  "steps": [...],
  "location": {...}
}
```

要求：

- 前端旧逻辑不应坏。
- 如果不是 workflow route，workflow=null。
- earthquake_location route 应尽量返回 workflow。

## A2. AgentService 返回 workflow payload

在 `AgentService.chat()` 中：

- 当 route == `earthquake_location`，调用 `run_location_workflow()`。
- 无论 status 是 `success`、`partial_success` 还是 `failed`，只要 workflow 有 steps，都应把 workflow payload 返回给 `/api/chat`。
- 只有 workflow 抛出 fatal exception 且没有 steps 时，才 fallback 到 LLM。

要求：

- ChatResult dataclass 增加 `workflow: dict | None`。
- chat route response schema 同步更新。
- failed workflow 也应给用户 answer，例如：

```text
地震定位工作流执行失败，但已完成部分步骤。请查看步骤详情。
```

## A3. Workflow result 清理

确保 `run_location_workflow()` 返回：

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

每个 step 必须包含：

```python
{
    "name": str,
    "status": "ok" | "warning" | "error" | "skipped",
    "required": bool,
    "message": str,
    "error": str | None,
    "data": dict,
    "artifacts": list,
    "duration_ms": int,
}
```

## A4. 测试

新增或修改：

- `tests/test_backend_chat_schema.py`
- `tests/test_location_workflow_route.py`

必须测试：

1. `/api/chat` 对普通问题返回 workflow 为 null 或缺省可接受。
2. `/api/chat` 对定位问题返回 workflow 字段。
3. workflow.steps 是 list。
4. workflow.status 是 success / partial_success / failed 之一。
5. 即使 workflow failed，也不返回 HTTP 500。

---

# Part B: 前端 Workspace 面板

## B1. 前端类型更新

在 `frontend/lib/api.ts` 或类型文件中增加：

```ts
export type WorkflowStep = {
  name: string;
  status: "ok" | "warning" | "error" | "skipped" | string;
  required?: boolean;
  message?: string;
  error?: string | null;
  duration_ms?: number;
};

export type WorkflowResult = {
  status: "success" | "partial_success" | "failed" | string;
  summary?: string;
  message?: string;
  steps?: WorkflowStep[];
  location?: Record<string, unknown>;
  artifacts?: Artifact[];
  error?: string | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  route: string;
  artifacts: Artifact[];
  error?: string | null;
  workflow?: WorkflowResult | null;
};
```

## B2. 新增 WorkflowSteps 组件

建议新增：

```text
frontend/components/workflow-steps.tsx
```

功能：

- 显示 workflow.status。
- 显示每个 step 的 name / status / duration_ms。
- error step 用明显样式。
- optional warning 不应显示为致命错误。
- 展示 summary。

不要引入大型 UI 依赖。

## B3. Chat 页面展示 workflow

在 `frontend/app/page.tsx` 中：

- 如果 assistant message 有 workflow，则显示 `WorkflowSteps`。
- artifacts 仍按原方式显示。
- error 不为空时显示 error card。

## B4. Workflow API 按钮（可选但推荐）

如果改动不大，可以在前端增加按钮：

```text
运行定位工作流
```

调用：

```text
POST /api/workflows/location/run
```

如果太复杂，先不做按钮，只展示 chat 返回的 workflow。

## B5. 前端验证

```bash
cd frontend
npm run build
```

---

# Part C: 真实 DeepSeek + Workflow 闭环 smoke

## C1. 新增集中 smoke 脚本

新增：

```text
scripts/smoke_full_web_agent.py
```

流程：

1. 检查 `/health`。
2. 如果存在 example_data 中的小型地震文件，则上传。
3. 调 `/api/chat`，message：

```text
请分析当前文件结构
```

4. 调 `/api/chat`，message：

```text
使用当前数据进行地震定位
```

5. 检查返回：

- HTTP 200
- JSON
- session_id
- route
- answer
- artifacts
- workflow 字段（定位请求时）

6. 如果 `DEEPSEEK_API_KEY` 存在，还要检查普通 chat answer 非空。
7. 如果没有 key，允许 structured error，但不允许 500。

## C2. smoke 输出格式

输出示例：

```text
[OK] health
[OK] upload file: demo.mseed
[OK] file_structure route=file_structure
[OK] location route=earthquake_location workflow_status=partial_success steps=7
[OK] artifacts=1
[DONE] full web agent smoke passed
```

## C3. 验证命令

后端启动后：

```bash
python scripts/smoke_full_web_agent.py
```

---

# Part D: 合并前清理

## D1. 清理重复文档

当前 `docs/codex_migration` 中可能同时存在：

- `16_PHASE2_WORKFLOW_STABILIZATION_CODEX.md`
- `16_PHASE2_WORKFLOW_STABILIZATION_CODEX_v2.md`

请保留详细版，删除过短的 v2 或在 index 中标记废弃。

## D2. 更新 README

README 应包含：

1. 旧 Streamlit 启动：

```bash
streamlit run app.py
```

2. FastAPI 启动：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

3. Frontend 启动：

```bash
cd frontend
npm install
npm run dev
```

4. DeepSeek：

```bash
export DEEPSEEK_API_KEY=your_key
```

5. 推荐模型：

```text
deepseek-v4-flash
```

6. 验证：

```bash
pytest tests -q
python scripts/smoke_deepseek_v4_flash.py
python scripts/smoke_location_workflow.py
python scripts/smoke_full_web_agent.py
cd frontend && npm run build
```

## D3. 安全检查

确认：

- `.env` 未提交。
- API key 未硬编码。
- artifact route 不能访问 data 外部。
- 上传文件名安全。
- `QUAKECORE_USE_LANGGRAPH` 默认关闭。

---

# 最终验收命令

```bash
pytest tests -q
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_location_workflow.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
cd frontend && npm run build
```

如果某个 smoke 依赖后端，先运行：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

# 完成后报告格式

请输出：

1. 修改了哪些文件。
2. `/api/chat` 是否返回 workflow。
3. 前端是否展示 workflow steps。
4. `smoke_full_web_agent.py` 结果。
5. DeepSeek v4 flash 真实调用结果。
6. 是否建议合并 refactor 到 main。
7. 合并前剩余风险。
