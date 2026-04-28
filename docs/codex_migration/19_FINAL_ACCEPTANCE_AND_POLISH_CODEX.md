# Task 19: 最终验收与产品化收口（集中任务）

推荐模型：`gpt-5.3-codex`。如果只修前端 TypeScript 或 CSS，可切换 `gpt-5.4-mini`。

## 当前状态判断

`refactor` 分支已经完成主要架构：

1. FastAPI 后端。
2. Next.js 前端。
3. Session file context。
4. ToolResult 标准化。
5. `quakecore_tools/*` facade 分层。
6. Deterministic location workflow。
7. `/api/workflows/location/run`。
8. `/api/chat` 已返回 `workflow` 字段。
9. 前端已引入 `WorkflowSteps`。
10. DeepSeek v4 Flash 真实 smoke 脚本。

本任务是合并前最后一次集中收口，不再新增大架构。目标是修复体验和验证链路，确保可以演示、测试、合并。

---

## 总目标

完成后应满足：

```text
上传文件
→ 发送中文问题
→ 后端用 deepseek-v4-flash 或 deterministic workflow 返回结果
→ 前端显示 route、answer、workflow steps、artifacts、error
→ smoke scripts 和 pytest 可验证
→ README 清楚说明本地运行方式
```

---

## 禁止事项

- 不要重写 `agent/tools.py`。
- 不要删除 `app.py`。
- 不要默认启用 LangGraph。
- 不要引入 Docker、Redis、Qdrant、数据库。
- 不要提交真实 API key。
- 不要让普通 `pytest tests -q` 依赖真实 DeepSeek API。
- 不要大规模重写前端 UI。
- 不要拆成多个新任务。

---

# Part A: 前端中文化与 API URL 修复

## A1. 默认中文

检查 `frontend/app/page.tsx`。

当前可能仍使用英文示例和 `lang: "en"`。

请修改为：

- 默认发送 `lang: "zh"`
- 示例 prompt 改为中文：
  - `请分析当前文件结构`
  - `对当前波形进行震相拾取`
  - `使用当前数据进行地震定位`

## A2. Artifact URL 拼接

如果 artifact.url 是相对路径，例如：

```text
/api/artifacts/location/map.png
```

前端在 `localhost:3000` 下直接访问会请求 Next.js，而不是 FastAPI。

请在 `frontend/lib/api.ts` 或前端组件中增加 helper：

```ts
export function toBackendUrl(url: string): string {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
  if (!url) return url;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${base}${url}`;
  return `${base}/${url}`;
}
```

然后所有 artifact 图片和下载链接都使用 `toBackendUrl(artifact.url)`。

## A3. Error 显示

如果 response.error 不为空：

- 不要只显示空 answer。
- 必须显示 error card。
- workflow steps 仍然显示。

---

# Part B: Workflow 展示检查

## B1. WorkflowSteps 组件

确认 `frontend/components/workflow-steps.tsx` 存在并能展示：

- workflow.status
- workflow.summary
- 每个 step 的 name
- step.status
- duration_ms
- error / warning

如果没有，请新增。

## B2. Chat 页面展示 workflow

在 assistant message 中：

```tsx
{message.workflow ? <WorkflowSteps workflow={message.workflow} /> : null}
```

已存在则检查类型和样式。

## B3. Location workflow 体验

当用户发送：

```text
使用当前数据进行地震定位
```

前端应展示：

- route: earthquake_location
- workflow status
- steps list
- artifacts（如果有）
- error（如果 failed）

---

# Part C: 后端 schema 和 route 最终检查

## C1. ChatResponse

确认 `backend/schemas.py` 中 `ChatResponse` 包含：

```python
workflow: WorkflowResultResponse | None = None
```

## C2. Chat route

确认 `backend/routes/chat.py` 返回：

```python
"workflow": result.workflow
```

## C3. AgentService

确认 `AgentService.chat()` 对 `earthquake_location`：

- workflow 有 steps 时返回 workflow。
- success / partial_success / failed 都能结构化返回。
- 只有 fatal exception 才 fallback。

---

# Part D: DeepSeek v4 Flash 最终验证

## D1. 模型统一

全仓库默认 DeepSeek 模型必须是：

```text
deepseek-v4-flash
```

检查：

- `agent/core.py`
- `app.py`
- `backend/services/config_service.py`
- `backend/services/agent_service.py`
- `frontend/app/settings/page.tsx`
- `scripts/smoke_deepseek_v4_flash.py`
- README

## D2. 环境变量

API key 读取规则：

1. 用户显式填写 config api_key 时优先使用。
2. 否则使用 `DEEPSEEK_API_KEY`。
3. 都没有时返回结构化 error。

不要把空字符串 config 覆盖环境变量。

## D3. 真实 smoke

用户本地有 `DEEPSEEK_API_KEY`。

必须支持：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
python scripts/smoke_deepseek_v4_flash.py
```

成功条件：

- HTTP 200
- error 为空
- answer 非空
- 模型为 deepseek-v4-flash

---

# Part E: Full Web Agent smoke

## E1. 检查或新增脚本

确认存在：

```text
scripts/smoke_full_web_agent.py
```

如果不存在，请新增。

## E2. 脚本流程

1. GET `/health`
2. 找到 example_data 或 data 下的小型地震文件
3. POST `/api/files/upload`
4. POST `/api/chat` message=`请分析当前文件结构`
5. POST `/api/chat` message=`使用当前数据进行地震定位`
6. 检查定位响应中包含：
   - session_id
   - route
   - answer
   - artifacts
   - workflow
   - workflow.status
   - workflow.steps

## E3. 成功标准

允许 workflow.status 为：

- success
- partial_success
- failed

但不允许：

- HTTP 500
- 非 JSON
- 缺少 workflow 字段
- 缺少 steps

---

# Part F: README 最终更新

README 必须包含以下内容。

## F1. 旧 Streamlit

```bash
streamlit run app.py
```

说明：旧版仍保留，可用于对比和回退。

## F2. 后端

```bash
pip install -r requirements.txt
pip install -r requirements-backend.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## F3. 前端

```bash
cd frontend
npm install
npm run dev
```

## F4. DeepSeek

```bash
export DEEPSEEK_API_KEY=your_key
```

当前唯一推荐模型：

```text
deepseek-v4-flash
```

## F5. 验证命令

```bash
pytest tests -q
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_location_workflow.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
cd frontend && npm run build
```

## F6. 当前限制

写清楚：

- 当前仍兼容旧 `agent.tools`。
- deterministic workflow 目前主要覆盖 earthquake_location。
- 其他 route 仍主要走 Agent + tools。
- LangGraph 默认关闭。
- RAG / Python Runner 尚未默认启用。

---

# Part G: 最终验证命令

请依次运行：

```bash
python -m py_compile app.py
python -m py_compile backend/main.py
python -m py_compile agent/core.py
pytest tests -q
```

启动后端后运行：

```bash
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_location_workflow.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
```

前端：

```bash
cd frontend
npm run build
```

---

# 完成后报告格式

请输出：

1. 修改了哪些文件。
2. 前端是否默认中文。
3. Artifact URL 是否已拼接 backend base URL。
4. `/api/chat` 是否返回 workflow。
5. 前端是否展示 workflow steps。
6. `smoke_full_web_agent.py` 结果。
7. `smoke_deepseek_v4_flash.py` 结果。
8. `pytest tests -q` 结果。
9. `npm run build` 结果。
10. 是否建议合并 `refactor` 到 `main`。
