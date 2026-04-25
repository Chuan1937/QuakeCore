# QuakeCore Phase 2 Codex 任务索引

当前分支：`refactor`

## Phase 2 总目标

Phase 1 已经完成 Web-first 迁移的主体：FastAPI 后端、Next.js 前端、文件上传、聊天 API、配置页面、skills 页面、artifact 下载、基础 smoke scripts。Phase 2 的目标不是继续堆 UI，而是把 QuakeCore 从“包装旧 Streamlit/旧 Agent 的 Web 应用”升级成“可扩展的地震 Agent 平台”。

核心方向：

1. 建立 session file context，减少旧 `agent.tools` 全局变量带来的多会话污染风险。
2. 标准化工具返回格式，避免字符串、Markdown、JSON 字符串混用导致前端和 workflow 难处理。
3. 逐步拆分 `agent/tools.py`，但不要一次性推倒重写。
4. 建立地震定位 workflow，让定位流程可测试、可复用、可被 LangGraph 接管。
5. 增加本地 RAG 的最小版本，优先 FAISS/Chroma 本地化，不使用 Docker/Qdrant。
6. 增加受控 Python runner 的最小版本，暂不开放危险自由执行。
7. 优化前端 Workspace 面板，用来展示地图、表格、图像和文件 artifact。

## 模型使用策略

| 文件 | 推荐模型 | 原因 |
|---|---|---|
| `12_SESSION_FILE_CONTEXT_CODEX.md` | `gpt-5.3-codex` | 涉及旧 agent 状态兼容，复杂度较高 |
| `13_TOOL_RESULT_STANDARD_CODEX.md` | `gpt-5.3-codex` | 需要理解后端返回结构与旧工具输出 |
| `14_TOOLS_FACADE_SPLIT_CODEX.md` | `gpt-5.3-codex` | 涉及工具分层迁移，必须谨慎 |
| `15_LOCATION_WORKFLOW_CODEX.md` | `gpt-5.3-codex` | 涉及定位流程和旧工具链 |
| `16_RAG_LOCAL_FAISS_MINI.md` | `gpt-5.4-mini` | 独立新增模块，尽量轻量 |
| `17_LANGGRAPH_LOCATION_WORKFLOW_CODEX.md` | `gpt-5.3-codex` | 涉及 Agent runtime 与 workflow |
| `18_PYTHON_RUNNER_SAFE_MINI.md` | `gpt-5.4-mini` | 先做受控 runner，避免复杂沙箱 |
| `19_FRONTEND_WORKSPACE_PANEL_MINI.md` | `gpt-5.4-mini` | 前端展示优化，适合 mini |
| `20_PHASE2_PRE_MERGE_HARDENING.md` | `gpt-5.4-mini` 或 `gpt-5.3-codex` | 合并前稳定性检查 |

尽量不要用 `gpt-5.5`，除非出现跨 backend / agent / frontend 的复杂阻塞问题，并且 `gpt-5.3-codex` 多次失败。

## 执行顺序

建议按以下顺序执行：

1. `12_SESSION_FILE_CONTEXT_CODEX.md`
2. `13_TOOL_RESULT_STANDARD_CODEX.md`
3. `14_TOOLS_FACADE_SPLIT_CODEX.md`
4. `15_LOCATION_WORKFLOW_CODEX.md`
5. `16_RAG_LOCAL_FAISS_MINI.md`
6. `17_LANGGRAPH_LOCATION_WORKFLOW_CODEX.md`
7. `18_PYTHON_RUNNER_SAFE_MINI.md`
8. `19_FRONTEND_WORKSPACE_PANEL_MINI.md`
9. `20_PHASE2_PRE_MERGE_HARDENING.md`

## Phase 2 禁止事项

- 不删除 `app.py`。
- 不破坏旧 Streamlit 入口。
- 不一次性删除 `agent/tools.py` 中的全局状态。
- 不一次性重写所有工具。
- 不默认启用 LangGraph。
- 不强制 Docker、Redis、Qdrant、数据库。
- 不让测试依赖真实 DeepSeek API key。
- 不让测试依赖外网模型下载。
- 不让 Python runner 默认执行任意危险代码。
- 不把 `.env`、真实 API key、用户数据提交到仓库。

## Phase 2 验收目标

执行完 Phase 2 后，应达到：

1. 上传文件会进入 session context。
2. 每次 chat 前能根据 session context 注入旧 agent 当前文件状态。
3. 后端 chat response 中 `artifacts`、`route`、`error` 结构稳定。
4. 新增 `ToolResult` / `NormalizedToolResult` 后，旧工具仍可兼容。
5. `tools_facade` 开始按领域分层，但旧工具仍可调用。
6. 地震定位 workflow 可单独测试，不完全依赖 LLM prompt。
7. 本地 RAG 可索引 `skills/` 和 `README.md`，即使没有外网也能 fallback。
8. 前端可以在 Workspace 面板展示 artifacts。
9. `pytest tests -q` 和 `cd frontend && npm run build` 通过。

## 阶段性验证命令

```bash
pytest tests -q
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_upload_then_chat.py
cd frontend && npm run build
```

如果启用了可选 RAG / Python runner，再运行对应 smoke scripts。
