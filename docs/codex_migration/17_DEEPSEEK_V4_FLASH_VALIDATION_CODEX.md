# Task 17: DeepSeek v4 Flash 真实环境验证与主线兼容

推荐模型：`gpt-5.3-codex`

## 背景

用户本地环境已经配置：

```bash
DEEPSEEK_API_KEY=<exists locally>
```

并且明确要求：

- 只能使用 `deepseek-v4-flash`
- FastAPI 后端必须默认使用 `deepseek-v4-flash`
- 旧 Streamlit `app.py` 也必须继续能使用 DeepSeek API
- 可以参考 main / legacy Streamlit 里已有的 DeepSeek API 调用方式
- 测试不能只覆盖“没有 API key 时返回结构化 error”，还必须提供真实 DeepSeek API smoke 验证路径

本任务集中解决 DeepSeek 配置一致性、真实 API smoke 测试、Streamlit 兼容和前后端配置锁定问题。

## 总目标

1. 统一默认模型为 `deepseek-v4-flash`。
2. 后端配置中不允许默认切到其它 DeepSeek 模型。
3. 增加真实 DeepSeek smoke test：读取 `DEEPSEEK_API_KEY`，调用 `/api/chat`，确认返回非空 answer。
4. 保持没有 API key 时普通单元测试仍可通过结构化 error。
5. 确认旧 Streamlit `app.py` 仍可使用 DeepSeek。
6. README 说明本地真实验证方式。

## 允许修改

- `backend/services/config_service.py`
- `backend/services/agent_service.py`
- `backend/routes/chat.py`
- `backend/schemas.py`
- `agent/core.py` 的小范围默认值修正
- `app.py` 的小范围默认值修正，仅限 DeepSeek 默认模型和读取环境变量
- `frontend/app/settings/page.tsx`
- `frontend/lib/api.ts`
- `scripts/*`
- `tests/*`
- `README.md`

## 禁止修改

- 不要删除或重写 `app.py`
- 不要把 API key 写死进代码
- 不要提交 `.env`
- 不要默认使用其它模型
- 不要默认使用 `deepseek-chat`、`deepseek-reasoner`、`deepseek-v3` 等其它模型
- 不要让普通 `pytest tests -q` 强制依赖真实网络
- 不要引入 Docker、Redis、数据库

## 详细实现要求

### 1. 统一默认模型

检查并确保这些位置默认模型一致：

```text
agent/core.py
backend/services/config_service.py
backend/services/agent_service.py
frontend 设置页默认值
README 示例
scripts/smoke_chat.py
```

必须统一为：

```text
deepseek-v4-flash
```

### 2. 后端配置服务约束

如果 `ConfigService` 有默认配置，必须包含：

```json
{
  "provider": "deepseek",
  "model_name": "deepseek-v4-flash",
  "base_url": "https://api.deepseek.com"
}
```

API key 优先从 config JSON 读取；如果 config JSON 为空，则从环境变量读取：

```python
os.getenv("DEEPSEEK_API_KEY")
```

注意：不要因为 `data/config/llm_config.json` 没有 `api_key` 就覆盖掉环境变量。

### 3. AgentService 调用规则

当 provider 为 `deepseek` 且 config 中 api_key 为空时，应 fallback 到：

```python
os.getenv("DEEPSEEK_API_KEY")
```

如果仍为空，返回结构化 error，而不是崩溃。

### 4. 旧 Streamlit 兼容

检查 `app.py` 中 DeepSeek 默认模型和 API key 读取方式。

要求：

- 默认模型为 `deepseek-v4-flash`
- API key 优先从 `DEEPSEEK_API_KEY` 读取
- UI 中如果用户手动填写 API key，仍允许覆盖环境变量
- 不重构 Streamlit 页面逻辑

### 5. 新增真实 smoke 脚本

新增：

```text
scripts/smoke_deepseek_v4_flash.py
```

脚本行为：

1. 检查 `DEEPSEEK_API_KEY` 是否存在。
2. 如果不存在，打印 `[SKIP] DEEPSEEK_API_KEY not set` 并退出 0。
3. 如果存在，调用：

```text
POST http://127.0.0.1:8000/api/chat
```

请求：

```json
{
  "message": "请用一句话说明 QuakeCore 可以做什么。",
  "language": "zh"
}
```

4. 验证：
   - HTTP 200
   - JSON 包含 `answer`
   - `answer` 非空
   - `error` 为 null 或空字符串
   - `route` 存在

5. 打印模型名和 answer 前 200 字。

如果后端没启动，应提示：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. 新增可选 pytest marker

可以新增：

```text
tests/test_deepseek_live.py
```

要求：

- 如果 `DEEPSEEK_API_KEY` 不存在，skip
- 如果存在，真实调用后端或 agent service
- 使用 pytest marker：`live_api`

运行方式：

```bash
pytest tests/test_deepseek_live.py -q
```

普通：

```bash
pytest tests -q
```

不应因为没有 key 而失败。

### 7. README 更新

README 增加：

```bash
export DEEPSEEK_API_KEY=your_key
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
python scripts/smoke_deepseek_v4_flash.py
```

并说明：

- 当前唯一推荐 DeepSeek 模型：`deepseek-v4-flash`
- 不要在仓库中提交真实 key
- Streamlit 和 FastAPI 共用 `DEEPSEEK_API_KEY`

## 验证命令

无 API key 环境下：

```bash
pytest tests -q
python scripts/smoke_chat.py
```

有 API key 环境下：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
python scripts/smoke_deepseek_v4_flash.py
pytest tests/test_deepseek_live.py -q
```

旧 Streamlit 验证：

```bash
streamlit run app.py
```

在 UI 中选择 DeepSeek，默认模型应为 `deepseek-v4-flash`，并能读取 `DEEPSEEK_API_KEY`。

## 常见失败点

1. config JSON 中空 api_key 覆盖了环境变量。
2. 前端设置页默认模型仍是旧模型。
3. smoke_chat.py 使用旧模型名。
4. Streamlit 默认模型没有同步。
5. live test 被纳入普通 pytest 导致没有网络时失败。

## 完成后报告

请输出：

1. 哪些位置已统一为 `deepseek-v4-flash`
2. `DEEPSEEK_API_KEY` 的读取优先级
3. FastAPI smoke 结果
4. Streamlit 兼容性说明
5. 测试命令与结果
