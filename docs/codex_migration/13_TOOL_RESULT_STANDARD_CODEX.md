# Task 13: Tool Result 标准化（详细版）

推荐模型：`gpt-5.3-codex`

## 背景

当前 QuakeCore 的工具返回类型不统一：

- 有的返回普通字符串
- 有的返回 JSON 字符串
- 有的返回 Markdown 表格
- 有的返回 Markdown 图片链接
- 有的直接返回错误文本

这导致：

1. 后端难以稳定提取 artifacts。
2. 前端难以统一展示图片、表格、文件和错误。
3. 后续 workflow / LangGraph 难以复用工具输出。
4. 测试只能检查字符串，无法检查结构化结果。

Phase 2 的目标不是马上重写所有旧工具，而是引入一个兼容层，把旧工具输出逐步标准化。

## 总目标

新增统一结构：

```python
class NormalizedToolResult:
    success: bool
    message: str
    data: dict
    artifacts: list
    raw: str | None
    error: str | None
```

并提供适配函数：

```python
normalize_tool_output(output: Any) -> NormalizedToolResult
```

## 允许修改

- `backend/services/tool_result.py`
- `backend/services/router_service.py`
- `backend/services/agent_service.py`
- `backend/schemas.py`
- `tests/*`

谨慎修改：

- `agent/tools_facade.py`

## 禁止修改

- 不要重写 `agent/tools.py`
- 不要改变旧工具的函数签名
- 不要让 LangChain tool 失效
- 不要破坏 Streamlit `app.py`

## 实现步骤

### Step 1: 扩展 tool_result.py

在 `backend/services/tool_result.py` 中定义：

```python
@dataclass
class NormalizedToolResult:
    success: bool = True
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None
```

同时保留现有 `ToolResult`，不要直接删除。

### Step 2: 实现 normalize_tool_output

支持输入：

1. `dict`
2. JSON 字符串
3. 普通字符串
4. Exception
5. None

规则：

- 如果是 dict 且包含 success/message/data/artifacts，直接规范化。
- 如果是 JSON 字符串，尝试解析成 dict。
- 如果是普通字符串，放到 `message` 和 `raw`。
- 如果字符串里包含 Markdown 图片，调用现有 artifact 提取逻辑或共用工具函数。
- 如果是 Exception，success=False，error=str(exc)。

### Step 3: artifact 兼容

仍然保留 Markdown 图片解析作为 fallback，避免旧工具失效。

但优先使用结构化 artifacts。

### Step 4: ChatResponse 兼容

`/api/chat` 返回仍保持：

```json
{
  "session_id": "...",
  "route": "...",
  "answer": "...",
  "artifacts": [],
  "error": null
}
```

不要因为内部结构变化而破坏前端。

### Step 5: 测试

新增或修改测试：

- 普通字符串 -> success true
- JSON 字符串 -> data 可解析
- Markdown 图片 -> artifacts 可提取
- Exception -> success false
- dict -> 保持 data/artifacts

建议新增：

- `tests/test_tool_result_normalization.py`

## 验证命令

```bash
pytest tests/test_tool_result_normalization.py -q
pytest tests -q
```

## 完成后报告

请输出：

1. 新增的数据结构
2. normalize_tool_output 支持哪些输入
3. 是否保持旧工具兼容
4. 测试结果
