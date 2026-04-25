# Task 20: ChatGPT-like 极简聊天界面 + 集成上传

推荐模型：`gpt-5.3-codex`。如果只修前端样式和 TypeScript，可切换 `gpt-5.4-mini`。

## 背景：必须修正的方向

当前 `refactor` 分支的后端能力已经比较完整：

- `/api/chat`
- `/api/files/upload`
- `/api/workflows/location/run`
- session file context
- ToolResult 标准化
- earthquake_location workflow
- DeepSeek `deepseek-v4-flash`

问题在于前端体验：当前页面像 landing page，不像 ChatGPT。用户希望的是：

```text
打开就是聊天窗口。
上传、拖拽、粘贴都集成在输入框。
所有地震能力由后端 router / agent / workflow 自动执行。
```

不要做传统软件式工具面板，也不要做一堆领域功能按钮。

---

## 核心目标

把首页改成 ChatGPT-like 极简界面：

```text
顶部：QuakeCore / Settings / Skills
中间：消息流
底部：composer 输入框
composer 左侧：附件上传按钮
全页面：支持拖拽上传
输入框：支持粘贴文件上传
```

用户只通过自然语言触发能力：

```text
请分析当前文件结构
对这个波形做初至拾取
使用当前数据进行地震定位
帮我做连续地震监测
画出台站和震中位置图
```

这些不需要 UI 按钮，全部交给后端自动 route。

---

## 严格禁止

- 不要添加“文件结构分析”按钮。
- 不要添加“初至拾取”按钮。
- 不要添加“地震定位”按钮。
- 不要添加“连续监测”按钮。
- 不要添加传统工具面板。
- 不要做复杂左侧功能栏。
- 不要把前端做成传统地震软件操作界面。
- 不要改动 `agent/tools.py`。
- 不要删除 `app.py`。
- 不要重写后端架构。
- 不要引入大型 UI 框架。
- 不要默认启用 LangGraph。
- 不要引入 Docker、Redis、数据库。
- 不要破坏已有 tests 和 smoke scripts。
- 不要删除 Settings / Skills 页面。

---

# Part A: 页面结构改成 ChatGPT-like

## A1. 移除 landing hero

当前 `frontend/app/page.tsx` 中有 hero / card 风格，例如：

```text
Seismic analysis, routed through chat.
Session card
大块展示区
```

请删除或重构这些 landing 元素。

首页应是一个聊天页面，而不是宣传页。

## A2. 推荐布局

实现类似：

```text
┌───────────────────────────────────────────────┐
│ QuakeCore                         Settings Skills │
├───────────────────────────────────────────────┤
│                                               │
│                message list                   │
│                                               │
│        今天要分析什么地震数据？                 │
│        上传数据，或直接输入问题。               │
│                                               │
├───────────────────────────────────────────────┤
│  +   输入消息...                         Send │
└───────────────────────────────────────────────┘
```

可以保留轻量顶部导航：

- QuakeCore
- Settings
- Skills

不要在首页展示复杂 sidebar。最多可以有一个很窄的 session/new chat 区，但不是必须。

## A3. 空状态提示

空消息时显示：

```text
今天要分析什么地震数据？
上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。
```

可以给自然语言示例，但必须只是提示文本，不是功能按钮：

```text
例如：请分析当前文件结构 / 对当前波形做初至拾取 / 使用当前数据进行地震定位
```

---

# Part B: 上传集成到 composer

## B1. API 封装

检查 `frontend/lib/api.ts`。如果没有上传函数，请新增：

```ts
export async function uploadFile(file: File, sessionId?: string | null): Promise<FileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) formData.append("session_id", sessionId);

  const res = await fetch(`${API_BASE}/api/files/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }

  return res.json();
}
```

类型：

```ts
export type FileUploadResponse = {
  session_id: string;
  filename: string;
  path: string;
  file_type: string;
  bound_to_agent: boolean;
};
```

## B2. 点击上传

在 composer 左侧放一个附件按钮：

```text
+
```

或 paperclip 图标。

要求：

- 使用隐藏 `<input type="file" multiple />`。
- 用户点击附件按钮后选择文件。
- 支持多文件上传。
- 每个文件调用 `/api/files/upload`。
- 上传成功后更新 `sessionId = response.session_id`。
- 后续 chat request 必须带这个 session_id。

## B3. 拖拽上传

支持把文件拖到整个聊天页面。

行为：

- drag over 时显示轻量 overlay：

```text
释放以上传地震数据文件
```

- drop 后自动上传。
- 支持常见文件：`.mseed`, `.miniseed`, `.sac`, `.sgy`, `.segy`, `.h5`, `.hdf5`, `.npy`, `.npz`, `.csv`, `.txt`。

不要做复杂文件管理 UI。

## B4. 粘贴上传

监听 paste 事件。

如果 clipboard 中有文件：

- 自动上传。
- 不阻止普通文本粘贴。

## B5. 上传后的聊天展示

上传成功后，不要显示成传统文件管理器。

要像 ChatGPT 一样，把它插入消息流：

用户消息：

```text
上传了 demo.mseed
```

助手/系统消息：

```text
已接收 demo.mseed，识别为 miniseed，已绑定到当前会话。你可以直接问我：分析文件结构、初至拾取或地震定位。
```

如果 `bound_to_agent=false`，提示：

```text
已接收文件，但该类型暂未自动绑定为当前地震数据。
```

---

# Part C: 自然语言触发领域能力

## C1. 不做按钮，只保留提示

不要添加功能按钮。

但可以在空状态或上传成功提示中给出自然语言示例：

```text
你可以这样问：
“请分析当前文件结构”
“对当前波形做初至拾取”
“使用当前数据进行地震定位”
“帮我做连续地震监测”
```

这些示例不要做成按钮。

## C2. 后端负责 route

前端只发送用户文本到 `/api/chat`。

后端负责判断：

- file_structure
- waveform_reading
- phase_picking
- earthquake_location
- continuous_monitoring
- map_plotting
- seismo_qa

如果 router 关键词缺失，可以小范围补充 `backend/services/router_service.py`，但不要把业务逻辑搬到前端。

建议补充关键词：

- 初至
- 初至拾取
- 到时拾取
- first arrival
- 连续地震监测
- 地震监测

---

# Part D: 消息体验

## D1. 消息流

消息应该像 ChatGPT：

- 用户消息和助手消息清晰区分。
- assistant 显示 route badge，但不要太显眼。
- workflow steps 嵌在 assistant 消息下方。
- artifacts 嵌在 assistant 消息下方。
- error 显示为警告卡片，不让页面崩溃。
- 上传文件显示为消息流中的 file chip。

## D2. 输入框行为

底部 composer：

- 大 textarea。
- Enter 发送。
- Shift+Enter 换行。
- 上传按钮。
- 发送按钮。
- loading 状态。

实现建议：

```ts
onKeyDown={(e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    void handleSend(input);
  }
}}
```

## D3. Artifact URL

如果 artifact.url 是相对路径：

```text
/api/artifacts/xxx.png
```

前端必须使用 `toBackendUrl(artifact.url)`，确保从 FastAPI 后端加载。

---

# Part E: Session 约束

前端必须保证：

1. 第一次上传返回的 `session_id` 存入 state。
2. 后续上传继续带同一个 session_id。
3. 后续 chat 也带同一个 session_id。
4. New chat 时才清空 session_id 和消息。

如果没有 New Chat 按钮，也可以先不做，但不要在每次发送消息时丢失 session。

---

# Part F: 不需要新增后端大功能

后端已有：

- `POST /api/files/upload`
- `POST /api/chat`
- `POST /api/workflows/location/run`

本任务主要修前端体验。

只有当 router 对“初至拾取 / 连续地震监测”识别不准确时，才允许小范围修改：

- `backend/services/router_service.py`
- `tests/test_router_service.py`

---

# Part G: 测试与验证

## G1. 前端 build

```bash
cd frontend
npm run build
```

必须通过。

## G2. 后端测试

```bash
pytest tests -q
```

必须通过。

## G3. Smoke

启动后端：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

运行：

```bash
python scripts/smoke_upload.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
```

## G4. 人工网页验收

打开：

```text
http://localhost:3000
```

人工测试：

1. 页面打开后是聊天窗口，不是 hero landing page。
2. 点击附件按钮上传 `.mseed`。
3. 拖拽 `.mseed` 到页面上传。
4. 粘贴文件上传。
5. 上传后消息流出现文件 chip / 上传提示。
6. 输入：`请分析当前文件结构`。
7. 输入：`对当前波形做初至拾取`。
8. 输入：`使用当前数据进行地震定位`。
9. 检查 route badge、workflow steps、artifacts。
10. 普通提问：`你是谁`，应由 DeepSeek 回复。

---

# Part H: README 更新

README 中说明新版前端支持：

- ChatGPT-like 极简聊天界面。
- 点击上传。
- 拖拽上传。
- 粘贴上传。
- 上传后自然语言触发分析。
- 后端自动 route 到文件结构、初至拾取、地震定位、连续监测等能力。

不要描述为“点击功能按钮执行工具”。

---

# 完成后报告格式

请输出：

1. 是否移除了 landing hero。
2. 是否实现 ChatGPT-like 单聊天窗口。
3. 是否支持点击上传、拖拽上传、粘贴上传。
4. 上传后 session_id 是否被保存并用于 chat。
5. 是否没有添加领域功能按钮。
6. 自然语言是否可以触发地震能力。
7. artifact 是否能打开。
8. workflow steps 是否能显示。
9. `npm run build` 结果。
10. `pytest tests -q` 结果。
11. smoke 脚本结果。
12. 仍然存在的限制。
