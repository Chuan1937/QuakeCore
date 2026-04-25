# Task 20: ChatGPT-like 极简聊天界面 + 集成上传 + 思考反馈修复

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

但是前端体验仍然不达标：

1. UI 仍不像 ChatGPT 当前网页风格。
2. 页面视觉还是卡片/容器感太重，不是一个简洁聊天窗口。
3. 上传虽然可做，但必须像 ChatGPT 一样集成在输入框。
4. 初至拾取、震相拾取、定位等长任务没有“正在思考/正在执行/工具调用中”的反馈。
5. 用户发出任务后，页面像卡住了一样，直到最终结果返回。

本任务不是新增地震功能按钮，而是修复：

```text
ChatGPT-like UI + 上传体验 + 思考/执行反馈
```

---

## 核心目标

把首页改成接近 ChatGPT 的极简聊天体验：

```text
顶部：QuakeCore / Settings / Skills / New Chat
中间：消息流
底部：固定 composer 输入框
composer 左侧：附件上传按钮
全页面：支持拖拽上传
输入框：支持粘贴文件上传
任务执行中：显示“正在思考/正在执行工具/正在拾取震相”等反馈
```

用户只通过自然语言触发能力：

```text
请分析当前文件结构
对这个波形做初至拾取
使用当前数据进行地震定位
帮我做连续地震监测
画出台站和震中位置图
```

这些不需要 UI 功能按钮，全部交给后端自动 route / agent / workflow。

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

# Part A: UI 改成真正 ChatGPT-like

## A1. 移除 landing / dashboard 感

当前页面仍然卡片边框很重，整体像 dashboard。请改成更接近 ChatGPT：

- 背景简洁。
- 消息流居中，最大宽度约 760-900px。
- 顶部导航轻量。
- 输入框固定在底部居中。
- 空状态居中显示。
- 不要大块 hero 标题。
- 不要大块 session card。
- 不要工具面板。

## A2. 推荐结构

```tsx
<div className="chat-shell">
  <header className="chat-topbar">...</header>
  <main className="chat-main">
    <div className="message-list">...</div>
  </main>
  <footer className="composer-bar">...</footer>
</div>
```

视觉目标：

```text
------------------------------------------------
QuakeCore                         Settings Skills New Chat
------------------------------------------------

                  今天要分析什么地震数据？
        上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。


user / assistant messages...

------------------------------------------------
[ + ]  输入消息...                         [发送]
------------------------------------------------
```

## A3. 空状态

空消息时显示：

```text
今天要分析什么地震数据？
上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。
例如：请分析当前文件结构 / 对当前波形做初至拾取 / 使用当前数据进行地震定位
```

这些示例只是文字，不是按钮。

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

在 composer 左侧放附件按钮：

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

# Part C: 思考反馈 / 执行反馈 / 类流式体验

## C1. 当前问题

用户执行：

```text
对当前波形做初至拾取
```

或：

```text
使用当前数据进行地震定位
```

这些任务可能耗时较长。当前 UI 没有“思考中/执行中/工具调用中”的反馈，看起来像卡住。

必须修复。

## C2. 前端立即插入 pending assistant 消息

当用户发送消息后，立即插入一条 assistant pending 消息，而不是等请求返回。

示例：

```text
QuakeCore 正在思考…
```

然后根据 route 或关键词显示更具体文案：

- 包含“初至/拾取/震相”：`正在分析波形并进行初至/震相拾取…`
- 包含“定位/震中/震源”：`正在执行地震定位工作流…`
- 包含“结构/采样率/header”：`正在读取文件结构…`
- 包含“连续/监测”：`正在准备连续地震监测任务…`
- 默认：`正在思考…`

请求完成后，用最终 assistant message 替换 pending message。

## C3. pending 消息样式

pending message 应该像 ChatGPT 的思考状态：

- 小圆点动画，或
- `正在思考…` 文本闪烁，或
- 简单 loading indicator。

不要使用全屏 loading。

## C4. workflow steps 到达后展示

后端当前不是 SSE 真流式，但返回中已有 workflow steps。请求完成后：

- 展示 answer。
- 展示 route。
- 展示 workflow steps。
- 展示 artifacts。

这属于“类流式体验”：等待期间有 thinking，完成后展示步骤。

## C5. 可选：轻量 SSE 规划但不实现

本任务不要强制实现 SSE。只需在 README 或注释中说明：

```text
当前为 pending 状态 + 最终结果展示；后续可加入 /api/chat/stream SSE。
```

不要为了流式传输大改后端。

---

# Part D: 自然语言触发领域能力

## D1. 不做按钮，只保留提示

不要添加功能按钮。

可以在空状态或上传成功提示中给自然语言示例：

```text
你可以这样问：
“请分析当前文件结构”
“对当前波形做初至拾取”
“使用当前数据进行地震定位”
“帮我做连续地震监测”
```

这些示例不要做成按钮。

## D2. 后端负责 route

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

# Part E: 消息体验

## E1. 消息流

消息应该像 ChatGPT：

- 用户消息和助手消息清晰区分，但不要过度卡片化。
- assistant 显示 route badge，但不要太显眼。
- workflow steps 嵌在 assistant 消息下方。
- artifacts 嵌在 assistant 消息下方。
- error 显示为警告卡片，不让页面崩溃。
- 上传文件显示为消息流中的 file chip。

## E2. 输入框行为

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

## E3. Artifact URL

如果 artifact.url 是相对路径：

```text
/api/artifacts/xxx.png
```

前端必须使用 `toBackendUrl(artifact.url)`，确保从 FastAPI 后端加载。

---

# Part F: Session 约束

前端必须保证：

1. 第一次上传返回的 `session_id` 存入 state。
2. 后续上传继续带同一个 session_id。
3. 后续 chat 也带同一个 session_id。
4. New Chat 时清空 session_id 和消息。

不要在每次发送消息时丢失 session。

---

# Part G: 参考旧 Streamlit UI 与行为

请检查旧 `app.py` 中的 UI / 执行反馈逻辑，借鉴：

- chat 消息展示方式
- DeepSeek 调用方式
- 用户提交后如何显示处理中状态
- 工具执行返回如何展示

不要复制 Streamlit 代码到 Next.js，但要借鉴其用户体验。

---

# Part H: 测试与验证

## H1. 前端 build

```bash
cd frontend
npm run build
```

必须通过。

## H2. 后端测试

```bash
pytest tests -q
```

必须通过。

## H3. Smoke

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

## H4. 人工网页验收

打开：

```text
http://localhost:3000
```

人工测试：

1. 页面打开后是 ChatGPT-like 单聊天窗口，不是 hero landing page。
2. 点击附件按钮上传 `.mseed`。
3. 拖拽 `.mseed` 到页面上传。
4. 粘贴文件上传。
5. 上传后消息流出现文件 chip / 上传提示。
6. 输入：`请分析当前文件结构`。
7. 输入：`对当前波形做初至拾取`。
8. 发送后立即看到“正在分析波形并进行初至/震相拾取…”。
9. 输入：`使用当前数据进行地震定位`。
10. 发送后立即看到“正在执行地震定位工作流…”。
11. 请求完成后显示 route badge、workflow steps、artifacts。
12. 普通提问：`你是谁`，应由 DeepSeek 回复。

---

# Part I: README 更新

README 中说明新版前端支持：

- ChatGPT-like 极简聊天界面。
- 点击上传。
- 拖拽上传。
- 粘贴上传。
- 上传后自然语言触发分析。
- 长任务有 pending/thinking 反馈。
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
6. 是否实现长任务 pending/thinking 反馈。
7. 初至拾取时是否立即显示“正在分析波形并进行初至/震相拾取…”。
8. 定位时是否立即显示“正在执行地震定位工作流…”。
9. 自然语言是否可以触发地震能力。
10. artifact 是否能打开。
11. workflow steps 是否能显示。
12. `npm run build` 结果。
13. `pytest tests -q` 结果。
14. smoke 脚本结果。
15. 仍然存在的限制。
