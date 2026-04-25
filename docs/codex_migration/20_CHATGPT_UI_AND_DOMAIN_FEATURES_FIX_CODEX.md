# Task 20: ChatGPT 风格 UI + 上传体验 + 领域功能入口修复

推荐模型：`gpt-5.3-codex`。如果只修前端样式，可用 `gpt-5.4-mini`，但本任务涉及前后端上传闭环，建议先用 Codex。

## 背景：当前问题

当前 `refactor` 分支的功能架构已经比较完整，但前端体验不达标：

1. 页面不是 ChatGPT 风格，而是 landing page / hero page。
2. 没有明显的文件上传入口。
3. 不支持像 ChatGPT 一样点击、拖拽、复制粘贴上传文件。
4. 用户看不到已上传文件列表。
5. 原有领域能力入口不明显，例如：
   - 地震监测
   - 初至拾取 / 震相拾取
   - 文件结构分析
   - 波形读取
   - 地震定位
   - 地图绘制
6. 当前前端虽然能 chat，但不像一个可用的专业 Agent 工作台。

当前后端其实已经有 `/api/files/upload`，并且上传后会写入 session context、绑定旧 `agent.tools` 当前文件状态。问题主要是前端没有把它产品化。

## 总目标

把前端从“展示页”改成“ChatGPT 风格地震 Agent 工作台”。

目标体验：

```text
左侧：会话 / 功能快捷入口 / 已上传文件
中间：ChatGPT 风格消息流
底部：输入框 + 上传按钮 + 拖拽/粘贴上传
右侧或消息内：workflow steps / artifacts / 文件卡片
```

必须支持：

- 点击上传文件
- 拖拽上传文件
- Ctrl+V / Cmd+V 粘贴文件上传
- 上传后显示文件列表
- 上传后保持同一个 session_id
- 后续 chat 使用同一个 session_id
- 快捷按钮触发地震领域任务

---

## 禁止事项

- 不要改动 `agent/tools.py`。
- 不要删除 `app.py`。
- 不要重写后端架构。
- 不要引入大型 UI 框架。
- 不要默认启用 LangGraph。
- 不要引入 Docker、Redis、数据库。
- 不要破坏已有 tests 和 smoke scripts。
- 不要删除 Settings / Skills 页面。

---

# Part A: 前端整体布局改造

## A1. 移除 hero landing 风格

当前 `frontend/app/page.tsx` 有 hero 区域：

- `Seismic analysis, routed through chat.`
- 大号标题
- session card

这不符合 ChatGPT 风格。

请改为三栏或两栏布局：

```text
┌──────────────────────────────────────────────┐
│ Top bar: QuakeCore | Settings | Skills       │
├──────────────┬───────────────────────────────┤
│ Sidebar      │ Chat messages                 │
│ - New Chat   │                               │
│ - Files      │                               │
│ - Tools      │                               │
│ - Shortcuts  │                               │
│              │ Composer + upload             │
└──────────────┴───────────────────────────────┘
```

可以保留深色主题，但视觉应接近 ChatGPT：

- 左侧窄 sidebar
- 中间消息流
- 底部固定 composer
- 用户/assistant 消息左右或块状区分
- 不要大幅 hero 标题占屏幕

## A2. Chat 页面结构建议

在 `frontend/app/page.tsx` 中组织为：

```tsx
<div className="chat-app">
  <aside className="sidebar">...</aside>
  <main className="chat-main">
    <header className="chat-header">...</header>
    <section className="message-list">...</section>
    <section className="composer-wrap">...</section>
  </main>
</div>
```

可以新增组件：

- `frontend/components/file-uploader.tsx`
- `frontend/components/uploaded-files.tsx`
- `frontend/components/domain-shortcuts.tsx`
- `frontend/components/artifact-card.tsx`

但不要过度拆分。

---

# Part B: 文件上传体验

## B1. API 封装

检查 `frontend/lib/api.ts`，如果没有上传函数，请新增：

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
export type UploadedFile = {
  session_id: string;
  filename: string;
  path: string;
  file_type: string;
  bound_to_agent: boolean;
};
```

## B2. 点击上传

在 composer 左侧或输入框内增加上传按钮：

```text
＋ / paperclip / 上传文件
```

支持多文件上传。对每个文件调用 `/api/files/upload`。

上传成功后：

- 更新 `sessionId` 为 response.session_id
- 添加到 uploadedFiles 列表
- 在消息流中插入一条 assistant/system 提示：

```text
已上传 demo.mseed，类型 miniseed，已绑定到当前会话。
```

## B3. 拖拽上传

支持把文件拖到 chat 主区域。

行为：

- drag over 时显示 overlay：`释放以上传地震数据文件`
- drop 后调用 upload
- 支持 `.mseed`, `.miniseed`, `.sac`, `.sgy`, `.segy`, `.h5`, `.hdf5`, `.npy`, `.npz`, `.csv`, `.txt`

## B4. 粘贴上传

监听 paste 事件。

如果 clipboard 中有 file：

- 自动上传
- 显示上传成功消息

注意：不要阻止普通文本粘贴。

## B5. 已上传文件列表

Sidebar 显示 uploaded files：

```text
已上传文件
- demo.mseed  miniseed  bound
- stations.csv unknown
```

显示字段：

- filename
- file_type
- bound_to_agent 状态

---

# Part C: 地震领域快捷入口恢复

在 sidebar 或 composer 上方添加快捷按钮：

## C1. 必须有这些快捷入口

```text
文件结构分析
读取/绘制波形
初至拾取 / 震相拾取
地震定位
连续地震监测
地图绘制
```

点击后直接发送对应 prompt：

```text
请分析当前文件结构
请读取当前文件第0道波形并绘图
请对当前波形进行初至拾取和P/S震相拾取
请使用当前数据进行地震定位并给出结果和地图
请基于当前配置进行连续地震监测
请绘制当前定位结果和台站分布地图
```

## C2. Route 预期

这些 prompt 应尽量命中：

- file_structure
- waveform_reading
- phase_picking
- earthquake_location
- continuous_monitoring
- map_plotting

如果 router 不准确，可微调 `backend/services/router_service.py` 关键词，但不要过度复杂化。

## C3. 功能提示

如果用户未上传文件就点击快捷入口，应发送也可以，但前端最好提示：

```text
建议先上传 MiniSEED/SAC/SEGY/HDF5 文件。
```

不要阻止发送，因为后端可能使用 example_data。

---

# Part D: 消息体验改造

## D1. 消息流

消息应该像 ChatGPT：

- user 消息靠右或显著区分
- assistant 消息靠左或全宽
- assistant 显示 route badge
- error 显示红色/警告卡片
- workflow steps 嵌在 assistant 消息下方
- artifacts 嵌在 assistant 消息下方

## D2. 输入框

底部 composer：

- 大 textarea
- Enter 发送，Shift+Enter 换行
- 上传按钮
- 发送按钮
- loading 状态

当前如果只支持 form submit，可以增强：

```ts
onKeyDown={(e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend(input);
  }
}}
```

## D3. 初始空状态

不要再显示巨大标题。

空状态显示：

```text
今天要分析什么地震数据？
上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。
```

---

# Part E: 后端确认与小修

## E1. 上传 route 已存在

确认 `backend/routes/files.py`：

- 接收 `session_id` form field
- 返回 `session_id`
- session_store 写入文件
- bind_uploaded_file_to_agent

如果已完成，不要大改。

## E2. Chat 同 session

前端必须确保：

- 上传文件返回的 `session_id` 存入 state
- 后续 chat request 带同一个 `session_id`

否则 Agent 无法知道用户上传过文件。

## E3. Router 关键词

如果连续监测或初至拾取命中不准，补充关键词：

- 初至
- 初至拾取
- 到时拾取
- first arrival
- continuous monitoring
- 连续地震监测

---

# Part F: 测试与验证

## F1. 前端 build

```bash
cd frontend
npm run build
```

必须通过。

## F2. 后端测试

```bash
pytest tests -q
```

必须通过。

## F3. smoke

后端启动：

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

运行：

```bash
python scripts/smoke_upload.py
python scripts/smoke_full_web_agent.py
python scripts/smoke_deepseek_v4_flash.py
```

## F4. 人工网页验收

打开：

```text
http://localhost:3000
```

手动测试：

1. 拖拽上传 `.mseed`。
2. 点击“文件结构分析”。
3. 点击“初至拾取 / 震相拾取”。
4. 点击“地震定位”。
5. 确认 workflow steps 显示。
6. 确认 artifacts 能打开。
7. 粘贴上传一个文件。
8. 普通问答：`你是谁`，应有 DeepSeek 回复。

---

# Part G: README 更新

README 增加新版 Web 使用说明：

```bash
export DEEPSEEK_API_KEY=your_key
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
cd frontend
npm run dev
```

说明前端支持：

- 点击上传
- 拖拽上传
- 粘贴上传
- 文件结构分析
- 初至/震相拾取
- 地震定位 workflow
- 连续监测入口
- artifacts 展示

---

# 完成后报告格式

请输出：

1. 修改了哪些前端文件。
2. 是否支持点击上传、拖拽上传、粘贴上传。
3. 上传后 session_id 是否被保存并用于 chat。
4. 已恢复哪些地震领域快捷入口。
5. artifact 是否能从前端打开。
6. workflow steps 是否能显示。
7. `npm run build` 结果。
8. `pytest tests -q` 结果。
9. smoke 脚本结果。
10. 仍然存在的限制。
