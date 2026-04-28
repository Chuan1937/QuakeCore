# Task 12: Session File Context 设计与实现

推荐模型：gpt-5.3-codex

## 背景

当前系统仍然依赖 `agent.tools` 中的全局变量来记录“当前文件”，例如：

- CURRENT_SEGY_PATH
- CURRENT_MINISEED_PATH
- CURRENT_HDF5_PATH
- CURRENT_SAC_PATH

这在单用户 Streamlit 场景下可行，但在 Web 多会话环境中会导致：

- 用户 A 上传的文件影响用户 B
- 并发请求互相污染
- 测试不可复现

## 目标

引入 Session File Context：

session_id -> uploaded_files -> current_file

但必须保持与旧 agent.tools 的兼容。

## 允许修改的文件

- backend/services/session_store.py
- backend/services/file_service.py
- backend/services/agent_service.py
- backend/routes/files.py
- backend/routes/chat.py
- tests/*（必要时）

## 禁止修改

- agent/tools.py（不允许大改）
- app.py

## 实现步骤

### Step 1: 扩展 session_store

为每个 session 保存：

- uploaded_files: List[str]
- current_file: Optional[str]

提供方法：

- add_file(session_id, path)
- set_current_file(session_id, path)
- get_current_file(session_id)

### Step 2: 上传时写入 session

在 upload API 中：

- 获取 session_id（若没有则生成）
- 调用 session_store.add_file
- 默认把最新文件设为 current_file

### Step 3: Chat 前注入旧 Agent 状态

在 agent_service 中：

- 获取 session current_file
- 根据后缀调用：
  - set_current_miniseed_path
  - set_current_segy_path
  - ...

注意：这里只是“临时注入”，不改变 session 结构。

### Step 4: 向后兼容

如果 session 没有文件：

- 保持旧行为
- 不抛异常

## 测试要求

新增测试：

- 两个 session 上传不同文件
- 确认不会互相污染

## 验证

pytest tests -q

## 完成标准

- session 能隔离文件
- 旧 Agent 仍能正常读取当前文件
