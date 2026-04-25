# Phase 3: 包装旧 Agent（必须使用 gpt-5.3-codex）

## 目标

把 agent/core.py 接入 API

---

## 任务

创建：

backend/services/agent_service.py  
backend/routes/chat.py  

---

## 必须复用

agent.core.get_agent_executor  
agent.tools.set_current_lang  

---

## 实现

agent.invoke({"input": message})

---

## API

POST /api/chat

返回：

{
  "session_id": "...",
  "answer": "...",
  "error": null
}

---

## 关键要求

- 必须读取 agent/core.py
- 不能修改 agent/tools.py
- 必须捕获异常

---

## 验证

python scripts/smoke_chat.py  
pytest tests/test_backend_chat_schema.py -q