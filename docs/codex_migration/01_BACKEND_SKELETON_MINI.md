# Phase 1: FastAPI 基础骨架（使用 gpt-5.4-mini）

## 目标

创建最小后端服务

## 任务

创建：

- backend/main.py
- backend/routes/health.py
- backend/schemas.py
- requirements-backend.txt
- tests/test_backend_health.py
- scripts/smoke_backend.py

---

## backend/main.py

要求：

- FastAPI
- CORS 支持 localhost:3000
- 注册 health route

---

## health API

路径：

GET /health

返回：

{
  "status": "ok"
}

---

## requirements-backend.txt

包含：

fastapi
uvicorn[standard]
pydantic
requests

---

## 测试

pytest test_backend_health.py

---

## 验证

必须执行：

uvicorn backend.main:app --reload  
python scripts/smoke_backend.py  
pytest tests/test_backend_health.py -q

---

## 限制

不要：

- 不要读取 agent/*
- 不要改旧代码