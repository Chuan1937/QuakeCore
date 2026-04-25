# Phase 5: Config + Skills（gpt-5.4-mini）

## Config API

GET /api/config/defaults  
GET /api/config/llm  
POST /api/config/llm  

保存：

data/config/llm_config.json  

---

## Skills

目录：

skills/*.md  

---

## API

GET /api/skills  
GET /api/skills/{name}

---

## 验证

pytest tests/test_backend_config.py  
pytest tests/test_backend_skills.py