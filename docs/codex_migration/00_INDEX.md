# QuakeCore Codex 迁移任务索引

## 模型使用策略

| 模型 | 用途 | 是否常用 |
|------|------|--------|
| gpt-5.4-mini | 默认开发（最省） | ✅ 主力 |
| gpt-5.3-codex | 复杂代码迁移 | ⚠️ 按需 |
| gpt-5.4 | mini 不行时兜底 | ⚠️ 少用 |
| gpt-5.5 | 不推荐 | ❌ |

---

## 执行顺序

1. 01_BACKEND_SKELETON_MINI.md
2. 02_FILE_UPLOAD_MINI.md
3. 03_CHAT_AGENT_WRAPPER_CODEX.md
4. 04_ROUTER_ARTIFACTS_CODEX.md
5. 05_CONFIG_SKILLS_MINI.md
6. 06_FRONTEND_BASE_MINI.md
7. 07_FRONTEND_SETTINGS_SKILLS_MINI.md
8. 08_TESTS_AND_DOCS_MINI.md
9. 09_PHASE2_REFACTOR_CODEX.md

---

## 规则（必须遵守）

- 不删除 app.py
- 不破坏 Streamlit
- 不做 CLI
- 不用 Docker
- 不改 agent/tools.py 结构（第一阶段）
- 每次只执行一个任务文件