# Task 16: Phase 2 Workflow 稳定化（集中版本）

推荐模型：gpt-5.3-codex

## 核心目标

集中解决：
- workflow 不稳定
- 过度 fallback LLM
- 无法解释失败

## 必做修改

1. workflow 返回 status：success / partial_success / failed
2. step 增加 duration_ms
3. plot 失败不影响整体成功
4. AgentService 不再仅在 success 时返回 workflow
5. 新增 API：/api/workflows/location/run
6. 新增 smoke：scripts/smoke_location_workflow.py

## 验证

pytest tests -q
python scripts/smoke_location_workflow.py
