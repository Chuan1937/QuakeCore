# Task 04: upload + chat smoke 测试

推荐模型：gpt-5.4-mini

新增脚本：

scripts/smoke_upload_then_chat.py

流程：

1. GET /health
2. 上传文件
3. POST /api/chat
4. 检查返回 JSON 结构

## 验证

python scripts/smoke_upload_then_chat.py
