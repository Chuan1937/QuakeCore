# Phase 4: Intent Router + Artifacts（gpt-5.3-codex）

## 目标

增加智能路由 + 结果展示

---

## Router

创建：

backend/services/router_service.py

规则：

定位 -> earthquake_location  
拾取 -> phase_picking  
结构 -> file_structure  

---

## Chat 返回

增加字段：

{
  "route": "...",
  "artifacts": []
}

---

## Artifacts

解析：

Markdown 图片路径

---

## 示例

![xxx](data/xxx.png)

→ 转为：

{
  "type": "image",
  "url": "/api/artifacts/xxx.png"
}

---

## 验证

手动调用 /api/chat  
检查 route 和 artifacts