# Task 03: 增强 Artifact 元数据

推荐模型：gpt-5.4-mini

## 目标

扩展 artifact 返回结构：

- type
- url
- name
- path

## 示例

![img](data/demo.png)

->

{
  type: image,
  name: demo.png,
  path: demo.png,
  url: /api/artifacts/demo.png
}

## 验证

pytest tests -q
