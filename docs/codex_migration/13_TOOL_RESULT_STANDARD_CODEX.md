# Task 13: Tool Result 标准化

推荐模型：gpt-5.3-codex

## 背景

当前工具返回存在严重不一致：

- 有的返回字符串
- 有的返回 JSON 字符串
- 有的返回 Markdown
- 有的返回带图片路径的文本

这导致：

- router 无法稳定解析 artifact
- 前端无法统一展示
- workflow 无法复用

## 目标

设计统一结构：

{
  "success": true,
  "message": "",
  "data": {},
  "artifacts": [],
  "raw": ""
}

## 实现步骤

### Step 1: 定义 ToolResult

在 backend/services/tool_result.py 扩展结构。

### Step 2: 包装旧工具输出

在 tools_facade 中统一包装。

### Step 3: Router 使用标准结果

router 不再解析字符串，而是解析 artifacts 字段。

### Step 4: 向后兼容

如果工具返回字符串：

- 自动包装为 ToolResult

## 测试

- 单个工具返回
- 多 artifact
- 错误情况

## 验证

pytest tests -q
