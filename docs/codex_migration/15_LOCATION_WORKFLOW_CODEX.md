# Task 15: 地震定位 Workflow 抽象（从 Prompt 到可执行流程）

推荐模型：`gpt-5.3-codex`

## 背景

当前 QuakeCore 的地震定位流程主要依赖 LLM prompt 自动串联工具，例如：

- 加载数据
- 拾取震相
- 加载台站坐标
- 计算定位
- 绘制地图

这种方式存在问题：

- 不可测试（只能靠聊天）
- 不可复现（LLM 可能换路径）
- 难以调试（中间步骤不可控）
- 无法接入 LangGraph

## 目标

把“地震定位”从纯 prompt 行为抽象为一个明确的 Python workflow：

```text
load data
→ pick phases
→ prepare taup
→ add station coordinates
→ locate
→ plot map
→ summarize
```

该 workflow：

- 可直接调用（不经过 LLM）
- 可被 LLM 调用
- 可被 LangGraph 接管
- 可单元测试

## 允许修改

- 新增 `backend/workflows/location_workflow.py`
- `backend/services/agent_service.py`
- `backend/services/router_service.py`
- tests/*

## 禁止修改

- 不修改 `agent/tools.py`
- 不破坏现有 tool 调用
- 不删除现有 prompt 路径

## 实现步骤

### Step 1: 新建 workflow 文件

创建：

```text
backend/workflows/location_workflow.py
```

定义函数：

```python
def run_location_workflow(session_id: str) -> dict:
    ...
```

返回结构：

```python
{
    "success": True,
    "steps": [...],
    "location": {...},
    "artifacts": [...],
    "message": "..."
}
```

### Step 2: 串联工具调用

调用顺序（必须严格）：

1. get_loaded_context
2. load_local_data（如无当前文件）
3. pick_all_miniseed_files
4. prepare_nearseismic_taup_cache
5. add_station_coordinates
6. locate_uploaded_data_nearseismic
7. plot_location_map

每一步：

- 捕获异常
- 记录 step 状态
- 将输出写入 steps

### Step 3: 与 session context 集成

workflow 不直接使用全局变量，而是：

- 从 session 获取 current_file
- 通过 facade 调用工具

### Step 4: Agent 调用 workflow

在 `agent_service.py` 中：

当 route == "earthquake_location" 时：

- 优先调用 workflow
- 如果 workflow 失败，再 fallback 到 LLM

### Step 5: Router 标记

router 不需要改变逻辑，但确保 earthquake_location route 被正确识别。

### Step 6: 输出格式统一

workflow 返回结果需转换为 ChatResponse：

- answer：summary 文本
- artifacts：地图路径
- route：earthquake_location

### Step 7: 测试

新增：

```text
tests/test_location_workflow.py
```

测试内容：

- workflow 可调用
- 不抛异常（允许无数据 fallback）
- 返回结构包含 success / steps / artifacts

不要要求真实定位成功。

## 验证命令

```bash
pytest tests/test_location_workflow.py -q
pytest tests -q
```

## 常见失败点

1. 工具返回字符串 → workflow 无法解析
2. 没有 session file → load_local_data 必须 fallback
3. taup cache 失败 → 不应中断整个流程
4. 地图生成失败 → 仍应返回结构化结果

## 完成后报告

请输出：

1. workflow 文件路径
2. 调用了哪些工具
3. fallback 策略
4. 测试结果
