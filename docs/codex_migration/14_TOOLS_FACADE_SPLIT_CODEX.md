# Task 14: 渐进式拆分 Tools Facade

推荐模型：`gpt-5.3-codex`

## 背景

当前 `agent/tools.py` 仍然是 QuakeCore 最大的技术债之一。它同时包含：

- 文件上下文与全局状态
- SEGY / MiniSEED / SAC / HDF5 结构读取
- 波形读取与绘图
- 格式转换
- 震相拾取
- 连续监测
- 近震定位
- 地图绘制

`refactor` 分支已经新增 `agent/tools_facade.py`，目前它只是从 `agent.tools` 重新导出旧工具。这个方向是正确的：不要直接重写 `agent/tools.py`，而是先通过 facade 做渐进式分层。

## 总目标

把 `agent/tools_facade.py` 从“单纯 re-export 文件”升级为“工具分层入口”，但保持所有现有 LangChain tool 名称、函数签名和 Agent 行为不变。

最终期望结构：

```text
quakecore_tools/
├── __init__.py
├── file_tools.py
├── waveform_tools.py
├── conversion_tools.py
├── picking_tools.py
├── location_tools.py
├── monitoring_tools.py
└── plotting_tools.py
```

第一阶段不要真正搬迁复杂实现，只做安全 re-export 和分组。

## 允许修改

- `agent/tools_facade.py`
- 新增 `quakecore_tools/*.py`
- `tests/*`

谨慎修改：

- `agent/core.py` 只允许在导入路径必要时小范围修改

## 禁止修改

- 不要大改 `agent/tools.py`
- 不要改变任何 tool 的名称
- 不要改变任何 tool 的调用参数
- 不要改变 Agent prompt 中的 tool 名称
- 不要删除旧 Streamlit 依赖
- 不要引入新框架

## 实现步骤

### Step 1: 创建 quakecore_tools 包

新增：

```text
quakecore_tools/__init__.py
quakecore_tools/file_tools.py
quakecore_tools/waveform_tools.py
quakecore_tools/conversion_tools.py
quakecore_tools/picking_tools.py
quakecore_tools/location_tools.py
quakecore_tools/monitoring_tools.py
quakecore_tools/plotting_tools.py
```

每个文件暂时只从 `agent.tools` 导入对应工具并 re-export。

### Step 2: 工具分组

建议分组如下。

#### file_tools.py

- `get_loaded_context`
- `load_local_data`
- `download_seismic_data`
- `get_file_structure`
- `get_hdf5_keys`
- `get_hdf5_structure`
- `get_segy_structure`
- `get_segy_text_header`
- `get_segy_binary_header`
- `get_miniseed_structure`
- `get_sac_structure`

#### waveform_tools.py

- `read_file_trace`
- `read_hdf5_trace`
- `read_trace_sample`
- `read_miniseed_trace`
- `read_sac_trace`

#### conversion_tools.py

- `convert_hdf5_to_numpy`
- `convert_hdf5_to_excel`
- `compress_hdf5_to_zfp`
- `convert_segy_to_numpy`
- `convert_segy_to_excel`
- `convert_segy_to_hdf5`
- `convert_miniseed_to_numpy`
- `convert_miniseed_to_hdf5`
- `convert_miniseed_to_sac`
- `convert_sac_to_numpy`
- `convert_sac_to_hdf5`
- `convert_sac_to_miniseed`
- `convert_sac_to_excel`

#### picking_tools.py

- `pick_first_arrivals`
- `pick_all_miniseed_files`

#### monitoring_tools.py

- `download_continuous_waveforms`
- `run_continuous_picking`
- `associate_continuous_events`
- `run_continuous_monitoring`

#### location_tools.py

- `prepare_nearseismic_taup_cache`
- `locate_earthquake`
- `locate_uploaded_data_nearseismic`
- `locate_place_data_nearseismic`
- `add_station_coordinates`

#### plotting_tools.py

- `plot_location_map`

### Step 3: 更新 agent/tools_facade.py

让 `agent/tools_facade.py` 从 `quakecore_tools.*` 导入，而不是直接从 `agent.tools` 导入所有内容。

重要：

- `agent/core.py` 中已有工具导入列表不应变化。
- 所有 `__all__` 必须保留。
- 工具对象本身必须仍是 LangChain tool 可调用对象。

### Step 4: 增加工具完整性测试

新增测试：

```text
tests/test_tools_facade_exports.py
```

测试内容：

1. `agent.tools_facade` 能导入。
2. 必须包含所有预期工具名。
3. 每个工具对象有 `.name` 属性或可调用。
4. `agent.core.get_agent_executor` 所需 import 不报错。

不要在测试里调用真正耗时工具。

### Step 5: 不搬迁实现

本任务只做 re-export 分层，不把旧 `agent/tools.py` 中的函数实现复制到新文件。

这样做的原因：

- 降低风险
- 保持 Streamlit 兼容
- 后续可逐步搬迁每一类工具

## 验证命令

```bash
pytest tests/test_tools_facade_exports.py -q
pytest tests -q
python -m py_compile agent/tools_facade.py
python -m py_compile agent/core.py
```

## 常见失败点

1. 循环导入：`quakecore_tools` 不要导入 `agent.core`。
2. 工具丢失：`__all__` 与 `agent/core.py` 的 import 列表必须一致。
3. 名称变化：LangChain ReAct Agent 依赖 tool.name，不能包装后改变名称。
4. 误删旧导入：不要让 Streamlit 旧入口失效。

## 完成后报告

请输出：

1. 新增了哪些 `quakecore_tools` 文件
2. 每类工具包含哪些旧工具
3. 是否保持 `agent/core.py` 兼容
4. 测试命令与结果
