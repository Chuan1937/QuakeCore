# Task 02: 扩展 Intent Router 规则

推荐模型：`gpt-5.4-mini`

扩展 RouterService，使其支持：

- earthquake_location
- phase_picking
- file_structure
- waveform_reading
- format_conversion
- continuous_monitoring
- map_plotting
- seismo_qa
- settings
- general_chat

## 要求

- 基于关键词规则实现
- 保持轻量，不引入 ML
- 添加 tests/test_router_service.py

## 验证

pytest tests/test_router_service.py -q
