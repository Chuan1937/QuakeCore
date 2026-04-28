# QuakeCore Plotting

必须严格遵守以下绘图规范。适用于所有地震波形图、震相拾取图、目录统计图、GMT/PyGMT 图。

## 通用规则

- 所有图片必须保存到 `data/plots/` 或当前 workspace 的 `outputs/`。
- 图片文件名使用英文、短横线或下划线，不要使用空格。
- 不要在正文中说"无法显示图像"；前端会自动显示 artifact。
- 正文只总结关键结论，图片由前端显示。

## 波形 + 拾取图规则

- 对单道波形，使用上下两栏布局：
  - 上栏：波形
  - 下栏：拾取置信度或方法对比
- P 波使用红色三角标记 (`#d62728`, marker='v')。
- S 波使用蓝色圆点标记 (`#1f77b4`, marker='o')。
- P/S 到时必须画垂直虚线 (`linestyle='--', alpha=0.85`)。
- 垂直线和文字标签不要遮挡主波形。
- 图例放在右上角或图外侧。
- x 轴使用 UTC 时间或相对秒，必须标明单位。
- y 轴标明 amplitude/counts。
- 标题格式：`Trace {index} ({channel}) — Phase Picking`
- 如果同一震相有多个模型结果，要在图中区分 method。

## 防止标注重叠

- 多个 pick 时间很接近时，文本标签应使用 y-offset 分层。
- 不要把所有文字都放在同一个 y 坐标。
- 可以只显示 method+phase+score，不显示完整时间。
- 对 score subplot，使用固定 y 位置分层：
  - eqtransformer-P: 0.95
  - phasenet-P: 0.82
  - eqtransformer-S: 0.62
  - phasenet-S: 0.35
- x 轴时间刻度最多显示 5 到 7 个。
- 使用 `fig.autofmt_xdate()` 或旋转 30 度。
- `fig.tight_layout()` 或 `constrained_layout=True` 必须使用。

## 拾取置信度标注

- 置信度低于 0.5 的拾取点降低透明度 (alpha=0.4) 或单独标注。
- 高置信度 (>0.8) 使用实线标记，中等置信度 (0.5-0.8) 使用虚线标记。

## 震相拾取 CSV 字段解析规则

- 绘图前必须自动识别字段，不允许硬编码只读取 `score`、`phase`、`sample`。

### 方法字段候选

- `method`
- `model`
- `picker`
- `algorithm`

### 震相字段候选

- `phase`
- `phase_type`
- `phase_name`
- `label`
- `type`

### 置信度字段候选

- `score`
- `confidence`
- `probability`
- `prob`
- `peak_value`
- `phase_score`
- `p_score`
- `s_score`

### 样本点字段候选

- `sample`
- `sample_index`
- `arrival_sample`
- `pick_sample`
- `index`
- `sample_id`

### 时间字段候选

- `time`
- `arrival_time`
- `relative_time`
- `time_sec`
- `seconds`

## 禁止事项

- 禁止使用 `row.get("score", 0)` 直接作为置信度。
- 禁止使用 `row.get("phase", "")` 后不检查空值。
- 禁止只读 `score`/`phase`/`sample` 单一字段。
- 禁止把 sample index 直接当秒。
- 禁止在图例中显示 `(0.00)`，除非原始 CSV 置信度确实为 0。
- 如果无法解析 score，图例应显示 `(NA)`，并在回答中说明字段无法识别。
- 单道拾取图必须检查 waveform 是否成功读取；如果没有读取 waveform，不要生成伪波形图。

## 输出

- 必须保存 PNG (dpi=150)。
- 如果生成了筛选表，另存 CSV。
- 最终回答中不要写本地绝对路径。
