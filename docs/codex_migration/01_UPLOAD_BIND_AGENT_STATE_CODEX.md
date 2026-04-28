# Task 01: 上传文件后绑定旧 Agent 当前文件状态

推荐模型：`gpt-5.3-codex`

## 背景

当前 `refactor` 分支已有：

- `backend/services/file_service.py`
- `backend/routes/files.py`

但是目前上传文件只保存到 `data/uploads/`，并返回 `filename/path/file_type`。

旧 QuakeCore 工具仍然依赖 `agent.tools` 中的全局当前文件状态：

- `set_current_segy_path`
- `set_current_miniseed_path`
- `set_current_hdf5_path`
- `set_current_sac_path`

如果上传文件后不调用这些函数，用户在新前端上传文件后再问：

- “分析当前文件结构”
- “读取第0道波形”
- “对当前文件进行震相拾取”

旧 Agent 可能不知道当前文件是谁。

## 目标

修改上传流程，使其在保存成功后自动绑定旧 Agent 当前文件状态。

## 需要修改的文件

优先修改：

- `backend/services/file_service.py`
- `backend/routes/files.py`
- `backend/schemas.py` 如有必要
- `tests/test_backend_files.py`

允许新增：

- `tests/test_file_service.py`

不要修改：

- `app.py`
- `agent/tools.py`
- `agent/core.py`
- `frontend/*`

## 具体实现要求

### 1. 支持 unknown 文件类型

当前 `FileService.infer_file_type()` 对未知扩展名抛 `ValueError`。

请改为：

```python
return FILE_TYPE_BY_SUFFIX.get(suffix, "unknown")
```

要求：

- `.mseed` -> `miniseed`
- `.miniseed` -> `miniseed`
- `.sgy` / `.segy` -> `segy`
- `.h5` / `.hdf5` -> `hdf5`
- `.sac` -> `sac`
- `.npy` -> `npy`
- `.npz` -> `npz`
- 其他 -> `unknown`

unknown 允许上传，但不绑定 agent 当前文件状态。

### 2. 新增绑定函数

在 `backend/services/file_service.py` 中新增函数或方法：

```python
def bind_uploaded_file_to_agent(path: str, file_type: str) -> bool:
    ...
```

逻辑：

```python
if file_type == "segy":
    set_current_segy_path(path)
    return True
elif file_type == "miniseed":
    set_current_miniseed_path(path)
    return True
elif file_type == "hdf5":
    set_current_hdf5_path(path)
    return True
elif file_type == "sac":
    set_current_sac_path(path)
    return True
return False
```

要求：

- 只导入必要函数
- 不要导入整个 `agent.tools`
- 绑定失败时不要让上传失败，除非是明显代码异常
- unknown/npy/npz 第一阶段不绑定，返回 `bound_to_agent=false`

### 3. UploadResponse 增加字段

如果 `backend/schemas.py` 中已有 `FileUploadResponse`，请增加：

```python
bound_to_agent: bool = False
```

返回示例：

```json
{
  "filename": "demo.mseed",
  "path": "data/uploads/xxx_demo.mseed",
  "file_type": "miniseed",
  "bound_to_agent": true
}
```

### 4. routes/files.py 调用绑定

在 `upload_file()` 保存成功后调用绑定逻辑。

## 测试要求

修改或新增 `tests/test_backend_files.py`：

### 必须测试 1：mseed 上传

上传临时文件 `demo.mseed`：

期望：

- status_code == 200
- file_type == "miniseed"
- bound_to_agent == True
- path 文件存在

### 必须测试 2：txt 上传

上传临时文件 `demo.txt`：

期望：

- status_code == 200
- file_type == "unknown"
- bound_to_agent == False
- path 文件存在

### 必须测试 3：扩展名推断

测试：

- `a.miniseed`
- `a.segy`
- `a.sgy`
- `a.h5`
- `a.hdf5`
- `a.sac`
- `a.npy`
- `a.npz`
- `a.txt`

## 验证命令

```bash
pytest tests/test_backend_files.py -q
pytest tests -q
```

## 完成后报告

请输出：

1. 修改了哪些文件
2. 上传后如何绑定旧 Agent
3. unknown 文件如何处理
4. 测试结果
