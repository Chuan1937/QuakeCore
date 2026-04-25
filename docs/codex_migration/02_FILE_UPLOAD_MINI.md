# Phase 2: 文件上传（gpt-5.4-mini）

## 目标

实现上传文件 + 推断类型

---

## 新增

backend/routes/files.py  
backend/services/file_service.py  
tests/test_backend_files.py  
scripts/smoke_upload.py  

---

## API

POST /api/files/upload

---

## 支持类型

.mseed -> miniseed  
.sgy/.segy -> segy  
.h5 -> hdf5  
.sac -> sac  

---

## 保存路径

data/uploads/

---

## 返回

{
  "filename": "...",
  "path": "...",
  "file_type": "..."
}

---

## 验证

pytest tests/test_backend_files.py -q  
python scripts/smoke_upload.py  

---

## 限制

不要调用 agent/tools.py（下一阶段做）