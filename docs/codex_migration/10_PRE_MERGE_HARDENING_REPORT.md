# Task 10 Pre-Merge Hardening Report

This report tracks the required hardening checklist before merge.

## Checklist

- [x] `pytest tests -q`
- [x] `frontend build`
- [x] smoke scripts
- [x] README completeness
- [x] security checks

## Executed commands

```bash
PYTHONPATH=. conda run -n quakecore pytest tests -q
cd frontend && npm run build
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_backend.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_upload.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_chat.py
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_upload_then_chat.py
```

## Security checks included

- Artifact route path traversal blocked (`tests/test_backend_artifacts_route.py`).
- Upload type handling:
  - known formats bind state
  - unknown formats remain uploadable and do not bind
- Artifact metadata normalization from markdown image paths.

## Notes

- In sandboxed environments where local loopback sockets are restricted, use `QUAKECORE_SMOKE_INPROCESS=1`.
