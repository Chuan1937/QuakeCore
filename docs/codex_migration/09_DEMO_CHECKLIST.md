# Task 09 Demo Checklist

This checklist validates the end-to-end refactor flow:

1. upload file
2. bind state
3. call chat
4. check route and artifacts

## Prerequisites

- Conda env `quakecore` is available.
- Dependencies from `requirements.txt` and `requirements-backend.txt` are installed.

## Run demo smoke (in-process)

```bash
PYTHONPATH=. QUAKECORE_SMOKE_INPROCESS=1 conda run -n quakecore python scripts/smoke_upload_then_chat.py
```

Expected result:

- Exit code `0`
- Output includes `Upload then chat smoke check passed`

## Manual API checks (optional)

If local network loopback is available and backend is running:

```bash
conda run -n quakecore uvicorn backend.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_backend.py
python scripts/smoke_upload.py
python scripts/smoke_chat.py
python scripts/smoke_upload_then_chat.py
```

## Acceptance criteria

- Upload returns `file_type=miniseed` and `bound_to_agent=true` for `.mseed`.
- Chat response contains keys:
  - `session_id`
  - `answer`
  - `error`
  - `route`
  - `artifacts`
- Route is `file_structure` for message `Analyze the current file structure.`
- `artifacts` is a list and follows metadata schema:
  - `type`
  - `name`
  - `path`
  - `url`
