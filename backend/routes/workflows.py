"""Workflow API routes."""

from uuid import uuid4

from fastapi import APIRouter

from backend.schemas import LocationWorkflowRunRequest
from backend.workflows.location_workflow import run_location_workflow

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.post("/location/run")
def run_location(payload: LocationWorkflowRunRequest):
    session_id = payload.session_id or uuid4().hex
    result = run_location_workflow(session_id)
    return {"session_id": session_id, **result}
