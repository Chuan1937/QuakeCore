"""FastAPI application entrypoint for the migration backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.artifacts import router as artifacts_router
from backend.routes.config import router as config_router
from backend.routes.chat import router as chat_router
from backend.routes.files import router as files_router
from backend.routes.health import router as health_router
from backend.routes.skills import router as skills_router
from backend.routes.workflows import router as workflows_router

app = FastAPI(title="QuakeCore Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(artifacts_router)
app.include_router(config_router)
app.include_router(skills_router)
app.include_router(workflows_router)
