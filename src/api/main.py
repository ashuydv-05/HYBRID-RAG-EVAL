import os
import sys
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.route.chat import router as chat_router
from src.api.route.health import router as health_router
from src.agent.workflow import MultiAgentWorkflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    try:
        app.state.workflow = MultiAgentWorkflow()
        logger.info("Workflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {e}")
    yield
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="arXiv Research Assistant API",
    description="API for querying arXiv papers using RAG and ReAct agent",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router, prefix="/api")
app.include_router(health_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "arXiv Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api")
async def api_info():
    return {
        "name": "arXiv Research Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream",
            "health": "/api/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
