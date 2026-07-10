"""
main.py - FastAPI application entry point.

Run with:
    uvicorn main:app --reload      (from the backend/ directory)
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env before anything else
load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Problem 8 - Industrial Knowledge Intelligence",
    description=(
        "RAG-powered backend for ingesting industrial documents "
        "(equipment specs, maintenance logs, operating procedures, "
        "inspection reports, compliance checklists) and making them "
        "queryable via a conversational AI copilot."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware - allow frontend to call backend ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
from api.routes.ingest import router as ingest_router   # noqa: E402

app.include_router(ingest_router)

# ── Built-in endpoints ──────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "problem-8-industrial-knowledge"}


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Industrial Knowledge Intelligence API",
        "version": "0.1.0",
        "docs": "/docs",
        "ingest_endpoint": "/ingest",
    }


# ── Dev entrypoint (python main.py) ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )