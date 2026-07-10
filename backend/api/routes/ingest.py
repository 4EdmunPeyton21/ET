"""
api/routes/ingest.py - Document ingestion endpoint.

POST /ingest
  Content-Type : multipart/form-data
  Field        : files (multiple UploadFile)

Response:
  {
    "documents_processed": int,
    "results": [
      {
        "filename":    str,
        "text_length": int,
        "preview":     str,       # first 200 chars
        "status":      "success" | "error",
        "error_msg":   str | null
      }
    ]
  }
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_ingestion.parser import parse_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Document Ingestion"])


# ── Response models (drives Swagger docs) ────────────────────────────────────

class DocumentResult(BaseModel):
    filename: str
    text_length: int
    preview: str
    status: str               # "success" | "error"
    error_msg: Optional[str] = None


class IngestResponse(BaseModel):
    documents_processed: int
    results: List[DocumentResult]


# ── Allowed extensions ───────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=IngestResponse,
    summary="Ingest one or more industrial documents",
    description=(
        "Upload PDF or TXT files for text extraction. "
        "Returns structured JSON with extracted text preview and metadata."
    ),
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="One or more PDF or TXT files"),
) -> IngestResponse:
    """Accept multipart file uploads, extract text, return structured JSON."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    results: List[DocumentResult] = []

    for upload in files:
        filename = upload.filename or "unknown"
        logger.info("Processing uploaded file: %s", filename)

        # Validate extension
        if not _is_allowed(filename):
            ext = Path(filename).suffix or "(none)"
            results.append(
                DocumentResult(
                    filename=filename,
                    text_length=0,
                    preview="",
                    status="error",
                    error_msg=f"Unsupported file type '{ext}'. Allowed: .pdf, .txt",
                )
            )
            continue

        # Save to temp file so parse_document can work with a path
        suffix = Path(filename).suffix.lower()
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                content = await upload.read()
                tmp.write(content)

            parsed = parse_document(tmp_path)

            results.append(
                DocumentResult(
                    filename=filename,
                    text_length=parsed["text_length"],
                    preview=parsed["text"][:200] if parsed["text"] else "",
                    status=parsed["status"],
                    error_msg=parsed.get("error_msg"),
                )
            )

        except Exception as exc:
            logger.error("Unexpected error processing '%s': %s", filename, exc)
            results.append(
                DocumentResult(
                    filename=filename,
                    text_length=0,
                    preview="",
                    status="error",
                    error_msg=str(exc),
                )
            )
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        documents_processed=len(results),
        results=results,
    )
