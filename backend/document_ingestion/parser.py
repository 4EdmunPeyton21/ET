"""
parser.py - Document text extraction.

Strategy
--------
PDF  -> pdfplumber (primary)
        If a page has no text (scanned), fall back to OCR via ocr.py.
     -> PyPDF2 (secondary, if pdfplumber cannot open the file at all)
TXT  -> plain UTF-8 read (with latin-1 fallback)
"""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ocr_page_image(page) -> str:
    """Render a pdfplumber Page to an image and run EasyOCR on it."""
    from document_ingestion.ocr import ocr_document

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pil_image = page.to_image(resolution=200).original
        pil_image.save(tmp_path, format="PNG")
        return ocr_document(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PDF parsers
# ---------------------------------------------------------------------------

def _parse_pdf_pdfplumber(file_path: Path) -> dict:
    """Parse a PDF with pdfplumber; fall back to OCR for image-only pages."""
    import pdfplumber

    page_texts: list[str] = []
    ocr_count = 0

    with pdfplumber.open(str(file_path)) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text()
            if raw and raw.strip():
                page_texts.append(raw.strip())
            else:
                logger.info(
                    "Page %d/%d in '%s' has no text - trying OCR.",
                    i, total_pages, file_path.name,
                )
                try:
                    ocr_text = _ocr_page_image(page)
                    if ocr_text.strip():
                        page_texts.append(ocr_text.strip())
                    ocr_count += 1
                except Exception as exc:
                    logger.warning("OCR failed for page %d: %s", i, exc)

    return {
        "text": "\n\n".join(page_texts),
        "pages": total_pages,
        "ocr_pages_count": ocr_count,
        "method": "pdfplumber" + ("+ocr" if ocr_count else ""),
    }


def _parse_pdf_pypdf2(file_path: Path) -> dict:
    """Fallback PDF parser using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(str(file_path))
    page_texts = [
        p.extract_text().strip()
        for p in reader.pages
        if p.extract_text()
    ]
    return {
        "text": "\n\n".join(page_texts),
        "pages": len(reader.pages),
        "ocr_pages_count": 0,
        "method": "pypdf2",
    }


def _parse_pdf(file_path: Path) -> dict:
    """Try pdfplumber first; fall back to PyPDF2."""
    try:
        return _parse_pdf_pdfplumber(file_path)
    except Exception as primary_exc:
        logger.warning(
            "pdfplumber failed for '%s' (%s) - trying PyPDF2.",
            file_path.name, primary_exc,
        )
        try:
            return _parse_pdf_pypdf2(file_path)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"All PDF parsers failed for '{file_path.name}'. "
                f"pdfplumber: {primary_exc} | PyPDF2: {fallback_exc}"
            ) from fallback_exc


# ---------------------------------------------------------------------------
# TXT parser
# ---------------------------------------------------------------------------

def _parse_txt(file_path: Path) -> dict:
    """Read a plain-text file (UTF-8 with latin-1 fallback)."""
    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_path.read_text(encoding=encoding)
            return {
                "text": text,
                "pages": None,
                "ocr_pages_count": 0,
                "method": f"plaintext:{encoding}",
            }
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode '{file_path.name}' as UTF-8 or latin-1.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(file_path: str) -> dict:
    """
    Extract text from a document and return a structured dict.

    Supported formats: .pdf, .txt

    Returns
    -------
    dict with keys:
        file_path, filename, file_type, text, text_length,
        pages, ocr_pages_count, method, status, error_msg
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    result_base = {
        "file_path": str(path),
        "filename": path.name,
        "file_type": suffix.lstrip("."),
    }

    try:
        if suffix == ".pdf":
            extracted = _parse_pdf(path)
        elif suffix == ".txt":
            extracted = _parse_txt(path)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported: .pdf, .txt"
            )

        return {
            **result_base,
            "text": extracted["text"],
            "text_length": len(extracted["text"]),
            "pages": extracted.get("pages"),
            "ocr_pages_count": extracted.get("ocr_pages_count", 0),
            "method": extracted["method"],
            "status": "success",
            "error_msg": None,
        }

    except Exception as exc:
        logger.error("parse_document failed for '%s': %s", path.name, exc)
        return {
            **result_base,
            "text": "",
            "text_length": 0,
            "pages": None,
            "ocr_pages_count": 0,
            "method": "none",
            "status": "error",
            "error_msg": str(exc),
        }
