"""
ocr.py - OCR extraction using EasyOCR.

Fallback for scanned/image-based PDF pages that contain no extractable text.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded singleton so the model is only downloaded/initialised once.
_reader = None


def _get_reader():
    """Return a cached EasyOCR Reader instance (English)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Initialising EasyOCR reader (first call - may take a moment)...")
            _reader = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR reader ready.")
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc
    return _reader


def ocr_document(image_path: str) -> str:
    """
    Extract text from an image file using EasyOCR.

    Parameters
    ----------
    image_path : str
        Path to a PNG / JPEG / TIFF image.

    Returns
    -------
    str
        Concatenated text detected by OCR, or empty string if nothing found.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = _get_reader()

    try:
        logger.info("Running OCR on: %s", path.name)
        results = reader.readtext(str(path), detail=0, paragraph=True)
        text = "\n".join(results)
        logger.info("OCR extracted %d characters from %s", len(text), path.name)
        return text
    except Exception as exc:
        logger.error("EasyOCR failed on %s: %s", path.name, exc)
        raise RuntimeError(f"OCR failed for {path.name}: {exc}") from exc
