# document_ingestion package
from .parser import parse_document
from .ocr import ocr_document

__all__ = ["parse_document", "ocr_document"]
