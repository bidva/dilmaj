"""Extractor interfaces and implementations for different document types."""

from .base import DocumentExtractor, ExtractionOptions
from .docx_extractor import WordExtractor
from .pdf_extractor import PDFExtractor

__all__ = [
    "DocumentExtractor",
    "ExtractionOptions",
    "PDFExtractor",
    "WordExtractor",
]
