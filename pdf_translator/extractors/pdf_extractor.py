"""PDF text extractor implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader

from ..utils import preprocess_text
from .base import DocumentExtractor, ExtractionOptions

logger = logging.getLogger(__name__)


class PDFExtractor(DocumentExtractor):
    """Extract text from PDFs with optional preprocessing."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def get_page_count(self, path: Path) -> int:
        try:
            reader = PdfReader(str(path))
            return len(reader.pages)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {path}: {e}")

    def extract_pages(self, path: Path, options: ExtractionOptions) -> List[str]:
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            # Normalize page range
            start = options.start_page if options.start_page is not None else 1
            end = options.end_page if options.end_page is not None else total_pages
            if start < 1:
                raise ValueError(f"Start page must be >= 1, got {start}")
            if end > total_pages:
                raise ValueError(f"End page {end} exceeds total pages {total_pages}")
            if start > end:
                raise ValueError(
                    f"Start page {start} cannot be greater than end page {end}"
                )

            pages: List[str] = []
            for page_num in range(start - 1, end):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        logger.warning(f"Page {page_num + 1} appears to be empty")
                        pages.append("")
                        continue

                    if options.preprocess_text:
                        cleaned = preprocess_text(
                            text,
                            remove_headers_footers=options.remove_headers_footers,
                            chunk_paragraphs=options.chunk_paragraphs,
                        ).strip()
                    else:
                        cleaned = text

                    if cleaned:
                        pages.append(cleaned)
                        if options.preprocess_text:
                            logger.debug(
                                "Extracted and cleaned page %s: %s chars -> %s chars",
                                page_num + 1,
                                len(text),
                                len(cleaned),
                            )
                        else:
                            logger.debug(
                                "Extracted page %s: %s characters",
                                page_num + 1,
                                len(text),
                            )
                    else:
                        logger.warning(
                            f"Page {page_num + 1} was empty after preprocessing"
                        )
                        pages.append("")
                except Exception as e:
                    logger.error(
                        f"Failed to extract text from page {page_num + 1}: {e}"
                    )
                    pages.append("")

            if not any(p.strip() for p in pages):
                raise RuntimeError(
                    f"No text content found in pages {start}-{end} of PDF: {path}"
                )

            return pages
        except Exception:
            raise
