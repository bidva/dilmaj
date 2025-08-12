"""PDF text extractor implementation."""

from __future__ import annotations

import logging
import re
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
        """Extract text as a list of paragraphs from the selected page range.

        Note: This implementation aggregates the selected pages and returns
        paragraph-level chunks instead of one string per page.
        """
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

            # Extract raw text for the requested range and aggregate
            raw_chunks: List[str] = []
            for page_num in range(start - 1, end):
                try:
                    page = reader.pages[page_num]
                    text = (page.extract_text() or "").strip()
                    if not text:
                        logger.warning(f"Page {page_num + 1} appears to be empty")
                    else:
                        logger.debug(
                            "Extracted page %s: %s characters",
                            page_num + 1,
                            len(text),
                        )
                    raw_chunks.append(text)
                except Exception as e:
                    logger.error(
                        f"Failed to extract text from page {page_num + 1}: {e}"
                    )
                    raw_chunks.append("")

            combined_text = "\n\n".join(chunk for chunk in raw_chunks if chunk)

            if not combined_text.strip():
                raise RuntimeError(
                    f"No text content found in pages {start}-{end} of PDF: {path}"
                )

            # Preprocess and split into paragraphs
            if options.preprocess_text:
                cleaned = preprocess_text(
                    combined_text,
                    remove_headers_footers=options.remove_headers_footers,
                    # Force paragraph chunking for paragraph output
                    chunk_paragraphs=True,
                ).strip()
                paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            else:
                # Naive paragraph split on blank lines when preprocessing is disabled
                paragraphs = [
                    p.strip() for p in re.split(r"\n\s*\n+", combined_text) if p.strip()
                ]

            if not paragraphs:
                raise RuntimeError(
                    (
                        f"No paragraph content could be derived from pages "
                        f"{start}-{end} of PDF: {path}"
                    )
                )

            logger.debug(
                "Aggregated pages %s-%s into %s paragraphs", start, end, len(paragraphs)
            )
            return paragraphs
        except Exception:
            raise
