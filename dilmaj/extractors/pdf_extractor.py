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
    """Extract text from PDFs with optional preprocessing (paragraph-only)."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract_paragraphs(self, path: Path, options: ExtractionOptions) -> List[str]:
        """Extract text as a list of paragraphs for the entire document."""
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            # Extract raw text for the whole document and aggregate
            raw_chunks: List[str] = []
            for page_num in range(0, total_pages):
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
                raise RuntimeError(f"No text content found in PDF: {path}")

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
                    (f"No paragraph content could be derived for PDF: {path}")
                )

            logger.debug(
                "Aggregated %s pages into %s paragraphs", total_pages, len(paragraphs)
            )
            return paragraphs
        except Exception:
            raise
