"""Base interfaces for document text extractors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


@dataclass(frozen=True)
class ExtractionOptions:
    """Options to control text extraction.

    Attributes:
        start_page: 1-based inclusive start page (for paged docs).
            If None, starts at 1.
        end_page: 1-based inclusive end page (for paged docs).
            If None, goes to last page.
        preprocess_text: Whether to perform general text preprocessing.
        remove_headers_footers: Whether to try removing headers/footers when
            preprocessing.
        chunk_paragraphs: Whether to chunk text into paragraphs during
            preprocessing.
    """

    start_page: int | None = None
    end_page: int | None = None
    preprocess_text: bool = True
    remove_headers_footers: bool = True
    chunk_paragraphs: bool = True


class DocumentExtractor(Protocol):
    """Protocol for document extractors.

    Implementations should be stateless and reusable.
    """

    def supports(self, path: Path) -> bool:
        """Return True if this extractor can handle the given file/path."""
        ...

    def get_page_count(self, path: Path) -> int:
        """Return number of pages/parts for the document (>=1).

        For non-paged documents, return 1.
        """
        ...

    def extract_pages(self, path: Path, options: ExtractionOptions) -> List[str]:
        """Extract textual content for each requested page (or unit).

        Return a list where index 0 corresponds to `options.start_page`.
        Empty strings may be used for pages with no extractable text.
        """
        ...
