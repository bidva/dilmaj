"""Base interfaces for document text extractors (paragraph-oriented)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


@dataclass(frozen=True)
class ExtractionOptions:
    """Options to control text extraction and preprocessing.

    Attributes:
        preprocess_text: Whether to perform general text preprocessing.
        remove_headers_footers: Whether to try removing headers/footers when
            preprocessing.
        chunk_paragraphs: Whether to chunk text into paragraphs during
            preprocessing.
    """

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

    def extract_paragraphs(self, path: Path, options: ExtractionOptions) -> List[str]:
        """Extract textual content as paragraphs for the entire document.

        Return a list of non-empty paragraph strings in reading order.
        """
        ...
