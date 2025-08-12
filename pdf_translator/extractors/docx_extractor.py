"""Word document text extractor implementation (.docx and .doc)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

from ..utils import preprocess_text
from .base import DocumentExtractor, ExtractionOptions

logger = logging.getLogger(__name__)


class WordExtractor(DocumentExtractor):
    """Extract text from Microsoft Word documents.

    - For .docx, uses python-docx to read paragraphs.
    - For legacy .doc, attempts to use the `antiword` CLI if available.
    """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".docx", ".doc"}

    def get_page_count(self, path: Path) -> int:
        # Word documents do not have a stable page concept without rendering.
        # Treat whole document as a single unit.
        return 1

    def _extract_docx_text(self, path: Path) -> str:
        try:
            from docx import Document  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "python-docx is required to read .docx files. " "Please install it."
            ) from e

        try:
            doc = Document(str(path))
            paragraphs = [
                p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()
            ]
            # Include text from tables
            for table in getattr(doc, "tables", []):
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            paragraphs.append(cell_text)
            return "\n\n".join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX {path}: {e}")

    def _extract_doc_text(self, path: Path) -> str:
        antiword = shutil.which("antiword")
        if not antiword:
            raise RuntimeError(
                "Reading .doc files requires 'antiword' to be installed. "
                "Either install antiword or convert the file to .docx."
            )
        try:
            result = subprocess.run(
                [antiword, "-f", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error("antiword failed with code %s: %s", e.returncode, e.stderr)
            raise RuntimeError(f"Failed to read DOC {path}: antiword error") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read DOC {path}: {e}")

    def extract_pages(self, path: Path, options: ExtractionOptions) -> List[str]:
        try:
            suffix = path.suffix.lower()
            if suffix == ".docx":
                raw_text = self._extract_docx_text(path)
            elif suffix == ".doc":
                raw_text = self._extract_doc_text(path)
            else:
                raise RuntimeError(f"Unsupported Word file type: {suffix}")

            if not raw_text or not raw_text.strip():
                raise RuntimeError(f"No text content found in document: {path}")

            if options.preprocess_text:
                cleaned = preprocess_text(
                    raw_text,
                    remove_headers_footers=options.remove_headers_footers,
                    chunk_paragraphs=True,
                ).strip()
                paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            else:
                paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

            if not paragraphs:
                raise RuntimeError(f"No paragraph content could be derived: {path}")

            logger.debug("Extracted %s paragraphs from %s", len(paragraphs), path)
            return paragraphs
        except Exception:
            raise
