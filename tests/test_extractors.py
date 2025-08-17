"""Tests for document extractors (PDF and DOCX)."""

from pathlib import Path
from typing import List

import pytest

from dilmaj.extractors.base import ExtractionOptions
from dilmaj.extractors.docx_extractor import WordExtractor
from dilmaj.extractors.pdf_extractor import PDFExtractor


@pytest.fixture()
def options() -> ExtractionOptions:
    return ExtractionOptions(
        preprocess_text=True, remove_headers_footers=True, chunk_paragraphs=True
    )


def _make_minimal_pdf(path: Path) -> None:
    # Create a minimal PDF with plain text using pypdf
    from pypdf import PdfWriter

    writer = PdfWriter()
    # Add a blank page to ensure valid PDF structure
    writer.add_blank_page(width=72, height=72)
    # pypdf doesn't write text easily without ReportLab; emulate by metadata
    # We'll instead rely on extractor gracefully handling empty/blank pages.
    with open(path, "wb") as f:
        writer.write(f)


def test_pdf_extractor_empty_pages_warns(tmp_path: Path, options: ExtractionOptions):
    pdf = tmp_path / "empty.pdf"
    _make_minimal_pdf(pdf)

    ex = PDFExtractor()
    assert ex.supports(pdf)

    # Even though the PDF is blank, extractor should error due to no content
    with pytest.raises(RuntimeError):
        ex.extract_paragraphs(pdf, options)


def test_word_extractor_supports_suffix(tmp_path: Path, options: ExtractionOptions):
    ex = WordExtractor()
    assert ex.supports(tmp_path / "x.docx")
    assert ex.supports(tmp_path / "x.DOC")


def test_word_extractor_doc_antiword_missing(
    monkeypatch, tmp_path: Path, options: ExtractionOptions
):
    # Force .doc path, and simulate antiword not found
    doc = tmp_path / "f.doc"
    doc.write_bytes(b"not a real doc")

    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError) as ei:
        WordExtractor().extract_paragraphs(doc, options)
    assert "antiword" in str(ei.value)


def test_word_extractor_docx_success(
    monkeypatch, tmp_path: Path, options: ExtractionOptions
):
    # Fake python-docx Document
    class Para:
        def __init__(self, text: str):
            self.text = text

    class Cell:
        def __init__(self, text: str):
            self.text = text

    class Row:
        def __init__(self, cells):
            self.cells = cells

    class Table:
        def __init__(self):
            self.rows = [Row([Cell("Table cell")])]

    class DummyDoc:
        def __init__(self, *_a, **_k):
            self.paragraphs = [Para(" First para "), Para("")]
            self.tables = [Table()]

    # Inject fake docx module so `from docx import Document` works
    import sys
    import types

    fake_docx = types.SimpleNamespace(Document=DummyDoc)
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    docx = tmp_path / "f.docx"
    docx.write_text("irrelevant", encoding="utf-8")

    out = WordExtractor().extract_paragraphs(docx, options)
    # Two pieces: paragraph + table cell
    assert any("First para" in p for p in out)
    assert any("Table cell" in p for p in out)


def test_word_extractor_no_preprocess_split(monkeypatch, tmp_path: Path):
    # Use the same DummyDoc as above
    class Para:
        def __init__(self, text: str):
            self.text = text

    class DummyDoc:
        def __init__(self, *_a, **_k):
            self.paragraphs = [Para("A\n\nB"), Para("C")]  # keep blank-line split
            self.tables = []

    import sys
    import types

    fake_docx = types.SimpleNamespace(Document=DummyDoc)
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    docx = tmp_path / "f.docx"
    docx.write_text("irrelevant", encoding="utf-8")

    opts = ExtractionOptions(preprocess_text=False)
    out = WordExtractor().extract_paragraphs(docx, opts)
    # naive split keeps A and B and C
    assert any(x == "A\n\nB" or x == "A" for x in out) or len(out) >= 2


def test_word_extractor_doc_success_via_antiword(
    monkeypatch, tmp_path: Path, options: ExtractionOptions
):
    doc = tmp_path / "f.doc"
    doc.write_bytes(b"fake")

    import shutil as _shutil
    import subprocess as _sub

    monkeypatch.setattr(_shutil, "which", lambda _: "/usr/bin/antiword")

    class DummyCompleted:
        def __init__(self):
            self.stdout = "Alpha\n\nBeta"

    def _run(*_a, **_k):
        return DummyCompleted()

    monkeypatch.setattr(_sub, "run", _run)

    out = WordExtractor().extract_paragraphs(doc, options)
    assert any("Alpha" in p for p in out)
