"""Tests for the processor module (paragraph-oriented)."""

import asyncio
import json
from pathlib import Path
from typing import List

import pytest

from dilmaj.config import Config
from dilmaj.exceptions import PDFProcessingError
from dilmaj.processor import PDFProcessor
from dilmaj.providers.base import LLMProvider, ProviderConfig


class FakeProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(ProviderConfig(model="fake"))
        self.requests: List[str] = []

    def init(self) -> None:  # pragma: no cover - not used
        pass

    async def ainvoke(self, prompt: str) -> str:
        self.requests.append(prompt)
        return f"Processed: {prompt[:12]}"


def test_select_extractor_by_suffix(tmp_path: Path):
    proc = PDFProcessor(Config(), init_llm=False)
    # pdf
    pdf_ex = proc._select_extractor(tmp_path / "file.pdf")
    assert pdf_ex.supports(tmp_path / "file.pdf") is True
    # docx
    docx_ex = proc._select_extractor(tmp_path / "file.docx")
    assert docx_ex.supports(tmp_path / "file.docx") is True


def test_extract_paragraphs_no_content_raises(monkeypatch, tmp_path: Path):
    proc = PDFProcessor(Config(), init_llm=False)

    class DummyExtractor:
        def supports(self, p: Path) -> bool:
            return True

        def extract_paragraphs(self, p: Path, options) -> List[str]:  # type: ignore[override]
            return [" ", "\n\n"]

    monkeypatch.setattr(proc, "extractors", [DummyExtractor()])

    with pytest.raises(PDFProcessingError):
        proc.extract_paragraphs(tmp_path / "any.txt")


def test_process_pages_async_saves_outputs(monkeypatch, tmp_path: Path):
    # Speed up: avoid real sleeping
    async def _noop_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    cfg = Config(rate_limit_rpm=10_000, concurrent_requests=2)
    proc = PDFProcessor(cfg, init_llm=False)
    fake = FakeProvider()
    proc.provider = fake

    pages = ["First paragraph.", "", "Second paragraph!"]
    outdir = tmp_path / "out"
    outdir.mkdir()

    completed_counts: List[int] = []

    def _progress(c: int) -> None:
        completed_counts.append(c)

    results = proc.process_pages_async(pages, outdir, progress_callback=_progress)

    # Only non-empty paragraphs processed
    assert len(results) == 2
    assert completed_counts[-1] == 2

    # Files saved
    json1 = outdir / "paragraph_001.json"
    json2 = outdir / "paragraph_003.json"
    txt1 = outdir / "paragraph_001_processed.txt"
    txt2 = outdir / "paragraph_003_processed.txt"
    assert json1.exists() and json2.exists()
    assert txt1.exists() and txt2.exists()

    # Summary and combined files
    summary = outdir / "processing_summary.json"
    combined = outdir / "combined_processed.txt"
    assert summary.exists() and combined.exists()

    data = json.loads(summary.read_text())
    assert data["successful_paragraphs"] == 2
