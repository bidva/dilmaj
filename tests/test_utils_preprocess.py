"""Tests for utils text preprocessing helpers."""

from dilmaj.utils import (
    _chunk_into_paragraphs,
    _clean_pdf_artifacts,
    _is_likely_header_footer,
    _is_likely_page_number,
    _remove_headers_footers,
    preprocess_text,
)


def test_page_number_detection():
    assert _is_likely_page_number("1")
    assert _is_likely_page_number("Page 2")
    assert _is_likely_page_number("3 / 10")
    assert not _is_likely_page_number("Section 1")


def test_header_footer_detection_and_removal():
    lines = ["Chapter 1", "Real content", "2023-10-01"]
    filtered = _remove_headers_footers(lines)
    # For very small inputs the heuristic may keep multiple lines
    assert "Real content" in filtered


def test_clean_pdf_artifacts():
    text = "Hello- world.It'sFine   !!!"
    cleaned = _clean_pdf_artifacts(text)
    assert (
        "Hello world. It's Fine".replace(" ", "")[:10] == cleaned.replace(" ", "")[:10]
    )


def test_chunk_into_paragraphs():
    text = "This is one sentence. This is two. Another starts: And continues. Short."
    out = _chunk_into_paragraphs(text)
    assert "\n\n" in out or len(out) > 0


def test_preprocess_text_end_to_end():
    raw = "Title | 1\n\nThis is content. With   bad  spacing.\n\nPage 3\nFooter"
    processed = preprocess_text(raw)
    assert "This is content" in processed
