#!/usr/bin/env python3
"""Test script to verify text preprocessing functionality."""

import sys
from pathlib import Path

from pdf_translator.utils import preprocess_text

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_text_preprocessing() -> None:
    """Test the text preprocessing functionality."""

    # Test data with common PDF extraction issues
    raw_text = """
    Chapter 1

    This is a sample text with multiple    spaces    and
    some- broken words that were
    hyphenated across lines.

    This is another paragraph with proper punctuation. It should be cleaned up!

    Page 1

    Some more text here with excessive...........dots and weird formatting.

    Footer Text | Page 1 | Confidential
    """

    print("=== Original Text ===")
    print(repr(raw_text))

    # Test with preprocessing enabled
    cleaned_text = preprocess_text(
        raw_text, remove_headers_footers=True, chunk_paragraphs=True
    )

    print("\n=== Preprocessed Text ===")
    print(repr(cleaned_text))

    # Test with preprocessing disabled components
    no_header_removal = preprocess_text(
        raw_text, remove_headers_footers=False, chunk_paragraphs=True
    )

    print("\n=== No Header/Footer Removal ===")
    print(repr(no_header_removal))

    # Test with no paragraph chunking
    no_chunking = preprocess_text(
        raw_text, remove_headers_footers=True, chunk_paragraphs=False
    )

    print("\n=== No Paragraph Chunking ===")
    print(repr(no_chunking))

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_text_preprocessing()
