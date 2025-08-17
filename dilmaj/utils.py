"""Utility functions for PDF Translator."""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_api_key() -> str:
    """Validate OpenAI API key from environment variables.

    Args:
        model_type: Type of model ("openai" by default)

    Returns:
        The valid API key string

    Raises:
        ConfigurationError: If API key is missing for OpenAI models
    """

    api_key = os.getenv("OPENAI_API_KEY")

    # Check if API key exists
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set your OpenAI API key in the .env file or environment."
        )

    # Remove whitespace
    api_key = api_key.strip()

    # Check if API key is empty after stripping whitespace
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is empty. "
            "Please set a valid OpenAI API key in the .env file or environment."
        )

    # Check for common placeholder values
    placeholder_values = {
        "your_openai_api_key_here",
        "your_api_key_here",
        "sk-your-api-key-here",
        "replace_with_your_api_key",
        "your-openai-api-key",
        "put_your_api_key_here",
    }

    if api_key.lower() in placeholder_values:
        raise ConfigurationError(
            f"OPENAI_API_KEY appears to be a placeholder value: '{api_key}'. "
            "Please replace it with your actual OpenAI API key from "
            "https://platform.openai.com/account/api-keys"
        )

    # Basic format validation - OpenAI API keys should start with 'sk-'
    if not api_key.startswith("sk-"):
        logger.warning(
            "API key does not start with 'sk-'. "
            "Please verify it's a valid OpenAI API key."
        )

    # Basic length validation - OpenAI API keys are typically 51 characters
    if len(api_key) < 20:
        logger.warning(
            f"API key appears unusually short ({len(api_key)} characters). "
            f"Please verify it's complete."
        )

    return api_key


def preprocess_text(
    text: str, remove_headers_footers: bool = True, chunk_paragraphs: bool = True
) -> str:
    """Pre-clean text extracted from PDF before processing.

    Args:
        text: Raw text content from PDF
        remove_headers_footers: Whether to remove likely headers/footers
        chunk_paragraphs: Whether to properly chunk text into paragraphs

    Returns:
        Cleaned and preprocessed text
    """
    if not text or not text.strip():
        return ""

    # Split into lines for processing
    lines = text.split("\n")
    processed_lines = []

    # Remove headers/footers and page numbers
    if remove_headers_footers:
        lines = _remove_headers_footers(lines)

    # Process each line
    for line in lines:
        # Skip empty lines initially
        if not line.strip():
            continue

        # Remove excessive whitespace
        cleaned_line = " ".join(line.split())

        # Skip lines that are likely page numbers or navigation
        if _is_likely_page_number(cleaned_line):
            continue

        # Skip lines that are likely headers/footers we missed
        if _is_likely_header_footer(cleaned_line):
            continue

        processed_lines.append(cleaned_line)

    # Join lines back together
    text = " ".join(processed_lines)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation repetitions
    text = re.sub(r"([.!?])\1+", r"\1", text)

    # Clean up common PDF artifacts
    text = _clean_pdf_artifacts(text)

    # Chunk into proper paragraphs if requested
    if chunk_paragraphs:
        text = _chunk_into_paragraphs(text)

    return text.strip()


def _remove_headers_footers(lines: List[str]) -> List[str]:
    """Remove likely headers and footers from lines.

    Args:
        lines: List of text lines

    Returns:
        Filtered list of lines
    """
    if len(lines) <= 3:
        return lines

    # Skip first and last few lines if they look like headers/footers
    start_idx = 0
    end_idx = len(lines)

    # Check first few lines for headers
    for i in range(min(3, len(lines))):
        line = lines[i].strip()
        if line and not _is_likely_header_footer(line):
            start_idx = i
            break
        start_idx = i + 1

    # Check last few lines for footers
    for i in range(len(lines) - 1, max(len(lines) - 4, -1), -1):
        line = lines[i].strip()
        if line and not _is_likely_header_footer(line):
            end_idx = i + 1
            break
        end_idx = i

    return lines[start_idx:end_idx]


def _is_likely_page_number(text: str) -> bool:
    """Check if text line is likely just a page number.

    Args:
        text: Text to check

    Returns:
        True if likely a page number
    """
    text = text.strip()

    # Simple page number patterns
    if re.match(r"^\d+$", text):  # Just a number
        return True

    if re.match(r"^Page\s+\d+$", text, re.IGNORECASE):  # "Page 1"
        return True

    if re.match(r"^\d+\s*/\s*\d+$", text):  # "1 / 10"
        return True

    if re.match(r"^\|\s*\d+\s*\|$", text):  # "| 1 |"
        return True

    return False


def _is_likely_header_footer(text: str) -> bool:
    """Check if text line is likely a header or footer.

    Args:
        text: Text to check

    Returns:
        True if likely a header or footer
    """
    text = text.strip().lower()

    # Too short to be meaningful content
    if len(text) < 3:
        return True

    # Common header/footer indicators
    header_footer_patterns = [
        r"^chapter\s+\d+",
        r"^section\s+\d+",
        r"^\d{4}-\d{2}-\d{2}",  # dates
        r"^copyright",
        r"^©",
        r"confidential",
        r"proprietary",
        r"^draft",
        r"^version\s+\d",
        r"^\w+\s*\|\s*\d+",  # "Title | 1"
        r"^\d+\s*\|\s*\w+",  # "1 | Title"
    ]

    for pattern in header_footer_patterns:
        if re.search(pattern, text):
            return True

    # Lines with only symbols or repeated characters
    if re.match(r"^[^\w\s]*$", text) and len(text) > 1:
        return True

    # Lines that are mostly punctuation
    punctuation_ratio = len(re.findall(r"[^\w\s]", text)) / len(text)
    if punctuation_ratio > 0.5:
        return True

    return False


def _clean_pdf_artifacts(text: str) -> str:
    """Clean common PDF extraction artifacts.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove soft hyphens and fix broken words
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Fix common PDF extraction issues
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Fix missing spaces
    text = re.sub(r"(\w)\.(\w)", r"\1. \2", text)  # Fix missing space after period

    # Remove excessive punctuation
    text = re.sub(r"[.]{3,}", "...", text)  # Multiple dots to ellipsis
    text = re.sub(r"[-]{3,}", "---", text)  # Multiple dashes

    # Clean up quotes and apostrophes
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r"[" "′]", "'", text)

    return text


def _chunk_into_paragraphs(text: str) -> str:
    """Chunk text into proper paragraphs.

    Args:
        text: Text to chunk

    Returns:
        Text organized into paragraphs
    """
    # Split on sentence endings followed by capital letters
    # (likely new sentences/paragraphs)
    sentences = re.split(r"([.!?]+\s+)(?=[A-Z])", text)

    # Group sentences into paragraphs (rough heuristic)
    paragraphs = []
    current_paragraph = []

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""

        current_paragraph.append(sentence + delimiter)

        # End paragraph on certain conditions
        if (
            len(current_paragraph) >= 3
            or len(" ".join(current_paragraph)) > 200  # At least 3 sentences
            or sentence.strip().endswith((":", "?", "!"))  # Or longer than 200 chars
        ):  # Or ends with certain punctuation
            paragraphs.append(" ".join(current_paragraph).strip())
            current_paragraph = []

    # Add any remaining content
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph).strip())

    # Join paragraphs with double newlines
    return "\n\n".join(p for p in paragraphs if p.strip())
