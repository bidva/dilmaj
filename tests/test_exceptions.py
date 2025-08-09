"""Tests for exceptions module."""

import pytest
from pdf_translator.exceptions import (
    PDFTranslatorError,
    PDFProcessingError,
    GPTProcessingError,
    RateLimitError,
    ConfigurationError,
)


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_base_exception(self):
        """Test base PDFTranslatorError exception."""
        error = PDFTranslatorError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_pdf_processing_error(self):
        """Test PDFProcessingError exception."""
        error = PDFProcessingError("PDF processing failed")
        assert str(error) == "PDF processing failed"
        assert isinstance(error, PDFTranslatorError)
        assert isinstance(error, Exception)
    
    def test_gpt_processing_error(self):
        """Test GPTProcessingError exception."""
        error = GPTProcessingError("GPT processing failed")
        assert str(error) == "GPT processing failed"
        assert isinstance(error, PDFTranslatorError)
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, PDFTranslatorError)
        assert isinstance(error, Exception)
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, PDFTranslatorError)
        assert isinstance(error, Exception)
