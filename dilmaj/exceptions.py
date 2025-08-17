"""Custom exceptions for PDF Translator."""


class PDFTranslatorError(Exception):
    """Base exception for PDF Translator."""

    pass


class PDFProcessingError(PDFTranslatorError):
    """Exception raised when PDF processing fails."""

    pass


class GPTProcessingError(PDFTranslatorError):
    """Exception raised when GPT processing fails."""

    pass


class RateLimitError(PDFTranslatorError):
    """Exception raised when rate limit is exceeded."""

    pass


class ConfigurationError(PDFTranslatorError):
    """Exception raised when configuration is invalid."""

    pass
