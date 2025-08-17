"""Provider interfaces and implementations for LLM backends.

This module exposes a common interface for text generation and the OpenAI
backend implementation.
"""

from .base import LLMProvider
from .factory import ProviderFactory
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ProviderFactory",
    "OpenAIProvider",
]
