"""Provider interfaces and implementations for LLM backends.

This module exposes a common interface for text generation and concrete
implementations for different backends (OpenAI, local llama-cpp).
"""

from .base import LLMProvider
from .factory import ProviderFactory
from .local_provider import LocalLlamaProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "LocalLlamaProvider",
]
