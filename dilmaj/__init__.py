"""PDF Translator CLI package."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Re-export provider interfaces for convenience
from .providers import LLMProvider, OpenAIProvider, ProviderFactory

__all__ = [
    "__version__",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderFactory",
]
