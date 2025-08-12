"""Base interface for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    """Normalized provider configuration.

    This mirrors a subset of pdf_translator.config.Config relevant to
    providers, so providers don't depend on the app-level Config directly.
    """

    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    # Local model specifics
    model_path: Optional[str] = None
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    # Extra fields for future providers (e.g., Bedrock):
    # region: Optional[str] = None
    # profile: Optional[str] = None


class LLMProvider(abc.ABC):
    """Abstract interface every provider must implement."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize the underlying client/session. Called once."""

    @abc.abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        """Asynchronously generate text for a given prompt.

        Returns the raw text response. Implementations should handle any
        backend-specific formatting and errors.
        """

    def invoke(self, prompt: str) -> str:
        """Optional sync helper for providers that are only sync
        (e.g., llama-cpp)."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook."""
        return None
